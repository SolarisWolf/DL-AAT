"""
MQTT client for IoT communication.
"""
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not installed. MQTT functionality will be limited.")

import json
import time
from typing import Callable, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, LoggerMixin


class MQTTClient(LoggerMixin):
    """MQTT client for publishing and subscribing to sensor data."""
    
    def __init__(self, config: Config = None):
        """
        Initialize MQTT client.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.broker = self.config.get('iot.mqtt_broker', 'localhost')
        self.port = self.config.get('iot.mqtt_port', 1883)
        self.topic = self.config.get('iot.mqtt_topic', 'smartgrid/sensors')
        
        self.client = None
        self.connected = False
        self.message_callback = None
        
        if MQTT_AVAILABLE:
            self._initialize_client()
        else:
            self.logger.warning("MQTT client not available. Install paho-mqtt package.")
    
    def _initialize_client(self):
        """Initialize MQTT client instance."""
        self.client = mqtt.Client(client_id="smartgrid_client")
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        self.logger.info(f"MQTT client initialized for broker {self.broker}:{self.port}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker."""
        if rc == 0:
            self.connected = True
            self.logger.info("Connected to MQTT broker")
            # Subscribe to topic
            client.subscribe(self.topic)
            self.logger.info(f"Subscribed to topic: {self.topic}")
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):
        """Callback for when message is received."""
        try:
            payload = json.loads(msg.payload.decode())
            self.logger.debug(f"Received message on topic {msg.topic}")
            
            # Call user-defined callback if set
            if self.message_callback:
                self.message_callback(payload)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def connect(self):
        """Connect to MQTT broker."""
        if not MQTT_AVAILABLE:
            self.logger.error("Cannot connect: MQTT not available")
            return False
        
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                self.logger.error("Connection timeout")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("Disconnected from MQTT broker")
    
    def publish(self, data: Dict, topic: str = None):
        """
        Publish data to MQTT topic.
        
        Args:
            data: Data dictionary to publish
            topic: Topic to publish to (uses default if None)
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker")
            return False
        
        topic = topic or self.topic
        
        try:
            payload = json.dumps(data)
            result = self.client.publish(topic, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Published message to {topic}")
                return True
            else:
                self.logger.error(f"Failed to publish message. Error code: {result.rc}")
                return False
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            return False
    
    def subscribe(self, topic: str = None):
        """
        Subscribe to MQTT topic.
        
        Args:
            topic: Topic to subscribe to (uses default if None)
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker")
            return False
        
        topic = topic or self.topic
        
        try:
            self.client.subscribe(topic)
            self.logger.info(f"Subscribed to topic: {topic}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to topic: {e}")
            return False
    
    def set_message_callback(self, callback: Callable[[Dict], None]):
        """
        Set callback function for received messages.
        
        Args:
            callback: Function to call when message is received
        """
        self.message_callback = callback
        self.logger.info("Message callback set")


class MockMQTTClient(LoggerMixin):
    """Mock MQTT client for testing without actual broker."""
    
    def __init__(self, config: Config = None):
        """Initialize mock client."""
        self.config = config or get_config()
        self.connected = False
        self.messages = []
        self.message_callback = None
        self.logger.info("Mock MQTT client initialized")
    
    def connect(self) -> bool:
        """Simulate connection."""
        self.connected = True
        self.logger.info("Mock: Connected to MQTT broker")
        return True
    
    def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        self.logger.info("Mock: Disconnected from MQTT broker")
    
    def publish(self, data: Dict, topic: str = None) -> bool:
        """Store published message."""
        if not self.connected:
            return False
        
        self.messages.append({
            'topic': topic,
            'data': data,
            'timestamp': time.time()
        })
        self.logger.debug(f"Mock: Published message to {topic}")
        return True
    
    def subscribe(self, topic: str = None) -> bool:
        """Simulate subscription."""
        self.logger.info(f"Mock: Subscribed to {topic}")
        return True
    
    def set_message_callback(self, callback: Callable[[Dict], None]):
        """Set callback."""
        self.message_callback = callback
    
    def get_messages(self):
        """Get all stored messages."""
        return self.messages


def get_mqtt_client(config: Config = None, mock: bool = False):
    """
    Get MQTT client instance.
    
    Args:
        config: Configuration object
        mock: If True, return mock client for testing
        
    Returns:
        MQTT client instance
    """
    if mock or not MQTT_AVAILABLE:
        return MockMQTTClient(config)
    else:
        return MQTTClient(config)


if __name__ == "__main__":
    # Test MQTT client
    print("Testing MQTT Client...")
    
    config = get_config()
    
    # Use mock client for testing
    client = get_mqtt_client(config, mock=True)
    
    # Connect
    if client.connect():
        print("✓ Connected successfully")
    
    # Publish test message
    test_data = {
        'sensor_id': 'V_A',
        'value': 230.5,
        'timestamp': time.time()
    }
    
    if client.publish(test_data):
        print("✓ Message published successfully")
    
    # Check messages (mock only)
    if isinstance(client, MockMQTTClient):
        messages = client.get_messages()
        print(f"✓ Stored {len(messages)} messages")
    
    # Disconnect
    client.disconnect()
    print("✓ Disconnected successfully")
    
    print("\n✓ MQTT Client tests completed!")
