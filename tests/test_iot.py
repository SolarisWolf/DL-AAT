"""
Test IoT sensor and real-time detection functionality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.iot import IoTSensor, SensorNetwork, MockMQTTClient
from src.utils import get_config


def test_iot_sensor_initialization():
    """Test IoT sensor initialization."""
    sensor = IoTSensor("V_A", "voltage")
    
    assert sensor.sensor_id == "V_A"
    assert sensor.sensor_type == "voltage"
    assert sensor.accuracy > 0
    assert len(sensor.range) == 2


def test_iot_sensor_reading():
    """Test IoT sensor reading with noise."""
    sensor = IoTSensor("V_A", "voltage")
    
    true_value = 230.0
    reading = sensor.read(true_value)
    
    assert isinstance(reading, float)
    assert sensor.range[0] <= reading <= sensor.range[1]
    # Reading should be close to true value (within reasonable noise)
    assert abs(reading - true_value) < true_value * 0.1


def test_sensor_network_initialization():
    """Test sensor network initialization."""
    network = SensorNetwork()
    
    assert len(network.sensors) == 7  # 3 voltage + 3 current + 1 temp
    assert 'V_A' in network.sensors
    assert 'I_A' in network.sensors
    assert 'temp' in network.sensors


def test_sensor_network_readings():
    """Test sensor network reading simulation."""
    network = SensorNetwork()
    
    readings = network.simulate_readings("Normal")
    
    assert 'V_A' in readings
    assert 'I_A' in readings
    assert 'temp' in readings
    assert 'fault_type' in readings
    assert 'timestamp' in readings


def test_sensor_network_signal_window():
    """Test sensor network signal window generation."""
    network = SensorNetwork()
    
    signal = network.get_signal_window("AG", window_size=200)
    
    assert signal.shape == (200, 6)
    assert np.all(np.isfinite(signal))


def test_mqtt_mock_client():
    """Test mock MQTT client."""
    client = MockMQTTClient()
    
    # Connect
    assert client.connect() == True
    assert client.connected == True
    
    # Publish
    data = {'sensor_id': 'V_A', 'value': 230.5}
    assert client.publish(data) == True
    
    # Check messages
    messages = client.get_messages()
    assert len(messages) == 1
    assert messages[0]['data'] == data
    
    # Disconnect
    client.disconnect()
    assert client.connected == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
