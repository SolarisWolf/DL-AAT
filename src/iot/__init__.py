"""Package initialization for IoT module.

Avoid importing heavy submodules (like real-time detector and models)
at package import time to prevent unnecessary dependencies during
lightweight operations (e.g., running the dashboard).
"""
from .sensor_simulator import IoTSensor, SensorNetwork
from .mqtt_client import MQTTClient, MockMQTTClient, get_mqtt_client

__all__ = [
    'IoTSensor',
    'SensorNetwork',
    'MQTTClient',
    'MockMQTTClient',
    'get_mqtt_client'
]
