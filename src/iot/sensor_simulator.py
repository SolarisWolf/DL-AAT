"""
IoT sensor simulation for power grid monitoring.
"""
import numpy as np
import time
import json
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, LoggerMixin
from src.data import GridFaultGenerator


class IoTSensor(LoggerMixin):
    """Simulated IoT sensor for grid monitoring."""
    
    def __init__(
        self,
        sensor_id: str,
        sensor_type: str,
        config: Config = None
    ):
        """
        Initialize IoT sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor ('voltage', 'current', 'temperature')
            config: Configuration object
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config or get_config()
        
        # Get sensor specifications
        sensor_config = self.config.get(f'iot.{sensor_type}_sensor', {})
        self.accuracy = sensor_config.get('accuracy', 0.01)
        self.range = sensor_config.get('range', [0, 100])
        
        self.is_active = False
        self.last_reading = None
    
    def read(self, true_value: float) -> float:
        """
        Simulate sensor reading with noise.
        
        Args:
            true_value: True value to measure
            
        Returns:
            Sensor reading with added noise
        """
        # Add measurement noise based on accuracy
        noise = np.random.normal(0, self.accuracy * abs(true_value))
        reading = true_value + noise
        
        # Clip to sensor range
        reading = np.clip(reading, self.range[0], self.range[1])
        
        self.last_reading = reading
        return reading
    
    def to_dict(self) -> Dict:
        """Convert sensor state to dictionary."""
        return {
            'sensor_id': self.sensor_id,
            'type': self.sensor_type,
            'active': self.is_active,
            'last_reading': self.last_reading,
            'timestamp': time.time()
        }


class SensorNetwork(LoggerMixin):
    """Network of IoT sensors for grid monitoring."""
    
    def __init__(self, config: Config = None):
        """
        Initialize sensor network.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.sensors = {}
        self.fault_generator = GridFaultGenerator(self.config)
        self.current_state = "Normal"
        self.update_interval = self.config.get('iot.sensor_update_interval', 0.1)
        
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize sensors for three-phase monitoring."""
        # Voltage sensors for each phase
        for phase in ['A', 'B', 'C']:
            sensor_id = f"V_{phase}"
            self.sensors[sensor_id] = IoTSensor(sensor_id, 'voltage', self.config)
        
        # Current sensors for each phase
        for phase in ['A', 'B', 'C']:
            sensor_id = f"I_{phase}"
            self.sensors[sensor_id] = IoTSensor(sensor_id, 'current', self.config)
        
        # Temperature sensor
        self.sensors['temp'] = IoTSensor('temp', 'temperature', self.config)
        
        self.logger.info(f"Initialized {len(self.sensors)} sensors")
    
    def simulate_readings(self, fault_type: str = "Normal") -> Dict:
        """
        Simulate sensor readings for given fault condition.
        
        Args:
            fault_type: Type of fault to simulate
            
        Returns:
            Dictionary of sensor readings
        """
        # Generate fault signals
        signals = self.fault_generator.generate_fault_sample(fault_type)
        
        # Extract current timestep (random point in signal)
        timestep = np.random.randint(0, signals.shape[0])
        current_values = signals[timestep]
        
        # Read from each sensor with type safety
        readings = {}
        
        # Voltage readings (scaled to sensor range)
        voltage_scale = self.config.get('grid.voltage_nominal', 11000) / 250  # Step down
        for i, phase in enumerate(['A', 'B', 'C']):
            sensor_id = f"V_{phase}"
            true_value = float(current_values[i]) / voltage_scale
            readings[sensor_id] = float(self.sensors[sensor_id].read(true_value))
        
        # Current readings
        for i, phase in enumerate(['A', 'B', 'C']):
            sensor_id = f"I_{phase}"
            true_value = float(current_values[3 + i])
            readings[sensor_id] = float(self.sensors[sensor_id].read(true_value))
        
        # Temperature reading (simulated equipment temperature)
        base_temp = 45  # Celsius
        temp_increase = 0 if fault_type == "Normal" else np.random.uniform(10, 30)
        readings['temp'] = self.sensors['temp'].read(base_temp + temp_increase)
        
        # Add metadata
        readings['fault_type'] = fault_type
        readings['timestamp'] = time.time()
        
        self.current_state = fault_type
        
        return readings
    
    def get_signal_window(self, fault_type: str = "Normal", window_size: int = 200) -> np.ndarray:
        """
        Get a window of sensor readings for model input.
        
        Args:
            fault_type: Type of fault to simulate
            window_size: Number of timesteps to generate
            
        Returns:
            Array of shape (window_size, 6) for model input
        """
        # Generate full signal
        signals = self.fault_generator.generate_fault_sample(fault_type)
        
        # Add sensor noise
        for i in range(6):
            sensor_type = 'voltage' if i < 3 else 'current'
            sensor_config = self.config.get(f'iot.{sensor_type}_sensor', {})
            accuracy = sensor_config.get('accuracy', 0.01)
            
            noise = np.random.normal(0, accuracy * np.std(signals[:, i]), signals.shape[0])
            signals[:, i] += noise
        
        return signals
    
    def start_monitoring(self):
        """Start continuous monitoring (placeholder for real implementation)."""
        self.logger.info("Sensor network monitoring started")
        for sensor in self.sensors.values():
            sensor.is_active = True
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.logger.info("Sensor network monitoring stopped")
        for sensor in self.sensors.values():
            sensor.is_active = False
    
    def get_status(self) -> Dict:
        """Get current network status."""
        return {
            'num_sensors': len(self.sensors),
            'active_sensors': sum(1 for s in self.sensors.values() if s.is_active),
            'current_state': self.current_state,
            'sensors': {sid: s.to_dict() for sid, s in self.sensors.items()}
        }


if __name__ == "__main__":
    # Test sensor network
    print("Testing IoT Sensor Network...")
    
    config = get_config()
    network = SensorNetwork(config)
    
    print(f"\nSensor Network Status:")
    status = network.get_status()
    print(f"  Total sensors: {status['num_sensors']}")
    print(f"  Sensor types: {list(network.sensors.keys())}")
    
    print("\nSimulating readings for different fault types:")
    for fault_type in ["Normal", "AG", "ABC", "ABCG"]:
        print(f"\n  {fault_type} fault:")
        readings = network.simulate_readings(fault_type)
        
        print(f"    Voltages: Va={readings['V_A']:.2f}V, Vb={readings['V_B']:.2f}V, Vc={readings['V_C']:.2f}V")
        print(f"    Currents: Ia={readings['I_A']:.2f}A, Ib={readings['I_B']:.2f}A, Ic={readings['I_C']:.2f}A")
        print(f"    Temperature: {readings['temp']:.2f}°C")
    
    print("\nTesting signal window generation...")
    signal = network.get_signal_window("AG", window_size=200)
    print(f"  Signal shape: {signal.shape}")
    print(f"  Signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    
    print("\n✓ IoT Sensor Network tests completed successfully!")
