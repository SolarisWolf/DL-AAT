"""
Real-time fault detection system.
"""
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, setup_logger
from src.models import ModelManager
from src.data import DataPreprocessor
from src.iot.sensor_simulator import SensorNetwork
from src.iot.mqtt_client import get_mqtt_client


class RealTimeFaultDetector:
    """Real-time fault detection system using trained CNN model."""
    
    def __init__(
        self,
        model_path: str,
        config: Config = None,
        use_mqtt: bool = False
    ):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            config: Configuration object
            use_mqtt: Whether to use MQTT for communication
        """
        self.config = config or get_config()
        self.logger = setup_logger("detector")
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = ModelManager.load_model(model_path)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.config)
        
        # Initialize sensor network
        self.sensor_network = SensorNetwork(self.config)
        
        # Initialize MQTT client if needed
        self.mqtt_client = None
        if use_mqtt:
            self.mqtt_client = get_mqtt_client(self.config, mock=True)
            self.mqtt_client.connect()
        
        # Detection parameters
        self.confidence_threshold = self.config.get('detection.confidence_threshold', 0.95)
        self.buffer_size = self.config.get('detection.buffer_size', 200)
        
        # Fault types
        self.fault_types = self.config.get('data.fault_types', [
            "Normal", "AG", "BG", "CG", "AB", "BC", "CA",
            "ABG", "BCG", "CAG", "ABC", "ABCG"
        ])
        
        # Statistics
        self.detection_count = 0
        self.detection_times = []
        
        self.logger.info("Real-time fault detector initialized")
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess signal for model input.
        
        Args:
            signal: Raw signal array of shape (timesteps, features)
            
        Returns:
            Preprocessed signal ready for model
        """
        # Add batch dimension
        signal = signal[np.newaxis, ...]
        
        # Normalize
        signal = self.preprocessor.normalize_data(signal, fit=False)
        
        return signal
    
    def detect_fault(self, signal: np.ndarray) -> Dict:
        """
        Detect fault from signal window.
        
        Args:
            signal: Signal window of shape (timesteps, features)
            
        Returns:
            Detection result dictionary
        """
        start_time = time.time()
        
        # Preprocess
        processed_signal = self.preprocess_signal(signal)
        
        # Predict
        predictions = self.model.predict(processed_signal, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Calculate detection time
        detection_time = (time.time() - start_time) * 1000  # milliseconds
        self.detection_times.append(detection_time)
        self.detection_count += 1
        
        # Determine alert level
        if confidence >= self.confidence_threshold:
            if self.fault_types[predicted_class] == "Normal":
                alert_level = "INFO"
            elif predicted_class in [1, 2, 3, 4, 5, 6]:  # Single/double line faults
                alert_level = "WARNING"
            else:  # Three-phase faults
                alert_level = "CRITICAL"
        else:
            alert_level = "UNCERTAIN"
        
        result = {
            'predicted_fault': self.fault_types[predicted_class],
            'confidence': float(confidence),
            'alert_level': alert_level,
            'detection_time_ms': detection_time,
            'timestamp': time.time(),
            'class_probabilities': {
                fault_type: float(prob)
                for fault_type, prob in zip(self.fault_types, predictions[0])
            }
        }
        
        return result
    
    def process_sensor_data(self, fault_type: str = None) -> Dict:
        """
        Get sensor data and perform detection.
        
        Args:
            fault_type: Simulated fault type (None for random)
            
        Returns:
            Detection result
        """
        # Get signal window from sensors
        if fault_type is None:
            # Random fault type for simulation
            fault_type = np.random.choice(self.fault_types)
        
        signal = self.sensor_network.get_signal_window(fault_type)
        
        # Detect fault
        result = self.detect_fault(signal)
        result['actual_fault'] = fault_type
        result['correct_prediction'] = (result['predicted_fault'] == fault_type)
        
        return result
    
    def start_monitoring(self, duration: float = 60, interval: float = 1.0):
        """
        Start continuous monitoring.
        
        Args:
            duration: Monitoring duration in seconds
            interval: Interval between detections in seconds
        """
        self.logger.info(f"Starting continuous monitoring for {duration}s...")
        
        self.sensor_network.start_monitoring()
        
        start_time = time.time()
        detection_results = []
        
        try:
            while (time.time() - start_time) < duration:
                # Process sensor data
                result = self.process_sensor_data()
                detection_results.append(result)
                
                # Log result
                self.logger.info(
                    f"Detection #{self.detection_count}: "
                    f"Predicted={result['predicted_fault']}, "
                    f"Actual={result['actual_fault']}, "
                    f"Confidence={result['confidence']:.3f}, "
                    f"Time={result['detection_time_ms']:.2f}ms, "
                    f"Alert={result['alert_level']}"
                )
                
                # Publish to MQTT if enabled
                if self.mqtt_client:
                    self.mqtt_client.publish(result)
                
                # Check for high-priority alerts
                if result['alert_level'] == 'CRITICAL':
                    self.logger.warning(
                        f"⚠️  CRITICAL FAULT DETECTED: {result['predicted_fault']} "
                        f"(Confidence: {result['confidence']:.1%})"
                    )
                
                # Wait for next interval
                time.sleep(interval)
        
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        
        finally:
            self.sensor_network.stop_monitoring()
            
            # Print statistics
            self._print_statistics(detection_results)
    
    def _print_statistics(self, results: List[Dict]):
        """Print detection statistics."""
        if not results:
            return
        
        print("\n" + "="*70)
        print("DETECTION STATISTICS")
        print("="*70)
        
        # Overall stats
        total_detections = len(results)
        correct_predictions = sum(1 for r in results if r['correct_prediction'])
        accuracy = correct_predictions / total_detections if total_detections > 0 else 0
        
        print(f"\nOverall Performance:")
        print(f"  Total detections: {total_detections}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Accuracy: {accuracy:.2%}")
        
        # Timing stats
        avg_time = np.mean(self.detection_times)
        max_time = np.max(self.detection_times)
        min_time = np.min(self.detection_times)
        
        print(f"\nTiming Performance:")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Min: {min_time:.2f} ms")
        print(f"  Max: {max_time:.2f} ms")
        print(f"  Target: <20 ms")
        
        if avg_time < 20:
            print(f"  ✓ MEETS TARGET")
        else:
            print(f"  ✗ DOES NOT MEET TARGET")
        
        # Alert level distribution
        alert_levels = [r['alert_level'] for r in results]
        print(f"\nAlert Distribution:")
        for level in ["INFO", "WARNING", "CRITICAL", "UNCERTAIN"]:
            count = alert_levels.count(level)
            percentage = count / len(alert_levels) * 100 if alert_levels else 0
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        print("="*70 + "\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Real-time power grid fault detection")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--duration', type=float, default=60,
                       help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Detection interval in seconds')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--mqtt', action='store_true',
                       help='Enable MQTT communication')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Create detector
    detector = RealTimeFaultDetector(
        model_path=args.model,
        config=config,
        use_mqtt=args.mqtt
    )
    
    print("\n" + "="*70)
    print("REAL-TIME FAULT DETECTION SYSTEM")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Monitoring duration: {args.duration}s")
    print(f"Detection interval: {args.interval}s")
    print(f"MQTT enabled: {args.mqtt}")
    print("\nPress Ctrl+C to stop monitoring early\n")
    print("="*70 + "\n")
    
    # Start monitoring
    detector.start_monitoring(duration=args.duration, interval=args.interval)
    
    print("\n✓ Monitoring completed successfully!")


if __name__ == "__main__":
    main()
