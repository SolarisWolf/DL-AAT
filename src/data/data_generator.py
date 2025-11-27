"""
Power grid fault simulation and data generation.

This module simulates various types of electrical faults in a power distribution
network and generates synthetic training data for the fault detection system.
"""
import numpy as np
from typing import Tuple, List, Dict
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, LoggerMixin


class GridFaultGenerator(LoggerMixin):
    """
    Generator for simulating power grid faults and creating training datasets.
    
    Simulates 12 different fault types:
    - Normal (no fault)
    - Single phase to ground: AG, BG, CG
    - Line to line: AB, BC, CA
    - Double line to ground: ABG, BCG, CAG
    - Three phase: ABC, ABCG
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize fault generator with configuration.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or get_config()
        
        # Get configuration parameters
        self.sampling_rate = self.config.get('data.sampling_rate', 1000)
        self.signal_duration = self.config.get('data.signal_duration', 0.2)
        self.num_points = self.config.get('data.num_points', 200)
        self.noise_level = self.config.get('data.noise_level', 0.05)
        
        # Grid parameters
        self.voltage_nominal = self.config.get('grid.voltage_nominal', 11000)
        self.frequency = self.config.get('grid.frequency', 50)
        self.phases = self.config.get('grid.phases', 3)
        
        # Fault types
        self.fault_types = self.config.get('data.fault_types', [
            "Normal", "AG", "BG", "CG", "AB", "BC", "CA",
            "ABG", "BCG", "CAG", "ABC", "ABCG"
        ])
        self.num_classes = len(self.fault_types)
        
        # Fault parameters ranges
        self.fault_resistance_range = self.config.get('data.fault_resistance_range', [0.001, 100.0])
        self.fault_inception_angle_range = self.config.get('data.fault_inception_angle_range', [0, 360])
        self.fault_location_range = self.config.get('data.fault_location_range', [0.1, 0.9])
        
        # Time array for signal generation
        self.time = np.linspace(0, self.signal_duration, self.num_points)
        
        self.logger.info(f"Initialized GridFaultGenerator with {self.num_classes} fault types")
    
    def generate_normal_operation(self) -> np.ndarray:
        """
        Generate signals for normal grid operation (no fault).
        
        Returns:
            Array of shape (num_points, 6) containing [Va, Vb, Vc, Ia, Ib, Ic]
        """
        # Phase angles for three-phase system (120 degrees apart)
        phase_angles = np.array([0, -120, -240]) * np.pi / 180
        
        # Generate three-phase voltages
        voltages = np.zeros((self.num_points, 3))
        for i in range(3):
            voltages[:, i] = self.voltage_nominal * np.sin(
                2 * np.pi * self.frequency * self.time + phase_angles[i]
            )
        
        # Generate three-phase currents (balanced load)
        # Assuming a nominal current of ~100A for 11kV system
        nominal_current = 100
        currents = np.zeros((self.num_points, 3), dtype=np.float64)
        for i in range(3):
            # Current lags voltage by ~30 degrees (power factor ~0.85)
            currents[:, i] = nominal_current * np.sin(
                2 * np.pi * self.frequency * self.time + phase_angles[i] - np.pi/6
            )
        
        # Add measurement noise
        voltages += np.random.normal(0, self.noise_level * self.voltage_nominal, voltages.shape).astype(np.float64)
        currents += np.random.normal(0, self.noise_level * nominal_current, currents.shape).astype(np.float64)
        
        # Combine voltages and currents
        signals = np.concatenate([voltages, currents], axis=1)
        
        return signals
    
    def generate_phase_to_ground_fault(self, phase: str) -> np.ndarray:
        """
        Generate single phase to ground fault (AG, BG, CG).
        
        Args:
            phase: Faulted phase ('A', 'B', or 'C')
            
        Returns:
            Array of shape (num_points, 6) containing fault signals
        """
        # Start with normal operation
        signals = self.generate_normal_operation()
        
        # Random fault parameters
        fault_resistance = np.random.uniform(*self.fault_resistance_range)
        inception_angle = np.random.uniform(*self.fault_inception_angle_range) * np.pi / 180
        fault_location = np.random.uniform(*self.fault_location_range)
        
        # Determine fault inception point
        inception_point = int(fault_location * self.num_points)
        
        # Phase index mapping
        phase_idx = {'A': 0, 'B': 1, 'C': 2}[phase]
        
        # Apply fault after inception point
        # Voltage drops significantly
        voltage_drop_factor = 0.3 + 0.3 * (fault_resistance / 100)  # 0.3 to 0.6
        signals[inception_point:, phase_idx] *= voltage_drop_factor
        
        # Current increases significantly on faulted phase
        current_increase = (10 + 40 * (1 - fault_resistance / 100))  # 10x to 50x
        signals[inception_point:, 3 + phase_idx] *= current_increase
        
        # Add transient components
        transient = 50 * np.exp(-50 * self.time[inception_point:])
        if len(transient) > 0:
            signals[inception_point:, phase_idx] += transient[:len(signals[inception_point:])] * self.voltage_nominal * 0.5
        
        return signals
    
    def generate_line_to_line_fault(self, phases: str) -> np.ndarray:
        """
        Generate line to line fault (AB, BC, CA).
        
        Args:
            phases: Faulted phases ('AB', 'BC', or 'CA')
            
        Returns:
            Array of shape (num_points, 6) containing fault signals
        """
        signals = self.generate_normal_operation()
        
        # Random fault parameters
        fault_resistance = np.random.uniform(*self.fault_resistance_range)
        inception_angle = np.random.uniform(*self.fault_inception_angle_range) * np.pi / 180
        fault_location = np.random.uniform(*self.fault_location_range)
        inception_point = int(fault_location * self.num_points)
        
        # Phase indices
        phase_map = {'A': 0, 'B': 1, 'C': 2}
        phase1_idx = phase_map[phases[0]]
        phase2_idx = phase_map[phases[1]]
        
        # Apply fault - voltages of both phases become equal
        voltage_avg = (signals[inception_point:, phase1_idx] + signals[inception_point:, phase2_idx]) / 2
        voltage_drop_factor = 0.4 + 0.4 * (fault_resistance / 100)
        
        signals[inception_point:, phase1_idx] = voltage_avg * voltage_drop_factor
        signals[inception_point:, phase2_idx] = voltage_avg * voltage_drop_factor
        
        # Currents increase on both faulted phases
        current_increase = 15 + 35 * (1 - fault_resistance / 100)
        signals[inception_point:, 3 + phase1_idx] *= current_increase
        signals[inception_point:, 3 + phase2_idx] *= current_increase
        
        return signals
    
    def generate_double_line_to_ground_fault(self, phases: str) -> np.ndarray:
        """
        Generate double line to ground fault (ABG, BCG, CAG).
        
        Args:
            phases: Faulted phases ('ABG', 'BCG', or 'CAG')
            
        Returns:
            Array of shape (num_points, 6) containing fault signals
        """
        signals = self.generate_normal_operation()
        
        # Random fault parameters
        fault_resistance = np.random.uniform(*self.fault_resistance_range)
        fault_location = np.random.uniform(*self.fault_location_range)
        inception_point = int(fault_location * self.num_points)
        
        # Phase indices (first two characters)
        phase_map = {'A': 0, 'B': 1, 'C': 2}
        phase1_idx = phase_map[phases[0]]
        phase2_idx = phase_map[phases[1]]
        
        # Severe voltage drop on both phases
        voltage_drop_factor = 0.2 + 0.2 * (fault_resistance / 100)
        signals[inception_point:, phase1_idx] *= voltage_drop_factor
        signals[inception_point:, phase2_idx] *= voltage_drop_factor
        
        # Very high currents on both phases
        current_increase = 20 + 60 * (1 - fault_resistance / 100)
        signals[inception_point:, 3 + phase1_idx] *= current_increase
        signals[inception_point:, 3 + phase2_idx] *= current_increase
        
        # Ground current (reflected in all phases)
        ground_current_effect = 1.2
        for i in range(3):
            signals[inception_point:, 3 + i] *= ground_current_effect
        
        return signals
    
    def generate_three_phase_fault(self, with_ground: bool = False) -> np.ndarray:
        """
        Generate three phase fault (ABC or ABCG).
        
        Args:
            with_ground: If True, generates ABCG fault, else ABC
            
        Returns:
            Array of shape (num_points, 6) containing fault signals
        """
        signals = self.generate_normal_operation()
        
        # Random fault parameters
        fault_resistance = np.random.uniform(*self.fault_resistance_range)
        fault_location = np.random.uniform(*self.fault_location_range)
        inception_point = int(fault_location * self.num_points)
        
        # Most severe fault - all voltages drop dramatically
        voltage_drop_factor = 0.1 + 0.15 * (fault_resistance / 100)
        signals[inception_point:, :3] *= voltage_drop_factor
        
        # Extremely high currents on all phases
        current_increase = 30 + 100 * (1 - fault_resistance / 100)
        signals[inception_point:, 3:] *= current_increase
        
        # If ground is involved, add more severe effects
        if with_ground:
            signals[inception_point:, :3] *= 0.8  # Even lower voltage
            signals[inception_point:, 3:] *= 1.3  # Even higher current
        
        return signals
    
    def generate_fault_sample(self, fault_type: str) -> np.ndarray:
        """
        Generate a single fault sample based on fault type.
        
        Args:
            fault_type: Type of fault to generate
            
        Returns:
            Array of shape (num_points, 6) containing fault signals
        """
        if fault_type == "Normal":
            return self.generate_normal_operation()
        elif fault_type in ["AG", "BG", "CG"]:
            phase = fault_type[0]
            return self.generate_phase_to_ground_fault(phase)
        elif fault_type in ["AB", "BC", "CA"]:
            return self.generate_line_to_line_fault(fault_type)
        elif fault_type in ["ABG", "BCG", "CAG"]:
            return self.generate_double_line_to_ground_fault(fault_type)
        elif fault_type == "ABC":
            return self.generate_three_phase_fault(with_ground=False)
        elif fault_type == "ABCG":
            return self.generate_three_phase_fault(with_ground=True)
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        balanced: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset with multiple fault samples.
        
        Args:
            num_samples: Total number of samples to generate
            balanced: If True, generate equal samples for each fault type
            
        Returns:
            Tuple of (X, y) where X is signals array and y is labels array
        """
        self.logger.info(f"Generating dataset with {num_samples} samples...")
        
        if balanced:
            samples_per_class = num_samples // self.num_classes
            remaining = num_samples % self.num_classes
            
            X_list = []
            y_list = []
            
            for class_idx, fault_type in enumerate(self.fault_types):
                # Generate samples for this class
                n_samples = samples_per_class + (1 if class_idx < remaining else 0)
                
                self.logger.info(f"Generating {n_samples} samples for {fault_type}")
                
                for _ in range(n_samples):
                    sample = self.generate_fault_sample(fault_type)
                    X_list.append(sample)
                    y_list.append(class_idx)
            
            X = np.array(X_list)
            y = np.array(y_list)
        else:
            # Random sampling
            X_list = []
            y_list = []
            
            for _ in range(num_samples):
                fault_type = np.random.choice(self.fault_types)
                class_idx = self.fault_types.index(fault_type)
                
                sample = self.generate_fault_sample(fault_type)
                X_list.append(sample)
                y_list.append(class_idx)
            
            X = np.array(X_list)
            y = np.array(y_list)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        self.logger.info(f"Dataset generated: X shape = {X.shape}, y shape = {y.shape}")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: str,
        metadata: Dict = None
    ):
        """
        Save dataset to disk.
        
        Args:
            X: Feature array
            y: Label array
            output_path: Path to save the dataset
            metadata: Additional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        save_dict = {
            'X': X,
            'y': y,
            'fault_types': self.fault_types,
            'sampling_rate': self.sampling_rate,
            'num_points': self.num_points,
            'feature_names': ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
        }
        
        if metadata:
            save_dict.update(metadata)
        
        # Save as compressed numpy file
        np.savez_compressed(output_path, **save_dict)
        self.logger.info(f"Dataset saved to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate power grid fault dataset")
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/fault_dataset.npz',
                       help='Output file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--balanced', action='store_true', default=True,
                       help='Generate balanced dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load configuration
    config = get_config(args.config)
    
    # Create generator
    generator = GridFaultGenerator(config)
    
    # Generate dataset
    X, y = generator.generate_dataset(
        num_samples=args.num_samples,
        balanced=args.balanced
    )
    
    # Save dataset
    generator.save_dataset(X, y, args.output)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Samples: {len(X)}")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {len(generator.fault_types)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
