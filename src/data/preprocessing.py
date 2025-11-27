"""
Data preprocessing and normalization utilities.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, LoggerMixin


class DataPreprocessor(LoggerMixin):
    """Preprocessing pipeline for power grid fault data."""
    
    def __init__(self, config: Config = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.scaler = None
        self.feature_names = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
    
    def normalize_data(
        self,
        X: np.ndarray,
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize input data.
        
        Args:
            X: Input data of shape (n_samples, n_timesteps, n_features)
            method: Normalization method ('standard' or 'minmax')
            fit: If True, fit the scaler. If False, use existing scaler
            
        Returns:
            Normalized data
        """
        original_shape = X.shape
        
        # Ensure float64 for numerical stability
        X = X.astype(np.float64)
        
        # Reshape to 2D for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            X_normalized = self.scaler.fit_transform(X_reshaped)
            self.logger.info(f"Fitted {method} scaler on data")
        else:
            X_normalized = self.scaler.transform(X_reshaped)
        
        # Reshape back to original shape
        X_normalized = X_normalized.reshape(original_shape).astype(np.float64)
        
        return X_normalized
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = None,
        val_split: float = None,
        test_split: float = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Label array
            train_split: Training set ratio
            val_split: Validation set ratio
            test_split: Test set ratio
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if train_split is None:
            train_split = self.config.get('data.train_split', 0.7)
        if val_split is None:
            val_split = self.config.get('data.val_split', 0.15)
        if test_split is None:
            test_split = self.config.get('data.test_split', 0.15)
        
        # Verify splits sum to 1
        total = train_split + val_split + test_split
        if not np.isclose(total, 1.0):
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_split,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_ratio = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.02,
        num_augmented: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation by adding noise and scaling.
        
        Args:
            X: Input data
            y: Labels
            noise_level: Standard deviation of Gaussian noise
            num_augmented: Number of augmented samples per original sample
            
        Returns:
            Augmented (X, y)
        """
        if num_augmented is None:
            num_augmented = 1
        
        X_augmented_list = [X]
        y_augmented_list = [y]
        
        for _ in range(num_augmented):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise * np.std(X, axis=(0, 1))
            
            # Random scaling
            scale_factor = np.random.uniform(0.95, 1.05, (X.shape[0], 1, X.shape[2]))
            X_scaled = X * scale_factor
            
            X_augmented_list.extend([X_noisy, X_scaled])
            y_augmented_list.extend([y, y])
        
        X_augmented = np.concatenate(X_augmented_list, axis=0)
        y_augmented = np.concatenate(y_augmented_list, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        self.logger.info(f"Data augmented: {X.shape} -> {X_augmented.shape}")
        
        return X_augmented, y_augmented
    
    def extract_features(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from raw signals.
        
        Args:
            X: Input data of shape (n_samples, n_timesteps, n_features)
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Time-domain features
        features['mean'] = np.mean(X, axis=1)
        features['std'] = np.std(X, axis=1)
        features['max'] = np.max(X, axis=1)
        features['min'] = np.min(X, axis=1)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(X**2, axis=1))
        
        # Peak features
        features['peak_to_peak'] = features['range']
        features['crest_factor'] = features['max'] / (features['rms'] + 1e-10)
        
        # Frequency-domain features (simplified)
        fft = np.fft.fft(X, axis=1)
        magnitude = np.abs(fft)
        features['spectral_energy'] = np.sum(magnitude**2, axis=1)
        
        self.logger.info(f"Extracted {len(features)} feature types")
        
        return features
    
    def preprocess_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        augment: bool = False,
        split: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            X: Input features
            y: Labels
            normalize: Whether to normalize data
            augment: Whether to apply data augmentation
            split: Whether to split into train/val/test
            
        Returns:
            Dictionary containing processed data
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        result = {}
        
        # Data augmentation
        if augment:
            X, y = self.augment_data(X, y)
        
        # Train/val/test split
        if split:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Normalize
            if normalize:
                X_train = self.normalize_data(X_train, fit=True)
                X_val = self.normalize_data(X_val, fit=False)
                X_test = self.normalize_data(X_test, fit=False)
            
            result['X_train'] = X_train
            result['X_val'] = X_val
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_val'] = y_val
            result['y_test'] = y_test
        else:
            if normalize:
                X = self.normalize_data(X, fit=True)
            
            result['X'] = X
            result['y'] = y
        
        self.logger.info("Preprocessing pipeline completed")
        
        return result


def load_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load dataset from file.
    
    Args:
        file_path: Path to dataset file (.npz)
        
    Returns:
        Tuple of (X, y, metadata)
    """
    data = np.load(file_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    
    metadata = {
        'fault_types': data.get('fault_types', None),
        'sampling_rate': data.get('sampling_rate', None),
        'num_points': data.get('num_points', None),
        'feature_names': data.get('feature_names', None)
    }
    
    return X, y, metadata


if __name__ == "__main__":
    # Test preprocessing
    from src.data.data_generator import GridFaultGenerator
    
    print("Generating test data...")
    generator = GridFaultGenerator()
    X, y = generator.generate_dataset(num_samples=1000)
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    print("\nTesting normalization...")
    X_norm = preprocessor.normalize_data(X)
    print(f"Normalized data - Mean: {X_norm.mean():.4f}, Std: {X_norm.std():.4f}")
    
    print("\nTesting data split...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("\nTesting full pipeline...")
    result = preprocessor.preprocess_pipeline(X, y, normalize=True, augment=False, split=True)
    for key, value in result.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✓ Preprocessing tests completed successfully!")
