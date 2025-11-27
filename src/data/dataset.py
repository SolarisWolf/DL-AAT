"""
Dataset utilities for loading and managing training data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import LoggerMixin


class FaultDataset(Dataset, LoggerMixin):
    """PyTorch Dataset for power grid fault data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Feature array of shape (n_samples, n_timesteps, n_features)
            y: Label array of shape (n_samples,)
            transform: Optional transform to apply to data
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
        self.logger.info(f"Initialized dataset with {len(self)} samples")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = FaultDataset(X_train, y_train)
    val_dataset = FaultDataset(X_val, y_val)
    test_dataset = FaultDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_and_prepare_data(
    data_path: str,
    batch_size: int = 32,
    normalize: bool = True
) -> dict:
    """
    Load dataset and prepare data loaders.
    
    Args:
        data_path: Path to dataset file
        batch_size: Batch size
        normalize: Whether to normalize data
        
    Returns:
        Dictionary containing loaders and metadata
    """
    from src.data.preprocessing import load_dataset, DataPreprocessor
    
    # Load data
    X, y, metadata = load_dataset(data_path)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_pipeline(
        X, y,
        normalize=normalize,
        augment=False,
        split=True
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        result['X_train'], result['y_train'],
        result['X_val'], result['y_val'],
        result['X_test'], result['y_test'],
        batch_size=batch_size
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'metadata': metadata,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Test dataset utilities
    print("Testing dataset utilities...")
    
    # Generate dummy data
    X = np.random.randn(100, 200, 6)
    y = np.random.randint(0, 12, 100)
    
    # Test PyTorch dataset
    dataset = FaultDataset(X, y)
    print(f"Dataset size: {len(dataset)}")
    
    x_sample, y_sample = dataset[0]
    print(f"Sample shape: {x_sample.shape}, Label: {y_sample}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
    
    print("\n✓ Dataset utilities tests completed!")
