"""
Test data generation functionality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data import GridFaultGenerator
from src.utils import get_config


def test_fault_generator_initialization():
    """Test fault generator initialization."""
    generator = GridFaultGenerator()
    
    assert generator is not None
    assert len(generator.fault_types) == 12
    assert generator.num_classes == 12
    assert generator.sampling_rate > 0
    assert generator.num_points > 0


def test_normal_operation_generation():
    """Test normal operation signal generation."""
    generator = GridFaultGenerator()
    signal = generator.generate_normal_operation()
    
    assert signal.shape == (generator.num_points, 6)
    assert np.all(np.isfinite(signal))


def test_fault_sample_generation():
    """Test fault sample generation for all fault types."""
    generator = GridFaultGenerator()
    
    for fault_type in generator.fault_types:
        signal = generator.generate_fault_sample(fault_type)
        
        assert signal.shape == (generator.num_points, 6)
        assert np.all(np.isfinite(signal))


def test_dataset_generation():
    """Test dataset generation."""
    generator = GridFaultGenerator()
    
    num_samples = 120  # Divisible by 12
    X, y = generator.generate_dataset(num_samples=num_samples, balanced=True)
    
    assert X.shape == (num_samples, generator.num_points, 6)
    assert y.shape == (num_samples,)
    assert len(np.unique(y)) == generator.num_classes
    
    # Check balanced distribution
    for class_idx in range(generator.num_classes):
        count = np.sum(y == class_idx)
        assert count == num_samples // generator.num_classes


def test_dataset_saving_loading(tmp_path):
    """Test dataset saving and loading."""
    generator = GridFaultGenerator()
    
    X, y = generator.generate_dataset(num_samples=100)
    
    # Save
    save_path = tmp_path / "test_dataset.npz"
    generator.save_dataset(X, y, str(save_path))
    
    assert save_path.exists()
    
    # Load
    from src.data import load_dataset
    X_loaded, y_loaded, metadata = load_dataset(str(save_path))
    
    assert np.array_equal(X, X_loaded)
    assert np.array_equal(y, y_loaded)
    assert metadata['fault_types'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
