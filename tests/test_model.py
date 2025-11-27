"""
Test model building and functionality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_1d_cnn_model, build_cnn_lstm_model, ModelManager
from src.utils import get_config


def test_1d_cnn_model_building():
    """Test 1D-CNN model building."""
    model = build_1d_cnn_model(input_shape=(200, 6), num_classes=12)
    
    assert model is not None
    assert model.input_shape == (None, 200, 6)
    assert model.output_shape == (None, 12)
    assert model.count_params() > 0


def test_cnn_lstm_model_building():
    """Test CNN-LSTM model building."""
    model = build_cnn_lstm_model(input_shape=(200, 6), num_classes=12)
    
    assert model is not None
    assert model.input_shape == (None, 200, 6)
    assert model.output_shape == (None, 12)
    assert model.count_params() > 0


def test_model_prediction():
    """Test model prediction."""
    model = build_1d_cnn_model(input_shape=(200, 6), num_classes=12)
    
    # Create dummy input
    X = np.random.randn(4, 200, 6).astype(np.float32)
    
    # Predict
    predictions = model.predict(X, verbose=0)
    
    assert predictions.shape == (4, 12)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    assert np.allclose(predictions.sum(axis=1), 1.0)


def test_model_saving_loading(tmp_path):
    """Test model saving and loading."""
    model = build_1d_cnn_model(input_shape=(200, 6), num_classes=12)
    
    # Save model
    save_path = tmp_path / "test_model.h5"
    ModelManager.save_model(model, str(save_path))
    
    assert save_path.exists()
    
    # Load model
    loaded_model = ModelManager.load_model(str(save_path))
    
    assert loaded_model is not None
    assert loaded_model.count_params() == model.count_params()


def test_model_compilation():
    """Test model compilation."""
    from src.models.cnn_1d import CNN1D
    
    builder = CNN1D()
    model = builder.build_model(input_shape=(200, 6), num_classes=12)
    model = builder.compile_model(model)
    
    assert model.optimizer is not None
    assert model.loss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
