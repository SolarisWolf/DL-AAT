"""
Model utilities and helper functions.
"""
import tensorflow as tf
import keras
from pathlib import Path
from typing import Dict, Any
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import LoggerMixin


class ModelManager(LoggerMixin):
    """Manager for saving, loading, and managing trained models."""
    
    @staticmethod
    def save_model(
        model: keras.Model,
        save_path: str,
        save_format: str = 'h5',
        save_weights_only: bool = False
    ):
        """
        Save model to disk.
        
        Args:
            model: Keras model to save
            save_path: Path to save model
            save_format: Format to save ('h5' or 'tf')
            save_weights_only: If True, only save weights
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if save_weights_only:
                model.save_weights(str(save_path))
                print(f"Model weights saved to {save_path}")
            else:
                model.save(str(save_path), save_format=save_format)
                print(f"Model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    @staticmethod
    def load_model(
        model_path: str,
        compile_model: bool = True
    ) -> keras.Model:
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model
            compile_model: Whether to compile the loaded model
            
        Returns:
            Loaded Keras model
        """
        model = keras.models.load_model(model_path, compile=compile_model)
        print(f"Model loaded from {model_path}")
        return model
    
    @staticmethod
    def save_training_history(
        history: Dict[str, Any],
        save_path: str
    ):
        """
        Save training history to JSON file.
        
        Args:
            history: Training history dictionary
            save_path: Path to save history
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in history.items():
            if hasattr(value, 'tolist'):
                history_dict[key] = value.tolist()
            elif isinstance(value, list):
                history_dict[key] = value
            else:
                history_dict[key] = str(value)
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {save_path}")
    
    @staticmethod
    def load_training_history(history_path: str) -> Dict[str, Any]:
        """
        Load training history from JSON file.
        
        Args:
            history_path: Path to history file
            
        Returns:
            Training history dictionary
        """
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print(f"Training history loaded from {history_path}")
        return history
    
    @staticmethod
    def get_model_size(model: keras.Model) -> Dict[str, Any]:
        """
        Get model size information.
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with size information
        """
        import tempfile
        
        # Save model temporarily to get file size
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            model.save(tmp.name)
            file_size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        
        return {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
            'file_size_mb': file_size_mb
        }
    
    @staticmethod
    def print_model_info(model: keras.Model):
        """
        Print comprehensive model information.
        
        Args:
            model: Keras model
        """
        print("\n" + "="*70)
        print("MODEL INFORMATION")
        print("="*70)
        
        # Basic info
        print(f"\nModel name: {model.name}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Size info
        size_info = ModelManager.get_model_size(model)
        print(f"\nParameters:")
        print(f"  Total: {size_info['total_params']:,}")
        print(f"  Trainable: {size_info['trainable_params']:,}")
        print(f"  Non-trainable: {size_info['non_trainable_params']:,}")
        print(f"  Model file size: {size_info['file_size_mb']:.2f} MB")
        
        # Layer info
        print(f"\nArchitecture:")
        print(f"  Total layers: {len(model.layers)}")
        
        layer_types = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        for layer_type, count in sorted(layer_types.items()):
            print(f"    {layer_type}: {count}")
        
        print("="*70 + "\n")


def create_model(
    model_type: str = '1D-CNN',
    input_shape: tuple = (200, 6),
    num_classes: int = 12,
    config = None
) -> keras.Model:
    """
    Create model based on type.
    
    Args:
        model_type: Type of model ('1D-CNN' or 'CNN-LSTM')
        input_shape: Input shape
        num_classes: Number of output classes
        config: Configuration object
        
    Returns:
        Compiled Keras model
    """
    if model_type.upper() == '1D-CNN':
        from src.models.cnn_1d import build_1d_cnn_model
        return build_1d_cnn_model(input_shape, num_classes, config)
    elif model_type.upper() == 'CNN-LSTM':
        from src.models.cnn_lstm import build_cnn_lstm_model
        return build_cnn_lstm_model(input_shape, num_classes, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model utilities
    from src.models.cnn_1d import build_1d_cnn_model
    
    print("Testing model utilities...")
    
    # Create a model
    model = build_1d_cnn_model(input_shape=(200, 6), num_classes=12)
    
    # Print model info
    ModelManager.print_model_info(model)
    
    # Test saving and loading
    print("Testing save/load...")
    save_path = "test_model.h5"
    ModelManager.save_model(model, save_path)
    loaded_model = ModelManager.load_model(save_path)
    
    # Cleanup
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up test file: {save_path}")
    
    print("\n✓ Model utilities tests completed!")
