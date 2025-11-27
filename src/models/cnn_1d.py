"""
1D-CNN model for power grid fault classification.
"""
import tensorflow as tf
import keras
from keras import layers, models
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, LoggerMixin


class CNN1D(LoggerMixin):
    """1D Convolutional Neural Network for fault classification."""
    
    def __init__(self, config: Config = None):
        """
        Initialize 1D-CNN model builder.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.model = None
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int
    ) -> keras.Model:
        """
        Build 1D-CNN model architecture.
        
        Args:
            input_shape: Shape of input (timesteps, features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building 1D-CNN model with input shape {input_shape}")
        
        # Get configuration
        conv_layers = self.config.get('model.conv_layers', [])
        dense_layers = self.config.get('model.dense_layers', [512, 256, 128])
        dropout_rate = self.config.get('model.dropout_rate', 0.3)
        pooling = self.config.get('model.pooling', 'max')
        pool_size = self.config.get('model.pool_size', 2)
        
        # Build model
        model = models.Sequential(name='CNN1D_FaultDetection')
        
        # Input layer
        model.add(layers.Input(shape=input_shape, name='input'))
        
        # Convolutional blocks
        for i, conv_config in enumerate(conv_layers):
            filters = conv_config.get('filters', 64)
            kernel_size = conv_config.get('kernel_size', 5)
            activation = conv_config.get('activation', 'relu')
            
            # Conv layer
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            
            # Batch normalization
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            
            # Pooling layer
            if pooling == 'max':
                model.add(layers.MaxPooling1D(
                    pool_size=pool_size,
                    name=f'maxpool_{i+1}'
                ))
            elif pooling == 'avg':
                model.add(layers.AveragePooling1D(
                    pool_size=pool_size,
                    name=f'avgpool_{i+1}'
                ))
            
            # Dropout
            model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
        
        # Flatten
        model.add(layers.Flatten(name='flatten'))
        
        # Dense layers
        for i, units in enumerate(dense_layers):
            model.add(layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        ))
        
        self.model = model
        self.logger.info(f"Model built with {model.count_params():,} parameters")
        
        return model
    
    def compile_model(
        self,
        model: keras.Model = None,
        learning_rate: float = None,
        optimizer: str = None
    ) -> keras.Model:
        """
        Compile model with optimizer and loss function.
        
        Args:
            model: Model to compile (uses self.model if None)
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name
            
        Returns:
            Compiled model
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model to compile. Build model first.")
        
        # Get configuration
        if learning_rate is None:
            learning_rate = self.config.get('training.learning_rate', 0.001)
        if optimizer is None:
            optimizer = self.config.get('training.optimizer', 'adam')
        
        loss = self.config.get('training.loss', 'sparse_categorical_crossentropy')
        metrics = self.config.get('training.metrics', ['accuracy'])
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Compile model
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
        
        return model
    
    def get_model_summary(self, model: keras.Model = None) -> str:
        """
        Get model architecture summary.
        
        Args:
            model: Model to summarize (uses self.model if None)
            
        Returns:
            Model summary string
        """
        if model is None:
            model = self.model
        
        if model is None:
            return "No model available"
        
        # Capture summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


def build_1d_cnn_model(
    input_shape: Tuple[int, int] = (200, 6),
    num_classes: int = 12,
    config: Config = None
) -> keras.Model:
    """
    Build and compile 1D-CNN model (convenience function).
    
    Args:
        input_shape: Input shape (timesteps, features)
        num_classes: Number of output classes
        config: Configuration object
        
    Returns:
        Compiled Keras model
    """
    builder = CNN1D(config)
    model = builder.build_model(input_shape, num_classes)
    model = builder.compile_model(model)
    return model


if __name__ == "__main__":
    # Test model building
    print("Building 1D-CNN model...")
    
    config = get_config()
    builder = CNN1D(config)
    
    # Build model
    model = builder.build_model(input_shape=(200, 6), num_classes=12)
    
    # Compile model
    model = builder.compile_model(model)
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    print(builder.get_model_summary())
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.randn(4, 200, 6).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sum per sample: {output.sum(axis=1)}")
    
    print("\n✓ 1D-CNN model built and tested successfully!")
