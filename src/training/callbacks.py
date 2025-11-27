"""
Custom callbacks for model training.
"""
import tensorflow as tf
import keras
import numpy as np
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import LoggerMixin


class TimeHistory(keras.callbacks.Callback, LoggerMixin):
    """Callback to track training time per epoch."""
    
    def on_train_begin(self, logs=None):
        self.times = []
        self.total_time = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        self.times.append(epoch_time)
        self.total_time += epoch_time
        
        self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")


class MetricsLogger(keras.callbacks.Callback, LoggerMixin):
    """Callback to log detailed metrics during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        metrics_str = f"Epoch {epoch + 1}: "
        metrics_str += f"loss={logs.get('loss', 0):.4f}, "
        metrics_str += f"acc={logs.get('accuracy', 0):.4f}, "
        metrics_str += f"val_loss={logs.get('val_loss', 0):.4f}, "
        metrics_str += f"val_acc={logs.get('val_accuracy', 0):.4f}"
        
        self.logger.info(metrics_str)


class LearningRateLogger(keras.callbacks.Callback, LoggerMixin):
    """Callback to log learning rate changes."""
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.logger.info(f"Learning rate at epoch {epoch + 1}: {lr:.6f}")


def get_callbacks(config, checkpoint_dir: str = "checkpoints"):
    """
    Create list of callbacks for training.
    
    Args:
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        List of Keras callbacks
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = []
    
    # ModelCheckpoint - save best model
    best_model_path = config.get('model_save.best_model_path', 'models/best_model.h5')
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # EarlyStopping - stop if no improvement
    early_stopping_patience = config.get('training.early_stopping_patience', 15)
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # ReduceLROnPlateau - reduce learning rate when plateauing
    reduce_lr_patience = config.get('training.reduce_lr_patience', 7)
    reduce_lr_factor = config.get('training.reduce_lr_factor', 0.5)
    min_lr = config.get('training.min_lr', 1e-7)
    
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )
    )
    
    # TensorBoard - visualization
    tensorboard_dir = config.get('logging.tensorboard_dir', 'logs/tensorboard')
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
    )
    
    # CSV Logger - save training history
    log_dir = Path(config.get('logging.log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / 'training_log.csv'
    
    callbacks.append(
        keras.callbacks.CSVLogger(
            csv_path,
            separator=',',
            append=False
        )
    )
    
    # Custom callbacks
    callbacks.append(TimeHistory())
    callbacks.append(MetricsLogger())
    callbacks.append(LearningRateLogger())
    
    return callbacks


if __name__ == "__main__":
    from src.utils import get_config
    
    print("Testing callbacks...")
    
    config = get_config()
    callbacks = get_callbacks(config)
    
    print(f"\nCreated {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
    
    print("\n✓ Callbacks created successfully!")
