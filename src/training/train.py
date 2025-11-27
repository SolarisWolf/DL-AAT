"""
Model training pipeline for power grid fault detection.
"""
import tensorflow as tf
import keras
import numpy as np
import argparse
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import Config, get_config, setup_logger, MetricsCalculator
from src.data import load_dataset, DataPreprocessor
from src.models import create_model, ModelManager
from src.models.registry import write_latest_metrics, register_model
from src.training.callbacks import get_callbacks


class FaultDetectionTrainer:
    """Trainer class for fault detection models."""
    
    def __init__(self, config: Config = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = setup_logger("trainer", level=self.config.get('logging.level', 'INFO'))
        self.model = None
        self.history = None
        self.preprocessor = None
    
    def prepare_data(self, data_path: str):
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Dictionary with prepared data
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Load dataset
        X, y, metadata = load_dataset(data_path)
        
        self.logger.info(f"Dataset loaded: {X.shape}, Classes: {len(np.unique(y))}")
        
        # Preprocess data
        self.preprocessor = DataPreprocessor(self.config)
        result = self.preprocessor.preprocess_pipeline(
            X, y,
            normalize=True,
            augment=False,
            split=True
        )
        
        # Add metadata
        result['metadata'] = metadata
        result['fault_types'] = metadata.get('fault_types', [])
        
        self.logger.info("Data preparation completed")
        
        return result
    
    def build_model(self, input_shape: tuple, num_classes: int):
        """
        Build and compile model.
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
        """
        model_type = self.config.get('model.type', '1D-CNN')
        
        self.logger.info(f"Building {model_type} model...")
        
        self.model = create_model(
            model_type=model_type,
            input_shape=input_shape,
            num_classes=num_classes,
            config=self.config
        )
        
        # Print model summary
        self.logger.info("Model architecture:")
        self.model.summary(print_fn=self.logger.info)
        
        # Print model info
        ModelManager.print_model_info(self.model)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.logger.info("Starting model training...")
        
        # Get training parameters
        batch_size = self.config.get('training.batch_size', 32)
        epochs = self.config.get('training.epochs', 100)
        
        # Get callbacks
        callbacks = get_callbacks(self.config)
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        self.logger.info(f"Training completed in {training_time:.2f}s ({training_time/60:.2f} minutes)")
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fault_types: list
    ):
        """
        Evaluate trained model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            fault_types: List of fault type names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")
        
        self.logger.info("Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        calculator = MetricsCalculator(fault_types)
        metrics = calculator.calculate_metrics(y_test, y_pred)
        
        # Print metrics
        from src.utils import print_metrics_summary
        print_metrics_summary(metrics)
        
        # Generate classification report
        print("\nDetailed Classification Report:")
        print(calculator.generate_classification_report(y_test, y_pred))
        
        # Save confusion matrix plot
        log_dir = Path(self.config.get('logging.log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        cm_path = log_dir / 'confusion_matrix.png'
        calculator.plot_confusion_matrix(y_test, y_pred, save_path=str(cm_path))
        
        # Calculate inference time
        self.logger.info("Measuring inference time...")
        sample_batch = X_test[:100]
        
        start_time = time.time()
        _ = self.model.predict(sample_batch, verbose=0)
        inference_time = (time.time() - start_time) / len(sample_batch) * 1000  # ms per sample
        
        metrics['inference_time_ms'] = inference_time
        self.logger.info(f"Average inference time: {inference_time:.2f} ms per sample")
        
        return metrics
    
    def save_model(self, save_path: str = None):
        """
        Save trained model.
        
        Args:
            save_path: Path to save model. If None, uses config path.
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if save_path is None:
            save_path = self.config.get('model_save.best_model_path', 'models/trained_model.h5')
        
        ModelManager.save_model(self.model, save_path)
        self.logger.info(f"Model saved to {save_path}")
        
        # Save training history
        if self.history is not None:
            history_path = Path(save_path).parent / 'training_history.json'
            ModelManager.save_training_history(self.history.history, str(history_path))
            
            # Plot training history
            plot_path = Path(save_path).parent / 'training_history.png'
            calculator = MetricsCalculator([])  # dummy class names
            calculator.plot_training_history(self.history.history, save_path=str(plot_path))


def main():
    """Main training function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train power grid fault detection model")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='1D-CNN',
                       choices=['1D-CNN', 'CNN-LSTM'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command-line arguments
    if args.model:
        config.config['model']['type'] = args.model
    if args.epochs:
        config.config['training']['epochs'] = args.epochs
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    
    # Create trainer
    trainer = FaultDetectionTrainer(config)
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    data = trainer.prepare_data(args.data)
    
    # Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    input_shape = data['X_train'].shape[1:]
    num_classes = len(data['fault_types'])
    trainer.build_model(input_shape, num_classes)
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Evaluate model
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    metrics = trainer.evaluate(
        data['X_test'], data['y_test'],
        data['fault_types']
    )
    # Persist latest metrics for CI baseline checks
    try:
        write_latest_metrics(metrics)
    except Exception as e:
        trainer.logger.warning(f"Failed to write latest metrics: {e}")
    
    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    trainer.save_model(args.output)

    # Register model in simple registry
    try:
        saved_path = args.output or config.get('model_save.best_model_path', 'models/trained_model.h5')
        register_model(
            model_path=saved_path,
            metrics=metrics,
            config=config.config,
            dataset_info={'path': args.data}
        )
    except Exception as e:
        trainer.logger.warning(f"Failed to register model: {e}")
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Test Recall: {metrics['recall_weighted']:.4f}")
    print(f"  Test F1-Score: {metrics['f1_weighted']:.4f}")
    print(f"  Inference Time: {metrics['inference_time_ms']:.2f} ms")
    print(f"\nModel saved to: {args.output or config.get('model_save.best_model_path')}")


if __name__ == "__main__":
    main()
