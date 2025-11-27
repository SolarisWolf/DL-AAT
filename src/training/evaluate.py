"""
Model evaluation utilities.
"""
import tensorflow as tf
import keras
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import setup_logger, MetricsCalculator, print_metrics_summary
from src.data import load_dataset, DataPreprocessor
from src.models import ModelManager


def evaluate_model(
    model_path: str,
    test_data_path: str,
    config_path: str = None
):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model file
        test_data_path: Path to test dataset
        config_path: Path to configuration file
    """
    logger = setup_logger("evaluator")
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ModelManager.load_model(model_path)
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    X_test, y_test, metadata = load_dataset(test_data_path)
    
    fault_types = metadata.get('fault_types', [f"Class_{i}" for i in range(len(np.unique(y_test)))])
    
    # Preprocess if needed
    if config_path:
        from src.utils import get_config
        config = get_config(config_path)
        preprocessor = DataPreprocessor(config)
        X_test = preprocessor.normalize_data(X_test, fit=False)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    calculator = MetricsCalculator(fault_types)
    metrics = calculator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Print metrics
    print_metrics_summary(metrics)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(calculator.generate_classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / "confusion_matrix.png"
    calculator.plot_confusion_matrix(y_test, y_pred, save_path=str(cm_path), normalize=True)
    
    # Measure inference time
    logger.info("\nMeasuring inference time...")
    import time
    
    num_samples = min(1000, len(X_test))
    sample_batch = X_test[:num_samples]
    
    # Warm-up
    _ = model.predict(sample_batch[:10], verbose=0)
    
    # Measure
    times = []
    for i in range(10):
        start = time.time()
        _ = model.predict(sample_batch, verbose=0)
        elapsed = time.time() - start
        times.append(elapsed / num_samples * 1000)  # ms per sample
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nInference Performance:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms per sample")
    print(f"  Throughput: {1000/avg_time:.2f} samples/second")
    print(f"  Target: <20 ms per sample")
    
    if avg_time < 20:
        print("  ✓ MEETS TARGET")
    else:
        print("  ✗ DOES NOT MEET TARGET")
    
    # Per-class performance
    print("\nPer-Class Performance:")
    print("-" * 70)
    print(f"{'Fault Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print("-" * 70)
    
    for i, fault_type in enumerate(fault_types):
        precision = metrics[f'precision_{fault_type}']
        recall = metrics[f'recall_{fault_type}']
        f1 = metrics[f'f1_{fault_type}']
        count = np.sum(y_test == i)
        
        print(f"{fault_type:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {count:<10}")
    
    print("-" * 70)
    
    logger.info("\n✓ Evaluation completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    return metrics


def main():
    """Main evaluation function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate power grid fault detection model")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
