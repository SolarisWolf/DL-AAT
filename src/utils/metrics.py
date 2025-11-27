"""
Performance metrics for fault detection and classification.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate and visualize performance metrics."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names for fault types
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
            normalize: Whether to normalize the confusion matrix
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: str = None
    ):
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: If True, return as dictionary
            
        Returns:
            Classification report (string or dict)
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def calculate_response_time_stats(
        self,
        response_times: List[float]
    ) -> Dict[str, float]:
        """
        Calculate response time statistics.
        
        Args:
            response_times: List of response times in milliseconds
            
        Returns:
            Dictionary of statistics
        """
        response_times = np.array(response_times)
        
        return {
            'mean_ms': np.mean(response_times),
            'median_ms': np.median(response_times),
            'std_ms': np.std(response_times),
            'min_ms': np.min(response_times),
            'max_ms': np.max(response_times),
            'p95_ms': np.percentile(response_times, 95),
            'p99_ms': np.percentile(response_times, 99)
        }


def print_metrics_summary(metrics: Dict[str, float]):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    # Overall metrics
    print("\nOverall Performance:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    
    print("\nWeighted Metrics:")
    print(f"  Precision:          {metrics['precision_weighted']:.4f}")
    print(f"  Recall:             {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:           {metrics['f1_weighted']:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test metrics calculation
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_classes=5, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train dummy model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    class_names = [f"Class_{i}" for i in range(5)]
    calculator = MetricsCalculator(class_names)
    
    metrics = calculator.calculate_metrics(y_test, y_pred)
    print_metrics_summary(metrics)
    
    print("\nClassification Report:")
    print(calculator.generate_classification_report(y_test, y_pred))
