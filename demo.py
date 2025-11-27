"""
Complete end-to-end demo of the Smart Grid Fault Detection System.
"""
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import get_config, setup_logger
from src.data import GridFaultGenerator, DataPreprocessor
from src.models import create_model, ModelManager
from src.training import FaultDetectionTrainer
from src.iot import SensorNetwork, RealTimeFaultDetector


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_data_generation():
    """Demo: Generate synthetic fault data."""
    print_section("STEP 1: DATA GENERATION")
    
    config = get_config()
    generator = GridFaultGenerator(config)
    
    print("Generating training dataset...")
    X, y = generator.generate_dataset(num_samples=1000, balanced=True)
    
    print(f"✓ Dataset generated:")
    print(f"  - Samples: {len(X)}")
    print(f"  - Shape: {X.shape}")
    print(f"  - Fault types: {len(generator.fault_types)}")
    print(f"  - Classes: {generator.fault_types}")
    
    # Save dataset
    output_path = "data/demo_dataset.npz"
    generator.save_dataset(X, y, output_path)
    print(f"  - Saved to: {output_path}")
    
    return output_path


def demo_data_preprocessing(data_path):
    """Demo: Preprocess data."""
    print_section("STEP 2: DATA PREPROCESSING")
    
    from src.data import load_dataset
    
    print(f"Loading dataset from {data_path}...")
    X, y, metadata = load_dataset(data_path)
    
    preprocessor = DataPreprocessor()
    print("Preprocessing data...")
    result = preprocessor.preprocess_pipeline(X, y, normalize=True, split=True)
    
    print(f"✓ Data preprocessing completed:")
    print(f"  - Train set: {result['X_train'].shape}")
    print(f"  - Val set: {result['X_val'].shape}")
    print(f"  - Test set: {result['X_test'].shape}")
    print(f"  - Normalized: Yes")
    
    return result, metadata


def demo_model_training(data, metadata):
    """Demo: Train CNN model."""
    print_section("STEP 3: MODEL TRAINING")
    
    config = get_config()
    # Override for demo (faster training)
    config.config['training']['epochs'] = 10
    config.config['training']['batch_size'] = 32
    
    trainer = FaultDetectionTrainer(config)
    
    print("Building 1D-CNN model...")
    input_shape = data['X_train'].shape[1:]
    num_classes = len(metadata['fault_types'])
    trainer.build_model(input_shape, num_classes)
    
    print("\nTraining model (10 epochs for demo)...")
    history = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    print("\n✓ Training completed:")
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    print(f"  - Final val accuracy: {final_acc:.4f}")
    print(f"  - Final val loss: {final_loss:.4f}")
    
    # Save model
    model_path = "models/demo_model.h5"
    trainer.save_model(model_path)
    print(f"  - Model saved to: {model_path}")
    
    return trainer, model_path


def demo_model_evaluation(trainer, data, metadata):
    """Demo: Evaluate trained model."""
    print_section("STEP 4: MODEL EVALUATION")
    
    print("Evaluating model on test set...")
    metrics = trainer.evaluate(
        data['X_test'],
        data['y_test'],
        metadata['fault_types']
    )
    
    print(f"\n✓ Evaluation results:")
    print(f"  - Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - Precision: {metrics['precision_weighted']:.4f}")
    print(f"  - Recall: {metrics['recall_weighted']:.4f}")
    print(f"  - F1-Score: {metrics['f1_weighted']:.4f}")
    print(f"  - Inference Time: {metrics['inference_time_ms']:.2f} ms")
    
    return metrics


def demo_iot_sensors():
    """Demo: IoT sensor network."""
    print_section("STEP 5: IoT SENSOR NETWORK")
    
    network = SensorNetwork()
    
    print("Sensor network initialized:")
    status = network.get_status()
    print(f"  - Total sensors: {status['num_sensors']}")
    print(f"  - Sensor types: {list(network.sensors.keys())}")
    
    print("\nSimulating sensor readings for different faults:")
    for fault_type in ["Normal", "AG", "ABC"]:
        readings = network.simulate_readings(fault_type)
        print(f"\n  {fault_type}:")
        print(f"    Va={readings['V_A']:.1f}V, Vb={readings['V_B']:.1f}V, Vc={readings['V_C']:.1f}V")
        print(f"    Ia={readings['I_A']:.1f}A, Ib={readings['I_B']:.1f}A, Ic={readings['I_C']:.1f}A")
        print(f"    Temp={readings['temp']:.1f}°C")
    
    print("\n✓ IoT sensor network operational")


def demo_real_time_detection(model_path):
    """Demo: Real-time fault detection."""
    print_section("STEP 6: REAL-TIME FAULT DETECTION")
    
    print(f"Loading model from {model_path}...")
    detector = RealTimeFaultDetector(model_path)
    
    print("\nRunning real-time detection (10 samples)...")
    print("-" * 80)
    print(f"{'#':<4} {'Predicted':<12} {'Actual':<12} {'Confidence':<12} {'Time (ms)':<12} {'Alert':<10}")
    print("-" * 80)
    
    for i in range(10):
        result = detector.process_sensor_data()
        
        print(f"{i+1:<4} {result['predicted_fault']:<12} {result['actual_fault']:<12} "
              f"{result['confidence']:.3f}{'':>7} {result['detection_time_ms']:>6.2f}{'':>5} "
              f"{result['alert_level']:<10}")
        
        time.sleep(0.5)  # Simulate real-time interval
    
    print("-" * 80)
    print("\n✓ Real-time detection demonstration completed")


def main():
    """Run complete demo."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "   SMART ENERGY GRID FAULT IDENTIFICATION SYSTEM".center(78) + "║")
    print("║" + "   Using CNN and IoT Sensor Networks".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "   Complete End-to-End Demonstration".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Step 1: Generate data
        data_path = demo_data_generation()
        
        # Step 2: Preprocess data
        data, metadata = demo_data_preprocessing(data_path)
        
        # Step 3: Train model
        trainer, model_path = demo_model_training(data, metadata)
        
        # Step 4: Evaluate model
        metrics = demo_model_evaluation(trainer, data, metadata)
        
        # Step 5: IoT sensors
        demo_iot_sensors()
        
        # Step 6: Real-time detection
        demo_real_time_detection(model_path)
        
        # Final summary
        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        
        print("Summary of Results:")
        print(f"  ✓ Dataset generated: {data_path}")
        print(f"  ✓ Model trained: {model_path}")
        print(f"  ✓ Test accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  ✓ Inference time: {metrics['inference_time_ms']:.2f} ms")
        print(f"  ✓ Real-time detection: Operational")
        
        print("\nNext Steps:")
        print("  1. Generate larger dataset: python src/data/data_generator.py --num-samples 10000")
        print("  2. Train full model: python src/training/train.py --data data/fault_dataset.npz")
        print("  3. Launch dashboard: python src/dashboard/app.py")
        print("  4. Run real-time monitoring: python src/iot/real_time_detector.py --model models/best_model.h5")
        
        print("\n" + "="*80)
        print("Thank you for exploring the Smart Grid Fault Detection System!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
