# Smart Energy Grid Fault Identification Using CNN and IoT Sensor Networks

A deep learning-based intelligent fault detection and classification system for smart energy grids, integrating Convolutional Neural Networks (CNN) with IoT sensor networks for real-time monitoring and fault identification.

## 🎯 Project Overview

This project implements an automated fault detection system capable of identifying and classifying 12 different types of electrical faults in power distribution networks with >98% accuracy and response times under 20ms.

### Fault Types Detected
- **Normal Operation**
- **Single-phase-to-ground:** AG, BG, CG
- **Line-to-line:** AB, BC, CA
- **Double-line-to-ground:** ABG, BCG, CAG
- **Three-phase:** ABC, ABCG

## 🏗️ Project Structure

```
DL AAT/
├── src/
│   ├── data/
│   │   ├── data_generator.py      # Fault simulation and data generation
│   │   ├── preprocessing.py       # Data preprocessing pipeline
│   │   └── dataset.py             # Dataset loading utilities
│   ├── models/
│   │   ├── cnn_1d.py             # 1D-CNN architecture
│   │   ├── cnn_lstm.py           # CNN-LSTM hybrid model
│   │   └── model_utils.py        # Model utilities
│   ├── training/
│   │   ├── train.py              # Training pipeline
│   │   ├── evaluate.py           # Model evaluation
│   │   └── callbacks.py          # Custom callbacks
│   ├── iot/
│   │   ├── sensor_simulator.py   # IoT sensor simulation
│   │   ├── mqtt_client.py        # MQTT communication
│   │   └── real_time_detector.py # Real-time fault detection
│   ├── dashboard/
│   │   ├── app.py                # Web dashboard
│   │   └── components.py         # Dashboard components
│   └── utils/
│       ├── config.py             # Configuration loader
│       ├── logger.py             # Logging utilities
│       └── metrics.py            # Performance metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   ├── test_data_generation.py
│   ├── test_model.py
│   └── test_iot.py
├── data/                         # Generated datasets
├── models/                       # Saved models
├── logs/                         # Training logs
├── checkpoints/                  # Model checkpoints
├── config.yaml                   # Configuration file
├── requirements.txt              # Dependencies
├── Problem_Statement.md          # Project problem statement
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "c:\Users\agarw\Downloads\DL AAT"
```

2. **Create a virtual environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

### Quick Start

#### 1. Generate Training Data
```powershell
python src/data/data_generator.py --num-samples 10000 --output data/train_data.npz
```

#### 2. Train the Model
```powershell
python src/training/train.py --config config.yaml --model 1D-CNN
```

#### 3. Evaluate Performance
```powershell
python src/training/evaluate.py --model-path models/best_model.h5 --test-data data/test_data.npz
```

#### 4. Run Real-time Detection
```powershell
python src/iot/real_time_detector.py --model models/best_model.h5
```

#### 5. Launch Dashboard
```powershell
python src/dashboard/app.py
```
Then open http://localhost:8050 in your browser.

## 📊 Model Architecture

### 1D-CNN Model
```
Input Layer (200 samples × 6 features)
    ↓
Conv1D (64 filters, kernel=5) + ReLU + MaxPool
    ↓
Conv1D (128 filters, kernel=5) + ReLU + MaxPool
    ↓
Conv1D (256 filters, kernel=3) + ReLU + MaxPool
    ↓
Flatten + Dropout(0.3)
    ↓
Dense (512) + ReLU + Dropout(0.3)
    ↓
Dense (256) + ReLU + Dropout(0.3)
    ↓
Dense (128) + ReLU
    ↓
Output Layer (12 classes) + Softmax
```

## 🔬 Features

- **Automated Data Generation:** Simulate various fault scenarios with configurable parameters
- **Deep Learning Models:** 1D-CNN and CNN-LSTM architectures
- **Real-time Detection:** IoT sensor integration with MQTT protocol
- **Web Dashboard:** Interactive monitoring and visualization
- **High Accuracy:** >98% fault classification accuracy
- **Fast Response:** <20ms detection latency
- **Scalable Architecture:** Deployable across distribution networks

## 📈 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | ≥98% | TBD |
| Precision | ≥97% | TBD |
| Recall | ≥97% | TBD |
| F1-Score | ≥97% | TBD |
| Response Time | <20ms | TBD |

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Data generation parameters
- Model architecture
- Training hyperparameters
- IoT sensor settings
- Dashboard configuration

## 📝 Usage Examples

### Generate Custom Dataset
```python
from src.data.data_generator import GridFaultGenerator

generator = GridFaultGenerator(config)
data, labels = generator.generate_dataset(num_samples=5000)
```

### Load and Train Model
```python
from src.models.cnn_1d import build_1d_cnn_model
from src.training.train import train_model

model = build_1d_cnn_model(input_shape=(200, 6), num_classes=12)
history = train_model(model, train_data, val_data, config)
```

### Real-time Prediction
```python
from src.iot.real_time_detector import RealTimeFaultDetector

detector = RealTimeFaultDetector(model_path="models/best_model.h5")
detector.start_monitoring()
```

## 🧪 Testing

Run all tests:
```powershell
pytest tests/ -v --cov=src
```

## 📚 Documentation

For detailed documentation, see:
- [Problem Statement](Problem_Statement.md)
- [API Documentation](docs/api.md) (coming soon)
- [User Guide](docs/user_guide.md) (coming soon)

## 🤝 Contributing

This is an academic project for BMSCE 5th Semester. Contributions, suggestions, and feedback are welcome!

## 📄 License

This project is developed for educational purposes at BMS College of Engineering.

## 👥 Authors

- **Department:** AI & Data Science
- **Semester:** 5th Semester
- **Institution:** BMS College of Engineering (BMSCE), Bangalore

## 🙏 Acknowledgments

- IEEE standards for power system fault analysis
- Kaggle Smart Grid Monitoring Dataset
- TensorFlow and Keras communities
- IoT sensor manufacturers (Arduino, Raspberry Pi)

## 📧 Contact

For questions or collaboration opportunities, please contact the project team.

---

**Last Updated:** November 26, 2025  
**Version:** 1.0.0
