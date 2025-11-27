# Quick Start Guide
## Smart Energy Grid Fault Identification System

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- GPU optional (for faster training)

### Installation

1. **Navigate to project directory:**
```powershell
cd "c:\Users\agarw\Downloads\DL AAT"
```

2. **Create and activate virtual environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

### Quick Demo

Run the complete end-to-end demonstration:
```powershell
python demo.py
```

This will:
- Generate synthetic fault data
- Train a CNN model
- Evaluate performance
- Demonstrate IoT sensors
- Show real-time fault detection

### Step-by-Step Usage

#### 1. Generate Training Data
```powershell
python src/data/data_generator.py --num-samples 10000 --output data/fault_dataset.npz
```

Options:
- `--num-samples`: Number of samples to generate (default: 10000)
- `--output`: Output file path
- `--balanced`: Generate balanced dataset (default: true)
- `--seed`: Random seed for reproducibility

#### 2. Train the Model
```powershell
python src/training/train.py --data data/fault_dataset.npz --model 1D-CNN --epochs 100
```

Options:
- `--data`: Path to training dataset (required)
- `--model`: Model type ('1D-CNN' or 'CNN-LSTM')
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--output`: Path to save trained model

#### 3. Evaluate the Model
```powershell
python src/training/evaluate.py --model-path models/best_model.h5 --test-data data/fault_dataset.npz
```

#### 4. Run Real-Time Detection
```powershell
python src/iot/real_time_detector.py --model models/best_model.h5 --duration 60
```

Options:
- `--model`: Path to trained model (required)
- `--duration`: Monitoring duration in seconds
- `--interval`: Detection interval in seconds
- `--mqtt`: Enable MQTT communication

#### 5. Launch Dashboard
```powershell
python src/dashboard/app.py --port 8050
```

Then open http://localhost:8050 in your browser.

Options:
- `--host`: Host address (default: localhost)
- `--port`: Port number (default: 8050)
- `--debug`: Enable debug mode

### Running Tests

Run all tests:
```powershell
pytest tests/ -v
```

Run specific test file:
```powershell
pytest tests/test_data_generation.py -v
```

### Project Structure

```
DL AAT/
├── src/                    # Source code
│   ├── data/              # Data generation and preprocessing
│   ├── models/            # CNN models
│   ├── training/          # Training pipeline
│   ├── iot/               # IoT sensors and real-time detection
│   ├── dashboard/         # Web dashboard
│   └── utils/             # Utilities
├── tests/                 # Unit tests
├── data/                  # Generated datasets
├── models/                # Saved models
├── logs/                  # Training logs
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── demo.py               # Complete demonstration
└── README.md             # Documentation
```

### Configuration

Edit `config.yaml` to customize:
- Data generation parameters
- Model architecture
- Training hyperparameters
- IoT sensor settings
- Dashboard configuration

### Troubleshooting

**Issue: Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**Issue: CUDA/GPU not detected**
- Install CUDA toolkit if available
- CPU-only mode will work (slower training)

**Issue: Dashboard not loading**
- Check if port 8050 is available
- Try different port: `python src/dashboard/app.py --port 8080`

**Issue: Low accuracy**
- Generate more training data (increase --num-samples)
- Train for more epochs
- Try different model architecture (CNN-LSTM)

### Expected Results

With default configuration (10,000 samples, 100 epochs):
- **Accuracy:** >98%
- **Precision:** >97%
- **Recall:** >97%
- **F1-Score:** >97%
- **Inference Time:** <20ms per sample

### Next Steps

1. **Experiment with hyperparameters:** Modify `config.yaml`
2. **Try different architectures:** Switch between 1D-CNN and CNN-LSTM
3. **Increase dataset size:** Generate 50,000+ samples for better accuracy
4. **Real hardware integration:** Connect actual IoT sensors
5. **Deploy to production:** Set up MQTT broker and continuous monitoring

### Support

For questions or issues:
- Check `Problem_Statement.md` for project details
- Review code documentation in source files
- Run demo.py for complete example

### Citations

If using this project for research, please cite:
```
Smart Energy Grid Fault Identification Using CNN and IoT Sensor Networks
Department of AI & Data Science
BMS College of Engineering (BMSCE), Bangalore
2025
```

---

**Version:** 1.0.0  
**Last Updated:** November 26, 2025
