# MNIST Classification with PyTorch

This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. The model is specifically designed to achieve >95% accuracy in a single epoch while maintaining less than 25,000 parameters.

## Model Architecture

The model uses a compact architecture to maintain high accuracy with minimal parameters:

- Input Layer: Takes 28x28 grayscale images
- First Conv Layer: 1 → 30 channels, 3x3 kernel with padding (28x28x30)
- Second Conv Layer: 30 → 40 channels, 3x3 kernel with padding (28x28x40)
- Third Conv Layer: 40 → 16 channels, 3x3 kernel with padding (14x14x16)
- MaxPool Layers: 2x2
- Batch Normalization after each conv layer
- Light Dropout (0.03)
- Final FC Layer: 784 → 10 neurons

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- pytest >= 6.0.0

## Local Setup and Running

1. Clone the repository
2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python model.py
```

5. Run tests:

```bash
pytest test_model.py -v
```

## Training Details

- Epochs: 1
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Learning Rate Schedule: OneCycleLR
  - max_lr: 0.01
  - div_factor: 25
  - final_div_factor: 1000
- Data Augmentation:
  - Random rotation (±15 degrees)
  - Random translation (±12%)
  - Random scaling (0.85-1.15)
- Gradient Clipping: max_norm=1.0

## Model Artifacts

Trained models are saved with timestamps in the format: `mnist_model_YYYYMMDD_HHMMSS.pth`

## Testing

The automated tests verify:

1. Model Architecture:
   - Input shape compatibility (28x28)
   - Output shape (10 classes)
   - Parameter count (< 25,000)
2. Model Performance:
   - Training accuracy > 95% in one epoch
