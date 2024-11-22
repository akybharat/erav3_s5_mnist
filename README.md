# MNIST Classification with PyTorch

This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. The model is specifically designed to achieve >95% accuracy in a single epoch while maintaining less than 25,000 parameters.

## Model Architecture

The model uses a compact architecture to maintain high accuracy with minimal parameters:

- Input Layer: Takes 28x28 grayscale images
- First Conv Layer: 1 → 4 channels, 3x3 kernel (26x26x4)
- Second Conv Layer: 4 → 8 channels, 3x3 kernel (24x24x8)
- MaxPool Layer: 2x2 (12x12x8)
- First FC Layer: 1152 → 32 neurons
- Output Layer: 32 → 10 neurons (one for each digit)

Total Parameters: < 25,000

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- pytest >= 6.0.0

## Local Setup and Running

1. Clone the repository

2. Create and activate virtual environment:

```bash
python -m venv mnist-env
source mnist-env/bin/activate
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
