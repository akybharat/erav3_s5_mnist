import torch
import pytest
from model import MNISTNet, count_parameters, train_model


def test_model_architecture():
    model = MNISTNet()

    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)

    # Test output shape
    assert output.shape == (1, 10), "Output shape should be (1, 10)"

    # Test parameter count
    param_count = count_parameters(model)
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_training():
    model, accuracy = train_model()
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is less than required 95%"


if __name__ == "__main__":
    pytest.main([__file__])
