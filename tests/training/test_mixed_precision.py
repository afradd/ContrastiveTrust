import pytest
import torch
import torch.nn as nn
from src.training.mixed_precision import MixedPrecisionManager

def test_mixed_precision_manager_init():
    manager = MixedPrecisionManager(enabled=False, device="cpu")
    assert manager.scaler is None
    
    manager = MixedPrecisionManager(enabled=True, device="cpu")
    assert manager.scaler is None  # AMP requires CUDA
    
    if torch.cuda.is_available():
        manager = MixedPrecisionManager(enabled=True, device="cuda")
        assert manager.scaler is not None

def test_mixed_precision_manager_autocast():
    manager = MixedPrecisionManager(enabled=False, device="cpu")
    with manager.autocast():
        # Just ensure context manager works without raising errors
        pass

def test_mixed_precision_manager_scale_and_backward():
    manager = MixedPrecisionManager(enabled=False, device="cpu")
    tensor = torch.tensor([1.0], requires_grad=True)
    manager.scale_and_backward(tensor * 2)
    assert tensor.grad is not None

def test_mixed_precision_manager_step_and_update():
    manager = MixedPrecisionManager(enabled=False, device="cpu")
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    loss = model(torch.tensor([[1.0]])).sum()
    manager.scale_and_backward(loss)
    manager.step_and_update(optimizer)
    
def test_mixed_precision_manager_unscale_():
    manager = MixedPrecisionManager(enabled=False, device="cpu")
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager.unscale_(optimizer) # Should be no-op if scaler is None
