import pytest
import torch
import torch.nn as nn
from src.training.config import OptimizerConfig
from src.training.optimizer_factory import create_optimizer

def test_create_optimizer_adam():
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="Adam", lr=1e-3, weight_decay=1e-4)
    opt = create_optimizer(model.parameters(), config)
    assert isinstance(opt, torch.optim.Adam)

def test_create_optimizer_adamw():
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="AdamW", lr=1e-3)
    opt = create_optimizer(model.parameters(), config)
    assert isinstance(opt, torch.optim.AdamW)

def test_create_optimizer_sgd():
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="SGD", lr=0.1)
    opt = create_optimizer(model.parameters(), config)
    assert isinstance(opt, torch.optim.SGD)

def test_create_optimizer_invalid_name():
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="RMSprop", lr=0.1)
    with pytest.raises(ValueError):
        create_optimizer(model.parameters(), config)

def test_create_optimizer_invalid_config_type():
    model = nn.Linear(10, 2)
    with pytest.raises(TypeError):
        create_optimizer(model.parameters(), "AdamW")
