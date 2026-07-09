import pytest
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from src.training.config import OptimizerConfig, SchedulerConfig
from src.training.optimizer_factory import create_optimizer
from src.training.scheduler_factory import create_scheduler

@pytest.fixture
def optimizer():
    model = nn.Linear(10, 2)
    return create_optimizer(model.parameters(), OptimizerConfig(name="AdamW"))

def test_create_scheduler_cosine(optimizer):
    config = SchedulerConfig(name="CosineAnnealingLR", kwargs={"T_max": 10})
    sched = create_scheduler(optimizer, config)
    assert isinstance(sched, lr_scheduler.CosineAnnealingLR)

def test_create_scheduler_cosine_missing_kwargs(optimizer):
    config = SchedulerConfig(name="CosineAnnealingLR")
    with pytest.raises(ValueError):
        create_scheduler(optimizer, config)

def test_create_scheduler_steplr(optimizer):
    config = SchedulerConfig(name="StepLR", kwargs={"step_size": 5})
    sched = create_scheduler(optimizer, config)
    assert isinstance(sched, lr_scheduler.StepLR)

def test_create_scheduler_reducelronplateau(optimizer):
    config = SchedulerConfig(name="ReduceLROnPlateau")
    sched = create_scheduler(optimizer, config)
    assert isinstance(sched, lr_scheduler.ReduceLROnPlateau)

def test_create_scheduler_invalid_name(optimizer):
    config = SchedulerConfig(name="UnknownScheduler")
    with pytest.raises(ValueError):
        create_scheduler(optimizer, config)

def test_create_scheduler_invalid_optimizer():
    config = SchedulerConfig(name="StepLR", kwargs={"step_size": 5})
    with pytest.raises(TypeError):
        create_scheduler("optimizer", config)

def test_create_scheduler_invalid_config(optimizer):
    with pytest.raises(TypeError):
        create_scheduler(optimizer, "StepLR")
