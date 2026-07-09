import pytest
from src.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.models.encoder import EncoderConfig
from src.losses.contrastive_trust_loss import ContrastiveTrustLossConfig
from src.data.view_generator import ContrastiveViewGeneratorConfig

def test_optimizer_config_valid():
    config = OptimizerConfig(name="AdamW", lr=1e-3, weight_decay=1e-4)
    assert config.name == "AdamW"
    assert config.lr == 1e-3
    assert config.weight_decay == 1e-4

def test_optimizer_config_invalid():
    with pytest.raises(ValueError):
        OptimizerConfig(name="")
    with pytest.raises(ValueError):
        OptimizerConfig(lr=-0.1)
    with pytest.raises(ValueError):
        OptimizerConfig(weight_decay=-1e-4)

def test_scheduler_config_valid():
    config = SchedulerConfig(name="StepLR", kwargs={"step_size": 10})
    assert config.name == "StepLR"
    assert config.kwargs["step_size"] == 10

def test_scheduler_config_invalid():
    with pytest.raises(ValueError):
        SchedulerConfig(name="  ")

def test_training_config_valid():
    config = TrainingConfig()
    assert config.epochs == 100
    assert config.batch_size == 256
    assert config.device == "cuda"
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.scheduler, SchedulerConfig)
    assert isinstance(config.encoder, EncoderConfig)
    assert isinstance(config.loss, ContrastiveTrustLossConfig)
    assert isinstance(config.view_generator, ContrastiveViewGeneratorConfig)

def test_training_config_invalid():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=-1)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(device="")
    with pytest.raises(TypeError):
        TrainingConfig(optimizer="Adam")
