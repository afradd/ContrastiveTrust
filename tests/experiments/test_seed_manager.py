"""Tests for SeedManager."""

import numpy as np
import pytest
import torch
import random
from src.experiments.seed_manager import SeedManager

def test_seed_manager_determinism():
    """Test that setting the seed produces deterministic outputs."""
    SeedManager.set_seed(42)
    val1_py = random.random()
    val1_np = np.random.rand()
    val1_torch = torch.rand(1).item()

    SeedManager.set_seed(42)
    val2_py = random.random()
    val2_np = np.random.rand()
    val2_torch = torch.rand(1).item()

    assert val1_py == val2_py
    assert val1_np == val2_np
    assert val1_torch == val2_torch

def test_seed_manager_cudnn_settings():
    """Test cuDNN settings for determinism."""
    SeedManager.set_seed(42, deterministic_cudnn=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    SeedManager.set_seed(42, deterministic_cudnn=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True

def test_rng_state_capture_and_restore():
    """Test capturing and restoring RNG states."""
    SeedManager.set_seed(123)
    val_before = random.random()
    
    state = SeedManager.get_rng_states()
    
    # Generate some random numbers to change the state
    _ = random.random()
    _ = random.random()
    
    SeedManager.set_rng_states(state)
    val_after = random.random()
    
    # The next random number should be identical to what it would have been
    # if we generated it right after the first `random.random()`
    # Wait, val_before consumed a number. 
    # Actually, we should test restoring the state gives the exact same sequence.
    SeedManager.set_seed(123)
    state = SeedManager.get_rng_states()
    seq1 = [random.random() for _ in range(3)]
    
    SeedManager.set_rng_states(state)
    seq2 = [random.random() for _ in range(3)]
    
    assert seq1 == seq2
