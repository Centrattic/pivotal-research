#!/usr/bin/env python3

import numpy as np
import torch
import psutil
import time
from pathlib import Path
import sys
sys.path.append('src')

from src.probes import LinearProbe

def test_memory_optimization():
    print("Testing memory optimization...")
    
    # Create a larger test setup to simulate your real scenario
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 512  # Larger model dimension
    n_samples = 10000  # More samples
    seq_len = 128  # Longer sequences
    batch_size = 64  # Smaller batch size to reduce memory pressure
    
    print(f"Creating synthetic data: {n_samples} samples, {seq_len} seq_len, {d_model} d_model")
    print(f"Device: {device}, Batch size: {batch_size}")
    
    # Monitor initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"Initial CPU memory: {initial_memory:.2f} GB")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_gpu = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_gpu:.2f} GB")
    
    # Create synthetic data
    print("Creating data...")
    start_time = time.time()
    X = np.random.randn(n_samples, seq_len, d_model).astype(np.float16)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    mask = np.ones((n_samples, seq_len), dtype=bool)
    
    data_time = time.time() - start_time
    memory_after_data = process.memory_info().rss / 1024**3
    print(f"Data creation time: {data_time:.2f}s")
    print(f"Memory after data creation: {memory_after_data:.2f} GB (+{memory_after_data - initial_memory:.2f} GB)")
    
    # Create a probe
    print("Creating probe...")
    probe = LinearProbe(d_model=d_model, device=device, task_type="classification", aggregation="mean")
    
    # Train the probe briefly
    print("Training probe...")
    start_time = time.time()
    probe.fit(X, y, mask=mask, epochs=3, lr=1e-2, batch_size=batch_size, verbose=True)
    train_time = time.time() - start_time
    
    final_memory = process.memory_info().rss / 1024**3
    if torch.cuda.is_available():
        final_gpu = torch.cuda.memory_allocated() / 1024**3
        print(f"Final GPU memory: {final_gpu:.2f} GB (+{final_gpu - initial_gpu:.2f} GB)")
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Final CPU memory: {final_memory:.2f} GB (+{final_memory - initial_memory:.2f} GB)")
    print(f"Memory efficiency: {(memory_after_data - initial_memory) / (final_memory - initial_memory):.1%} of total memory used for data")
    
    # Test prediction
    print("Testing prediction...")
    pred_start = time.time()
    predictions = probe.predict(X[:100], mask[:100], batch_size=32)
    pred_time = time.time() - pred_start
    print(f"Prediction time for 100 samples: {pred_time:.2f}s")
    
    print("✅ Memory optimization test complete!")

if __name__ == "__main__":
    test_memory_optimization() 