# Job Configuration in main.py

The main.py file now supports different job limits for CPU and GPU operations through environment variables.

## Environment Variables

### CPU Jobs
- `EP_CPU_NJOBS`: Controls both CPU threading and process parallelism (default: 1)
  - When = 1: Single-threaded operations within a single process
  - When > 1: Multiple processes with single-threaded operations (process parallelism)

### GPU Jobs  
- `EP_GPU_NJOBS`: Controls CUDA job behavior (default: 1)
  - When set to 1: Enables `CUDA_LAUNCH_BLOCKING=1` for synchronous GPU operations
  - When set to >1: Enables `CUDA_LAUNCH_BLOCKING=0` for asynchronous GPU operations

## Usage Examples

### Conservative Settings (Default)
```bash
# Single CPU job, single GPU job, single process
python -m src.main -c my_config
```

### Multi-threaded CPU Operations
```bash
# Allow 4 CPU threads for BLAS/OMP and PyTorch operations (single process)
EP_CPU_NJOBS=1 python -m src.main -c my_config
```

### Multi-process Execution
```bash
# Run 8 parallel processes (each with single-threaded operations)
EP_CPU_NJOBS=8 python -m src.main -c my_config
```

### GPU-Optimized Settings
```bash
# Allow asynchronous GPU operations
EP_GPU_NJOBS=4 python -m src.main -c my_config
```

### Combined Settings
```bash
# Multi-process with asynchronous GPU operations
EP_CPU_NJOBS=8 EP_GPU_NJOBS=4 python -m src.main -c my_config
```

## Notes

- **CPU Jobs**: Controls both threading within processes and the number of parallel processes
  - Use `EP_CPU_NJOBS=1` for single-threaded operations in one process
  - Use `EP_CPU_NJOBS>1` for multiple processes (each with single-threaded operations)
- **GPU Jobs**: Controls CUDA concurrency within each process
- The default settings are conservative to avoid oversubscription and memory issues
- Monitor system resources (CPU, memory, GPU) when increasing job limits
- For most workloads, process parallelism (`EP_CPU_NJOBS>1`) is more effective than threading
