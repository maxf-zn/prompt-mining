"""
GPU utilities for memory management.
"""
import gc
import torch


def clear_gpu_memory():
    """
    Clear GPU memory by emptying cache and running garbage collection.

    This pattern is useful after processing large models or when recovering
    from GPU OOM errors.

    Usage:
        try:
            result = model(inputs)
        except torch.cuda.OutOfMemoryError:
            clear_gpu_memory()
            # Retry with smaller batch or skip
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_gpu_memory_stats(device_id: int = 0) -> dict:
    """
    Get GPU memory statistics.

    Args:
        device_id: CUDA device ID

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {'available': False}

    device = torch.device(f'cuda:{device_id}')
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    return {
        'available': True,
        'device_id': device_id,
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - allocated,
    }
