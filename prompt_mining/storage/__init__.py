"""
Storage backends for prompt mining platform.
"""
from .base import StorageBackend
from .local_storage import LocalStorage

__all__ = [
    'StorageBackend',
    'LocalStorage',
]
