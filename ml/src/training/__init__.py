"""
Training modules with multi-node distributed support
"""
from training.distributed import setup_distributed, cleanup_distributed
from training.trainer import Trainer

__all__ = ["Trainer", "setup_distributed", "cleanup_distributed"]
