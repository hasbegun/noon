"""
Distributed training utilities for Apple Silicon (MPS) multi-node setup
"""
import os
import socket
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from loguru import logger

from config import config


def setup_distributed(
    backend: str = "gloo",
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Setup distributed training

    Args:
        backend: Distributed backend ('gloo' for MPS, 'nccl' for CUDA)
        init_method: Initialization method (e.g., 'tcp://localhost:29500')
        rank: Process rank
        world_size: Total number of processes

    Returns:
        Tuple of (rank, world_size)
    """
    # Check if already initialized
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Get environment variables
    rank = rank or int(os.environ.get("RANK", 0))
    world_size = world_size or int(os.environ.get("WORLD_SIZE", config.num_nodes))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Single node/device case
    if world_size == 1:
        logger.info("Single node training (no distribution)")
        return 0, 1

    # Setup init method
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{master_addr}:{master_port}"

    logger.info(f"Initializing distributed training: rank={rank}, world_size={world_size}")
    logger.info(f"Backend: {backend}, Init method: {init_method}")

    try:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        logger.info(f"Distributed training initialized successfully")
        logger.info(f"Rank: {rank}/{world_size}, Local rank: {local_rank}")

        return rank, world_size

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        logger.warning("Falling back to single-node training")
        return 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        logger.info("Cleaning up distributed training")
        dist.destroy_process_group()


def is_main_process(rank: int = None) -> bool:
    """Check if this is the main process"""
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    return rank == 0


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronization barrier"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce operation"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """All-gather operation"""
    if not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_multinode_env(
    num_nodes: int,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: Optional[int] = None,
):
    """
    Setup environment variables for multi-node training

    Args:
        num_nodes: Total number of nodes
        node_rank: Rank of this node
        master_addr: Address of master node
        master_port: Port for communication
    """
    if master_port is None:
        master_port = find_free_port() if node_rank == 0 else 29500

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(num_nodes)
    os.environ["RANK"] = str(node_rank)
    os.environ["LOCAL_RANK"] = "0"  # Single GPU per node for MPS

    logger.info(f"Multi-node environment configured:")
    logger.info(f"  Nodes: {num_nodes}")
    logger.info(f"  Node rank: {node_rank}")
    logger.info(f"  Master: {master_addr}:{master_port}")
