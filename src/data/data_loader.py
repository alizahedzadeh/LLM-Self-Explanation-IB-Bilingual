"""
Main data loading module for the self-explanation thesis project.

This module provides convenient imports for dataset and dataloader functionality,
specifically for the AI2 ARC Challenge question answering tasks.
"""

# Import PyTorch Dataset and DataLoader implementations for ARC Challenge
from .arc_dataset import (
    ARCChallengeDataset,
    create_arc_dataloaders,
    create_single_dataloader
)

# Make the main classes available at module level
__all__ = [
    "ARCChallengeDataset",
    "create_arc_dataloaders", 
    "create_single_dataloader"
]
