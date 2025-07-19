"""
PyTorch Dataset and DataLoader for AI2 ARC Challenge question answering tasks.

This module provides a custom Dataset class and DataLoader factory function
for loading and processing the AI2 ARC Challenge dataset.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from typing import Dict, List, Tuple, Union, Optional


class ARCChallengeDataset(Dataset):
    """
    PyTorch Dataset for AI2 ARC Challenge data.
    
    This dataset loads questions, multiple choice answers, and correct answer keys
    from the AI2 ARC Challenge dataset. It handles train/validation/test splits
    and processes the choices structure appropriately.
    """
    
    def __init__(self, csv_path: str, split: str = "train", transform=None):
        """
        Initialize the ARC Challenge Dataset.
        
        Args:
            csv_path (str): Path to the ARC Challenge CSV file
            split (str): Data split to use ('train', 'validation', 'test')
            transform: Optional transform to be applied on a sample
        """
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        
        # Load and filter data
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and filter data for the specified split."""
        df = pd.read_csv(self.csv_path)
        
        # Filter by split
        filtered_df = df[df['split'] == self.split].copy()
        
        # Reset index
        filtered_df = filtered_df.reset_index(drop=True)
        
        return filtered_df
    
    def _parse_choices(self, choices_str: str) -> Dict[str, List[str]]:
        """
        Parse the choices string into a structured format.
        
        Args:
            choices_str (str): String representation of choices
            
        Returns:
            Dict with 'text' and 'label' keys containing lists of choices and labels
        """
        try:
            # Method 1: Use pandas eval for most cases
            import pandas as pd
            choices_dict = pd.eval(choices_str)
            
            # Extract text and label arrays
            text_choices = choices_dict['text'].tolist()
            label_choices = choices_dict['label'].tolist()
            
            return {
                'text': text_choices,
                'label': label_choices
            }
            
        except Exception as e1:
            # Method 2: Try regex-based parsing for complex cases
            try:
                import re
                
                # Extract text array using regex
                text_pattern = r"'text': array\(\[([^\]]+)\]"
                text_match = re.search(text_pattern, choices_str, re.DOTALL)
                if text_match:
                    text_content = text_match.group(1)
                    # Split by quotes and clean up
                    text_parts = re.findall(r"'([^']+)'", text_content)
                    text_choices = [part.strip() for part in text_parts if part.strip()]
                else:
                    raise ValueError("Could not extract text choices with regex")
                
                # Extract label array using regex
                label_pattern = r"'label': array\(\[([^\]]+)\]"
                label_match = re.search(label_pattern, choices_str, re.DOTALL)
                if label_match:
                    label_content = label_match.group(1)
                    # Split by quotes and clean up
                    label_parts = re.findall(r"'([^']+)'", label_content)
                    label_choices = [part.strip() for part in label_parts if part.strip()]
                else:
                    raise ValueError("Could not extract label choices with regex")
                
                return {
                    'text': text_choices,
                    'label': label_choices
                }
                
            except Exception as e2:
                # Method 3: Manual parsing with string operations
                try:
                    # Find text section
                    text_start = choices_str.find("'text': array([")
                    if text_start == -1:
                        raise ValueError("No text array found")
                    
                    # Find the end of text array (look for closing bracket before 'label')
                    label_pos = choices_str.find("'label':", text_start)
                    if label_pos == -1:
                        raise ValueError("No label section found")
                    
                    # Extract text section and find the array content
                    text_section = choices_str[text_start:label_pos]
                    array_start = text_section.find("[")
                    array_end = text_section.rfind("]")
                    
                    if array_start == -1 or array_end == -1:
                        raise ValueError("Could not find text array boundaries")
                    
                    text_array_content = text_section[array_start+1:array_end]
                    
                    # Parse text choices more carefully
                    text_choices = []
                    current_choice = ""
                    in_quote = False
                    i = 0
                    while i < len(text_array_content):
                        char = text_array_content[i]
                        if char == "'" and (i == 0 or text_array_content[i-1] != "\\"):
                            if in_quote:
                                # End of choice
                                text_choices.append(current_choice)
                                current_choice = ""
                                in_quote = False
                            else:
                                # Start of choice
                                in_quote = True
                        elif in_quote:
                            current_choice += char
                        i += 1
                    
                    # Extract label section
                    label_start = choices_str.find("'label': array([")
                    if label_start == -1:
                        raise ValueError("No label array found")
                    
                    label_section = choices_str[label_start:]
                    array_start = label_section.find("[")
                    array_end = label_section.find("]")
                    
                    if array_start == -1 or array_end == -1:
                        raise ValueError("Could not find label array boundaries")
                    
                    label_array_content = label_section[array_start+1:array_end]
                    
                    # Parse labels (usually simpler)
                    import re
                    label_matches = re.findall(r"'([A-Z])'", label_array_content)
                    label_choices = label_matches if label_matches else ['A', 'B', 'C', 'D'][:len(text_choices)]
                    
                    # Ensure we have the same number of choices and labels
                    if len(text_choices) == 0:
                        raise ValueError("No text choices found")
                    
                    # If we have fewer labels than choices, generate them
                    if len(label_choices) < len(text_choices):
                        label_choices = [chr(ord('A') + i) for i in range(len(text_choices))]
                    
                    return {
                        'text': text_choices,
                        'label': label_choices[:len(text_choices)]
                    }
                    
                except Exception as e3:
                    # Final fallback: create default choices
                    # print(f"Warning: Could not parse choices (tried 3 methods): {e1} / {e2} / {e3}")
                    return {
                        'text': ['Choice A', 'Choice B', 'Choice C', 'Choice D'],
                        'label': ['A', 'B', 'C', 'D']
                    }
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str], int]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict containing:
                - 'id': Question ID
                - 'question': Question text
                - 'choices': List of choice texts
                - 'choice_labels': List of choice labels (A, B, C, D)
                - 'answer_key': Correct answer label
                - 'answer_idx': Index of correct answer (0-based)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.data.iloc[idx]
        
        # Parse choices
        choices = self._parse_choices(row['choices'])
        
        # Find answer index
        answer_key = row['answerKey']
        try:
            answer_idx = choices['label'].index(answer_key)
        except ValueError:
            # If answer key not found, default to 0
            answer_idx = 0
        
        sample = {
            'id': row['id'],
            'question': row['question'],
            'choices': choices['text'],
            'choice_labels': choices['label'],
            'answer_key': answer_key,
            'answer_idx': answer_idx
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_arc_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    transform=None
) -> Dict[str, DataLoader]:
    """
    Create DataLoader instances for train, validation, and test splits.
    
    Args:
        csv_path (str): Path to the ARC Challenge CSV file
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        shuffle_train (bool): Whether to shuffle the training data
        transform: Optional transform to be applied on samples
        
    Returns:
        Dict containing DataLoader instances for each split
    """
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length choice lists."""
        # Initialize the result dictionary
        result = {}
        
        # Get all the keys from the first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['choices', 'choice_labels']:
                # For lists of lists, keep as list of lists
                result[key] = [sample[key] for sample in batch]
            else:
                # For regular fields, create a list
                result[key] = [sample[key] for sample in batch]
        
        return result
    
    dataloaders = {}
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        # Create dataset
        dataset = ARCChallengeDataset(
            csv_path=csv_path,
            split=split,
            transform=transform
        )
        
        # Determine shuffle setting
        shuffle = shuffle_train if split == 'train' else False
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


def create_single_dataloader(
    csv_path: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = None,
    transform=None
) -> DataLoader:
    """
    Create a single DataLoader for a specific split.
    
    Args:
        csv_path (str): Path to the ARC Challenge CSV file
        split (str): Data split ('train', 'validation', 'test')
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle data. If None, defaults to True for train, False for others
        transform: Optional transform to be applied on samples
        
    Returns:
        DataLoader instance
    """
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length choice lists."""
        # Initialize the result dictionary
        result = {}
        
        # Get all the keys from the first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['choices', 'choice_labels']:
                # For lists of lists, keep as list of lists
                result[key] = [sample[key] for sample in batch]
            else:
                # For regular fields, create a list
                result[key] = [sample[key] for sample in batch]
        
        return result
    
    # Default shuffle behavior
    if shuffle is None:
        shuffle = split == 'train'
    
    # Create dataset
    dataset = ARCChallengeDataset(
        csv_path=csv_path,
        split=split,
        transform=transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return dataloader