"""
Example usage of the ARC Challenge Dataset and DataLoader.

This script demonstrates how to use the PyTorch Dataset and DataLoader
implementations for the AI2 ARC Challenge question answering task.
"""

import torch
from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCChallengeDataset, create_arc_dataloaders, create_single_dataloader


def main():
    """Demonstrate usage of ARC Challenge Dataset and DataLoader."""
    
    # Path to the ARC Challenge CSV file
    csv_path = 'data/raw/ai2_arc/ARC-Challenge.csv'
    
    print("=== ARC Challenge Dataset Demo ===\n")
    
    # 1. Create individual datasets
    print("1. Creating individual datasets for each split:")
    train_dataset = ARCChallengeDataset(csv_path, split='train')
    val_dataset = ARCChallengeDataset(csv_path, split='validation')
    test_dataset = ARCChallengeDataset(csv_path, split='test')
    
    print(f"   Train dataset: {len(train_dataset)} samples")
    print(f"   Validation dataset: {len(val_dataset)} samples")
    print(f"   Test dataset: {len(test_dataset)} samples")
    print()
    
    # 2. Examine a sample
    print("2. Examining a sample from the training dataset:")
    sample = train_dataset[0]
    print(f"   ID: {sample['id']}")
    print(f"   Question: {sample['question']}")
    print(f"   Choices:")
    for i, (label, choice) in enumerate(zip(sample['choice_labels'], sample['choices'])):
        marker = " ★" if i == sample['answer_idx'] else "  "
        print(f"     {label}. {choice}{marker}")
    print(f"   Correct Answer: {sample['answer_key']} (index {sample['answer_idx']})")
    print()
    
    # 3. Create DataLoaders using the factory function
    print("3. Creating DataLoaders using factory function:")
    batch_size = 8
    dataloaders = create_arc_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for compatibility
        shuffle_train=True
    )
    
    print(f"   Created DataLoaders for: {list(dataloaders.keys())}")
    print()
    
    # 4. Iterate through a batch
    print("4. Processing a batch from the training DataLoader:")
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))
    
    print(f"   Batch size: {len(batch['id'])}")
    print(f"   Sample questions from batch:")
    for i in range(min(3, len(batch['id']))):
        print(f"     {i+1}. {batch['question'][i][:60]}...")
        print(f"        Answer: {batch['answer_key'][i]}")
    print()
    
    # 5. Create a single DataLoader with custom parameters
    print("5. Creating a single DataLoader with custom parameters:")
    custom_loader = create_single_dataloader(
        csv_path=csv_path,
        split='validation',
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    print(f"   Validation DataLoader batch size: {custom_loader.batch_size}")
    print(f"   Shuffle: {custom_loader.sampler is not None}")
    print()
    
    # 6. Demonstrate iteration over multiple batches
    print("6. Iterating over multiple batches (first 3 batches of validation):")
    batch_count = 0
    for batch in custom_loader:
        batch_count += 1
        print(f"   Batch {batch_count}: {len(batch['id'])} samples")
        if batch_count >= 3:
            break
    print()
    
    # 7. Show dataset statistics
    print("7. Dataset statistics:")
    all_dataloaders = [dataloaders['train'], dataloaders['validation'], dataloaders['test']]
    total_samples = sum(len(loader.dataset) for loader in all_dataloaders)
    
    print(f"   Total samples across all splits: {total_samples}")
    print(f"   Average batch size: {batch_size}")
    print(f"   Total batches (approx): {total_samples // batch_size}")
    print()
    
    # 8. Demonstrate custom transform (optional)
    print("8. Example with custom transform:")
    
    def sample_transform(sample):
        """Example transform that adds question length."""
        sample['question_length'] = len(sample['question'])
        return sample
    
    transformed_dataset = ARCChallengeDataset(
        csv_path, 
        split='train', 
        transform=sample_transform
    )
    
    transformed_sample = transformed_dataset[0]
    print(f"   Question: {transformed_sample['question'][:50]}...")
    print(f"   Question length: {transformed_sample['question_length']} characters")
    print()
    
    print("=== Demo Complete ===")


if __name__ == '__main__':
    main()