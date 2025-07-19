# PyTorch Dataset and DataLoader for AI2 ARC Challenge

This implementation provides a complete PyTorch Dataset and DataLoader solution for the AI2 ARC Challenge question answering dataset. The implementation follows PyTorch best practices and provides efficient data loading capabilities for training and evaluation.

## Features

- **Custom PyTorch Dataset**: `ARCChallengeDataset` class that handles the AI2 ARC Challenge CSV data
- **Automatic split handling**: Support for train/validation/test splits
- **Robust choice parsing**: Handles complex choice structures in the dataset
- **DataLoader factories**: Convenient functions to create DataLoaders with optimal configurations
- **Efficient data loading**: Supports batching, shuffling, and multi-worker processing
- **Transform support**: Optional transform functions for data preprocessing
- **Comprehensive testing**: Full test suite to ensure reliability

## Installation

Make sure you have PyTorch installed:

```bash
pip install torch pandas
```

## Quick Start

```python
from src.data.arc_dataset import ARCChallengeDataset, create_arc_dataloaders

# Create DataLoaders for all splits
csv_path = 'data/raw/ai2_arc/ARC-Challenge.csv'
dataloaders = create_arc_dataloaders(
    csv_path=csv_path,
    batch_size=32,
    num_workers=4,
    shuffle_train=True
)

# Access individual dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['validation']
test_loader = dataloaders['test']

# Iterate through batches
for batch in train_loader:
    questions = batch['question']  # List of question strings
    choices = batch['choices']     # List of choice lists
    answers = batch['answer_key']  # List of correct answer labels
    answer_indices = batch['answer_idx']  # List of correct answer indices
    
    # Your training code here
    break
```

## Dataset Structure

Each sample from the dataset contains:

- `id`: Unique question identifier
- `question`: Question text
- `choices`: List of answer choice texts
- `choice_labels`: List of choice labels (e.g., ['A', 'B', 'C', 'D'])
- `answer_key`: Correct answer label (e.g., 'B')
- `answer_idx`: Index of correct answer (0-based)

## API Reference

### ARCChallengeDataset

```python
dataset = ARCChallengeDataset(
    csv_path='data/raw/ai2_arc/ARC-Challenge.csv',
    split='train',           # 'train', 'validation', or 'test'
    transform=None          # Optional transform function
)
```

### create_arc_dataloaders

```python
dataloaders = create_arc_dataloaders(
    csv_path='path/to/ARC-Challenge.csv',
    batch_size=32,          # Batch size for all dataloaders
    num_workers=4,          # Number of worker processes
    shuffle_train=True,     # Whether to shuffle training data
    transform=None          # Optional transform function
)
```

### create_single_dataloader

```python
dataloader = create_single_dataloader(
    csv_path='path/to/ARC-Challenge.csv',
    split='train',          # Data split to use
    batch_size=32,          # Batch size
    num_workers=4,          # Number of worker processes
    shuffle=None,           # Shuffle (None = auto: True for train, False for others)
    transform=None          # Optional transform function
)
```

## Custom Transforms

You can apply custom transforms to the data:

```python
def question_transform(sample):
    """Add question length to the sample."""
    sample['question_length'] = len(sample['question'])
    sample['question_words'] = len(sample['question'].split())
    return sample

dataset = ARCChallengeDataset(
    csv_path='data/raw/ai2_arc/ARC-Challenge.csv',
    split='train',
    transform=question_transform
)
```

## Dataset Statistics

- **Training set**: ~1,119 samples
- **Validation set**: ~299 samples  
- **Test set**: ~1,172 samples
- **Total**: ~2,590 samples
- **Choice format**: Typically 4 multiple choice options (A, B, C, D)

## Running Tests

Run the comprehensive test suite:

```bash
python -m pytest tests/test_data/test_arc_dataset.py -v
```

## Example Usage

See `examples/arc_dataset_demo.py` for a complete demonstration:

```bash
cd /path/to/self-explaination-thesis
PYTHONPATH=. python examples/arc_dataset_demo.py
```

## Performance Considerations

- **Batch size**: Recommended batch sizes: 16-64 for most use cases
- **Number of workers**: Set `num_workers=0` for debugging, 2-8 for production
- **Memory usage**: Each sample contains text data, memory usage scales with batch size
- **Caching**: The dataset loads all data into memory for fast access

## Error Handling

The implementation includes robust error handling:

- **Invalid splits**: Return empty datasets
- **Parsing errors**: Fall back to default choices with warning messages
- **Missing files**: Raise clear error messages
- **Malformed data**: Attempt multiple parsing strategies before falling back

## Integration with Training Loops

Example training loop integration:

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# Load data
dataloaders = create_arc_dataloaders('data/raw/ai2_arc/ARC-Challenge.csv', batch_size=32)
train_loader = dataloaders['train']

# Your model, loss, optimizer
model = YourQuestionAnsweringModel()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        questions = batch['question']
        choices = batch['choices']
        targets = torch.tensor(batch['answer_idx'])
        
        # Forward pass
        outputs = model(questions, choices)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Contributing

When contributing to this dataset implementation:

1. Run tests to ensure compatibility
2. Follow PyTorch best practices
3. Update documentation for any API changes
4. Consider performance implications of changes

## License

This implementation is part of the self-explanation thesis project and follows the project's licensing terms.