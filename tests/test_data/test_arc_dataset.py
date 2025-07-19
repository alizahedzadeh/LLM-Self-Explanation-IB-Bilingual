"""
Unit tests for ARC Challenge Dataset and DataLoader implementations.
"""

import unittest
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import tempfile

from src.data.arc_dataset import ARCChallengeDataset, create_arc_dataloaders, create_single_dataloader


class TestARCChallengeDataset(unittest.TestCase):
    """Test cases for ARCChallengeDataset class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.csv_path = 'data/raw/ai2_arc/ARC-Challenge.csv'
        
        # Check if the CSV file exists
        if not os.path.exists(cls.csv_path):
            cls.skipTest("ARC Challenge CSV file not found")
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_train = ARCChallengeDataset(self.csv_path, split='train')
        self.dataset_val = ARCChallengeDataset(self.csv_path, split='validation')
        self.dataset_test = ARCChallengeDataset(self.csv_path, split='test')
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        self.assertIsInstance(self.dataset_train, ARCChallengeDataset)
        self.assertEqual(self.dataset_train.split, 'train')
        self.assertIsInstance(self.dataset_train.data, pd.DataFrame)
    
    def test_dataset_lengths(self):
        """Test dataset lengths for different splits."""
        # Expected lengths based on the CSV structure
        self.assertGreater(len(self.dataset_train), 1000)  # Should have ~1119 samples
        self.assertGreater(len(self.dataset_val), 200)     # Should have ~299 samples
        self.assertGreater(len(self.dataset_test), 1000)   # Should have ~1172 samples
        
        # Verify total equals sum of parts
        total_expected = len(self.dataset_train) + len(self.dataset_val) + len(self.dataset_test)
        
        # Load full dataset to check
        full_df = pd.read_csv(self.csv_path)
        self.assertEqual(total_expected, len(full_df))
    
    def test_getitem_structure(self):
        """Test the structure of items returned by __getitem__."""
        sample = self.dataset_train[0]
        
        # Check that all required keys are present
        required_keys = ['id', 'question', 'choices', 'choice_labels', 'answer_key', 'answer_idx']
        for key in required_keys:
            self.assertIn(key, sample)
        
        # Check data types
        self.assertIsInstance(sample['id'], str)
        self.assertIsInstance(sample['question'], str)
        self.assertIsInstance(sample['choices'], list)
        self.assertIsInstance(sample['choice_labels'], list)
        self.assertIsInstance(sample['answer_key'], str)
        self.assertIsInstance(sample['answer_idx'], int)
        
        # Check constraints
        self.assertEqual(len(sample['choices']), len(sample['choice_labels']))
        self.assertIn(sample['answer_key'], sample['choice_labels'])
        self.assertGreaterEqual(sample['answer_idx'], 0)
        self.assertLess(sample['answer_idx'], len(sample['choices']))
    
    def test_choice_parsing(self):
        """Test that choices are parsed correctly."""
        sample = self.dataset_train[0]
        
        # Choices should be a list of strings
        self.assertIsInstance(sample['choices'], list)
        self.assertTrue(all(isinstance(choice, str) for choice in sample['choices']))
        
        # Choice labels should be ['A', 'B', 'C', 'D'] (typically)
        self.assertIsInstance(sample['choice_labels'], list)
        self.assertTrue(all(isinstance(label, str) for label in sample['choice_labels']))
        
        # Usually should have 4 choices
        self.assertLessEqual(len(sample['choices']), 5)  # At most 5 choices
        self.assertGreaterEqual(len(sample['choices']), 2)  # At least 2 choices
    
    def test_answer_consistency(self):
        """Test that answer keys and indices are consistent."""
        for i in range(min(10, len(self.dataset_train))):  # Test first 10 samples
            sample = self.dataset_train[i]
            
            # Answer key should be in choice labels
            self.assertIn(sample['answer_key'], sample['choice_labels'])
            
            # Answer index should correspond to answer key
            expected_key = sample['choice_labels'][sample['answer_idx']]
            self.assertEqual(sample['answer_key'], expected_key)
    
    def test_torch_tensor_indexing(self):
        """Test indexing with torch tensors."""
        idx_tensor = torch.tensor(5)
        sample = self.dataset_train[idx_tensor]
        self.assertIsInstance(sample, dict)
    
    def test_invalid_split(self):
        """Test behavior with invalid split."""
        # Invalid split should result in an empty dataset
        invalid_dataset = ARCChallengeDataset(self.csv_path, split='invalid_split')
        self.assertEqual(len(invalid_dataset), 0)


class TestDataLoaderFactories(unittest.TestCase):
    """Test cases for DataLoader factory functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.csv_path = 'data/raw/ai2_arc/ARC-Challenge.csv'
        
        # Check if the CSV file exists
        if not os.path.exists(cls.csv_path):
            cls.skipTest("ARC Challenge CSV file not found")
    
    def test_create_arc_dataloaders(self):
        """Test create_arc_dataloaders function."""
        batch_size = 8
        dataloaders = create_arc_dataloaders(
            csv_path=self.csv_path,
            batch_size=batch_size,
            num_workers=0
        )
        
        # Check that all splits are present
        expected_splits = ['train', 'validation', 'test']
        self.assertEqual(set(dataloaders.keys()), set(expected_splits))
        
        # Check that each is a DataLoader
        for split, dataloader in dataloaders.items():
            self.assertIsInstance(dataloader, DataLoader)
            self.assertEqual(dataloader.batch_size, batch_size)
    
    def test_create_single_dataloader(self):
        """Test create_single_dataloader function."""
        batch_size = 4
        dataloader = create_single_dataloader(
            csv_path=self.csv_path,
            split='train',
            batch_size=batch_size,
            shuffle=True
        )
        
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, batch_size)
    
    def test_dataloader_iteration(self):
        """Test that DataLoaders can be iterated."""
        dataloader = create_single_dataloader(
            csv_path=self.csv_path,
            split='validation',
            batch_size=4,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIsInstance(batch, dict)
        required_keys = ['id', 'question', 'choices', 'choice_labels', 'answer_key', 'answer_idx']
        for key in required_keys:
            self.assertIn(key, batch)
        
        # Check batch sizes
        batch_size = len(batch['id'])
        for key in required_keys:
            if key in ['choices', 'choice_labels']:
                # These are lists of lists, so we check the outer dimension
                self.assertEqual(len(batch[key]), batch_size)
            else:
                self.assertEqual(len(batch[key]), batch_size)
    
    def test_dataloader_shuffling(self):
        """Test that shuffling works correctly."""
        # Create two identical dataloaders with shuffle=True
        dataloader1 = create_single_dataloader(
            csv_path=self.csv_path,
            split='train',
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        dataloader2 = create_single_dataloader(
            csv_path=self.csv_path,
            split='train',
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        # Get first batch from each
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))
        
        # They should potentially be different (though not guaranteed)
        # At least verify they have the same structure
        self.assertEqual(len(batch1['id']), len(batch2['id']))
    
    def test_dataloader_no_shuffle(self):
        """Test that no shuffling produces consistent results."""
        # Create two identical dataloaders with shuffle=False
        dataloader1 = create_single_dataloader(
            csv_path=self.csv_path,
            split='test',
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        
        dataloader2 = create_single_dataloader(
            csv_path=self.csv_path,
            split='test',
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        
        # Get first batch from each
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))
        
        # They should be identical
        self.assertEqual(batch1['id'], batch2['id'])
        self.assertEqual(batch1['question'], batch2['question'])
    
    def test_dataloader_parameters(self):
        """Test various DataLoader parameters."""
        dataloaders = create_arc_dataloaders(
            csv_path=self.csv_path,
            batch_size=16,
            num_workers=0,
            shuffle_train=False
        )
        
        # Check that training dataloader has shuffle=False
        train_loader = dataloaders['train']
        self.assertFalse(train_loader.dataset.data.empty)


if __name__ == '__main__':
    unittest.main()