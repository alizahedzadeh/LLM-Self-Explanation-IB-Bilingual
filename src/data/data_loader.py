import logging
import pandas as pd
from datasets import load_dataset

class DataLoader:
    """
    Flexible DataLoader for ARC-Challenge dataset supporting both CSV and Huggingface datasets.
    - If a CSV path is provided, loads data from CSV into a pandas DataFrame.
    - If not, loads from Huggingface's 'allenai/ai2_arc', subset 'ARC-Challenge', and processes splits.
    """

    def __init__(self, csv_path=None, dataset_name="allenai/ai2_arc", subset="ARC-Challenge"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.subset = subset
        self.df = None
        self.train_df = None
        self.test_df = None
        self.validation_df = None

    def load(self):
        if self.csv_path:
            self.logger.info(f"Loading dataset from CSV: {self.csv_path}")
            try:
                self.df = pd.read_csv(self.csv_path)
                self.logger.info(f"CSV loaded successfully. Shape: {self.df.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load CSV: {e}")
                raise
            # Optionally extract splits if present
            if "split" in self.df.columns:
                self.train_df = self.df[self.df["split"] == "train"]
                self.test_df = self.df[self.df["split"] == "test"]
                self.validation_df = self.df[self.df["split"] == "validation"]
        else:
            self.logger.info(f"Loading dataset from Huggingface: {self.dataset_name}, subset: {self.subset}")
            try:
                ds = load_dataset(self.dataset_name, self.subset)
                self.train_df = ds["train"].to_pandas()
                self.train_df["split"] = "train"
                self.test_df = ds["test"].to_pandas()
                self.test_df["split"] = "test"
                self.validation_df = ds["validation"].to_pandas()
                self.validation_df["split"] = "validation"
                self.df = pd.concat([self.train_df, self.test_df, self.validation_df], ignore_index=True)
                self.logger.info(f"Dataset loaded and concatenated. Shape: {self.df.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load Huggingface dataset: {e}")
                raise

    def save_csv(self, filename="ARC-Challenge.csv"):
        if self.df is None:
            self.logger.error("No DataFrame loaded. Call load() first.")
            raise RuntimeError("No DataFrame loaded. Call load() first.")
        self.df.to_csv(filename, index=False)
        self.logger.info(f"DataFrame saved to {filename}")

    def get_split(self, split):
        if self.df is None:
            self.logger.error("No DataFrame loaded. Call load() first.")
            raise RuntimeError("No DataFrame loaded. Call load() first.")
        if "split" not in self.df.columns:
            self.logger.warning("No 'split' column found in DataFrame. Returning full DataFrame.")
            return self.df
        split_df = self.df[self.df["split"] == split]
        self.logger.info(f"Returning split '{split}' with shape {split_df.shape}")
        return split_df

    def get_full(self):
        if self.df is None:
            self.logger.error("No DataFrame loaded. Call load() first.")
            raise RuntimeError("No DataFrame loaded. Call load() first.")
        return self.df

# Example usage:
# loader = DataLoader(csv_path="ARC-Challenge.csv")  # For CSV
# loader = DataLoader()  # For Huggingface dataset
# loader.load()
# train_df = loader.get_split("train")
# full_df = loader.get_full()
# loader.save_csv("my_dataset.csv")