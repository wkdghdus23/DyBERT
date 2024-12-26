import pandas as pd
import torch
from typing import List, Tuple
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BatchEncoding
from dybert.tokenizer import Tokenizer

class MyDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling string data and creating inputs for Masked Language Modeling (MLM)
    or downstream regression/classification tasks.

    Args:
        task_type (str): Type of task ('mlm' or 'downstream').
        input_df (pd.DataFrame): DataFrame containing input strings and optionally target values.
        vocab_file (str): Path to the vocabulary file used for tokenizing strings.
        tokenizer (BertTokenizer): A tokenizer used to tokenize and encode the input strings.
        max_len (int): Maximum length for tokenizing the input string sequences.
        mask_ratio (float): Ratio of tokens to be masked for MLM. Required if task_type is 'mlm'.
        target_name (List[str]): List of target column names for downstream task. Required if task_type is 'downstream'.
    """
    def __init__(self, 
                 task_type: str,
                 input_df: pd.DataFrame, 
                 vocab_file: str,
                 tokenizer: BertTokenizer,
                 max_len: int,
                 mask_ratio: float,
                 target_name: List[str]):
        self.task_type = task_type
        self.input_df = input_df
        self.vocab_file = vocab_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.target_name = target_name

        # Validate arguments based on task type
        if self.task_type == 'mlm' and self.mask_ratio == 0.0:
            raise ValueError("mask_ratio must be provided for MLM task.")
        if self.task_type == 'downstream' and self.target_name is None:
            raise ValueError("target_name must be provided for downstream task.")

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of input strings in the dataset.
        """
        return len(self.input_df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a data sample at the given index, including the tokenized input and optionally masked tokens or target values.

        Args:
            idx (int): Index of the input string to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
                  'input_ids' (Tensor): Token IDs for the input string.
                  'attention_mask' (Tensor): Attention mask indicating valid tokens.
                  'labels' (Tensor): Masked tokens or target values.
        """
        # Get the input string at the specified index
        text = self.input_df['input'].iloc[idx]

        # Tokenize and encode the input string using the tokenizer
        inputs: BatchEncoding = Tokenizer(smiles=text,
                                          vocab_file=self.vocab_file, 
                                          tokenizer=self.tokenizer,
                                          max_len=self.max_len)

        # Convert the encoded input_ids and attention mask to tensors
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).squeeze(0)  # Remove extra dimension
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).squeeze(0)  # Remove extra dimension

        # For MLM Task
        if self.task_type == 'mlm':
            labels = input_ids.clone()
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand < self.mask_ratio) & (input_ids != self.tokenizer.cls_token_id) & \
                       (input_ids != self.tokenizer.sep_token_id) & (input_ids != self.tokenizer.pad_token_id)

            input_ids[mask_arr] = self.tokenizer.mask_token_id

        # For Downstream Task
        elif self.task_type == 'downstream':
            target_list = self.input_df[self.target_name].values
            labels = torch.tensor(target_list[idx], dtype=torch.float32)

        return {'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,}


def MyDataLoader(task_type: str,
                 vocab_file: str,
                 tokenizer: BertTokenizer,
                 max_len: int,
                 batch_size: int,
                 mask_ratio: float,
                 target_name: List[str],
                 df_train: pd.DataFrame,
                 df_val: pd.DataFrame,
                 df_test: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and optionally testing datasets for Masked Language Modeling (MLM)
    or downstream tasks.

    Args:
        task_type (str): Type of task ('mlm' or 'downstream').
        vocab_file (str): Path to the vocabulary file used for tokenizing string.
        tokenizer (BertTokenizer): Tokenizer used for text tokenization.
        max_len (int): Maximum length for input sequences.
        batch_size (int): Batch size for the DataLoader.
        mask_ratio (float): Ratio of tokens to be masked for MLM.
        target_name (List[str]): Target names for downstream tasks. Required if task_type is 'downstream'.
        df_train (pd.DataFrame): DataFrame for training data.
        df_val (pd.DataFrame): DataFrame for validation data.
        df_test (pd.DataFrame): DataFrame for test data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and optionally testing datasets.
    """
    # Create datasets for training, validation, and optionally testing
    train_dataloader = None

    if df_train is not None:
        train_dataset = MyDataset(task_type=task_type,
                                  input_df=df_train, 
                                  vocab_file=vocab_file,
                                  tokenizer=tokenizer,
                                  max_len=max_len,
                                  mask_ratio=mask_ratio, 
                                  target_name=target_name)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = None

    if df_val is not None:
        val_dataset = MyDataset(task_type=task_type,
                                input_df=df_val,
                                vocab_file=vocab_file,
                                tokenizer=tokenizer,
                                max_len=max_len,
                                mask_ratio=mask_ratio,
                                target_name=target_name)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = None

    if df_test is not None:
        test_dataset = MyDataset(task_type=task_type,
                                 input_df=df_test,
                                 vocab_file=vocab_file,
                                 tokenizer=tokenizer,
                                 max_len=max_len,
                                 mask_ratio=mask_ratio,
                                 target_name=target_name)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
