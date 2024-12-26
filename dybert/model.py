import torch
import torch.nn as nn
from typing import List
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizer

class BertForMLM(nn.Module):
    """
    Masked Language Model (MLM) using a pre-trained BERT model.

    Args:
        tokenizer (BertTokenizer): Tokenizer used for text tokenization.
        hidden_size (int): Dimensionality of the encoder layers.
        num_attention_heads (int): Number of attention heads in each attention layer.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
    """
    def __init__(self,
                 tokenizer: BertTokenizer,
                 hidden_size: int = 768, 
                 num_attention_heads: int = 12, 
                 num_hidden_layers: int = 12):
        super(BertForMLM, self).__init__()

        # Configure BERT model for Masked Language Modeling
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.is_decoder = False  # Specify that this model is not a decoder
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.config.num_hidden_layers = num_hidden_layers
        self.config.vocab_size = self.tokenizer.vocab_size  # Set vocabulary size based on the tokenizer

        # Initialize the BERT model for Masked Language Modeling
        self.model = BertForMaskedLM(self.config)

    def forward(self, input_ids, labels, attention_mask):
        """
        Performs a forward pass through the MLM model.

        Args:
            input_ids (Tensor): Tensor containing token IDs for input sequences.
            labels (Tensor): Tensor containing the labels for MLM (same as input_ids but with masked tokens).
            attention_mask (Tensor): Tensor indicating which tokens should be attended to (1 for valid tokens, 0 for padding).

        Returns:
            MaskedLMOutput: The output of the BERT MLM model, including loss (if labels are provided) and logits.
        """
        return self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: BertTokenizer):
        """
        Load a pre-trained MLM model from a given path.

        Args:
            pretrained_path (str): Path to the pre-trained model directory.
            tokenizer (BertTokenizer): Tokenizer used for text tokenization.

        Returns:
            BertForMLM: An instance of BertForMLM with the pre-trained weights loaded.
        """
        model = cls(tokenizer=tokenizer)
        model.model = BertForMaskedLM.from_pretrained(pretrained_path)
        return model

class BertForDownstream(nn.Module):
    """
    A custom model for downstream task using a pre-trained BERT model or an untrained BERT model.

    Args:
        tokenizer (BertTokenizer): Tokenizer used for text tokenization.
        target_name (List[str]): List of target columns.
        hidden_size (int): Dimensionality of the encoder layers.
        num_attention_heads (int): Number of attention heads in each attention layer.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
    """
    def __init__(self,
                 tokenizer: BertTokenizer,
                 target_name: List[str],
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12):
        super(BertForDownstream, self).__init__()

        # Configure BERT model from scratch (random initialization)
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.is_decoder = False  # Specify that this model is not a decoder
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads
        self.config.num_hidden_layers = num_hidden_layers
        self.config.vocab_size = self.tokenizer.vocab_size  # Set vocabulary size based on the tokenizer

        self.bert = BertModel(self.config)  # BERT model (randomly initialized)
        self.target_name = target_name  # List of target columns
        self.fc = nn.Linear(self.bert.config.hidden_size, len(self.target_name))  # Fully connected layer to predict target values
        self.dropout = nn.Dropout(0.3)  # Dropout layer for regularization

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model to predict target values.

        Args:
            input_ids (Tensor): Tensor containing token IDs for input sequences.
            attention_mask (Tensor): Tensor indicating which tokens should be attended to.

        Returns:
            Tensor: A tensor containing the predicted target values for each input sequence.
        """
        # Forward pass through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token output from the last hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Pass through the fully connected layer to predict target values
        prediction = self.fc(cls_output)  # Shape: (batch_size, len(target_name))

        return prediction

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: BertTokenizer, target_name: List[str]):
        """
        Load a pre-trained BERT model for downstream task from a given path.

        Args:
            pretrained_path (str): Path to the pre-trained model directory.
            tokenizer (BertTokenizer): Tokenizer used for text tokenization.
            target_name (List[str]): name of target columns.

        Returns:
            BertForDownstream: An instance of BertForDownstream with the pre-trained weights loaded.
        """
        model = cls(tokenizer=tokenizer, target_name=target_name)
        model.bert = BertModel.from_pretrained(pretrained_path)
        return model

