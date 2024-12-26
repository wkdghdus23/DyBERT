import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from typing import Union, List
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from transformers import BertTokenizer, BertForMaskedLM, BertModel, get_cosine_schedule_with_warmup
from dybert.dataset import MyDataLoader
from dybert.utils import adjust_learning_rate, loss_function

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(task_type: str,
          model: Union[BertForMaskedLM, BertModel],
          vocab_file: str,
          tokenizer: BertTokenizer,
          max_len: int,
          batch_size: int,
          epochs: int,
          mask_ratio: float,
          target_name: List[str],
          model_save_path: str,
          df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          df_test: pd.DataFrame) -> None:
    """
    Trains a BERT model for either masked language modeling (MLM) or target prediction (downstream).

    Args:
        task_type (str): Type of task - "mlm" for Masked Language Modeling, "downstream" for target prediction.
        model (Union[BertForMaskedLM, BertModel]): The BERT model to be trained.
        vocab_file (str): Path to the vocabulary file for the tokenizer.
        tokenizer (BertTokenizer): Tokenizer used to encode input data.
        max_len (int): Maximum length for input sequences.
        batch_size (int): The batch size for training and validation.
        epochs (int): The number of epochs for training.
        mask_ratio (float, optional): The ratio of tokens to mask during MLM training.
        target_name (List[str], optional): List of target columns (only used for downstream tasks).
        model_save_path (str, optional): Path to save the trained model.
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.

    Returns:
        None
    """
    # Move model to the specified device
    model.to(device)

    # Create data loaders based on the task type
    train_dataloader, val_dataloader, test_dataloader = MyDataLoader(task_type=task_type,
                                                                     vocab_file=vocab_file,
                                                                     tokenizer=tokenizer,
                                                                     max_len=max_len,
                                                                     batch_size=batch_size,
                                                                     mask_ratio=mask_ratio,
                                                                     target_name=target_name,
                                                                     df_train=df_train,
                                                                     df_val=df_val,
                                                                     df_test=df_test)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
    total_steps = int((len(df_train) / batch_size) * epochs)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps,
                                                num_cycles=0.5)

    # Training loop
    for epoch in range(epochs):
        # Choose loss function based on task type and epoch (for downstream)
        if task_type == 'downstream':
            loss_fn = loss_function(epoch, epochs)
        total_loss = 0

        # Training step
        model.train()

        # Iterate over each batch of training data
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if task_type == 'mlm':
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
            elif task_type == 'downstream':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
           
            # Update learning rate with scheduler
            scheduler.step()
            adjust_learning_rate(optimizer)

            # Accumulate training loss
            total_loss += loss.item()

        # Print average training loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.5f}")

        # Validation step
        avg_val_loss = evaluate(task_type, model, val_dataloader, device, loss_fn if task_type == 'downstream' else None)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.5f}")

    # Test step (only for downstream tasks)
    if task_type == 'downstream' and df_test is not None:
        avg_test_loss = evaluate(task_type, model, test_dataloader, device, loss_fn)
        print(f"Test Loss: {avg_test_loss:.5f}")

    # Save the trained model
    if task_type == 'mlm':
        savepath=f"{model_save_path}/{task_type}_batchsize{batch_size}_epochs{epochs}_{mask_ratio}masking/"
        model.model.save_pretrained(savepath)
    else:
        savepath=f"{model_save_path}/{task_type}_batchsize{batch_size}_epochs{epochs}_{''.join(target_name)}/"
        model.bert.save_pretrained(savepath)
        torch.save(model.state_dict(), f"{savepath}/DownstreamModel.pt")

def evaluate(task_type: str,
             model: Union[BertForMaskedLM, BertModel],
             dataloader: DataLoader,
             device: torch.device,
             loss_fn: nn.Module = None) -> float:
    """
    Evaluates the model on the given dataset.

    Args:
        task_type (str): Type of task - "mlm" for Masked Language Modeling, "downstream" for target prediction.
        model (Union[BertForMaskedLM, BertModel]): The BERT model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        device (torch.device): The device to use for evaluation (CPU or GPU).
        loss_fn (nn.Module, optional): The loss function used for calculating the evaluation loss (only for downstream).

    Returns:
        float: The average loss over the evaluation dataset.
    """
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Disable gradient calculation for evaluation to save memory
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs and labels to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass through the model
            if task_type == 'mlm':
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
            elif task_type == 'downstream':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            # Accumulate the loss and increment the batch counter
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss over all batches
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return average_loss

