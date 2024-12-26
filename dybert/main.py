import os
import torch
import argparse
import pandas as pd
from torch.utils.data import random_split
from transformers import BertTokenizer, BertModel
from dybert import BertForMLM, BertForDownstream
from dybert import train
from dybert.utils import set_seed
from dybert.tokenizer import vocabulary

# Set the random seed for reproducibility
SEED = 42
set_seed(SEED)

def main():
    # Parse command line arguments for model training configuration
    parser = argparse.ArgumentParser(description="Unified BERT Training for MLM and Downstream Tasks")
    parser.add_argument('--task', type=str, required=True, choices=['mlm', 'downstream'],
                        help="Task type: 'mlm' for Masked Language Modeling, 'downstream' for target prediction.")
    parser.add_argument('--pretrained', type=str, default=None, help='Path of pre-trained Model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset with CSV format')
    parser.add_argument('--vocabfile', type=str, default='./vocab.txt', help='Vocabulary file for tokenizing')
    parser.add_argument('--max_len', type=int, default=256, help='Vocabulary file for tokenizing')
    parser.add_argument('--batchsize', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--modelsavepath', type=str, default='./results', help='Save path of finetuned model')
    parser.add_argument('--masking', type=float, default=0.0, help='Masking ratio for MLM task')
    parser.add_argument('--target', type=str, nargs='+', default=None,
                        help='List of target names for downstream prediction (e.g., --target_list HOMO LUMO)')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Check if the vocabulary file exists; if not, create it using vocabulary() function
    if not os.path.exists(args.vocabfile):
        print(f"Vocabulary file '{args.vocabfile}' not found. Creating vocabulary...")
        vocabulary(path = args.vocabfile)

    # Load dataset CSV file
    df = pd.read_csv(args.dataset)

    tokenizer = BertTokenizer(vocab_file=args.vocabfile, clean_up_tokenization_spaces=True)

    # Task type: Masked Language Modeling (MLM)
    if args.task == 'mlm':
        # Split data into training and validation sets
        train_size = int(0.99 * len(df))
        val_size = len(df) - train_size

        train_subset, val_subset = random_split(df, [train_size, val_size])
        
        # Convert Subsets back to DataFrame
        df_train = df.iloc[train_subset.indices].reset_index(drop=True)
        df_val = df.iloc[val_subset.indices].reset_index(drop=True)
        df_test = None

        # Load pre-trained BERT model or initialize a new one
        if args.pretrained is None:
            model = BertForMLM(tokenizer=tokenizer)
        else:
            model = BertForMLM.from_pretrained_model(tokenizer=tokenizer,
                                                     pretrained_path=args.pretrained)

    # Task type: Downstream task for target prediction
    elif args.task == 'downstream':
        # Load training, validation, and test datasets using pandas
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        test_size = len(df) - train_size - val_size

        train_subset, val_subset, test_subset = random_split(df, [train_size, val_size, test_size])

        # Convert Subsets back to DataFrame
        df_train = df.iloc[train_subset.indices].reset_index(drop=True)
        df_val = df.iloc[val_subset.indices].reset_index(drop=True)
        df_test = df.iloc[test_subset.indices].reset_index(drop=True)

        # Load pre-trained BERT model
        if args.pretrained is None:
            model = BertForDownstream(tokenizer=tokenizer, target_name=args.target)
        else:
            model = BertForDownstream.from_pretrained_model(tokenizer=tokenizer,
                                                            target_name=args.target,
                                                            pretrained_path=args.pretrained)

    # Start training the model
    train(task_type=args.task,
            model=model,
            vocab_file=args.vocabfile,
            tokenizer=tokenizer,
            max_len=args.max_len,
            batch_size=args.batchsize,
            epochs=args.epochs,
            mask_ratio=args.masking,
            target_name=args.target,
            model_save_path=args.modelsavepath,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test)

if __name__ == '__main__':
    main()
