import os
import re
from typing import Union, List
from transformers import BertTokenizer, BatchEncoding

def vocabulary(path: str) -> None:
    """
    Creates a vocabulary file for tokenizing SMILES strings, containing element symbols, special symbols, and aromatic symbols.

    Args:
        path (str): The directory path where the vocabulary file will be saved.

    Returns:
        None
    """
    # List of element symbols commonly used in SMILES strings
    element_symbols: List[str] = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                                  'K', 'Sc', 'Zn', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Ag', 'In', 'Sn', 'Te', 'I', 'Xe',
                                  'Cs', 'Ba', 'At', 'Ra', 'Np']

    # List of special symbols used in SMILES notation
    special_symbols: List[str] = ['@', '@@', '/', '\\', '-', '=', '#', '(', ')', '[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '%', '+', '.']

    # List of aromatic symbols used in SMILES notation
    aromatic_symbols: List[str] = ['c', 'n', 'o', 'p', 's']

    # List of vocabulary tokens, including special BERT-like tokens
    vocab_list: List[str] = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + element_symbols + aromatic_symbols + special_symbols

    # Write each token from vocab_list to the vocabulary file
    with open(path, 'w') as f:
        for token in vocab_list:
            f.write(token + '\n')

def TokenGenerator(smiles: str, vocab_file: str) -> List[str]:
    """
    Tokenizes a SMILES string based on a custom vocabulary.

    Args:
        smiles (str): The SMILES string to tokenize.
        vocab_file (str): Path to the vocabulary file used for tokenizing SMILES.

    Returns:
        List[str]: A list of tokens representing the SMILES string.
    """
    # Load vocabulary symbols from file
    with open(vocab_file, 'r') as f:
        sorted_symbols = sorted([line.strip() for line in f.readlines()], key=len, reverse=True)

    # Tokenize the SMILES string using the generated pattern
    token_pattern = '(' + '|'.join(map(re.escape, sorted_symbols)) + '|.)'
    tokens = re.findall(token_pattern, smiles)

    return tokens

def Tokenizer(smiles: Union[str, List[str]], 
              vocab_file: str,
              tokenizer: BertTokenizer,
              max_len: int) -> BatchEncoding:
    """
    Tokenizer for SMILES strings with options for BERT and GPT.

    Args:
        smiles (str or List[str]): The SMILES string(s) to tokenize.
        vocab_file (str): Path to the vocabulary file used for tokenizing SMILES.
        tokenizer (BertTokenizer): Pre-trained tokenizer to encode the tokens.
        max_len (int): Maximum length for sequences.
        add_special_tokens (bool): Whether to add special tokens (e.g., [CLS], [SEP], [EOS]).
        return_token_type_ids (bool): Whether to return token_type_ids (BERT specific).

    Returns:
        BatchEncoding: Encoded SMILES input with input_ids, attention_mask, and optional token_type_ids.
    """
    # Ensure smiles is a list, even if a single SMILES string is provided
    smiles_list = smiles if isinstance(smiles, list) else [smiles]

    # Initialize lists to store tokenized and encoded information
    all_input_ids, all_token_type_ids, all_attention_mask = [], [], []

    # Iterate over each SMILES string in the list
    for smiles in smiles_list:
        # Tokenize the SMILES string using the custom SMILES tokenizer
        tokens = TokenGenerator(smiles, vocab_file)

        # Encode the tokens using the provided tokenizer
        output = tokenizer.encode_plus(
                tokens,
                is_split_into_words=True,   # Tokens are already split
                add_special_tokens=True,    # Add special tokens such as [CLS] and [SEP]
                return_token_type_ids=True, # Return token type IDs
                return_attention_mask=True, # Return attention mask
                padding='max_length',       # Pad sequences to the maximum length
                max_length=max_len,         # Maximum length for sequences (truncates if longer)
                truncation=True             # Truncate sequences longer than max_length
                )

        # Append the encoded components to their respective lists
        all_input_ids.append(output['input_ids'])
        all_token_type_ids.append(output['token_type_ids'])
        all_attention_mask.append(output['attention_mask'])

    # Create a BatchEncoding object containing the encoded inputs
    batch_encoding = BatchEncoding(data={
                        'input_ids': all_input_ids,
                        'token_type_ids': all_token_type_ids,
                        'attention_mask': all_attention_mask
                        })

    return batch_encoding

