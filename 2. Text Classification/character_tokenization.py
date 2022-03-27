import pandas as pd
import torch
import torch.nn.functional as F

def char_tkn(text, logs=False):
    """
        text to be tokenized to a list of characters

    Args:
        text (string): sentence to be tokenized
    """
    
    # String --> Characters
    tokenized_text = list(text)
    
    # Characters --> Integers
    # mapping for token to index
    token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
    
    # map the tokenized text using token2idx
    input_ids = [token2idx[token] for token in tokenized_text]
    
    if logs:
        print(tokenized_text)
        print(token2idx)
        
    return input_ids, token2idx

def get_one_hot_encoding(input_ids, token2idx):
    """generate 2-d tensors from input ids

    Args:
        input_ids (list `int`): list of token indicies
        token2idx (dict) : mapping of tokens to indicies
    """
    
    input_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    return one_hot_encodings
    
if __name__ == '__main__':
    
    text = "Tokenizing text is a core task of NLP."
    input_ids, token2idx = char_tkn(text)
    
    # -------------- Example for One-Hot encoding ---------------------
    # categorical_df = pd.DataFrame(
    #     {"Name" : ["Bumblebee", "Optimus Prime", "Megatron"], 
    #      "Label ID" : [0, 1, 2]}
    # )
    
    # print(categorical_df.head())
    
    # Creating one-hot vectors
    # print(pd.get_dummies(categorical_df["Name"]))
    # -----------------------------------------------------------------
    
    # Creating one-hot vectors using PyTorch
    one_hot_encodings = get_one_hot_encoding(input_ids=input_ids, token2idx=token2idx)
    
    print(one_hot_encodings.shape)
    print(f"Tensor index: {input_ids[0]}")
    print(f"One-hot: {one_hot_encodings[0]}")
    