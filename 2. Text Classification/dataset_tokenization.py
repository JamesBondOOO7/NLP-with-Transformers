# from previous script
from hf_datasets import loading_dataset

from transformers import AutoTokenizer

# NOTE: SPECIAL TOKEN       SPECIAL TOKEN ID
#       [PAD]               0
#       [UNK]               100
#       [CLS]               101
#       [SEP]               102
#       [MASK]              103

def tokenize(batch):
    """tokenizer applies to a batch of examples

    Args:
        batch (dict): batch of examples
    """
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    return tokenizer(batch["text"], padding=True, truncation=True)

def getTokenizer():
    """
        returns the required tokenizer
    """
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    return tokenizer
    
if __name__ == '__main__':
    
    dataset = loading_dataset("emotion", info=False)
    
    # print(tokenize(dataset["train"][:2]))
    
    # for the entire corpus
    # By default, the map() method operates individually on every example in the corpus
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    
    print(dataset_encoded)