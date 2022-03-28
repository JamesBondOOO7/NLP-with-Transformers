from transformers import AutoTokenizer
from transformers import DistilBertTokenizer

# NOTE: When using pretrained models, it is really important to make sure that you use the same tokenizer that the model was trained with. From the model’s perspective, switching the tokenizer is like shuffling the vocabulary.

# Fixing Environment :
# If conda install huggingface-hub fails, use pip
# Eg: pip install huggingface-hub==0.4.0

def subword_tkn_auto(text, logs=False):
    """combining best aspects of word and character tokenizations.

    Args:
        text (list): sentence to be tokenized
    """
    
    # The AutoTokenizer class belongs to a larger set of “auto” classes whose job is to automatically retrieve the model’s configuration, pretrained weights, or vocabulary from the name of the checkpoint.
    
    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # tokenizer.save_pretrained('./tokenizer1')
    
    # ----------------- Extra -------------------
    # FASTER !!
    # to save the tokenizer locally
    # model_ckpt = "distilbert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # tokenizer.save_pretrained('./tokenizer1')
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer1')
    # -------------------------------------------
    
    # text --> integer ids
    encoded_text = tokenizer(text)
    
    # ids --> tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    
    if logs:
        print(encoded_text)
        print(tokens)
        
        # tokens --> string
        print(tokenizer.convert_tokens_to_string(tokens))
        
        # Vocab size
        print(tokenizer.vocab_size)
        
        # model's max length
        print(tokenizer.model_max_length)
        
        # name of the fields that the model expects in forward pass
        print(tokenizer.model_input_names)
        

def subword_tkn_manual(text, logs=False):
    """combining best aspects of word and character tokenizations.

    Args:
        text (list): sentence to be tokenized
    """
    
    # if you wish to load the specific class manually you can do so as well. For example, we could have loaded the DistilBERT tokenizer :
    model_ckpt = "distilbert-base-uncased"
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
    encoded_text = distilbert_tokenizer(text)
    
    if logs:
        print(encoded_text)

if __name__ == '__main__':
    
    text = "Tokenizing text is a core task of NLP."
    subword_tkn_auto(text, True)
    # subword_tkn_manual(text, True)