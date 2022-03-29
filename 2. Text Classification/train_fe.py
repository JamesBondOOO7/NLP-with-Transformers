# TRANSFORMERS AS FEATURE EXTRACTORS

# from previous script
from numpy import extract
from hf_datasets import loading_dataset
from dataset_tokenization import getTokenizer, tokenize
import config

from transformers import AutoModel
import torch

def getModel():
    """
        load the pretrained model
    """
    
    # model checkpoint
    model_ckpt = "distilbert-base-uncased"
    
    # loading the model
    model = AutoModel.from_pretrained(model_ckpt).to(config.device)
    
    return model
    
def extract_hidden_states(batch):
    """Extracting all hidden states for the batch in one go

    Args:
        batch (dict): batch of examples
    """
    
    # Place the model inputs on the GPU
    inputs = {k:v.to(config.device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
    
if __name__ == '__main__':
    
    model = getModel()
    tokenizer = getTokenizer()
    
    text = "this is a test"
    # return_tensors="pt" => return PyTorch Tensor
    inputs = tokenizer(text, return_tensors="pt")
    
    # print(f"Input tensor shape : {inputs['input_ids'].size()}") # (batch_size, n_tokens)
    
    # -----------------------------------------------
    # Input tensor shape : torch.Size([1, 6])
    # -----------------------------------------------
    
    
    inputs = {k:v.to(config.device) for k,v in inputs.items()}
    
    # torch.no_grad() context manager to disable the automatic calculation of the gradient.
    # This is useful for inference since it reduces the memory footprint of the computations.
    with torch.no_grad():
        outputs = model(**inputs)
        
    # print(outputs)
    
    # -----------------------------------------------
    # BaseModelOutput(last_hidden_state=tensor([[[-0.1565, -0.1862,  0.0528,  ..., -0.1188,  0.0662,  0.5470],
    #      [-0.3575, -0.6484, -0.0618,  ..., -0.3040,  0.3508,  0.5221],
    #      [-0.2772, -0.4459,  0.1818,  ..., -0.0948, -0.0076,  0.9958],
    #      [-0.2841, -0.3917,  0.3753,  ..., -0.2151, -0.1173,  1.0526],
    #      [ 0.2661, -0.5094, -0.3180,  ..., -0.4203,  0.0144, -0.2149],
    #      [ 0.9441,  0.0112, -0.4714,  ...,  0.1439, -0.7288, -0.1619]]],
    #    device='cuda:0'), hidden_states=None, attentions=None)
    # -----------------------------------------------
    
    # The current model returns only one attribute, which is the last hidden state
    
    # print(outputs.last_hidden_state.size()) # (batch_size, n_tokens, hid_dim)
    
    # For classification tasks, it is common practice to just use the hidden state associated with the [CLS] token as the input feature.
    # print(outputs.last_hidden_state[:, 0].size()) # (1, hid_dim)
    
    # The dataset
    dataset = loading_dataset("emotion", info=False)
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    
    # Since our model expects tensors as inputs, we have to convert the input_ids and attention_mask columns to the "torch" format
    dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Extract the hidden states across all splits in one go
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True, batch_size=500) # by default, batch_size=1000
    
    print(dataset_hidden)
    # print(dataset_hidden["train"].column_names)