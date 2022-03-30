# Creating a model

# from previous scripts
from train_fe import getTokenizer, getModel
from hf_datasets import loading_dataset
from dataset_tokenization import tokenize
from train_fe import extract_hidden_states
from utils import *

import torch
import config
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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

def classifier(X_train, X_valid, y_train, y_valid):
    """
        training a simple classifier

    Args:
        X_train (numpy array): processed and dimensionaly reduced training data
        X_valid (numpy array): processed and dimensionaly reduced training labels
        y_train (numpy array): processed and dimensionaly reduced validation data
        y_valid (numpy array): processed and dimensionaly reduced validation labels
    """
    
    # We increase the `max_iter` to gurantee convergence
    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    print(lr_clf.score(X_valid, y_valid))
    return lr_clf
    
def dummy_classifier(X_train, X_valid, y_train, y_valid):
    """
        training a dummy classifier for comparing performance

    Args:
        X_train (numpy array): processed and dimensionaly reduced training data
        X_valid (numpy array): processed and dimensionaly reduced training labels
        y_train (numpy array): processed and dimensionaly reduced validation data
        y_valid (numpy array): processed and dimensionaly reduced validation labels
    """
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    print(dummy_clf.score(X_valid, y_valid))

def plot_confusion_matrix(y_preds, y_true, labels):
    """plot the confusion matrix

    Args:
        y_preds (numpy array): predicted class
        y_true (numpy array): actual classes
        labels (list): list of classes
    """
    
    cm = confusion_matrix(y_preds, y_true, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

if __name__ == '__main__':
    
    model = getModel()
    tokenizer = getTokenizer()
    
    dataset = loading_dataset("emotion", info=False)
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    
    dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True, batch_size=500)
    
    X_train, X_valid, y_train, y_valid = get_numpy_array(dataset_hidden)
    
    lr_clf = classifier(X_train, X_valid, y_train, y_valid)
    dummy_classifier(X_train, X_valid, y_train, y_valid)
    
    # Predictions
    y_preds = lr_clf.predict(X_valid)
    labels = dataset["train"].features["label"].names
    plot_confusion_matrix(y_preds, y_valid, labels)