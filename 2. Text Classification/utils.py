import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_numpy_array(dataset_hidden):
    """convert the processed dataset for some eda

    Args:
        dataset_encoded (DatasetDict): processed dataset
    """
    
    X_train = np.array(dataset_hidden["train"]["hidden_state"])
    X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
    y_train = np.array(dataset_hidden["train"]["label"])
    y_valid = np.array(dataset_hidden["validation"]["label"])
    
    print(X_train.shape, X_valid.shape)
    return X_train, X_valid, y_train, y_valid

def visualize_train_set(X_train, y_train):
    
    # Scale features to [0, 1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)
    
    # Initialize and fit UMAP
    mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)
    
    # Create a dataframe of 2d embeddings
    df_emb = pd.DataFrame(mapper.embedding_)