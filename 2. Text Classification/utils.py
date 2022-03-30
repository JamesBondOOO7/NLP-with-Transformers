import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_numpy_array(dataset_hidden):
    """convert the processed dataset for some eda

    Args:
        dataset_hidden (DatasetDict): processed dataset
    """
    
    X_train = np.array(dataset_hidden["train"]["hidden_state"])
    X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
    y_train = np.array(dataset_hidden["train"]["label"])
    y_valid = np.array(dataset_hidden["validation"]["label"])
    
    print(X_train.shape, X_valid.shape)
    return X_train, X_valid, y_train, y_valid

def get_visualization_dataframe(X_train, y_train):
    """convert data to some lower dimension for plotting

    Args:
        X_train (numpy array): training points
        y_train (numpy array): labels
    """
    
    # Scale features to [0, 1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)
    
    # Initialize and fit UMAP
    mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)
    
    # Create a dataframe of 2d embeddings
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train
    print(df_emb.head())
    return df_emb
    
def visualize_data(dataset, df_emb):
    """_summary_

    Args:
        dataset (DatasetDict): the Dataset
        df_emb (Pandas Dataframe): Lower dimensional data for visualization --> dataframe of 2d embeddings
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()
    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
    labels = dataset["train"].features["label"].names
    
    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, 
                       gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
    plt.tight_layout()
    plt.show()