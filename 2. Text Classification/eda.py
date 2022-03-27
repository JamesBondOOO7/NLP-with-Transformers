# from previous scripts
from this import d
from pyzstd import train_dict
from hf_datasets import loading_dataset
from datasets_to_dataframe import dt_to_df

# new libraries import
import matplotlib.pyplot as plt
import pandas as pd

def label_int2str(row):
    """
        converts `int` labels to `string` labels in a dataframe

    Args:
        row (pandas dataframe row): row of the dataframe
    """
    return dataset["train"].features["label"].int2str(row)

def class_distr(df):
    """
        generates horizontal bar graph for `labels` in df

    Args:
        df (pandas df): the dataframe
    """
    
    df["label_name"].value_counts(ascending=True).plot.barh()
    plt.title("Frequency of Classes")
    plt.show()
    
def tweet_len_plot(df):
    """
        generates box plot graph for `Words Per tweet` in df

    Args:
        df (pandas df): the dataframe
    """
    
    df["Words Per Tweet"] = df["text"].str.split().apply(len)
    df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()
    
    
if __name__ == '__main__':
    
    # Prepare the dataframe
    dataset = loading_dataset(name="emotion", info=False)
    df = dt_to_df(dataset)
    df["label_name"] = df["label"].apply(label_int2str)
    
    # Observation : the dataset is heavily imbalanced
    class_distr(df=df)
    
    # Observation : Most tweets are 15 words long
    # Tweets longer than model's context size need to be truncated
    tweet_len_plot(df=df)