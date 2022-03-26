# from previous script
from hf_datasets import loading_dataset

# new libraries import
import pandas as pd


def dt_to_df(dt_obj):
    """
        converts the given dataset object to pandas Dataframe

    Args:
        dt_obj (Dataset): dataset object
    """
    
    dt_obj.set_format(type="pandas")
    df = dt_obj["train"][:]
    return df
    
def label_int2str(row):
    """
        converts `int` labels to `string` labels in a dataframe

    Args:
        row (pandas dataframe row): row of the dataframe
    """
    return dataset["train"].features["label"].int2str(row)


if __name__ == "__main__":
    
    dataset = loading_dataset("emotion", info=False)
    train_df = dt_to_df(dataset)
    # print(train_df.head())
    
    train_df["label_name"] = train_df["label"].apply(label_int2str)
    print(train_df.head())