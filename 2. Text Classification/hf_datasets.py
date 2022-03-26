from datasets import list_datasets
from datasets import load_dataset

def check():
    # THE `DATASET` Hub
    all_datasets = list_datasets()

    print(f"There are {len(all_datasets)} datasets currently available on the Hub")
    print(f"The first 10 are :{all_datasets[:10]}")

def loading_dataset(name, info=True):
    """
        Loading the dataset
    Args:
        name (string): name of the dataset
        info (boolean): print statements
    """
    dataset = load_dataset(name)
    
    if info:
        print(dataset)
        
        # train data
        # returns an instance of the Dataset class
        train_ds = dataset["train"]
        
        print(train_ds)
        print(len(train_ds))
        print(train_ds[0]) # to see an example
        print(train_ds.column_names) # to see the column names
        print(train_ds.features)
        print(train_ds[:5]) # printing 5 dictionary values
        print(train_ds["text"][:5]) # list of text elements
    
    return dataset
    
def load_dataset_notInHub(dataset_url):
    """
        Loading the dataset not present in the HF Hub
    Args:
        dataset_url (string): url of the dataset
        
    Eg: 
    Run it in terminal
    
    >> dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt"
    >> wget {dataset_url}
    
    """
    
    # for local device
    dataset_local = load_dataset("csv", data_files="train.txt", sep=";",
                                names=["text", "label"])
    
    print(dataset_local)
    # directly from url
    # automatically download and cache the dataset for you
    dataset_remote = load_dataset("csv", data_files=dataset_url, sep=";", 
                                  names=["text", "label"])
    
    print(dataset_remote)

if __name__ == '__main__':
    check()
    loading_dataset("emotion")
    dataset_url = "https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1"
    load_dataset_notInHub(dataset_url=dataset_url)
