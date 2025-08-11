import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import setup_logger

DEFAULT_DATA_PATH = os.path.join(os.getcwd(), "..", "data", "raw", "cs-training.csv")
log = setup_logger()

def load_and_split(path: str = DEFAULT_DATA_PATH, 
                   val_size: float = 0.15, test_size: float = 0.15, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to load the data and split it into train, validation and testing datasets, stratified
    on the output label, so that dataset is balanced at the same ratio of positive and negative across
    for fairness.

    Args:
        path (str, optional): The path to the dataset csv file. Defaults to DEFAULT_DATA_PATH.
        val_size (float, optional): The ratio of records to be used for validation. Defaults to 0.10.
        test_size (float, optional): The ratio of records to be used for evaluation. Defaults to 0.10.
        random_state (int, optional): The random integer seed to be used for reproducibility. Defaults to 42.
    """
    log.info(f"Reading dataset from: {path}")

    df = pd.read_csv(path, index_col=0)

    log.info("Dataset loaded successfully!")
    df_train_val, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["SeriousDlqin2yrs"],
            random_state=random_state
        )
        
    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)  # adjust for already-removed test set
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        stratify=df_train_val["SeriousDlqin2yrs"],
        random_state=random_state
    )
    
    return df_train, df_val, df_test

def save_splits(train: pd.DataFrame, val: pd.DataFrame, 
                test: pd.DataFrame, out_dir: str="../data/processed") -> None:
    """To save the three sets under out_dir as processed datasets.

    Args:
        train (pd.DataFrame): The training dataframe to be saved.
        val (pd.DataFrame): The validation dataframe to be saved.
        test (pd.DataFrame): The test set dataframe to be saved.
        out_dir (str, optional): The output directory. Defaults to "data/processed".
    """
    log.info(f"Creating folder: {out_dir}.")
    os.makedirs(out_dir, exist_ok=True)
    log.info("Processing train set.")
    train.to_csv(f"{out_dir}/train.csv", index=False)
    log.info("Processing validation set.")
    val.to_csv(f"{out_dir}/val.csv", index=False)
    log.info("Processing test set.")
    test.to_csv(f"{out_dir}/test.csv", index=False)
    log.info("All files processed sucessfully!")


if __name__ == "__main__":
    train, val, test = load_and_split()
    save_splits(train=train, val=val, test=test)