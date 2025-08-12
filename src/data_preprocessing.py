import os
from typing import Tuple
import sys

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

def impute_missing_values() -> None:
    """Impute missing values using training set statistics only, updating it in place.
    """
    base_path = os.path.join(os.getcwd(), "..", "data", "processed")
    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")
    
    log.info("Loading datasets")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    log.info("Removing age outliers")
    # Remove the age == 0 records from all data
    train_df = train_df[train_df["age"] > 0]
    val_df = val_df[val_df["age"] > 0]
    test_df = test_df[test_df["age"] > 0]

    log.info("Imputing MonthlyIncome with median")
    # Impute MonthlyIncome with median (check eda notebook for reason)
    income_median = train_df["MonthlyIncome"].median()
    log.info(f"Imputing MonthlyIncome with train median: {income_median:.2f}")
    for df in [train_df, val_df, test_df]:
        df["MonthlyIncome"].fillna(income_median, inplace=True)
    
    log.info("Imputing NumberOfDependents with mode")
    # Impute NumberOfDependents with mode (again eda notebook)
    dependents_mode = train_df["NumberOfDependents"].mode()[0]
    log.info(f"Imputing NumberOfDependents with train mode: {dependents_mode}")
    for df in [train_df, val_df, test_df]:
        df["NumberOfDependents"].fillna(dependents_mode, inplace=True)
    
    log.info("Clipping the outliers of other columns with hard values or upper percentile")
    # Cap the values to prevent throwing off my model
    for col in ["NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTime30-59DaysPastDueNotWorse"]:
        for df in [train_df, val_df, test_df]:
            # Handle those weird 98 values
            df[f"{col}_extreme"] = (df[col] >= 90).astype(int)
            df[col] = df[col].clip(upper=10)
    
    for df in [train_df, val_df, test_df]:
        # Its a percent so anything above 1 itself is weird.
        df["overutilized"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
        df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.0)

    clip_threshold = train_df["DebtRatio"].quantile(0.99)
    log.info(f"Clipping DebtRatio at 99th percentile: {clip_threshold:.2f}")
    for df in [train_df, val_df, test_df]:
        # The max value is 326442.0, assuming the person has less debt than a country
        df["DebtRatio"] = df["DebtRatio"].clip(upper=clip_threshold)
    
    save_splits(train=train_df, val=val_df, test=test_df)


    



if __name__ == "__main__":
    allowed_methods = ["split", "impute", "all"]

    if len(sys.argv) < 2:
        log.error("Requires an argument to select the type of preprocessing.")
        sys.exit()

    method = sys.argv[1]

    if method not in allowed_methods:
        log.error(f"Specified method is wrong, allowed methods are: {allowed_methods}")
        sys.exit()

    if method != "impute" or not os.path.exists(os.path.join(os.getcwd(), "..", "data", "processed")):
        train, val, test = load_and_split()
        save_splits(train=train, val=val, test=test)
    else:
        log.info("Skipping data split and save.")
    
    if method == "impute":
        log.info("Handling outliers and missing values in data.")
        impute_missing_values()