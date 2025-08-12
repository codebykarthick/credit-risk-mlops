import os

import pandas as pd

from utils.logger import setup_logger
from data_preprocessing import save_splits

log = setup_logger()

def create_features():
    """Just a few engineered features from existing data to see if model can use it for 
    better performance. Also allows us to make it more interpretable as well.
    """
    base_path = os.path.join(os.getcwd(), "..", "data", "processed")
    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")
    
    log.info("Loading datasets")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    log.info("Creating new features")
    for df in [train_df, val_df, test_df]:
        # Get MonthlyDebt from the ratio
        df["MonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]
        # Binary feature to mark if a person is a senior citizen, might help
        df["IsSeniorCitizen"] = (df["age"] > 60).astype(int)
        # Combine the total defaults as a single sum.
        df["DelinquencyTotal"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"] +
            df["NumberOfTime60-89DaysPastDueNotWorse"] +
            df["NumberOfTimes90DaysLate"]
        )
    
    log.info("Updating saved files")
    save_splits(train=train_df, val=val_df, test=test_df)


if __name__ == "__main__":
    create_features()