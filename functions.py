import math
import random

import numpy as np
import pandas as pd

def distinct_values(dataframe: pd.DataFrame, col_name: str, coef: float) -> pd.DataFrame:
    """Takes in dataframe with missing values and randomly fill NaNs of specified column with most frequent values.
    The most frequent values and total number of them are calculated by coef argument which states the fraction of samples.
    
    Keyword arguments:
    dataframe -- dataframe with missing values.
    col_name -- name of the column with missing values in given a dataframe.
    coef -- coeficient that specifies the number of most frequent values.
    """
    
    unique_values = len(dataframe[col_name].value_counts())
    samples = sum(dataframe[col_name].value_counts())
    
    for i in range(unique_values):
        if sum(dataframe[col_name].value_counts().iloc[:i]) > samples * coef:
            ext_source_list = dataframe[col_name].value_counts().iloc[:i].reset_index()["index"].to_list()
            dataframe[col_name] = dataframe[col_name].apply(lambda x: random.choice(ext_source_list) if math.isnan(x) == True else x)
            return dataframe
        
        
def missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Takes in dataframe and returns new dataframe with calculated NaN values and corresponding features.
    
    Keyword arguments:
    dataframe -- dataframe with missing values.
    """
    
    nan_values = dataframe.isna().sum().reset_index().rename(columns={"index": "Feature", 0: "NaN values"})
    nan_values["NaN values, %"] = nan_values["NaN values"] / len(dataframe) * 100
    nan_values = nan_values[nan_values["NaN values"] > 0].sort_values(by=["NaN values, %"], ascending=False)
    
    return nan_values


def export_predictions(id: pd.Series, predictions: np.ndarray, name_prefix: str):
    """Exports final predictions for submission as a .csv file.
    
    Keyword arguments:
    id -- column as a pandas series with the IDs.
    predictions -- predictions as an array.
    name_prefix -- name prefix in final URL. For example "submissions/{name_prefix}_predictions.
    """
    
    output = pd.DataFrame({"SK_ID_CURR": id, "TARGET": predictions[:, 1]})
    output.to_csv(f"submissions/{name_prefix}_predictions.csv", index=False)
    
    
def imbalanced_features(table: pd.DataFrame) -> pd.DataFrame:
    """Takes in a table and returns another table with column names 
    and information about most frequent values in those columns.
    """
    
    feature_names = [column for column in table]
    qty_most_freq_val = [table[column].value_counts().iloc[0] for column in table]
    qty_most_freq_val_perc = [table[column].value_counts().iloc[0] / len(table) * 100 for column in table]
    
    most_freq_val_table = pd.DataFrame({"Feature Name": feature_names, "QTY of most freq. value": qty_most_freq_val,
                                        "% of Total Values": qty_most_freq_val_perc}).sort_values(by="% of Total Values", ascending=False)
    
    return most_freq_val_table