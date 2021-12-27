import math
import random

import pandas as pd

def distinct_values(dataframe: pd.DataFrame, col_name: str, coef: float) -> pd.DataFrame:
    """Takes in dataframe with missing values and randomly fill that NaN values with most frequent values
    The number of most frequent values is specified by coef argument which states what fraction off samples
    should contain that most frequent values.
    
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