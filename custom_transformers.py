import numpy as np
import pandas as pd

def organization_type(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe and returns pandas dataframe with 
    customized values of ORGANIZATION_TYPE feature.
    """
    
    condlist = [input_df["ORGANIZATION_TYPE"].str.startswith("Business") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Trade") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Transport") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Industry") == True]

    choicelist = ["Business", "Trade", "Transport", "Industry"]

    input_df["ORGANIZATION_TYPE"] = np.select(condlist, choicelist, input_df["ORGANIZATION_TYPE"])
    
    return input_df