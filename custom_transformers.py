import numpy as np
import pandas as pd

def blend_organization_type(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe and returns pandas dataframe with 
    blended values of ORGANIZATION_TYPE feature.
    """
    
    condlist = [input_df["ORGANIZATION_TYPE"].str.startswith("Business") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Trade") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Transport") == True,
                input_df["ORGANIZATION_TYPE"].str.startswith("Industry") == True]

    choicelist = ["Business", "Trade", "Transport", "Industry"]

    input_df["ORGANIZATION_TYPE"] = np.select(condlist, choicelist, input_df["ORGANIZATION_TYPE"])
    
    return input_df


def credit_card_dpd(input_df: pd.DataFrame, credit_card_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with credit cards information
    and returns input dataframe with additional column with DPD flag values.
    
    Keyword arguments:
    input_df -- primary dataframe from sklearn pipeline.
    credit_card_df -- additional dataframe with information about credit cards. 
    """
    
    credit_card_dpd = credit_card_df[(credit_card_df["MONTHS_BALANCE"] > -12) & (credit_card_df["NAME_CONTRACT_STATUS"] == "Active")][["SK_ID_CURR", "SK_DPD"]]
    credit_card_dpd["FLAG_DPD"] = [1 if x > 0 else 0 for x in credit_card_dpd["SK_DPD"]]
    count_credit_card_dpd = credit_card_dpd.groupby("SK_ID_CURR")["FLAG_DPD"].sum().reset_index()

    input_df = pd.merge(input_df, count_credit_card_dpd, on="SK_ID_CURR", how="left").fillna(0)
    
    return input_df


def pos_cash_dpd(input_df: pd.DataFrame, pos_cash_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with information of 
    POS (point of sales) and cash loans and returns input dataframe with additional column with sum of DPD.
    
    Keyword arguments:
    input_df -- primary dataframe from sklearn pipeline.
    pos_cash_df -- additional dataframe with information about POS (point of sales) and cash loans. 
    """
    
    pos_cash_dpd = pos_cash_df[(pos_cash_df["MONTHS_BALANCE"] > -12) & (pos_cash_df["NAME_CONTRACT_STATUS"] == "Active")].groupby("SK_ID_CURR")["SK_DPD"].sum().reset_index()

    input_df = pd.merge(input_df, pos_cash_dpd, on="SK_ID_CURR", how="left").fillna(0)
    
    return input_df


def flag_insurance(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and returns input dataframe with FLAG_INSURANCE column."""
    
    input_df["FLAG_INSURANCE"] = np.where(input_df["AMT_CREDIT"] - input_df["AMT_GOODS_PRICE"] > 0, 1, 0)
    
    return input_df