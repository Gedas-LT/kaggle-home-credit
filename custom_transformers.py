import numpy as np
import pandas as pd

def blend_organization_type(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe and returns pandas dataframe with 
    blended values of ORGANIZATION_TYPE feature.
    """
    
    result = input_df.copy()
    
    condlist = [result["ORGANIZATION_TYPE"].str.startswith("Business") == True,
                result["ORGANIZATION_TYPE"].str.startswith("Trade") == True,
                result["ORGANIZATION_TYPE"].str.startswith("Transport") == True,
                result["ORGANIZATION_TYPE"].str.startswith("Industry") == True]

    choicelist = ["Business", "Trade", "Transport", "Industry"]

    result["ORGANIZATION_TYPE"] = np.select(condlist, choicelist, result["ORGANIZATION_TYPE"])
    
    return result


def credit_card_dpd(input_df: pd.DataFrame, credit_card_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with credit cards information
    and returns input dataframe with additional column with DPD flag values.
    
    Keyword arguments:
    input_df -- primary dataframe from sklearn pipeline.
    credit_card_df -- additional dataframe with information about credit cards. 
    """
    
    result = input_df.copy()
    
    credit_card_dpd = credit_card_df[(credit_card_df["MONTHS_BALANCE"] > -12) & (credit_card_df["NAME_CONTRACT_STATUS"] == "Active")][["SK_ID_CURR", "SK_DPD"]]
    credit_card_dpd["FLAG_DPD"] = [1 if x > 0 else 0 for x in credit_card_dpd["SK_DPD"]]
    count_credit_card_dpd = credit_card_dpd.groupby("SK_ID_CURR")["FLAG_DPD"].sum().reset_index()

    result = pd.merge(result, count_credit_card_dpd, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def pos_cash_dpd(input_df: pd.DataFrame, pos_cash_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with information of 
    POS (point of sales) and cash loans and returns input dataframe with additional column with sum of DPD.
    
    Keyword arguments:
    input_df -- primary dataframe from sklearn pipeline.
    pos_cash_df -- additional dataframe with information about POS (point of sales) and cash loans. 
    """
    
    result = input_df.copy()
    
    pos_cash_dpd = pos_cash_df[(pos_cash_df["MONTHS_BALANCE"] > -12) & (pos_cash_df["NAME_CONTRACT_STATUS"] == "Active")].groupby("SK_ID_CURR")["SK_DPD"].sum().reset_index()

    result = pd.merge(result, pos_cash_dpd, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def flag_insurance(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and returns input dataframe with FLAG_INSURANCE column."""
    
    result = input_df.copy()
    
    result["FLAG_INSURANCE"] = np.where(result["AMT_CREDIT"] - result["AMT_GOODS_PRICE"] > 0, 1, 0)
    
    return result


def credit_card_drawings(input_df: pd.DataFrame, credit_card_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with credit cards information and 
    returns input dataframe with additional column with total amount of drawings for the last half of year.
    
    Keyword arguments:
    input_df -- primary dataframe from sklearn pipeline.
    credit_card_df -- additional dataframe with information about credit cards. 
    """
    
    result = input_df.copy()
    
    credit_card_drawings = credit_card_df[["SK_ID_CURR", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT", "MONTHS_BALANCE"]].fillna(0)
    credit_card_drawings["ALL_DRAWINGS"] = credit_card_drawings[credit_card_drawings["MONTHS_BALANCE"] > -7].iloc[:,-4:].sum(axis=1)
    credit_card_drawings = credit_card_drawings.groupby("SK_ID_CURR")["ALL_DRAWINGS"].sum().reset_index()
    result = pd.merge(result, credit_card_drawings, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def flag_insurance(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and returns input dataframe with FLAG_INSURANCE column."""
    
    result = input_df.copy()
    
    result["FLAG_INSURANCE"] = np.where(result["AMT_CREDIT"] - result["AMT_GOODS_PRICE"] > 0, 1, 0)
    
    return result


def pandas_binning(input_df: pd.DataFrame, feature: str, bins: list, labels: list) -> pd.DataFrame:
    """Takes in pandas dataframe within scikit-learn pipeline and 
    returns input dataframe with binned values of specified feature.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    feature -- name of the binning feature.
    bins -- list of bins used in pandas cut function.
    labels -- list of labels used in pandas cut function. 
    """
    
    result = input_df.copy()
    
    result[feature] = pd.cut(result[feature] , bins=bins, labels=labels, include_lowest=True).astype(object)
    
    return result


def bureau_credit_type_counter(input_df: pd.DataFrame, bureau_df:pd.DataFrame, scarce_values: list) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with data provided by other financial institutions 
    and returns input dataframe with additional column with count of different credit type for each client.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    bureau_df -- additional dataframe with data provided by other financial institutions.
    scarce_values -- list of values which should be named under one name. 
    """
    
    result = input_df.copy()
    
    bureau_credit_type = bureau_df[["SK_ID_CURR", "CREDIT_TYPE"]]
    bureau_credit_type["CREDIT_TYPE"] = np.where(bureau_credit_type["CREDIT_TYPE"]
                                                 .isin(scarce_values), "Other", bureau_credit_type["CREDIT_TYPE"])
    bureau_credit_type = (pd.get_dummies(bureau_credit_type[["SK_ID_CURR", "CREDIT_TYPE"]], prefix="BUREAU_CREDIT")
                          .groupby("SK_ID_CURR")
                          .sum()
                          .reset_index())
    
    result = pd.merge(result, bureau_credit_type, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def prev_credit_type_counter(input_df: pd.DataFrame, previous_application_df:pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with data about previous applications 
    and returns input dataframe with additional column with count of different credit type for each client.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    bureau_df -- additional dataframe with data about previous applications in Home Credit. 
    """
    
    result = input_df.copy()
    
    prev_app_type = previous_application_df[["SK_ID_CURR", "NAME_CONTRACT_TYPE"]]
    prev_app_type = (pd.get_dummies(prev_app_type[["SK_ID_CURR", "NAME_CONTRACT_TYPE"]], prefix="PREV_APP")
                     .groupby("SK_ID_CURR")
                     .sum()
                     .reset_index())
    
    result = pd.merge(result, prev_app_type, on="SK_ID_CURR", how="left").fillna(0)
    
    return result