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
    """Takes in pandas dataframe from pipeline, additional dataframe with credit cards information
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


def bureau_credit_type_counter(input_df: pd.DataFrame, bureau_df: pd.DataFrame, scarce_values: list) -> pd.DataFrame:
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


def prev_credit_type_counter(input_df: pd.DataFrame, previous_application_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with data about previous applications 
    and returns input dataframe with additional column with count of different credit type for each client.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    previous_application_df -- additional dataframe with data about previous applications in Home Credit. 
    """
    
    result = input_df.copy()
    
    prev_app_type = previous_application_df[["SK_ID_CURR", "NAME_CONTRACT_TYPE"]]
    prev_app_type = (pd.get_dummies(prev_app_type[["SK_ID_CURR", "NAME_CONTRACT_TYPE"]], prefix="PREV_APP")
                     .groupby("SK_ID_CURR")
                     .sum()
                     .reset_index())
    
    result = pd.merge(result, prev_app_type, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def prev_flag_insurance(input_df: pd.DataFrame, previous_application_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with data about previous applications 
    and returns input dataframe with additional column with insurance flag for previous application.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    previous_application_df -- additional dataframe with data about previous applications in Home Credit. 
    """
    
    result = input_df.copy()
    
    flag_previous_insurance = previous_application_df.groupby("SK_ID_CURR")["NFLAG_INSURED_ON_APPROVAL"].max().reset_index()
    
    result = pd.merge(result, flag_previous_insurance, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def annuity_income_ratio(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and returns input dataframe
    with additional column ANNUITY_VERSUS_INCOME.
    """
    
    result = input_df.copy()
    
    result["ANNUITY_VS_INCOME"] = result["AMT_ANNUITY"] / result["AMT_INCOME_TOTAL"] * 100
    
    return result


def prev_annuity_income_ratio(input_df: pd.DataFrame, previous_application_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and additional dataframe with data about previous applications 
    and returns input dataframe with additional column PREV_ANNUITY_VERSUS_INCOME.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    previous_application_df -- additional dataframe with data about previous applications in Home Credit. 
    """
    
    result = input_df.copy()
    
    avg_previous_annuity = (previous_application_df
                            .groupby("SK_ID_CURR")["AMT_ANNUITY"]
                            .mean()
                            .reset_index()
                            .rename(columns={"AMT_ANNUITY": "PREV_ANNUITY"}))

    result = pd.merge(result, avg_previous_annuity, on="SK_ID_CURR", how="left").fillna(0)

    result["PREV_ANNUITY_VS_INCOME"] = result["PREV_ANNUITY"] / result["AMT_INCOME_TOTAL"] * 100
    
    return result


def enquiries(input_df: pd.DataFrame, enquiries_list: list) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and list of columns' name and returns 
    input dataframe with additional column AMT_REQ_CREDIT_BUREAU - number of all 
    enquiries to Credit Bureau about the client before application.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    enquiries_list -- list of columns' name about enquirties. 
    """
    
    result = input_df.copy()
    
    result["AMT_REQ_CREDIT_BUREAU"] = 0
    
    for enquire in enquiries_list:
        result["AMT_REQ_CREDIT_BUREAU"] += result[enquire]
    
    return result


def prev_dpd_flag(input_df: pd.DataFrame, bureau_balance_df: pd.DataFrame, bureau_df: pd.DataFrame, dpd_notation: list) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline, two additional dataframes with information about previous credits 
    provided by other financial institutions and list of DPD notations and returns input dataframe with additional column DPD_STATUS.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    bureau_df - additional dataframe with information about previous credits provided by other financial institutions.
    bureau_balance_df - additional dataframe with information about monthly balances of previous credits.
    enquiries_list -- list of columns' name about enquirties. 
    """
    
    result = input_df.copy()
    
    bureau_balance_df["DPD_STATUS"] = np.where(bureau_balance_df["STATUS"].isin(dpd_notation), 1, 0)
    bureau_dpd_status = bureau_balance_df.groupby("SK_ID_BUREAU")["DPD_STATUS"].sum().reset_index()
    
    bureau_balance_dpd = pd.merge(bureau_df, bureau_dpd_status, on="SK_ID_BUREAU", how="left").fillna(0)
    bureau_balance_dpd = bureau_balance_dpd.groupby("SK_ID_CURR")["DPD_STATUS"].sum().reset_index()
    
    result = pd.merge(result, bureau_balance_dpd, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def down_payment_rate(input_df: pd.DataFrame, previous_application_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline, additional dataframe with data about previous applications
    and returns input dataframe with additional column RATE_DOWN_PAYMENT.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    previous_application_df - additional dataframe with data about previous applications in Home Credit. 
    """
    
    result = input_df.copy()
    
    down_payment_rate = (previous_application_df[["SK_ID_CURR", "RATE_DOWN_PAYMENT"]]
                         .groupby("SK_ID_CURR")["RATE_DOWN_PAYMENT"]
                         .mean()
                         .reset_index())
    
    result = pd.merge(result, down_payment_rate, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def installments_version(input_df: pd.DataFrame, installments_payments_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline, additional dataframe with data about repayment history for 
    the previously disbursed credits in Home Credit and returns input dataframe with additional column NUM_INSTALMENT_VERSION.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    installments_payments_df - additional dataframe with data about repayment history for the previously disbursed credits in Home Credit. 
    """
    
    result = input_df.copy()
    
    installments_payments_df["INSTALMENT_VERSION_CHANGE"] = np.where((installments_payments_df["NUM_INSTALMENT_VERSION"] > 1), 1, 0)
    avg_installment_version = (installments_payments_df[installments_payments_df["INSTALMENT_VERSION_CHANGE"] == 1]
                               .groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"]
                               .mean()
                               .reset_index())
    
    result = pd.merge(result, avg_installment_version, on="SK_ID_CURR", how="left").fillna(0)
    
    return result


def debt_income_ratio(input_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline, additional dataframe with data provided by other financial institutions
    and returns input dataframe with additional column with ratio of client's total monthly debt and monthly income.
    
    Keyword arguments:
    input_df -- primary dataframe within sklearn pipeline.
    bureau_df - additional dataframe with information about previous credits provided by other financial institutions. 
    """
    
    result = input_df.copy()
    
    bureau_debt = (bureau_df[(bureau_df["AMT_CREDIT_SUM_DEBT"] > 0)
                             & (bureau_df["CREDIT_ACTIVE"] == "Active")
                             & (bureau_df["DAYS_CREDIT_ENDDATE"] > 0)]
                   .groupby("SK_ID_CURR")[["AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT_ENDDATE"]]
                   .sum()
                   .reset_index())
    bureau_debt["CREDIT_DEBT_ANNUITY"] = bureau_debt["AMT_CREDIT_SUM_DEBT"] / (bureau_debt["DAYS_CREDIT_ENDDATE"] / 30)
    
    result = pd.merge(result, bureau_debt, on="SK_ID_CURR", how="left").fillna(0)
    
    result["TOTAL_ANNUITY"] = result["CREDIT_DEBT_ANNUITY"] + result["AMT_ANNUITY"]
    result["INCOME_DEBT_RATIO"] = result["TOTAL_ANNUITY"] / result["AMT_INCOME_TOTAL"] * 100

    result = result.drop(columns=["AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT_ENDDATE", "CREDIT_DEBT_ANNUITY", "TOTAL_ANNUITY"])
    
    return result


def client_social_circle(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline, sums up OBS_30_CNT_SOCIAL_CIRCLE with OBS_60_CNT_SOCIAL_CIRCLE,
    DEF_30_CNT_SOCIAL_CIRCLE with DEF_60_CNT_SOCIAL_CIRCLE and returns input dataframe with two new columns
    OBS_CNT_SOCIAL_CIRCLE and DEF_CNT_SOCIAL_CIRCLE. Also drops aforementioned primary columns.
    """
    
    result = input_df.copy()
    
    result["OBS_CNT_SOCIAL_CIRCLE"] = result["OBS_30_CNT_SOCIAL_CIRCLE"] * result["OBS_60_CNT_SOCIAL_CIRCLE"]
    result["DEF_CNT_SOCIAL_CIRCLE"] = result["DEF_30_CNT_SOCIAL_CIRCLE"] * result["DEF_60_CNT_SOCIAL_CIRCLE"]
    
    result = result.drop(columns=["OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
                                  "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"])
    
    return result


def drop_id(input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes in pandas dataframe from pipeline and drops SK_ID_CURR column which is irrelevant for final modelling.
    """
    
    result = input_df.copy()
    
    result = result.drop(columns=["SK_ID_CURR"])
    
    return result 