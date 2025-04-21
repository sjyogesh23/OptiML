import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from dateutil.parser import parse

def basic_wraggling(df, dupli=True):
    if dupli:
        df.drop_duplicates(inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])
    return df

def clean_money_columns(df):
    def is_money(value):
        try:
            return bool(re.match(r'^\s*[\$€₹]?\s*-?\d+(\.\d+)?\s*$', str(value)))
        except:
            return False

    money_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: is_money(str(x))).mean() > 0.5:
            money_cols.append(col)
    
    for col in money_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    return df

def clean_date_column(df):
    def is_date(value):
        try:
            parse(value, fuzzy=False)
            return True
        except Exception:
            return False
        
    def parse_date(value):
        try:
            date = parse(value, dayfirst=False, yearfirst=False, fuzzy=False)
            return date.day, date.month, date.year
        except Exception:
            return None, None, None
    
    potential_date_cols = []
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            if df[col].apply(lambda x: isinstance(x, str) and is_date(str(x))).mean() > 0.8:
                potential_date_cols.append(col)

    for col in potential_date_cols:
        parsed_dates = df[col].astype(str).apply(parse_date)
        
        df[f"{col}_dd"] = parsed_dates.apply(lambda x: x[0])
        df[f"{col}_mm"] = parsed_dates.apply(lambda x: x[1])
        df[f"{col}_yyyy"] = parsed_dates.apply(lambda x: x[2])
        
        df[f"{col}_dd"] = df[f"{col}_dd"].replace({None: np.nan})
        df[f"{col}_mm"] = df[f"{col}_mm"].replace({None: np.nan})
        df[f"{col}_yyyy"] = df[f"{col}_yyyy"].replace({None: np.nan})
        df.drop(columns=[col], inplace=True)
    return df

def clean_time_column(df):
    def is_time(value):
        try:
            time = parse(value, fuzzy=False)
            return time.time() is not None
        except Exception:
            return False

    def parse_time(value):
        try:
            time = parse(value, fuzzy=False)
            return time.hour, time.minute, time.second
        except Exception:
            return None, None, None

    potential_time_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and is_time(str(x))).mean() > 0.5:
            potential_time_cols.append(col)

    for col in potential_time_cols:
        parsed_times = df[col].astype(str).apply(parse_time)
        
        df[f"{col}_hh"] = parsed_times.apply(lambda x: x[0])
        df[f"{col}_mm"] = parsed_times.apply(lambda x: x[1])
        df[f"{col}_ss"] = parsed_times.apply(lambda x: x[2])

        df[f"{col}_hh"] = df[f"{col}_hh"].replace({None: np.nan})
        df[f"{col}_mm"] = df[f"{col}_mm"].replace({None: np.nan})
        df[f"{col}_ss"] = df[f"{col}_ss"].replace({None: np.nan})
        df.drop(columns=[col], inplace=True)
    return df

def clean_text_column(df):
    object_cols = df.select_dtypes(include=[object]).columns
    for col in object_cols:
        df[col] = df[col].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()) if isinstance(x, str) else x
        )
    return df

def autocleandata(df):
    df = basic_wraggling(df)
    df = clean_date_column(df)
    df = clean_time_column(df)
    df = clean_money_columns(df)
    df = clean_text_column(df)
    df = basic_wraggling(df, dupli=False)
    df = impute_missing_values(df)
    df = df.loc[:, df.nunique() > 1]
    return df