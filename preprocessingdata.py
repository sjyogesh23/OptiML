import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from col_datatype import detect_variable_type

def remove_outliers_zscore(df, threshold=3):
    numeric_cols = [col for col in df.columns if detect_variable_type(df, col) == "Numeric"]
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    filtered_df = df[(z_scores < threshold).all(axis=1)]
    return filtered_df

def remove_outliers_iqr(df, factor=1.5):
    numeric_cols = [col for col in df.columns if detect_variable_type(df, col) == "Numeric"]
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    filtered_df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]
    return filtered_df

def encode_categorical_columns(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=[object]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    for col in numeric_cols:
        variable_type = detect_variable_type(df, col)
        if variable_type not in ["Category", "Binary"]:
            df[col] = scaler.fit_transform(df[[col]])
    return df

def preprocessingdata(df):
    method = 1

    if method == 0:
        df = remove_outliers_zscore(df)
    elif method == 1:
        df = remove_outliers_iqr(df)
    else:
        pass
    
    df, label_encoders = encode_categorical_columns(df)
    df = scale_numeric_columns(df)
    return df, label_encoders