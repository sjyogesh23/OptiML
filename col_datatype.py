import numpy as np

def detect_variable_type(df, col):
    unique_values = df[col].dropna().unique()
    if len(unique_values) == 2:
        return "Binary"
    elif df[col].dtype == "object" or df[col].dtype.name == "category":
        return "Categorical" if len(unique_values) < 50 else "Text"
    elif np.issubdtype(df[col].dtype, np.number):
        return "Numeric"
    else:
        return "Text"