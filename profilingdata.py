import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from col_datatype import detect_variable_type

def overview(df):
    c1, c2 = st.columns(2)
    with c1:
        stats = {
            "Number of variables": df.shape[1],
            "Number of observations": df.shape[0],
            "Missing cells": df.isnull().sum().sum(),
            "Missing cells (%)": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "Duplicate rows": df.duplicated().sum(),
            "Duplicate rows (%)": (df.duplicated().sum() / df.shape[0]) * 100,
            "Total size in memory": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "Average record size in memory": f"{df.memory_usage(deep=True).sum() / df.shape[0]:.2f} bytes",
        }
        for key, value in stats.items():
            st.write(f"- {key}: {value}")

    with c2:
        st.write("\n**Variable Types**")
        variable_types = {
            "Text": 0,
            "Categorical": 0,
            "Binary": 0,
            "Numeric": 0,
        }
        for col in df.columns:
            var_type = detect_variable_type(df, col)
            variable_types[var_type] += 1

        for var_type, count in variable_types.items():
            st.write(f"- {var_type}: {count}")

def variable_overview(df):
    st.write("### Variables Overview")
    columns = sorted(df.columns)
    col = st.selectbox("Search or Select Column", options=columns, index=0, key="variable_select")
    if col:
        st.write(f"### {col}")
        var_type = detect_variable_type(df, col)
        st.write(f"**Variable Type:** {var_type}")
        c3, c4 = st.columns(2)
        if var_type in ["Text", "Categorical"]:
            with c3:
                content = {
                    "Distinct": df[col].nunique(),
                    "Distinct (%)": (df[col].nunique() / len(df)) * 100,
                    "Missing": df[col].isnull().sum(),
                    "Missing (%)": (df[col].isnull().sum() / len(df)) * 100,
                    "Memory size": f"{df[col].memory_usage(deep=True) / 1024:.2f} KB",
                }
                for key, value in content.items():
                    st.write(f"- {key}: {value}")

            with c4:
                if var_type == "Categorical":
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_ylabel("")
                    st.pyplot(fig)

        elif var_type == "Binary":
            with c3:
                content = {
                    "Distinct": df[col].nunique(),
                    "Distinct (%)": (df[col].nunique() / len(df)) * 100,
                    "Missing": df[col].isnull().sum(),
                    "Missing (%)": (df[col].isnull().sum() / len(df)) * 100,
                    "Memory size": f"{df[col].memory_usage(deep=True) / 1024:.2f} KB",
                }
                for key, value in content.items():
                    st.write(f"- {key}: {value}")
            with c4:
                fig, ax = plt.subplots()
                df[col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

        elif var_type == "Numeric":
            with c3:
                content = {
                    "Distinct": df[col].nunique(),
                    "Distinct (%)": (df[col].nunique() / len(df)) * 100,
                    "Missing": df[col].isnull().sum(),
                    "Missing (%)": (df[col].isnull().sum() / len(df)) * 100,
                    "Infinite": np.isinf(df[col]).sum(),
                    "Infinite (%)": (np.isinf(df[col]).sum() / len(df)) * 100,
                    "Mean": df[col].mean(),
                    "Min": df[col].min(),
                    "Max": df[col].max(),
                    "Zeros": (df[col] == 0).sum(),
                    "Zeros (%)": ((df[col] == 0).sum() / len(df)) * 100,
                    "Negatives": (df[col] < 0).sum(),
                    "Negatives (%)": ((df[col] < 0).sum() / len(df)) * 100,
                    "Memory size": f"{df[col].memory_usage(deep=True) / 1024:.2f} KB",
                }
                for key, value in content.items():
                    st.write(f"- {key}: {value}")
            with c4:
                fig, ax = plt.subplots()
                df[col].plot.hist(bins=20, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

def correlations_overview(df):
    encoded_df = df.copy()
    for col in df.columns:
        var_type = detect_variable_type(df, col)
        if var_type == "Categorical" or var_type == "Binary":
            encoded_df[col] = df[col].astype("category").cat.codes

    numeric_cols = encoded_df.select_dtypes(include=["number"])
    if not numeric_cols.empty:
        corr = numeric_cols.corr()
        corr = corr[sorted(corr.columns)].loc[sorted(corr.columns)]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric, binary, or categorical columns available for correlation analysis.")

def profiledata(df):
    menu_choice = option_menu(
        menu_title=None,
        options=["Dataset Statistics", "Variables", "Correlations"],
        icons=["diagram-3", "columns", "kanban"],
        default_index=0,
        orientation="horizontal",
    )

    if menu_choice == "Dataset Statistics":
        with st.spinner("Loading overview..."):
            overview(df)

    elif menu_choice == "Variables":
        variable_overview(df)

    elif menu_choice == "Correlations":
        with st.spinner("Loading correlation overview..."):
            correlations_overview(df)
