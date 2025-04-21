import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from profilingdata import profiledata
from mlmodels import mlmodels
from autocleandata import autocleandata
from data_ana import data_analysis_section

if 'original_df' not in st.session_state:
    st.session_state.original_df = None

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

def main():
    st.set_page_config(page_title="OptiML Suite", layout="wide")
    st.title("OptiML Suite")
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Upload File", "Data Profiling and Cleaning", "Data Analysis", "Model Generation"],
            icons=["upload", "bar-chart-line", "graph-up","cpu"],
            menu_icon="cast",
            default_index=0
        )

    # 1. Upload File
    if selected == "Upload File":
        datacsv = st.file_uploader("Upload data file", type="csv")
        if datacsv:
            df = pd.read_csv(datacsv, low_memory=False)
            st.session_state.original_df = df
            st.session_state.cleaned_df = None
            st.success("File uploaded successfully!")
            st.dataframe(df)

    # 2. Data Profiling and Cleaning
    elif selected == "Data Profiling and Cleaning":
        if st.session_state.original_df is not None:
            df = st.session_state.original_df.copy()
            st.subheader("Original Data")
            st.dataframe(df)
            st.write("Shape before cleaning:", df.shape)

            if st.button("Clean Data"):
                cleaned = autocleandata(df)
                st.session_state.cleaned_df = cleaned
                st.success("Data cleaned and saved!")

            if st.session_state.cleaned_df is not None:
                st.subheader("Cleaned Data")
                cleaned_df = st.session_state.cleaned_df
                st.dataframe(cleaned_df)
                st.write("Shape after cleaning:", cleaned_df.shape)
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

            if st.checkbox("Show Profile Report"):
                profile_target = (
                    st.session_state.cleaned_df
                    if st.session_state.cleaned_df is not None
                    else st.session_state.original_df
                )
                if profile_target is not None:
                    st.subheader("Profiling Data")
                    profiledata(profile_target)
                else:
                    st.warning("No data available for profiling.")
        else:
            st.warning("Please upload a file first in 'Upload File' section.")
            
    elif selected == "Data Analysis":
        df = None
        if st.session_state.cleaned_df is not None:
            df = st.session_state.cleaned_df
        elif st.session_state.original_df is not None:
            df = st.session_state.original_df

        if df is not None:
            data_analysis_section(df)
        else:
            st.warning("Please upload and clean the data first.")
            
    # 3. Model Generation
    elif selected == "Model Generation":
        df = None
        if st.session_state.cleaned_df is not None:
            df = st.session_state.cleaned_df
        elif st.session_state.original_df is not None:
            df = st.session_state.original_df
        
        if df is not None:
            mlmodels(df)
        else:
            st.warning("Please upload and clean the data first.")

if __name__ == '__main__':
    main()
