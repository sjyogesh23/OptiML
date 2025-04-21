import streamlit as st
import json
import pickle
import zipfile
import os
import time
from pycaret.regression import setup as regression_setup, compare_models as regression_compare, pull as regression_pull
from pycaret.classification import setup as classification_setup, compare_models as classification_compare, pull as classification_pull
from sklearn.preprocessing import LabelEncoder
from col_datatype import detect_variable_type
from preprocessingdata import preprocessingdata

def trainmodels(df, target_type, target):
    if target_type == "Numeric":
        start_time = time.time()
        regression_setup(df, target=target, session_id=1)
        st.write("✅ Setup completed in", round(time.time() - start_time, 2), "seconds")
        
        with st.spinner("Initializing regression setup pull..."):
            start_time = time.time()
            setup_df = regression_pull()
            st.write("✅ Regression setup is ready")
            st.dataframe(setup_df)
            st.write("✅ Model completed in", round(time.time() - start_time, 2), "seconds")

        with st.spinner("Training regression models..."):
            best_model = regression_compare()
            compare_df = regression_pull()
            st.write("✅ Best Regression Model")
    else:
        with st.spinner("Initializing classification setup..."):
            classification_setup(df, target=target, session_id=1, fold_shuffle=True)
            setup_df = classification_pull()
            st.write("✅ Classification setup is ready")
            st.dataframe(setup_df)

        with st.spinner("Training classification models..."):
            best_model = classification_compare()
            compare_df = classification_pull()
            st.write("✅ Best Classification Model")

    best_model.target_name = target
    st.dataframe(compare_df)
    st.write(best_model)
    return best_model


def create_model_inputs(df, target, column_types):
    model_inputs = {
        "target": {},
        "input_columns": {}
    }

    def build_meta(col, col_type):
        meta = {
            "variable_name": col,
            "variable_type": col_type,
            "unique_percentage": round(df[col].nunique() / df.shape[0] * 100, 2)
        }
        if col_type in ["Binary", "Categorical"]:
            unique_vals = df[col].dropna().unique()
            meta["inputs"] = {i + 1: str(val) for i, val in enumerate(unique_vals)}
        return meta

    model_inputs["target"] = build_meta(target, column_types[target])

    input_cols = [col for col in df.columns if col != target]
    for idx, col in enumerate(input_cols, start=1):
        model_inputs["input_columns"][str(idx)] = build_meta(col, column_types[col])

    return model_inputs


def mlmodels(df):
    st.subheader('ML Models')

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

    column_types = {col: detect_variable_type(df, col) for col in df.columns}
    filtered_columns = {col: col_type for col, col_type in column_types.items() if col_type != "Text"}
    columns_with_types = [f"{col} ({col_type})" for col, col_type in filtered_columns.items()]

    target_label = st.selectbox("Select Target Variable", options=sorted(columns_with_types), key="target_select")
    target = target_label.split(" (")[0]
    target_type = filtered_columns[target]

    model_inputs = create_model_inputs(df, target, column_types)

    encoder = LabelEncoder()
    if target_type in ["Binary", "Categorical"]:
        df[target] = encoder.fit_transform(df[target])

    if st.button("Train Model"):
        df, label_encoders = preprocessingdata(df)
        df = df.loc[:, df.nunique() > 1]
        st.session_state.df = df
        
        st.write("Dataframe after preprocessing and encoding:")
        st.dataframe(df)
        st.write("Shape after preprocessing:", df.shape)
        try:
            best_model = trainmodels(df, target_type, target)
            if best_model is None:
                return
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during model training: {e}")
            return

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model_inputs.json", "w") as f:
            json.dump(model_inputs, f, indent=4)
        with open("artifacts/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        with open("artifacts/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        zip_filename = "model_package.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            zipf.write("artifacts/model_inputs.json")
            zipf.write("artifacts/label_encoders.pkl")
            zipf.write("artifacts/best_model.pkl")

        
        st.write("To demostrate the model:")
        st.write("1. Open this link [OptiML Suite - Prediction App](https://optimlsuite-app.streamlit.app/)")
        st.write("2. Download the model package you just created.")
        st.write("3. Upload the model package you just created.")
        st.write("4. Enter the inputs for prediction.")
        with open(zip_filename, "rb") as f:
            st.download_button(
                label="Download Trained Model Package",
                data=f,
                file_name="model_package.zip",
                mime="application/zip"
            )

        try:
            os.remove(zip_filename)
            os.remove("artifacts/model_inputs.json")
            os.remove("artifacts/label_encoders.pkl")
            os.remove("artifacts/best_model.pkl")
            os.rmdir("artifacts")
        except Exception as e:
            st.warning(f"⚠️ Cleanup failed: {e}")
            
        return best_model