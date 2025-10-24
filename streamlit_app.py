import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import dill
import io
from pathlib import Path


st.set_page_config(page_title="Student Marks Predictor", layout="wide")

BASE_DIR = Path(__file__).parent

st.title("Student Marks Prediction — Streamlit Deployment")

st.write(
    "This app looks for .pkl model files in the same folder. Upload a CSV (rows=samples, cols=features) or enter a single sample as comma-separated values to predict."
)

# find .pkl files in folder
# look for common serialized model files in the folder
pkl_files = sorted([p.name for p in BASE_DIR.glob("*.pkl")] + [p.name for p in BASE_DIR.glob("*.joblib")])

if not pkl_files:
    st.warning(
        "No .pkl files found in the app folder. Place your model files (e.g., 'model.pkl') in the same folder as this app."
    )
    st.info("If you want, upload a .pkl directly via the sidebar (select 'Upload a .pkl').")

upload_pkl = st.sidebar.file_uploader("Upload a .pkl (optional, will not overwrite files on disk)", type=["pkl"])


def load_model_from_path(path):
    """Try several loaders for a path: joblib, pickle (with fallbacks), dill."""
    # Try joblib first (common for sklearn models)
    try:
        return joblib.load(path)
    except Exception:
        pass

    # Try pickle with a couple of encoding options (helps with py2->py3 pickles)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception:
        pass

    # Try dill as a last resort
    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise e


def load_model_from_bytes(bts):
    """Try to load a model from raw bytes using pickle/dill. Returns object or raises last exception."""
    last_exc = None
    try:
        return joblib.load(io.BytesIO(bts))
    except Exception:
        pass

    try:
        return pickle.loads(bts)
    except Exception as e:
        last_exc = e

    try:
        return pickle.loads(bts, encoding="latin1")
    except Exception as e:
        last_exc = e

    try:
        return dill.loads(bts)
    except Exception as e:
        last_exc = e

    if last_exc:
        raise last_exc


loaded_model = None
loaded_scaler = None

if upload_pkl:
    try:
        # read bytes and attempt several loaders
        content = upload_pkl.read()
        obj = load_model_from_bytes(content)
        # heuristics: decide if uploaded object is a scaler/preprocessor or a model
        if hasattr(obj, "predict"):
            loaded_model = obj
            st.sidebar.success("Model loaded from upload (temporary)")
        elif hasattr(obj, "transform"):
            loaded_scaler = obj
            st.sidebar.success("Scaler/preprocessor loaded from upload (temporary)")
        else:
            loaded_model = obj
            st.sidebar.info("Uploaded object loaded (no predict/transform detected).")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded pkl: {e}")

if pkl_files:
    st.sidebar.markdown("**Local serialized files**")
    # allow selecting multiple files (model + optional scaler)
    choices = st.sidebar.multiselect("Select files from folder (model and/or scaler)", pkl_files)
    for choice in choices:
        try:
            obj = load_model_from_path(BASE_DIR / choice)
            # Heuristics to categorize loaded object
            if hasattr(obj, "predict"):
                if loaded_model is None:
                    loaded_model = obj
                    st.sidebar.success(f"Model loaded: {choice}")
                else:
                    st.sidebar.warning(f"Multiple model-like objects selected; '{choice}' was ignored (already have a model).")
            elif hasattr(obj, "transform") and not hasattr(obj, "predict"):
                if loaded_scaler is None:
                    loaded_scaler = obj
                    st.sidebar.success(f"Scaler/preprocessor loaded: {choice}")
                else:
                    st.sidebar.warning(f"Multiple scaler-like objects selected; '{choice}' was ignored (already have a scaler).")
            else:
                # fallback: treat as model if no other model present
                if loaded_model is None:
                    loaded_model = obj
                    st.sidebar.info(f"Loaded object (treated as model): {choice}")
                else:
                    st.sidebar.info(f"Loaded object '{choice}' ignored (no obvious role)")
        except Exception as e:
            msg = str(e)
            if "STACK_GLOBAL requires str" in msg or "STACK_GLOBAL" in msg:
                msg += (
                    " — this often happens when a model was pickled with a different Python version or with Python2 compatibility. "
                    "Try re-saving the model using `joblib.dump(model, 'model.pkl')` from your training environment, or run the included `convert_model.py` script."
                )
            st.sidebar.error(f"Failed to load '{choice}': {msg}")


def try_predict(model, df):
    """Try to call predict on model and return array or raise exception."""
    if hasattr(model, "predict"):
        return model.predict(df)
    # fallback for estimators stored as dicts or custom wrappers
    if isinstance(model, dict) and "model" in model and hasattr(model["model"], "predict"):
        return model["model"].predict(df)
    raise AttributeError("Loaded object has no 'predict' method")


if loaded_model is not None:
    st.header("Predict from CSV")
    upload = st.file_uploader("Upload CSV (rows = samples, columns = features)", type=["csv"]) 
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            st.subheader("Input preview")
            st.dataframe(df.head())

            # Prepare input for prediction: apply scaler if provided and model is not a pipeline
            X = df
            X_for_pred = X
            if loaded_scaler is not None and not hasattr(loaded_model, "named_steps"):
                try:
                    X_for_pred = loaded_scaler.transform(X)
                except Exception as e:
                    st.warning(f"Failed to apply scaler to input DataFrame: {e}. Attempting to predict on raw input.")
                    X_for_pred = X

            with st.spinner("Running prediction..."):
                preds = try_predict(loaded_model, X_for_pred)
            st.subheader("Predictions")
            # show results as dataframe
            out = df.copy()
            out["prediction"] = preds
            st.dataframe(out.head())
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", csv, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.header("Manual single-sample prediction")
    st.write("Enter feature values in order, comma-separated (e.g. 4.5, 10.0, 1)")
    sample_text = st.text_input("Features (comma-separated)")
    if st.button("Predict manual input"):
        if not sample_text:
            st.error("Enter feature values first.")
        else:
            try:
                values = np.array([float(x.strip()) for x in sample_text.split(",")]).reshape(1, -1)
                values_for_pred = values
                if loaded_scaler is not None and not hasattr(loaded_model, "named_steps"):
                    try:
                        values_for_pred = loaded_scaler.transform(values)
                    except Exception as e:
                        st.warning(f"Failed to apply scaler to manual input: {e}. Predicting on raw values.")
                        values_for_pred = values

                with st.spinner("Predicting..."):
                    pred = try_predict(loaded_model, values_for_pred)
                st.success(f"Prediction: {pred}")
                st.write("Raw output:")
                st.write(pred)
            except Exception as e:
                st.error(f"Failed to predict from manual input: {e}")

    st.markdown("---")
    if hasattr(loaded_model, "predict_proba"):
        st.info("The loaded model supports predict_proba (probabilities will be available when using the CSV flow).")

st.markdown("\n---\n\nHints:\n- The model must accept a 2D array or DataFrame for .predict().\n- For sklearn pipelines, include any preprocessing/scaler inside the pipeline before saving.\n- If your model uses a scaler saved separately, you can load both .pkl files and apply scaler manually in a custom script.")
