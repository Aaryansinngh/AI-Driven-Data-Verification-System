import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.preprocessing import preprocess_data
from src.anomaly_models import isolation_forest_model

# Page config
st.set_page_config(
    page_title="AI Data Verification Dashboard",
    layout="centered"
)

# Title and description
st.title("AI Data Verification Dashboard")
st.write("Upload a CSV file to detect anomalies using AI models.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# If no file uploaded
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Handle empty file
if uploaded_file.size == 0:
    st.error("❌ Uploaded file is empty. Please upload a valid CSV file.")
    st.stop()

# Read CSV safely
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading CSV file: {e}")
    st.stop()

# Check if dataframe is empty
if df.empty:
    st.error("❌ CSV file has no data rows.")
    st.stop()

# Show uploaded data
st.subheader("Uploaded Data")
st.dataframe(df)

# Preprocess data
try:
    processed_data = preprocess_data(df)
except Exception as e:
    st.error(f"❌ Error during preprocessing: {e}")
    st.stop()

# Apply anomaly detection
try:
    predictions = isolation_forest_model(processed_data)
except Exception as e:
    st.error(f"❌ Error during anomaly detection: {e}")
    st.stop()

# Add results
df["Anomaly"] = predictions

# Show results
st.subheader("Anomaly Detection Results")
st.dataframe(df)

# Visualization
st.subheader("Anomaly Distribution")
st.bar_chart(df["Anomaly"].value_counts())

# Explanation
st.markdown("""
### 🔍 Interpretation
- **1** → Normal data point  
- **-1** → Anomalous data point  

This model uses **Isolation Forest**, an unsupervised machine learning algorithm.
""")
