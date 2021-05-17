import streamlit as st
import pandas as pd
from model import AnomalyDetection
DATA_URL = ("ex_data_v2.csv")

st.set_page_config(layout="wide")
st.title("anomaly detection app")
conf_side_bar = st.sidebar.beta_expander("select values")
with conf_side_bar:
    ci = st.slider(
        "select confidence interval:",
        min_value=0.5,
        max_value=0.999,
        step=0.001,
        value=0.95
    )
    split_ratio = st.slider(
        "select percentage for train-test",
        min_value=50,
        max_value=90,
        step=10,
        value=70
    )
col1, col2 = st.beta_columns([2,4])
with col1:
    uploaded_file = st.file_uploader("choose file to train on:")

with col2:
    df = pd.DataFrame()

@st.cache
def load_data(file=None):
    if file is None:
        file = DATA_URL
    data = pd.read_csv(file)
    return data

st.markdown("here is the original data:")
df = load_data(uploaded_file)
st.write(df)

anomaly_detection = AnomalyDetection(ci)
anomaly_rows = anomaly_detection.run(df, split_ratio)

st.markdown("anomaly rows are: ")
st.write(df.iloc[anomaly_rows, :])

btn = st.button('celebrate!')
if btn:
    st.balloons()


