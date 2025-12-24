import streamlit as st
import pandas as pd

st.title("My First Streamlit App")

uploaded = st.file_uploader("Upload a CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df)

