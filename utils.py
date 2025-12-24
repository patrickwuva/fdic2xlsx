import streamlit as st
import pandas as pd

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

