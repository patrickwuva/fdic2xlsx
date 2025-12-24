import pandas as pd

def open_file(path:str):
    return pd.read_csv(path)

