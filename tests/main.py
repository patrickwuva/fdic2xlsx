from helpers import open_file
import streamlit as st
PATH = '../data/sod_data.csv'

def combine_data(df):
    df["cert_state_total"]  = df.groupby(["STNAMEBR", "CERT"])["DEPSUMBR"].transform("sum")
    df["state_total"]       = df.groupby(["STNAMEBR"])["DEPSUMBR"].transform("sum")
    df["county_state_total"]= df.groupby(["STNAMEBR", "CNTYNAMB"])["DEPSUMBR"].transform("sum")

    df["pct_cert_in_county"] = (df["DEPSUMBR"] / df["cert_state_total"]).fillna(0)

    df["pct_county_of_state"] = (df["county_state_total"] / df["state_total"]).fillna(0)

    df_sorted = df.sort_values(
        ["STNAMEBR", "CNTYNAMB", "cert_state_total", "DEPSUMBR"],
        ascending=[True, True, False, False]
    )
    return df_sorted

def main():
    df = open_file(PATH)
    new_df = combine_data(df)
    st.dataframe(new_df)

if __name__ == '__main__':
    main()
