import streamlit as st
import pandas as pd



st.title("CERT Rank Buckets Across Counties")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV to continue.")
    st.stop()

# ✅ Now it's safe
df = pd.read_csv(uploaded)
st.dataframe(df, use_container_width=True)

df["DEPSUMBR"] = pd.to_numeric(df["DEPSUMBR"], errors="coerce").fillna(0)

# UI: choose a CERT
certs = sorted(df["CERT"].dropna().unique().tolist())
cert = st.selectbox("Pick a CERT", certs)

# ---------------------------
# Pre-aggregate to COUNTY x CERT
# ---------------------------
cc = (
    df.groupby(["STNAMEBR", "CNTYNAMB", "CERT"], as_index=False)
      .agg(cert_county_deps=("DEPSUMBR", "sum"))
)

# County totals (all CERTs) (still useful if you later want market share in county)
county_totals = (
    cc.groupby(["STNAMEBR", "CNTYNAMB"], as_index=False)
      .agg(county_total_deps=("cert_county_deps", "sum"))
)
cc = cc.merge(county_totals, on=["STNAMEBR", "CNTYNAMB"], how="left")

# Rank each CERT within each county
cc["rank_in_county"] = (
    cc.groupby(["STNAMEBR", "CNTYNAMB"])["cert_county_deps"]
      .rank(method="dense", ascending=False)
)

# ---------------------------
# Filter to chosen CERT (all counties where this CERT has deposits)
# ---------------------------
my = cc[cc["CERT"] == cert].copy()
if my.empty:
    st.warning(f"No rows found for CERT {cert}")
    st.stop()

# CERT total deposits across all counties (for % of CERT total)
cert_total_deps = my["cert_county_deps"].sum()

# ---------------------------
# ✅ State-level summary (percent is % of CERT total, not % of state)
# ---------------------------
my_state = (
    my.groupby("STNAMEBR", as_index=False)
      .agg(state_cert_deps=("cert_county_deps", "sum"))
)

# % of CERT total that is in that state
my_state["pct_of_cert_total_in_state"] = (my_state["state_cert_deps"] / cert_total_deps).fillna(0)

# Rank of this state within the CERT's states (1 = biggest state for this CERT)
my_state["rank_state_for_cert"] = my_state["state_cert_deps"].rank(method="dense", ascending=False)

my_state = my_state.sort_values("state_cert_deps", ascending=False)

state_display = my_state[["STNAMEBR", "state_cert_deps", "pct_of_cert_total_in_state", "rank_state_for_cert"]].copy()
state_display["state_cert_deps"] = state_display["state_cert_deps"].map(lambda x: f"{x:,.0f}")
state_display["pct_of_cert_total_in_state"] = state_display["pct_of_cert_total_in_state"].map(lambda x: f"{x:.2%}")
state_display["rank_state_for_cert"] = state_display["rank_state_for_cert"].map(lambda x: f"{int(x)}")

st.subheader(f"CERT {cert} — State distribution (share of CERT total)")
st.dataframe(state_display, use_container_width=True)

# ---------------------------
# ✅ County rank buckets (percent is % of CERT total, not % of county totals)
# ---------------------------
def rank_bucket(r):
    r = int(r)
    return str(r) if r <= 9 else "10+"

my["rank_bucket"] = my["rank_in_county"].map(rank_bucket)

bucket = (
    my.groupby("rank_bucket", as_index=False)
      .agg(
          counties=("CNTYNAMB", "count"),
          cert_deps=("cert_county_deps", "sum"),
      )
)

# % of CERT total deposits that fall into counties where rank == bucket
bucket["pct_of_cert_total_in_bucket"] = (bucket["cert_deps"] / cert_total_deps).fillna(0)

# Ensure buckets ordered 1..9 then 10+
order = [str(i) for i in range(1, 10)] + ["10+"]
bucket["rank_bucket"] = pd.Categorical(bucket["rank_bucket"], categories=order, ordered=True)
bucket = bucket.sort_values("rank_bucket")

bucket_display = bucket.copy()
bucket_display["cert_deps"] = bucket_display["cert_deps"].map(lambda x: f"{x:,.0f}")
bucket_display["pct_of_cert_total_in_bucket"] = bucket_display["pct_of_cert_total_in_bucket"].map(lambda x: f"{x:.2%}")

st.subheader(f"CERT {cert} — County rank buckets (share of CERT total)")
st.dataframe(bucket_display, use_container_width=True)

# Optional: show CERT total
st.markdown(f"**CERT {cert} total deposits across all counties:** `{cert_total_deps:,.0f}`")

