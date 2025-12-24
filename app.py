import streamlit as st
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------
import io
import csv
import pandas as pd
import streamlit as st

@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    # Read bytes once (Streamlit UploadedFile is file-like; caching likes deterministic inputs)
    raw = uploaded_file.getvalue()

    # Try decode with utf-8-sig first (handles BOM), then fallback
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore")
            break
        except Exception:
            pass
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    # Sniff delimiter from a sample
    sample = text[:200_000]
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", "|", ";"])
        delimiter = dialect.delimiter
    except Exception:
        # keep default ","
        pass

    # Parse with tolerant settings
    buf = io.StringIO(text)

    try:
        df = pd.read_csv(
            buf,
            sep=delimiter,
            engine="python",          # more tolerant than C engine
            on_bad_lines="skip",      # skip malformed lines instead of crashing
        )
        return df
    except Exception as e:
        # Try a last-resort parse (no quoting assumptions)
        buf2 = io.StringIO(text)
        try:
            df = pd.read_csv(
                buf2,
                sep=delimiter,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
            )
            return df
        except Exception:
            # Surface a helpful message in the app
            st.error(
                "Could not parse this file as a CSV. "
                "It may have inconsistent columns, broken quoting, or the wrong delimiter."
            )
            st.exception(e)
            raise



def fmt_int(x) -> str:
    return f"{x:,.0f}"


def fmt_pct(x) -> str:
    return f"{x:.2%}"


def build_cc(df: pd.DataFrame) -> pd.DataFrame:
    """
    County x RSSDHCR deposits + county totals + rank within county.
    This rolls up multiple CERTs to the same RSSDHCR automatically.
    """
    cc = (
        df.groupby(["STNAMEBR", "CNTYNAMB", "RSSDHCR"], as_index=False)
          .agg(bank_county_deps=("DEPSUMBR", "sum"))
    )

    county_totals = (
        cc.groupby(["STNAMEBR", "CNTYNAMB"], as_index=False)
          .agg(county_total_deps=("bank_county_deps", "sum"))
    )
    cc = cc.merge(county_totals, on=["STNAMEBR", "CNTYNAMB"], how="left")

    cc["rank_in_county"] = (
        cc.groupby(["STNAMEBR", "CNTYNAMB"])["bank_county_deps"]
          .rank(method="dense", ascending=False)
    )
    return cc


def build_rssdhcr_name_map(df: pd.DataFrame) -> pd.Series:
    """
 -> display name.

    Since NAMEFULL can differ across CERTs under the same RSSDHCR,
    pick the most common NAMEFULL for that RSSDHCR.
    """
    tmp = df[["RSSDHCR", "NAMEFULL"]].dropna(subset=["RSSDHCR"]).copy()
    tmp["NAMEFULL"] = tmp["NAMEFULL"].fillna("").astype(str)
    return tmp.groupby("RSSDHCR")["NAMEFULL"].agg(lambda s: s.value_counts().index[0])


def build_bank_lookup(df: pd.DataFrame, rssdhcr_name_map: pd.Series) -> pd.DataFrame:
    """
    Build RSSDHCR lookup with labels for searching by RSSDHCR or name.
    """
    ids = df[["RSSDHCR"]].dropna().drop_duplicates().copy()
    ids["RSSDHCR_STR"] = ids["RSSDHCR"].astype(str)
    ids["BANK_NAME"] = ids["RSSDHCR"].map(rssdhcr_name_map).fillna("")
    ids["LABEL"] = ids["RSSDHCR_STR"] + " — " + ids["BANK_NAME"]
    ids = ids.sort_values(["BANK_NAME", "RSSDHCR_STR"])
    return ids[["RSSDHCR", "RSSDHCR_STR", "BANK_NAME", "LABEL"]]


def bank_state_distribution(my: pd.DataFrame, bank_total_deps: float) -> pd.DataFrame:
    """State table for the selected RSSDHCR: deposits + % of bank total + rank."""
    my_state = (
        my.groupby("STNAMEBR", as_index=False)
          .agg(state_bank_deps=("bank_county_deps", "sum"))
    )

    my_state["pct_of_bank_total_in_state"] = (my_state["state_bank_deps"] / bank_total_deps).fillna(0)
    my_state["rank_state_for_bank"] = my_state["state_bank_deps"].rank(method="dense", ascending=False)
    my_state = my_state.sort_values("state_bank_deps", ascending=False)

    out = my_state[["STNAMEBR", "state_bank_deps", "pct_of_bank_total_in_state", "rank_state_for_bank"]].copy()
    out["state_bank_deps"] = out["state_bank_deps"].map(fmt_int)
    out["pct_of_bank_total_in_state"] = out["pct_of_bank_total_in_state"].map(fmt_pct)
    out["rank_state_for_bank"] = out["rank_state_for_bank"].astype(int)
    return out


def bank_state_market_share_table(cc: pd.DataFrame, rssdhcr: int | str) -> pd.DataFrame:
    # State x Bank deposits (all banks)
    state_bank = (
        cc.groupby(["STNAMEBR", "RSSDHCR"], as_index=False)
          .agg(bank_state_deps=("bank_county_deps", "sum"))
    )

    # Total deposits per state (all banks)
    state_totals = (
        state_bank.groupby("STNAMEBR", as_index=False)
                  .agg(state_total_deps=("bank_state_deps", "sum"))
    )

    # Rank banks within each state by deposits (1 = largest)
    state_bank["market_share_rank_in_state"] = (
        state_bank.groupby("STNAMEBR")["bank_state_deps"]
                  .rank(method="dense", ascending=False)
    )

    # Merge totals + compute market share in state
    state_bank = state_bank.merge(state_totals, on="STNAMEBR", how="left")
    state_bank["market_share_pct_in_state"] = (
        state_bank["bank_state_deps"] / state_bank["state_total_deps"]
    ).fillna(0)

    # Filter to selected bank
    me = state_bank[state_bank["RSSDHCR"] == rssdhcr].copy()
    if me.empty:
        return pd.DataFrame()

    # Bank total deposits across all states
    bank_total_deps = float(me["bank_state_deps"].sum())

    # Bank footprint in each state
    me["pct_of_bank_total_in_state"] = (me["bank_state_deps"] / bank_total_deps).fillna(0)

    # Weighted var
    me["weighted_var"] = me["market_share_rank_in_state"] * me["pct_of_bank_total_in_state"]

    # Sort rows (largest bank presence first)
    me = me.sort_values("bank_state_deps", ascending=False)

    # ---- TOTAL metrics you asked for ----
    avg_rank_total = float(me["market_share_rank_in_state"].mean())
    sum_weighted_var = float(me["weighted_var"].sum())
    avg_pct_of_bank_total_in_state = float(me["pct_of_bank_total_in_state"].mean())

    totals_row = {
        "STNAMEBR": "TOTAL / AVG",
        "state_total_deps": me["state_total_deps"].sum(),     # optional, still useful context
        "bank_state_deps": me["bank_state_deps"].sum(),       # equals bank_total_deps
        "market_share_rank_in_state": avg_rank_total,         # ✅ avg rank across states
        "market_share_pct_in_state": None,                    # not meaningful to average unless you want it
        "pct_of_bank_total_in_state": avg_pct_of_bank_total_in_state,  # ✅ avg pct across states
        "weighted_var": sum_weighted_var,                     # ✅ sum of weighted var
    }

    out = pd.concat([me, pd.DataFrame([totals_row])], ignore_index=True)

    # ---------- Formatting ----------
    out["state_total_deps"] = out["state_total_deps"].map(fmt_int)
    out["bank_state_deps"] = out["bank_state_deps"].map(fmt_int)

    # Rank: per-state ranks are ints; total row is avg -> show 2 decimals there
    def fmt_rank(x):
        if pd.isna(x):
            return ""
        # if it's basically an int, show int; else show 2 decimals
        return str(int(x)) if float(x).is_integer() else f"{float(x):.2f}"

    out["market_share_rank_in_state"] = out["market_share_rank_in_state"].apply(fmt_rank)

    out["market_share_pct_in_state"] = out["market_share_pct_in_state"].apply(
        lambda x: "" if pd.isna(x) else fmt_pct(x)
    )

    # pct_of_bank_total_in_state: per-state sum to 100%; total row is avg %
    out["pct_of_bank_total_in_state"] = out["pct_of_bank_total_in_state"].apply(
        lambda x: "" if pd.isna(x) else fmt_pct(x)
    )

    # weighted_var: per-state + total sum
    out["weighted_var"] = out["weighted_var"].apply(
        lambda x: "" if pd.isna(x) else f"{float(x):.4f}"
    )

    return out

def bank_county_rank_buckets_raw(my: pd.DataFrame, bank_total_deps: float) -> pd.DataFrame:
    """Return bucket table with numeric pct for charting, plus a TOTAL/AVG row."""
    def rank_bucket(r):
        r = int(r)
        return str(r) if r <= 9 else "10+"

    tmp = my.copy()
    tmp["rank_bucket"] = tmp["rank_in_county"].map(rank_bucket)

    bucket = (
        tmp.groupby("rank_bucket", as_index=False)
           .agg(
               counties=("CNTYNAMB", "count"),
               bank_deps=("bank_county_deps", "sum"),
           )
    )

    # Percent of bank total in each bucket
    bucket["pct_of_bank_total_in_bucket"] = (bucket["bank_deps"] / bank_total_deps).fillna(0)

    # Order 1..9 then 10+
    order = [str(i) for i in range(1, 10)] + ["10+"]
    bucket["rank_bucket"] = pd.Categorical(bucket["rank_bucket"], categories=order, ordered=True)
    bucket = bucket.sort_values("rank_bucket")

    # -----------------------
    # TOTAL / AVG row
    # -----------------------
    # map bucket -> numeric rank (10+ => 10)
    bucket_numeric = bucket.copy()
    bucket_numeric["rank_bucket_num"] = bucket_numeric["rank_bucket"].astype(str).apply(
        lambda x: 10 if x == "10+" else int(x)
    )

    # weighted average rank using deposit share weights
    weight = bucket_numeric["pct_of_bank_total_in_bucket"].fillna(0)
    avg_rank_bucket = float((bucket_numeric["rank_bucket_num"] * weight).sum() / max(weight.sum(), 1e-12))

    bank_total = float(bucket["bank_deps"].sum())
    avg_dep_share_per_bucket = float(bucket["pct_of_bank_total_in_bucket"].mean()) if len(bucket) else 0.0
    pct_total = float(bucket["pct_of_bank_total_in_bucket"].sum())

    totals_row = {
        "rank_bucket": "TOTAL / AVG",
        "counties": int(bucket["counties"].sum()),
        "bank_deps": bank_total,
        "pct_of_bank_total_in_bucket": pct_total,   # should be 1.0
        "avg_rank_bucket": avg_rank_bucket,
        "avg_dep_share_per_bucket": avg_dep_share_per_bucket,
    }

    bucket = pd.concat([bucket, pd.DataFrame([totals_row])], ignore_index=True)
    return bucket


def bank_county_rank_buckets_display(bucket_raw: pd.DataFrame) -> pd.DataFrame:
    """Format the bucket table for display (including TOTAL/AVG row)."""
    out = bucket_raw.copy()

    # Format money
    out["bank_deps"] = out["bank_deps"].apply(lambda x: "" if pd.isna(x) else fmt_int(float(x)))

    # Format pct (note: TOTAL row should be 100%)
    out["pct_of_bank_total_in_bucket"] = out["pct_of_bank_total_in_bucket"].apply(
        lambda x: "" if pd.isna(x) else fmt_pct(float(x))
    )

    # Optional columns: only totals row has these
    if "avg_rank_bucket" in out.columns:
        out["avg_rank_bucket"] = out["avg_rank_bucket"].apply(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )

    if "avg_dep_share_per_bucket" in out.columns:
        out["avg_dep_share_per_bucket"] = out["avg_dep_share_per_bucket"].apply(
            lambda x: "" if pd.isna(x) else fmt_pct(float(x))
        )

    return out

# ---------------------------
# App
# ---------------------------
st.title("RSSDHCR Rank Buckets Across Counties")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded is None:
    st.info("Please upload a CSV to continue.")
    st.stop()

df = load_csv(uploaded)

required_cols = {"STNAMEBR", "CNTYNAMB", "RSSDHCR", "NAMEFULL", "DEPSUMBR"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

df["DEPSUMBR"] = pd.to_numeric(df["DEPSUMBR"], errors="coerce").fillna(0)

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(df.head(200), use_container_width=True)

# Build aggregates once (by RSSDHCR)
cc = build_cc(df)

# Build RSSDHCR -> canonical name map + lookup for searching
rssdhcr_name_map = build_rssdhcr_name_map(df)
lookup = build_bank_lookup(df, rssdhcr_name_map)

selected_label = st.selectbox(
    "Search bank by RSSDHCR or NAMEFULL (type to search):",
    options=lookup["LABEL"].tolist(),
)

selected_row = lookup[lookup["LABEL"] == selected_label].iloc[0]
rssdhcr = selected_row["RSSDHCR"]
bank_name = selected_row["BANK_NAME"]

st.markdown(f"**Selected bank (rolled up by RSSDHCR):** `{bank_name}` (RSSDHCR `{rssdhcr}`)")

# Filter to chosen RSSDHCR
my = cc[cc["RSSDHCR"] == rssdhcr].copy()
if my.empty:
    st.warning(f"No rows found for RSSDHCR {rssdhcr}")
    st.stop()

bank_total_deps = float(my["bank_county_deps"].sum())

# ---------------------------
# Outputs
# ---------------------------
st.subheader(f"RSSDHCR {rssdhcr} — State distribution (share of bank total)")
st.dataframe(bank_state_distribution(my, bank_total_deps), use_container_width=True)

st.subheader(f"RSSDHCR {rssdhcr} — County rank buckets (share of bank total)")
bucket_raw = bank_county_rank_buckets_raw(my, bank_total_deps)

# Table (formatted)
st.dataframe(bank_county_rank_buckets_display(bucket_raw), use_container_width=True)

# Bar chart (x=bucket, y=%)
bucket_raw = bank_county_rank_buckets_raw(my, bank_total_deps)

chart_bucket = bucket_raw[bucket_raw["rank_bucket"] != "TOTAL / AVG"].copy()

chart_df = chart_bucket.set_index("rank_bucket")[["pct_of_bank_total_in_bucket"]]
st.bar_chart(chart_df)
st.subheader("Bucket distribution chart (% of bank total deposits)")
chart_df = bucket_raw.set_index("rank_bucket")[["pct_of_bank_total_in_bucket"]]
st.bar_chart(chart_df)

st.markdown(f"**RSSDHCR {rssdhcr} total deposits across all counties:** `{fmt_int(bank_total_deps)}`")

# ✅ NEW TABLE
st.divider()
st.subheader(f"RSSDHCR {rssdhcr} — State market share (bank vs total state deposits)")
st.dataframe(bank_state_market_share_table(cc, rssdhcr), use_container_width=True)

