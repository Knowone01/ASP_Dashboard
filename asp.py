import streamlit as st
import pandas as pd

st.set_page_config(page_title="LID Level ASP Dashboard", layout="wide")
st.title("LID Level ASP Analysis Dashboard")

JA_REQUIRED = ["listing_id", "seller_id", "seller_name", "brand", "gmv", "units"]
D1_REQUIRED = ["listing_id", "unit_creation_timestamp", "amount", "units"]


def safe_div(num, den):
    """num/den, returning 0 where den is 0 or missing (avoids inf/NaN)."""
    return num.div(den.where(den > 0)).fillna(0)


# 1. Load raw data (cached)
@st.cache_data
def load_raw():
    df_ja = pd.read_csv("lid_JA.csv", low_memory=False)
    df_d1 = pd.read_csv("lid_d-1.csv", low_memory=False)
    df_ja.columns = df_ja.columns.str.strip()
    df_d1.columns = df_d1.columns.str.strip()
    return df_ja, df_d1


df_ja, df_d1 = load_raw()

# 2. Validate columns up front so the error is readable
missing_ja = [c for c in JA_REQUIRED if c not in df_ja.columns]
missing_d1 = [c for c in D1_REQUIRED if c not in df_d1.columns]
if missing_ja or missing_d1:
    st.error(
        f"Missing columns -> lid_JA.csv: {missing_ja or 'none'} | "
        f"lid_d-1.csv: {missing_d1 or 'none'}"
    )
    st.write("JA columns:", list(df_ja.columns))
    st.write("d-1 columns:", list(df_d1.columns))
    st.stop()

# 3. Clean / rename
df_ja = df_ja.rename(columns={"units": "JA_units", "gmv": "JA_revenue"})
df_d1 = df_d1.rename(columns={"units": "d-1_units", "amount": "d-1_revenue"})

def to_num(s):
    """Convert to numbers; strips commas and turns junk text ('-', 'NA') into 0."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


for col in ["JA_revenue", "JA_units"]:
    df_ja[col] = to_num(df_ja[col]).fillna(0)
for col in ["d-1_revenue", "d-1_units"]:
    df_d1[col] = to_num(df_d1[col]).fillna(0)

df_ja["listing_id"] = df_ja["listing_id"].astype(str).str.strip()
df_d1["listing_id"] = df_d1["listing_id"].astype(str).str.strip()

# Date from timestamp (first 10 chars = YYYY-MM-DD)
df_d1["date"] = pd.to_datetime(
    df_d1["unit_creation_timestamp"].astype(str).str[:10], errors="coerce"
).dt.date
df_d1 = df_d1.dropna(subset=["date"])

# JA: one row per listing (guards against duplicate listing rows inflating the merge)
df_ja_agg = df_ja.groupby(
    ["listing_id", "seller_id", "seller_name", "brand"], as_index=False, dropna=False
).agg({"JA_revenue": "sum", "JA_units": "sum"})

# 4. Sidebar date filter
st.sidebar.header("Filters")
available_dates = sorted(df_d1["date"].unique())
selected_dates = st.sidebar.multiselect(
    "Select Date(s) from d-1:", options=available_dates, default=available_dates
)
if not selected_dates:
    st.warning("Please select at least one date from the sidebar.")
    st.stop()

# 5. Aggregate d-1 per listing only, then merge on listing_id
df_d1_agg = (
    df_d1[df_d1["date"].isin(selected_dates)]
    .groupby("listing_id", as_index=False)
    .agg({"d-1_units": "sum", "d-1_revenue": "sum"})
)

basefile = pd.merge(df_ja_agg, df_d1_agg, on="listing_id", how="inner")

st.caption(
    f"Matched {len(basefile):,} of {len(df_ja_agg):,} JA listings "
    f"with d-1 sales on the selected dates."
)
if basefile.empty:
    st.error("No listings matched between JA and d-1. Check that listing_id formats are the same in both files.")
    st.stop()

# 6. Calculations
basefile["JA_ASP"] = safe_div(basefile["JA_revenue"], basefile["JA_units"])
basefile["d-1_ASP"] = safe_div(basefile["d-1_revenue"], basefile["d-1_units"])
basefile["JA_ASP*JA_units"] = basefile["JA_ASP"] * basefile["JA_units"]
basefile["d-1_ASP*JA_units"] = basefile["d-1_ASP"] * basefile["JA_units"]

base_columns = [
    "listing_id", "seller_id", "seller_name", "brand",
    "JA_revenue", "JA_units", "d-1_revenue", "d-1_units",
    "JA_ASP", "d-1_ASP", "JA_ASP*JA_units", "d-1_ASP*JA_units",
]
basefile_display = basefile[base_columns]


# 7. Pivots
def make_pivot(df, keys):
    pivot = df.groupby(keys, dropna=False).agg(
        Total_JA_units=("JA_units", "sum"),
        Sum_JA_ASP_x_JA_units=("JA_ASP*JA_units", "sum"),
        Sum_d1_ASP_x_JA_units=("d-1_ASP*JA_units", "sum"),
    ).reset_index()
    pivot["Fixed_JA_ASP"] = safe_div(pivot["Sum_JA_ASP_x_JA_units"], pivot["Total_JA_units"])
    pivot["Fixed_d-1_ASP"] = safe_div(pivot["Sum_d1_ASP_x_JA_units"], pivot["Total_JA_units"])
    return pivot


seller_pivot = make_pivot(basefile, ["seller_id", "seller_name"])
brand_pivot = make_pivot(basefile, ["brand"])

# 8. UI
tab1, tab2, tab3 = st.tabs(["Basefile Data", "Seller Level Pivot", "Brand Level Pivot"])

with tab1:
    st.subheader("Basefile (Filtered by Date)")
    st.dataframe(basefile_display, use_container_width=True)

with tab2:
    st.subheader("ASP Fixed by JA Units - Seller Level")
    st.dataframe(seller_pivot, use_container_width=True)

with tab3:
    st.subheader("ASP Fixed by JA Units - Brand Level")
    st.dataframe(brand_pivot, use_container_width=True)