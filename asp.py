import streamlit as st
import pandas as pd
import numpy as np

# Set page layout
st.set_page_config(page_title="LID Level ASP Dashboard", layout="wide")
st.title("LID Level ASP Analysis Dashboard")


def to_num(s):
    """Convert text column to numbers (strips commas; junk like '-' / 'NA' becomes NaN)."""
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


# 1. Load Data (cached once, shared across sessions -> no per-rerun copies)
@st.cache_resource
def load_data():
    id_cols = ['listing_id', 'seller_id', 'seller_name', 'brand']
    str_dtype = {c: str for c in id_cols}

    # --- JA file ---
    df_ja = pd.read_csv(
        "lid_JA.csv",
        usecols=id_cols + ['units', 'gmv'],
        dtype={**str_dtype, 'units': str, 'gmv': str},
        low_memory=False,
    )
    for c in ['units', 'gmv']:
        df_ja[c] = to_num(df_ja[c])
    df_ja = df_ja.rename(columns={'units': 'JA_units', 'gmv': 'JA_revenue'})

    # --- d-1 file: read in chunks and pre-sum per listing/seller/brand/date ---
    # (only this small summary is kept in memory, never the raw rows)
    parts = []
    reader = pd.read_csv(
        "lid_d-1.csv",
        usecols=id_cols + ['unit_creation_timestamp', 'units', 'amount'],
        dtype={**str_dtype, 'unit_creation_timestamp': str, 'units': str, 'amount': str},
        chunksize=100_000,
    )
    for chunk in reader:
        chunk['units'] = to_num(chunk['units'])
        chunk['amount'] = to_num(chunk['amount'])
        chunk['date'] = chunk['unit_creation_timestamp'].str.split('T').str[0]
        chunk = chunk.drop(columns=['unit_creation_timestamp'])
        parts.append(
            chunk.groupby(id_cols + ['date'], as_index=False).agg(
                {'units': 'sum', 'amount': 'sum'}
            )
        )
    df_d1 = pd.concat(parts, ignore_index=True)
    df_d1 = df_d1.groupby(id_cols + ['date'], as_index=False).agg(
        {'units': 'sum', 'amount': 'sum'}
    )
    df_d1 = df_d1.rename(columns={'units': 'd-1_units', 'amount': 'd-1_revenue'})

    # Convert the date text to real dates (done on the small summary table)
    df_d1['date'] = pd.to_datetime(df_d1['date'], errors='coerce').dt.date
    df_d1 = df_d1.dropna(subset=['date'])

    return df_ja, df_d1


df_ja, df_d1 = load_data()

# 2. Sidebar Date Filter
st.sidebar.header("Filters")
available_dates = sorted(df_d1['date'].unique())
selected_dates = st.sidebar.multiselect(
    "Select Date(s) from d-1:",
    options=available_dates,
    default=available_dates
)

# Stop execution if no dates are selected
if not selected_dates:
    st.warning("Please select at least one date from the sidebar.")
    st.stop()

# 3. Filter and Aggregate d-1 Data
df_d1_filtered = df_d1[df_d1['date'].isin(selected_dates)]
df_d1_agg = df_d1_filtered.groupby(['listing_id', 'seller_id', 'seller_name', 'brand'], as_index=False).agg(
    {'d-1_units': 'sum', 'd-1_revenue': 'sum'}
)

# 4. Create the Basefile
basefile = pd.merge(df_ja, df_d1_agg, on=['listing_id', 'seller_id', 'seller_name', 'brand'], how='inner')

# Perform required calculations (inf from divide-by-zero treated like NaN -> 0)
basefile['JA_ASP'] = (basefile['JA_revenue'] / basefile['JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
basefile['d-1_ASP'] = (basefile['d-1_revenue'] / basefile['d-1_units']).replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate weighted metrics fixing JA_units
basefile['JA_ASP*JA_units'] = basefile['JA_ASP'] * basefile['JA_units']
basefile['d-1_ASP*JA_units'] = basefile['d-1_ASP'] * basefile['JA_units']

# Keep only the requested columns for the basefile display
base_columns = [
    'listing_id', 'seller_id', 'seller_name', 'brand',
    'JA_revenue', 'JA_units', 'd-1_revenue', 'd-1_units',
    'JA_ASP', 'd-1_ASP', 'JA_ASP*JA_units', 'd-1_ASP*JA_units'
]
basefile_display = basefile[base_columns]

# 5. Create Seller Level Pivot
seller_pivot = basefile.groupby(['seller_id', 'seller_name']).agg(
    Total_JA_units=('JA_units', 'sum'),
    Sum_JA_ASP_x_JA_units=('JA_ASP*JA_units', 'sum'),
    Sum_d1_ASP_x_JA_units=('d-1_ASP*JA_units', 'sum')
).reset_index()

seller_pivot['Fixed_JA_ASP'] = (seller_pivot['Sum_JA_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
seller_pivot['Fixed_d-1_ASP'] = (seller_pivot['Sum_d1_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)

# 6. Create Brand Level Pivot
brand_pivot = basefile.groupby(['brand']).agg(
    Total_JA_units=('JA_units', 'sum'),
    Sum_JA_ASP_x_JA_units=('JA_ASP*JA_units', 'sum'),
    Sum_d1_ASP_x_JA_units=('d-1_ASP*JA_units', 'sum')
).reset_index()

brand_pivot['Fixed_JA_ASP'] = (brand_pivot['Sum_JA_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
brand_pivot['Fixed_d-1_ASP'] = (brand_pivot['Sum_d1_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)

# 7. UI Dashboard Layout with Navigable Tabs
# (.round(2) replaces Styler: same 2-decimal display, far lighter on memory)
tab1, tab2, tab3 = st.tabs(["Basefile Data", "Seller Level Pivot", "Brand Level Pivot"])

with tab1:
    st.subheader("Basefile (Filtered by Date)")
    st.dataframe(basefile_display.round(2), use_container_width=True)

with tab2:
    st.subheader("ASP Fixed by JA Units - Seller Level")
    st.dataframe(seller_pivot.round(2), use_container_width=True)

with tab3:
    st.subheader("ASP Fixed by JA Units - Brand Level")
    st.dataframe(brand_pivot.round(2), use_container_width=True)