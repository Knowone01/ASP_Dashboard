import streamlit as st
import pandas as pd

# Set page layout
st.set_page_config(page_title="LID Level ASP Dashboard", layout="wide")
st.title("LID Level ASP Analysis Dashboard")

# 1. Load Data (Cached so it only runs once per session)
@st.cache_data
def load_data():
    # Load the files (assumes they are in the same folder as this script)
    df_ja = pd.read_csv("lid_JA.csv")
    df_d1 = pd.read_csv("lid_d-1.csv")

    # Rename columns to match the standard
    df_ja = df_ja.rename(columns={'units': 'JA_units', 'gmv': 'JA_revenue'})
    df_d1 = df_d1.rename(columns={'units': 'd-1_units', 'amount': 'd-1_revenue'})

    # Extract date from the timestamp for filtering

    df_d1['date'] = pd.to_datetime(df_d1['unit_creation_timestamp'].astype(str).str.split('T').str[0]).dt.date

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
# Filter by selected dates and sum the units/revenue per listing to create the base mapping
df_d1_filtered = df_d1[df_d1['date'].isin(selected_dates)]
df_d1_agg = df_d1_filtered.groupby(['listing_id', 'seller_id', 'seller_name', 'brand'], as_index=False).agg(
    {'d-1_units': 'sum', 'd-1_revenue': 'sum'}
)

# 4. Create the Basefile
basefile = pd.merge(df_ja, df_d1_agg, on=['listing_id', 'seller_id', 'seller_name', 'brand'], how='inner')

# Perform required calculations
basefile['JA_ASP'] = (basefile['JA_revenue'] / basefile['JA_units']).fillna(0)
basefile['d-1_ASP'] = (basefile['d-1_revenue'] / basefile['d-1_units']).fillna(0)

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

seller_pivot['Fixed_JA_ASP'] = (seller_pivot['Sum_JA_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).fillna(0)
seller_pivot['Fixed_d-1_ASP'] = (seller_pivot['Sum_d1_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).fillna(0)

# 6. Create Brand Level Pivot
brand_pivot = basefile.groupby(['brand']).agg(
    Total_JA_units=('JA_units', 'sum'),
    Sum_JA_ASP_x_JA_units=('JA_ASP*JA_units', 'sum'),
    Sum_d1_ASP_x_JA_units=('d-1_ASP*JA_units', 'sum')
).reset_index()

brand_pivot['Fixed_JA_ASP'] = (brand_pivot['Sum_JA_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).fillna(0)
brand_pivot['Fixed_d-1_ASP'] = (brand_pivot['Sum_d1_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).fillna(0)

# 7. UI Dashboard Layout with Navigable Tabs
tab1, tab2, tab3 = st.tabs(["Basefile Data", "Seller Level Pivot", "Brand Level Pivot"])

with tab1:
    st.subheader("Basefile (Filtered by Date)")
    st.dataframe(basefile_display.style.format(precision=2), use_container_width=True)

with tab2:
    st.subheader("ASP Fixed by JA Units - Seller Level")
    st.dataframe(seller_pivot.style.format(precision=2), use_container_width=True)

with tab3:
    st.subheader("ASP Fixed by JA Units - Brand Level")
    st.dataframe(brand_pivot.style.format(precision=2), use_container_width=True)