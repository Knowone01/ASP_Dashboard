import gc
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="LID Level ASP Dashboard", layout="wide")
st.title("LID Level ASP Analysis Dashboard")


def to_num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


def normalize(columns):
    return columns.str.strip().str.lower().str.replace(' ', '_', regex=False)


BUCKET_BINS   = [0, 200, 300, 500, 1000, np.inf]
BUCKET_LABELS = ['0-200', '200-300', '300-500', '500-1000', 'Above 1000']


def assign_bucket(asp_series):
    return pd.cut(
        asp_series,
        bins=BUCKET_BINS,
        labels=BUCKET_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)


@st.cache_resource
def load_data():
    id_cols    = ['listing_id', 'seller_id', 'seller_name', 'brand']
    extra_cols = ['super_category', 'vertical', 'alpha_flag', 'asp_bucket']

    # ── JA ────────────────────────────────────────────────────────────────────
    df_ja = pd.read_csv(
        "lid_JA.csv",
        usecols=id_cols + ['units', 'gmv'],
        dtype=str,
        low_memory=False,
    )
    df_ja.columns = normalize(df_ja.columns)
    df_ja['units'] = to_num(df_ja['units']).astype('float32')
    df_ja['gmv']   = to_num(df_ja['gmv']).astype('float32')
    df_ja = df_ja.rename(columns={'units': 'JA_units', 'gmv': 'JA_revenue'})

    # ── d-1: peek header → exact usecols ─────────────────────────────────────
    _peek    = pd.read_csv("lid_d-1.csv", nrows=0)
    _col_map = dict(zip(normalize(_peek.columns), _peek.columns))

    needed      = id_cols + extra_cols + ['unit_creation_timestamp', 'units', 'amount']
    usecols_act = [_col_map[n] for n in needed if n in _col_map]
    rename_map  = {_col_map[n]: n for n in needed if n in _col_map}

    d1_parts, meta_parts = [], []

    for chunk in pd.read_csv(
        "lid_d-1.csv", usecols=usecols_act, dtype=str, chunksize=100_000
    ):
        chunk = chunk.rename(columns=rename_map)
        chunk['units']  = to_num(chunk['units']).astype('float32')
        chunk['amount'] = to_num(chunk['amount']).astype('float32')
        chunk['date']   = chunk['unit_creation_timestamp'].str.split('T').str[0]
        chunk.drop(columns=['unit_creation_timestamp'], inplace=True)

        d1_parts.append(
            chunk.groupby(['listing_id', 'date'], as_index=False)
                 .agg({'units': 'sum', 'amount': 'sum'})
        )
        meta_parts.append(
            chunk[id_cols + extra_cols].drop_duplicates('listing_id')
        )

    df_d1 = pd.concat(d1_parts, ignore_index=True)
    del d1_parts; gc.collect()

    df_d1 = (df_d1.groupby(['listing_id', 'date'], as_index=False)
                  .agg({'units': 'sum', 'amount': 'sum'}))
    df_d1 = df_d1.rename(columns={'units': 'd-1_units', 'amount': 'd-1_revenue'})
    df_d1['date']        = pd.to_datetime(df_d1['date'], errors='coerce').dt.date
    df_d1['d-1_units']   = df_d1['d-1_units'].astype('float32')
    df_d1['d-1_revenue'] = df_d1['d-1_revenue'].astype('float32')
    df_d1.dropna(subset=['date'], inplace=True)

    lid_meta = (pd.concat(meta_parts, ignore_index=True)
                  .drop_duplicates('listing_id')
                  .reset_index(drop=True))
    del meta_parts; gc.collect()

    for col in id_cols + extra_cols:
        if col in lid_meta.columns:
            lid_meta[col] = lid_meta[col].str.strip().astype('category')

    # ── Lookup ────────────────────────────────────────────────────────────────
    df_lookup = pd.read_csv("Lookup.csv", dtype=str)
    df_lookup.columns = df_lookup.columns.str.strip()
    df_lookup['Brand']     = df_lookup['Brand'].str.strip()
    df_lookup['Brand_Tag'] = df_lookup['Brand_Tag'].str.strip()

    return df_ja, df_d1, lid_meta, df_lookup


df_ja, df_d1, lid_meta, df_lookup = load_data()

_lookup_map = df_lookup.drop_duplicates('Brand').set_index('Brand')['Brand_Tag']


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
available_dates = sorted(df_d1['date'].unique())
if not available_dates:
    st.error("No valid dates found in the d-1 data.")
    st.stop()

with st.sidebar.form("date_filter_form"):
    start_input = st.selectbox("Start Date (d-1):", options=available_dates, index=0)
    end_input   = st.selectbox(
        "End Date (d-1):", options=available_dates, index=len(available_dates) - 1
    )
    refresh = st.form_submit_button("Refresh Data")

if refresh:
    if start_input > end_input:
        st.sidebar.warning("Start Date must be on or before End Date.")
    else:
        st.session_state['applied_range'] = (start_input, end_input)

if 'applied_range' not in st.session_state:
    st.info("Select a Start Date and End Date in the sidebar, then click **Refresh Data**.")
    st.stop()

start_date, end_date = st.session_state['applied_range']
st.caption(f"Showing d-1 data from {start_date} to {end_date}")


# ── d-1 aggregate across date range ──────────────────────────────────────────
df_d1_agg = (
    df_d1[(df_d1['date'] >= start_date) & (df_d1['date'] <= end_date)]
    .groupby('listing_id', as_index=False)
    .agg({'d-1_units': 'sum', 'd-1_revenue': 'sum'})
)


# ── Basefile ──────────────────────────────────────────────────────────────────
basefile = pd.merge(df_ja, df_d1_agg, on='listing_id', how='inner')

basefile['JA_ASP'] = (
    (basefile['JA_revenue'] / basefile['JA_units'])
    .replace([np.inf, -np.inf], np.nan).fillna(0)
)
basefile['d-1_ASP'] = (
    (basefile['d-1_revenue'] / basefile['d-1_units'])
    .replace([np.inf, -np.inf], np.nan).fillna(0)
)
basefile['JA_ASP*JA_units']  = basefile['JA_ASP']  * basefile['JA_units']
basefile['d-1_ASP*JA_units'] = basefile['d-1_ASP'] * basefile['JA_units']

# JA ASP bucket: derived from JA_ASP, fixed per LID
basefile['JA_asp_bucket'] = assign_bucket(basefile['JA_ASP'])

basefile_display = basefile[[
    'listing_id', 'seller_id', 'seller_name', 'brand',
    'JA_revenue', 'JA_units', 'd-1_revenue', 'd-1_units',
    'JA_ASP', 'd-1_ASP', 'JA_ASP*JA_units', 'd-1_ASP*JA_units',
]]


# ── LID attrs ─────────────────────────────────────────────────────────────────
common_lids = set(basefile['listing_id'])

lid_attrs = (
    lid_meta[lid_meta['listing_id'].isin(common_lids)]
    .copy()
    .reset_index(drop=True)
)
lid_attrs['Brand_Tag']  = lid_attrs['brand'].astype(str).map(_lookup_map).fillna('Unbranded')
lid_attrs['alpha_flag'] = lid_attrs['alpha_flag'].astype(str).str.strip()

# Add JA_asp_bucket from basefile (computed from JA_ASP)
_ja_bucket_map        = basefile.set_index('listing_id')['JA_asp_bucket']
lid_attrs['JA_asp_bucket'] = lid_attrs['listing_id'].map(_ja_bucket_map).fillna('Unknown')

valid_lids = set(
    lid_attrs[
        (lid_attrs['alpha_flag'] == 'Non-Alpha') &
        (lid_attrs['Brand_Tag']  == 'Unbranded')
    ]['listing_id']
)


# ── Seller Pivot ──────────────────────────────────────────────────────────────
seller_pivot = (
    basefile.groupby(['seller_id', 'seller_name'])
    .agg(
        Total_JA_units       =('JA_units',         'sum'),
        Sum_JA_ASP_x_JA_units=('JA_ASP*JA_units',  'sum'),
        Sum_d1_ASP_x_JA_units=('d-1_ASP*JA_units', 'sum'),
    ).reset_index()
)
seller_pivot['Fixed_JA_ASP']  = (seller_pivot['Sum_JA_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
seller_pivot['Fixed_d-1_ASP'] = (seller_pivot['Sum_d1_ASP_x_JA_units'] / seller_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
seller_pivot['Disc %'] = ((seller_pivot['Fixed_d-1_ASP'] / seller_pivot['Fixed_JA_ASP'] - 1) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)


# ── Brand Pivot ───────────────────────────────────────────────────────────────
brand_pivot = (
    basefile.groupby(['brand'])
    .agg(
        Total_JA_units       =('JA_units',         'sum'),
        Sum_JA_ASP_x_JA_units=('JA_ASP*JA_units',  'sum'),
        Sum_d1_ASP_x_JA_units=('d-1_ASP*JA_units', 'sum'),
    ).reset_index()
)
brand_pivot['Fixed_JA_ASP']  = (brand_pivot['Sum_JA_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
brand_pivot['Fixed_d-1_ASP'] = (brand_pivot['Sum_d1_ASP_x_JA_units'] / brand_pivot['Total_JA_units']).replace([np.inf, -np.inf], np.nan).fillna(0)
brand_pivot['Disc %'] = ((brand_pivot['Fixed_d-1_ASP'] / brand_pivot['Fixed_JA_ASP'] - 1) * 100).replace([np.inf, -np.inf], np.nan).fillna(0)


# ── df_dated: shared by all ASP tabs ─────────────────────────────────────────
df_dated = pd.DataFrame()

if valid_lids:
    _ja_ref = (
        basefile[basefile['listing_id'].isin(valid_lids)]
        [['listing_id', 'JA_units', 'JA_ASP', 'JA_revenue']]
        .drop_duplicates('listing_id')
    )

    _attr_cols = lid_attrs[
        lid_attrs['listing_id'].isin(valid_lids)
    ][['listing_id', 'super_category', 'vertical', 'asp_bucket', 'JA_asp_bucket']].copy()

    for col in ['super_category', 'vertical', 'asp_bucket', 'JA_asp_bucket']:
        _attr_cols[col] = _attr_cols[col].astype(str)

    df_dated = (
        df_d1[
            df_d1['listing_id'].isin(valid_lids) &
            (df_d1['date'] >= start_date) &
            (df_d1['date'] <= end_date)
        ]
        .merge(_ja_ref,    on='listing_id', how='inner')
        .merge(_attr_cols, on='listing_id', how='left')
    )

    if not df_dated.empty:
        df_dated['d-1_ASP'] = (
            (df_dated['d-1_revenue'] / df_dated['d-1_units'])
            .replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        # Input ASP weights  — fixed by JA units
        df_dated['jaw']     = df_dated['JA_ASP']  * df_dated['JA_units']
        df_dated['d1w']     = df_dated['d-1_ASP'] * df_dated['JA_units']
        # Output ASP weights — fixed by d-1 units
        df_dated['jaw_out'] = df_dated['JA_ASP']  * df_dated['d-1_units']
        df_dated['d1w_out'] = df_dated['d-1_revenue']   # = d-1_ASP × d-1_units


# ── Shared helpers ────────────────────────────────────────────────────────────
def style_pivot(df):
    def color_disc(v):
        return ("background-color: #f8d7da; color: #842029" if v < 0
                else "background-color: #d1e7dd; color: #0f5132")
    styler = df.style.format(precision=2)
    fn = styler.map if hasattr(styler, 'map') else styler.applymap
    return fn(color_disc, subset=['Disc %'])


def style_date_disc(df, label_col):
    date_cols = [c for c in df.columns if c != label_col]
    def color(v):
        try:
            return ("background-color: #f8d7da; color: #842029" if float(v) < 0
                    else "background-color: #d1e7dd; color: #0f5132")
        except (TypeError, ValueError):
            return ""
    s  = df.style.format({c: '{:.2f}' for c in date_cols})
    fn = s.map if hasattr(s, 'map') else s.applymap
    return fn(color, subset=date_cols)


def make_disc_pivot(group_col, label_col, jaw_col, d1w_col, unit_col,
                    sort_by_gmv=False, sort_order=None):
    """
    sort_order : list of label values in desired display order (e.g. bucket order)
    sort_by_gmv: sort descending by JA GMV (used for vertical)
    """
    sub = df_dated.dropna(subset=[group_col])
    sub = sub[sub[group_col].astype(str) != 'nan']
    if sub.empty:
        return pd.DataFrame()

    agg = (sub.groupby([group_col, 'date'])
              .agg(jaw=(jaw_col, 'sum'), d1w=(d1w_col, 'sum'), ju=(unit_col, 'sum'))
              .reset_index())
    agg['Fixed_JA_ASP'] = (agg['jaw'] / agg['ju']).replace([np.inf, -np.inf], np.nan).fillna(0)
    agg['Fixed_d1_ASP'] = (agg['d1w'] / agg['ju']).replace([np.inf, -np.inf], np.nan).fillna(0)
    agg['Disc%'] = (
        (agg['Fixed_d1_ASP'] / agg['Fixed_JA_ASP'] - 1) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    pivot = agg.pivot(index=group_col, columns='date', values='Disc%')
    pivot.columns = [str(c) for c in pivot.columns]
    pivot.columns.name = None
    pivot = pivot.reset_index().rename(columns={group_col: label_col})

    if sort_order is not None:
        # Keep only rows that exist in data, in the defined order
        order_df = pd.DataFrame({label_col: [b for b in sort_order if b in pivot[label_col].values]})
        pivot = order_df.merge(pivot, on=label_col, how='left')
    elif sort_by_gmv:
        gmv_order = (sub.groupby(group_col)['JA_revenue']
                        .sum()
                        .reset_index()
                        .rename(columns={group_col: label_col})
                        .sort_values('JA_revenue', ascending=False))
        pivot = gmv_order[[label_col]].merge(pivot, on=label_col, how='left')

    return pivot


def render_asp_tab(title, jaw_col, d1w_col, unit_col, caption_suffix, sections):
    """
    sections: list of (group_col, label_col, sort_by_gmv, sort_order)
    """
    st.subheader(title)

    if not valid_lids:
        st.warning(
            "No listings match Non-Alpha + Unbranded. "
            "Check that alpha_flag in d-1 is exactly 'Non-Alpha' "
            "and Lookup.csv covers the relevant brands."
        )
        with st.expander("Debug"):
            st.write("alpha_flag values:", lid_attrs['alpha_flag'].unique().tolist())
            st.write("Brand_Tag values:",  lid_attrs['Brand_Tag'].unique().tolist())
        return

    if df_dated.empty:
        st.info('No d-1 data for the selected date range after applying filters.')
        return

    for i, (group_col, label_col, sort_by_gmv, sort_order) in enumerate(sections):
        pivot = make_disc_pivot(
            group_col, label_col, jaw_col, d1w_col, unit_col, sort_by_gmv, sort_order
        )
        if not pivot.empty:
            st.markdown(f'**{label_col}**')
            st.dataframe(style_date_disc(pivot, label_col), use_container_width=True)
        else:
            st.info(f'No {label_col} data available.')
        if i < len(sections) - 1:
            st.divider()

    st.caption(
        f'Disc% = (Fixed d-1 ASP / Fixed JA ASP − 1) × 100  ·  {caption_suffix}  ·  '
        'Common LIDs · Non-Alpha · Unbranded only'
    )


# ── Section definitions ───────────────────────────────────────────────────────
# (group_col, label_col, sort_by_gmv, sort_order)
SC_V_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('vertical',       'Vertical',       True,  None),
]

# Input ASP bucket: JA_asp_bucket (computed from JA_ASP, fixed per LID)
INPUT_BUCKET_SECTIONS = [
    ('JA_asp_bucket', 'JA ASP Bucket', False, BUCKET_LABELS),
]

# Output ASP bucket: asp_bucket from d-1 sheet (can change by date)
OUTPUT_BUCKET_SECTIONS = [
    ('asp_bucket', 'D-1 ASP Bucket', False, BUCKET_LABELS),
]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    'Output ASP',
    'Input ASP',
    'Output ASP (ASP Bucket)',
    'Input ASP (ASP Bucket)',
    'Seller Level Pivot',
    'Brand Level Pivot',
    'Basefile Data',
])

with tab1:
    render_asp_tab(
        title          = 'Output ASP — Fixed by d-1 Units',
        jaw_col        = 'jaw_out',
        d1w_col        = 'd1w_out',
        unit_col       = 'd-1_units',
        caption_suffix = 'weighted by d-1 units · verticals sorted by JA GMV',
        sections       = SC_V_SECTIONS,
    )

with tab2:
    render_asp_tab(
        title          = 'Input ASP — Fixed by JA Units',
        jaw_col        = 'jaw',
        d1w_col        = 'd1w',
        unit_col       = 'JA_units',
        caption_suffix = 'weighted by JA units · verticals sorted by JA GMV',
        sections       = SC_V_SECTIONS,
    )

with tab3:
    render_asp_tab(
        title          = 'Output ASP — D-1 ASP Bucket (Fixed by d-1 Units)',
        jaw_col        = 'jaw_out',
        d1w_col        = 'd1w_out',
        unit_col       = 'd-1_units',
        caption_suffix = 'weighted by d-1 units · d-1 ASP bucket from d-1 sheet',
        sections       = OUTPUT_BUCKET_SECTIONS,
    )

with tab4:
    render_asp_tab(
        title          = 'Input ASP — JA ASP Bucket (Fixed by JA Units)',
        jaw_col        = 'jaw',
        d1w_col        = 'd1w',
        unit_col       = 'JA_units',
        caption_suffix = 'weighted by JA units · JA ASP bucket derived from JA_ASP',
        sections       = INPUT_BUCKET_SECTIONS,
    )

with tab5:
    st.subheader('ASP Fixed by JA Units - Seller Level')
    st.dataframe(style_pivot(seller_pivot), use_container_width=True)

with tab6:
    st.subheader('ASP Fixed by JA Units - Brand Level')
    st.dataframe(style_pivot(brand_pivot), use_container_width=True)

with tab7:
    st.subheader('Basefile (Filtered by Date)')
    st.dataframe(basefile_display.round(2), use_container_width=True)
