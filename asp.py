import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ASP Disc% Dashboard", layout="wide")
st.title("ASP Discount % Dashboard")

GREEN = "background-color: #d1e7dd; color: #0f5132"
RED   = "background-color: #f8d7da; color: #842029"

BUCKET_LABELS    = ['0-200', '200-300', '300-500', '500-1000', 'Above 1000']
SEGMENT_OPTIONS  = ['Overall', 'Unbranded Non-Alpha', 'Branded Non-Alpha', 'Alpha']
TARGET_SUPERCATS = ['WomenWesternCore', 'WomenWesternGrowth']

NUM_COLS = ['num_op_u', 'den_op_u', 'num_ip_u', 'den_ip_u', 'units', 'gmv', 'mrp']


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(filepath):
    df = pd.read_csv(filepath, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(',', '', regex=False), errors='coerce'
            ).fillna(0).astype('float32')

    # Convert date to readable string: 20260924 → "24 Sep"
    df['date_'] = (pd.to_datetime(df['date_'].str.strip(), format='%Y%m%d', errors='coerce')
                     .dt.strftime('%d %b'))

    # Normalize segment columns
    df['alpha_flag']   = df['alpha_flag'].str.strip().str.upper()
    df['branded_flag'] = df['branded_flag'].str.strip()
    df['super_category'] = df['super_category'].str.strip()
    df['vertical']       = df['vertical'].str.strip()
    df['ja_asp_bucket']  = df['ja_asp_bucket'].str.strip()
    df['d1_asp_bucket']  = df['d1_asp_bucket'].str.strip()

    # Filter to target super categories only
    df = df[df['super_category'].isin(TARGET_SUPERCATS)]

    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

df_raw = load_csv("data.csv")

if df_raw.empty:
    st.warning(
        f"No data found for super categories: {TARGET_SUPERCATS}. "
        "Check the exact values in your `super_category` column."
    )
    st.stop()

segment = st.sidebar.selectbox("Segment:", options=SEGMENT_OPTIONS)

available_dates = sorted(df_raw['date_'].dropna().unique())
st.sidebar.caption(f"Dates found: {', '.join(available_dates)}")


# ── Segment filter ────────────────────────────────────────────────────────────
def filter_segment(df, seg):
    if seg == 'Overall':
        return df
    elif seg == 'Alpha':
        return df[df['alpha_flag'] == 'TRUE']
    elif seg == 'Branded Non-Alpha':
        return df[(df['alpha_flag'] == 'FALSE') & (df['branded_flag'] == 'Branded')]
    elif seg == 'Unbranded Non-Alpha':
        return df[(df['alpha_flag'] == 'FALSE') & (df['branded_flag'] == 'Unbranded')]
    return df

df = filter_segment(df_raw, segment)
st.caption(f"Segment: **{segment}**  ·  Super categories: {', '.join(TARGET_SUPERCATS)}")

if df.empty:
    st.warning(f"No data for segment **{segment}**.")
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_pivot(data, group_col, label_col, num_col, den_col,
               sort_by_gmv=False, sort_order=None):
    sub = data.dropna(subset=[group_col])
    sub = sub[sub[group_col].str.strip() != '']
    if sub.empty:
        return pd.DataFrame()

    agg = (sub.groupby([group_col, 'date_'], as_index=False)
              .agg(num=(num_col, 'sum'), den=(den_col, 'sum'), gmv=('gmv', 'sum')))

    agg['disc'] = (agg['num'] / agg['den'].replace(0, np.nan)).fillna(0) * 100

    pivot = agg.pivot(index=group_col, columns='date_', values='disc')
    pivot.columns.name = None

    # Sort date columns chronologically
    def sort_key(d):
        try:
            return pd.to_datetime(d, format='%d %b', errors='coerce')
        except Exception:
            return d

    pivot = pivot[sorted(pivot.columns, key=sort_key)]
    pivot = pivot.reset_index().rename(columns={group_col: label_col})

    if sort_order is not None:
        order_df = pd.DataFrame({label_col: [b for b in sort_order if b in pivot[label_col].values]})
        pivot = order_df.merge(pivot, on=label_col, how='left')
    elif sort_by_gmv:
        gmv_order = (sub.groupby(group_col)['gmv'].sum().reset_index()
                        .rename(columns={group_col: label_col})
                        .sort_values('gmv', ascending=False))
        pivot = gmv_order[[label_col]].merge(pivot, on=label_col, how='left')

    return pivot


def style_disc(df, label_col):
    df = df.set_index(label_col)
    date_cols = df.columns.tolist()
    def color(v):
        try:
            return GREEN if float(v) < 0 else RED
        except (TypeError, ValueError):
            return ""
    s  = df.style.format({c: '{:.2f}%' for c in date_cols})
    fn = s.map if hasattr(s, 'map') else s.applymap
    return fn(color, subset=date_cols)


def render_tab(title, num_col, den_col, sections, caption_suffix):
    st.subheader(title)
    if df.empty:
        st.info(f'No data for **{segment}**.')
        return

    for i, (group_col, label_col, sort_by_gmv, sort_order) in enumerate(sections):
        pivot = make_pivot(df, group_col, label_col, num_col, den_col, sort_by_gmv, sort_order)
        if not pivot.empty:
            st.markdown(f'**{label_col}**')
            st.dataframe(style_disc(pivot, label_col), use_container_width=True)
        else:
            st.info(f'No {label_col} data for **{segment}**.')
        if i < len(sections) - 1:
            st.divider()

    st.caption(f'Disc% = {num_col} / {den_col} × 100  ·  {segment}  ·  {caption_suffix}')


# ── Section definitions ───────────────────────────────────────────────────────
SC_V_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('vertical',       'Vertical',       True,  None),
]
OUTPUT_BUCKET_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('ja_asp_bucket',  'JA ASP Bucket',  False, BUCKET_LABELS),
]
INPUT_BUCKET_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('ja_asp_bucket',  'JA ASP Bucket',  False, BUCKET_LABELS),
]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    'Output ASP', 'Input ASP',
    'Output ASP (JA Bucket)', 'Input ASP (JA Bucket)',
])

with tab1:
    render_tab(
        title='Output ASP Disc% — Fixed by d-1 Units',
        num_col='num_op_u', den_col='den_op_u',
        sections=SC_V_SECTIONS,
        caption_suffix='verticals sorted by GMV',
    )

with tab2:
    render_tab(
        title='Input ASP Disc% — Fixed by JA Units',
        num_col='num_ip_u', den_col='den_ip_u',
        sections=SC_V_SECTIONS,
        caption_suffix='verticals sorted by GMV',
    )

with tab3:
    render_tab(
        title='Output ASP Disc% — JA ASP Bucket (Fixed by d-1 Units)',
        num_col='num_op_u', den_col='den_op_u',
        sections=OUTPUT_BUCKET_SECTIONS,
        caption_suffix='bucket order: 0-200 → Above 1000',
    )

with tab4:
    render_tab(
        title='Input ASP Disc% — JA ASP Bucket (Fixed by JA Units)',
        num_col='num_ip_u', den_col='den_ip_u',
        sections=INPUT_BUCKET_SECTIONS,
        caption_suffix='bucket order: 0-200 → Above 1000',
    )
