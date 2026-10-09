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
NUM_COLS         = ['num_op_u', 'den_op_u', 'num_ip_u', 'den_ip_u', 'units', 'gmv', 'mrp']


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

    # Keep as YYYYMMDD string — sorts correctly as plain string
    df['date_']          = df['date_'].str.strip()
    df['alpha_flag']     = df['alpha_flag'].str.strip().str.upper()
    df['branded_flag']   = df['branded_flag'].str.strip()
    df['super_category'] = df['super_category'].str.strip()
    df['vertical']       = df['vertical'].str.strip()
    df['ja_asp_bucket']  = df['ja_asp_bucket'].str.strip()
    df['d1_asp_bucket']  = df['d1_asp_bucket'].str.strip()

    return df[df['super_category'].isin(TARGET_SUPERCATS)]


def fmt_date(d):
    """YYYYMMDD string → '24 Sep' display string."""
    try:
        return pd.to_datetime(str(d), format='%Y%m%d').strftime('%d %b')
    except Exception:
        return str(d)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
segment = st.sidebar.selectbox("Segment:", options=SEGMENT_OPTIONS)


# ── Load both years ───────────────────────────────────────────────────────────
df_2026 = load_csv("data.csv")
df_2025 = load_csv("data_2025.csv")


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

df26 = filter_segment(df_2026, segment)
df25 = filter_segment(df_2025, segment)

st.caption(f"Segment: **{segment}**  ·  Super categories: {', '.join(TARGET_SUPERCATS)}")


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

    # Sort YYYYMMDD strings — correct order guaranteed
    pivot = pivot[sorted(pivot.columns)]

    # Rename columns to display format
    pivot.columns = [fmt_date(c) for c in pivot.columns]
    pivot = pivot.reset_index().rename(columns={group_col: label_col})

    if sort_order is not None:
        order_df = pd.DataFrame({label_col: [b for b in sort_order if b in pivot[label_col].values]})
        pivot = order_df.merge(pivot, on=label_col, how='left')
    elif sort_by_gmv:
        gmv_order = (sub.groupby(group_col)['gmv'].sum().reset_index()
                        .rename(columns={group_col: label_col})
                        .sort_values('gmv', ascending=False))
        pivot = gmv_order[[label_col]].merge(pivot, on=label_col, how='left')

    # Overall row at top
    date_cols    = [c for c in pivot.columns if c != label_col]
    overall_agg  = sub.groupby('date_', as_index=False).agg(num=(num_col, 'sum'), den=(den_col, 'sum'))
    overall_agg['disc'] = (overall_agg['num'] / overall_agg['den'].replace(0, np.nan)).fillna(0) * 100

    overall_row = {label_col: 'Overall'}
    for _, r in overall_agg.iterrows():
        d = fmt_date(r['date_'])
        if d in date_cols:
            overall_row[d] = r['disc']

    pivot = pd.concat([pd.DataFrame([overall_row]), pivot], ignore_index=True)
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


def render_sections(data, sections, num_col, den_col):
    """Render a list of (group_col, label_col, sort_by_gmv, sort_order) sections."""
    if data.empty:
        st.info(f'No data for **{segment}**.')
        return
    for i, (group_col, label_col, sort_by_gmv, sort_order) in enumerate(sections):
        pivot = make_pivot(data, group_col, label_col, num_col, den_col, sort_by_gmv, sort_order)
        if not pivot.empty:
            st.markdown(f'**{label_col}**')
            st.dataframe(style_disc(pivot, label_col), use_container_width=True)
        else:
            st.info(f'No {label_col} data for **{segment}**.')
        if i < len(sections) - 1:
            st.divider()


def render_bucket_tab(num_col, den_col):
    """Show 2026 super_category + bucket, then 2025 super_category + bucket below."""
    BUCKET_SECTIONS = [
        ('super_category', 'Super Category', False, None),
        ('ja_asp_bucket',  'JA ASP Bucket',  False, BUCKET_LABELS),
    ]

    st.markdown('### 2026')
    render_sections(df26, BUCKET_SECTIONS, num_col, den_col)

    st.divider()

    st.markdown('### 2025')
    render_sections(df25, BUCKET_SECTIONS, num_col, den_col)

    st.caption(f'Disc% = {num_col} / {den_col} × 100  ·  {segment}')


# ── Section definitions ───────────────────────────────────────────────────────
SC_V_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('vertical',       'Vertical',       True,  None),
]


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Output ASP 2026', 'Output ASP 2025',
    'Input ASP 2026',  'Input ASP 2025',
    'Output ASP (JA Bucket)', 'Input ASP (JA Bucket)',
])

with tab1:
    st.subheader('Output ASP Disc% — 2026')
    render_sections(df26, SC_V_SECTIONS, 'num_op_u', 'den_op_u')
    st.caption(f'Disc% = num_op_u / den_op_u × 100  ·  {segment}  ·  verticals sorted by GMV')

with tab2:
    st.subheader('Output ASP Disc% — 2025')
    render_sections(df25, SC_V_SECTIONS, 'num_op_u', 'den_op_u')
    st.caption(f'Disc% = num_op_u / den_op_u × 100  ·  {segment}  ·  verticals sorted by GMV')

with tab3:
    st.subheader('Input ASP Disc% — 2026')
    render_sections(df26, SC_V_SECTIONS, 'num_ip_u', 'den_ip_u')
    st.caption(f'Disc% = num_ip_u / den_ip_u × 100  ·  {segment}  ·  verticals sorted by GMV')

with tab4:
    st.subheader('Input ASP Disc% — 2025')
    render_sections(df25, SC_V_SECTIONS, 'num_ip_u', 'den_ip_u')
    st.caption(f'Disc% = num_ip_u / den_ip_u × 100  ·  {segment}  ·  verticals sorted by GMV')

with tab5:
    st.subheader('Output ASP Disc% — JA ASP Bucket (Fixed by d-1 Units)')
    render_bucket_tab('num_op_u', 'den_op_u')

with tab6:
    st.subheader('Input ASP Disc% — JA ASP Bucket (Fixed by JA Units)')
    render_bucket_tab('num_ip_u', 'den_ip_u')
