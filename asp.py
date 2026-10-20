import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("styler.render.max_elements", 50_000_000)

st.set_page_config(page_title="ASP Disc% Dashboard", layout="wide")
st.title("ASP Discount % Dashboard")

GREEN = "background-color: #d1e7dd; color: #0f5132"
RED   = "background-color: #f8d7da; color: #842029"

BUCKET_LABELS     = ['0-200', '200-300', '300-500', '500-1000', 'Above 1000']
BUCKET_ORDER_FULL = ['0-200', '200-300', '0-300', '300-500', '500-1000', 'Above 500', 'Above 1000']
EXTRA_BUCKETS     = [
    ('0-300',     ['0-200', '200-300']),
    ('Above 500', ['500-1000', 'Above 1000']),
]
SEGMENT_OPTIONS  = ['Overall', 'Unbranded Non-Alpha', 'Branded Non-Alpha', 'Alpha']
TARGET_SUPERCATS = ['WomenWesternCore', 'WomenWesternGrowth']
NUM_COLS         = ['num_op_u', 'den_op_u', 'num_ip_u', 'den_ip_u', 'units', 'gmv', 'mrp']
TOP_N_BRANDS     = 100    # brand tabs 7-10
TOP_N_BBD        = 1000   # BBD tab: shown by default, rest via search

# Like-to-like windows are inclusive; None means use the last date in that year's file.
EVENT_WINDOWS = {
    'BBD': {'2026': ('20261008', '20261018'), '2025': ('20250922', '20251002')},
    'SPL': {'2026': ('20260924', '20261007'), '2025': ('20250908', '20250921')},
    'BDS': {'2026': ('20261019', None), '2025': ('20251004', None)},
}
TOP_N_LTL = 1000
EXCLUDED_BRANDS = {'', 'unknown'}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dashboard data...")
def load_parquet(filepath, modified_time):
    # modified_time makes Streamlit reload if the Parquet file changes.
    return pd.read_parquet(filepath, engine="pyarrow")


def fmt_date(d):
    try:
        return pd.to_datetime(str(d), format='%Y%m%d').strftime('%d %b')
    except Exception:
        return str(d)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
segment = st.sidebar.selectbox("Segment:", options=SEGMENT_OPTIONS)

BASE_DIR = Path(__file__).resolve().parent
path_2026 = BASE_DIR / "data.parquet"
path_2025 = BASE_DIR / "data_2025.parquet"

for path in (path_2026, path_2025):
    if not path.exists():
        st.error(f"Missing {path.name}. Run: python convert_to_parquet.py")
        st.stop()

df_2026 = load_parquet(str(path_2026), path_2026.stat().st_mtime_ns)
df_2025 = load_parquet(str(path_2025), path_2025.stat().st_mtime_ns)


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


# ── Pivot builder ─────────────────────────────────────────────────────────────
def make_pivot(data, group_col, label_col, num_col, den_col,
               sort_by_gmv=False, sort_order=None, extra_buckets=None):
    sub = data.dropna(subset=[group_col])
    sub = sub[sub[group_col].str.strip() != '']
    if sub.empty:
        return pd.DataFrame()

    agg = (sub.groupby([group_col, 'date_'], as_index=False)
              .agg(num=(num_col, 'sum'), den=(den_col, 'sum'), gmv=('gmv', 'sum')))
    agg['disc'] = (agg['num'] / agg['den'].replace(0, np.nan)).fillna(0) * 100

    pivot = agg.pivot(index=group_col, columns='date_', values='disc')
    pivot.columns.name = None
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [fmt_date(c) for c in pivot.columns]
    pivot = pivot.reset_index().rename(columns={group_col: label_col})

    # Derived bucket rows (computed from raw num/den, not from disc%)
    if extra_buckets:
        date_cols = [c for c in pivot.columns if c != label_col]
        derived_rows = []
        for extra_label, source_vals in extra_buckets:
            src = sub[sub[group_col].isin(source_vals)]
            if src.empty:
                continue
            da = src.groupby('date_', as_index=False).agg(
                num=(num_col, 'sum'), den=(den_col, 'sum')
            )
            da['disc'] = (da['num'] / da['den'].replace(0, np.nan)).fillna(0) * 100
            row = {label_col: extra_label}
            for _, r in da.iterrows():
                d = fmt_date(r['date_'])
                if d in date_cols:
                    row[d] = r['disc']
            derived_rows.append(row)
        if derived_rows:
            pivot = pd.concat([pivot, pd.DataFrame(derived_rows)], ignore_index=True)

    # Row ordering
    if sort_order is not None:
        order_df = pd.DataFrame({label_col: [b for b in sort_order if b in pivot[label_col].values]})
        pivot = order_df.merge(pivot, on=label_col, how='left')
    elif sort_by_gmv:
        gmv_order = (sub.groupby(group_col)['gmv'].sum().reset_index()
                        .rename(columns={group_col: label_col})
                        .sort_values('gmv', ascending=False))
        if group_col == 'brand':
            gmv_order = gmv_order.head(TOP_N_BRANDS)
        pivot = gmv_order[[label_col]].merge(pivot, on=label_col, how='left')

    # Overall row at top (computed from all rows, incl. brands beyond top N)
    date_cols   = [c for c in pivot.columns if c != label_col]
    overall_agg = sub.groupby('date_', as_index=False).agg(num=(num_col,'sum'), den=(den_col,'sum'))
    overall_agg['disc'] = (overall_agg['num'] / overall_agg['den'].replace(0, np.nan)).fillna(0) * 100
    overall_row = {label_col: 'Overall'}
    for _, r in overall_agg.iterrows():
        d = fmt_date(r['date_'])
        if d in date_cols:
            overall_row[d] = r['disc']
    pivot = pd.concat([pd.DataFrame([overall_row]), pivot], ignore_index=True)

    return pivot


# ── Styler ────────────────────────────────────────────────────────────────────
def style_disc(df, label_col):
    df = df.set_index(label_col)
    date_cols = df.columns.tolist()
    def color(v):
        try:
            f = float(v)
            if np.isnan(f):
                return ""
            return GREEN if f < 0 else RED
        except (TypeError, ValueError):
            return ""
    s  = df.style.format({c: '{:.2f}%' for c in date_cols}, na_rep='-')
    fn = s.map if hasattr(s, 'map') else s.applymap
    return fn(color, subset=date_cols)


# ── Section renderer ──────────────────────────────────────────────────────────
def render_sections(data, sections, num_col, den_col):
    """sections: list of (group_col, label_col, sort_by_gmv, sort_order[, extra_buckets])"""
    if data.empty:
        st.info(f'No data for **{segment}**.')
        return
    for i, section in enumerate(sections):
        group_col, label_col, sort_by_gmv, sort_order = section[:4]
        extra_buckets = section[4] if len(section) > 4 else None
        pivot = make_pivot(data, group_col, label_col, num_col, den_col,
                           sort_by_gmv, sort_order, extra_buckets)
        if not pivot.empty:
            st.markdown(f'**{label_col}**')
            st.dataframe(style_disc(pivot, label_col), use_container_width=True)
        else:
            st.info(f'No {label_col} data for **{segment}**.')
        if i < len(sections) - 1:
            st.divider()


# ── Section definitions ───────────────────────────────────────────────────────
SC_V_SECTIONS = [
    ('super_category', 'Super Category', False, None),
    ('vertical',       'Vertical',       True,  None),
]

BKT_SECTIONS = [
    ('ja_asp_bucket', 'JA ASP Bucket', False, BUCKET_ORDER_FULL, EXTRA_BUCKETS),
]

BRAND_SECTIONS = [
    ('brand', 'Brand', True, None),
]


# ── Bucket tab renderer ───────────────────────────────────────────────────────
def render_bucket_tab(num_col, den_col):
    SC = [('super_category', 'Super Category', False, None)]

    st.markdown('**Super Category — 2026**')
    render_sections(df26, SC, num_col, den_col)

    st.divider()

    st.markdown('**Super Category — 2025**')
    render_sections(df25, SC, num_col, den_col)

    st.divider()

    st.markdown('**JA ASP Bucket — 2026**')
    render_sections(df26, BKT_SECTIONS, num_col, den_col)

    st.divider()

    st.markdown('**JA ASP Bucket — 2025**')
    render_sections(df25, BKT_SECTIONS, num_col, den_col)

    st.caption(
        f'Disc% = {num_col} / {den_col} × 100  ·  {segment}  ·  '
        '0-300 = sum(0-200, 200-300)  ·  500+ = sum(500-1000, Above 1000)'
    )


# ── Event like-to-like comparisons ──────────────────────────────────────────
def event_dates(df, window):
    """Calendar dates, not just available rows, so a missing day never shifts pairs."""
    start, end = window
    if end is None:
        available = df['date_'].dropna().astype(str)
        available = available[available.str.fullmatch(r'\d{8}')]
        if available.empty:
            return []
        end = available.max()
    if end < start:
        return []
    return [d.strftime('%Y%m%d') for d in pd.date_range(
        pd.to_datetime(start, format='%Y%m%d'),
        pd.to_datetime(end, format='%Y%m%d'), freq='D')]


def event_day_table(d26, d25, date26, date25):
    cols = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25',
            'BBD GMV 26', 'BBD GMV 25']

    def agg(df, day, year):
        names = [f'I/P ASP {year}', f'O/P ASP {year}', f'BBD GMV {year}']
        if day is None:
            return pd.DataFrame(columns=names)
        sub = df[df['date_'] == day]
        if sub.empty:
            return pd.DataFrame(columns=names)
        g = sub.groupby('brand').agg(
            nop=('num_op_u', 'sum'), dop=('den_op_u', 'sum'),
            nip=('num_ip_u', 'sum'), dip=('den_ip_u', 'sum'), gmv=('gmv', 'sum'))
        overall = g.sum()
        g.loc['Overall'] = overall
        return pd.DataFrame({
            names[0]: g['nip'] / g['dip'].replace(0, np.nan) * 100,
            names[1]: g['nop'] / g['dop'].replace(0, np.nan) * 100,
            names[2]: g['gmv'] / overall['gmv'] * 100 if overall['gmv'] else np.nan,
        })

    return agg(d26, date26, '26').join(agg(d25, date25, '25'), how='outer').reindex(columns=cols)


@st.cache_data(show_spinner='Preparing event comparison...')
def event_all_days(event, seg, mtime26, mtime25):
    d26 = filter_segment(df_2026, seg)
    d25 = filter_segment(df_2025, seg)
    dates26 = event_dates(df_2026, EVENT_WINDOWS[event]['2026'])
    dates25 = event_dates(df_2025, EVENT_WINDOWS[event]['2025'])
    parts = []
    for i in range(max(len(dates26), len(dates25))):
        day26 = dates26[i] if i < len(dates26) else None
        day25 = dates25[i] if i < len(dates25) else None
        t = event_day_table(d26, d25, day26, day25)
        t.columns = pd.MultiIndex.from_product([[f'{event} D{i+1}'], t.columns])
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, axis=1)
    if full.empty:
        return full
    if 'Overall' not in full.index:
        full.loc['Overall'] = np.nan
    top = full.loc[['Overall']]
    rest = full.drop('Overall')
    rest = rest[~rest.index.astype(str).str.strip().str.lower().isin(EXCLUDED_BRANDS)]
    first_gmv = (f'{event} D1', 'BBD GMV 26')
    rest = rest.sort_values(first_gmv, ascending=False, na_position='last')
    return pd.concat([top, rest])


def render_event(event):
    st.subheader(f'{event} Like-to-Like Comparison')
    dates26 = event_dates(df_2026, EVENT_WINDOWS[event]['2026'])
    dates25 = event_dates(df_2025, EVENT_WINDOWS[event]['2025'])
    full = event_all_days(event, segment,
                          path_2026.stat().st_mtime_ns, path_2025.stat().st_mtime_ns)
    if full.empty:
        st.info(f'No data in the {event} comparison windows.')
        return
    all_brands = sorted(b for b in full.index if b != 'Overall')
    picked = st.multiselect(
        f'Search brand (empty = top {TOP_N_LTL} by D1 2026 GMV):',
        all_brands, placeholder='Type to search brand...', key=f'{event}_brand_search')
    t = full.loc[['Overall'] + picked].copy() if picked else full.head(TOP_N_LTL + 1).copy()
    t.index.name = 'Brand'
    disc_names = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25']
    gmv_names = ['BBD GMV 26', 'BBD GMV 25']
    disc_cols = [c for c in t.columns if c[1] in disc_names]
    gmv_cols = [c for c in t.columns if c[1] in gmv_names]

    def color(v):
        try:
            f = float(v)
            return '' if np.isnan(f) else (GREEN if f < 0 else RED)
        except (ValueError, TypeError):
            return ''

    styled = (t.style.format('{:.0f}%', subset=pd.IndexSlice[:, disc_cols], na_rep='-')
                    .format('{:.1f}%', subset=pd.IndexSlice[:, gmv_cols], na_rep='-'))
    fn = styled.map if hasattr(styled, 'map') else styled.applymap
    st.dataframe(fn(color, subset=pd.IndexSlice[:, disc_cols]), use_container_width=True)
    pairs = [
        f'D{i+1}: {fmt_date(dates25[i]) if i < len(dates25) else "-"} 2025'
        f' vs {fmt_date(dates26[i]) if i < len(dates26) else "-"} 2026'
        for i in range(max(len(dates25), len(dates26)))
    ]
    with st.expander('View exact day-to-day date mapping', expanded=False):
        st.write(' | '.join(pairs))
    if len(dates25) != len(dates26):
        st.warning(f'Different window lengths: {len(dates25)} days in 2025 and '
                   f'{len(dates26)} days in 2026. Unpaired dates show as blank (-).')
    st.caption(f'{segment} · BBD GMV = brand share of segment GMV that day · '
               f'showing {"selected brands" if picked else f"top {TOP_N_LTL} brands"} '
               '· sorted by D1 2026 GMV · - = no matched sales or unpaired day')


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    'Output ASP 2026 (Vertical Level)', 'Output ASP 2025 (Vertical Level)',
    'Input ASP 2026(Vertical Level)',  'Input ASP 2025 (Vertical Level)',
    'Output ASP (JA Bucket)', 'Input ASP (JA Bucket)',
    'Output ASP 2026 (Brand Level)', 'Output ASP 2025 (Brand Level)',
    'Input ASP 2026 (Brand Level)',  'Input ASP 2025 (Brand Level)',
    'BBD Like-to-Like', 'SPL Like-to-Like', 'BDS Like-to-Like'
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

with tab7:
    st.subheader('Output ASP Disc% — Brand — 2026')
    render_sections(df26, BRAND_SECTIONS, 'num_op_u', 'den_op_u')
    st.caption(f'Disc% = num_op_u / den_op_u × 100  ·  {segment}  ·  top {TOP_N_BRANDS} brands by GMV')

with tab8:
    st.subheader('Output ASP Disc% — Brand — 2025')
    render_sections(df25, BRAND_SECTIONS, 'num_op_u', 'den_op_u')
    st.caption(f'Disc% = num_op_u / den_op_u × 100  ·  {segment}  ·  top {TOP_N_BRANDS} brands by GMV')

with tab9:
    st.subheader('Input ASP Disc% — Brand — 2026')
    render_sections(df26, BRAND_SECTIONS, 'num_ip_u', 'den_ip_u')
    st.caption(f'Disc% = num_ip_u / den_ip_u × 100  ·  {segment}  ·  top {TOP_N_BRANDS} brands by GMV')

with tab10:
    st.subheader('Input ASP Disc% — Brand — 2025')
    render_sections(df25, BRAND_SECTIONS, 'num_ip_u', 'den_ip_u')
    st.caption(f'Disc% = num_ip_u / den_ip_u × 100  ·  {segment}  ·  top {TOP_N_BRANDS} brands by GMV')

with tab11:
    render_event('BBD')

with tab12:
    render_event('SPL')

with tab13:
    render_event('BDS')
