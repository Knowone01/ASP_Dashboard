import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("styler.render.max_elements", 250_000)

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
TOP_N_LTL = 300  # show 300 brands by default; search includes every brand
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


EVENT_METRICS = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25',
                 'BBD GMV 26', 'BBD GMV 25']
EVENT_AGGS = {'nop': ('num_op_u', 'sum'), 'dop': ('den_op_u', 'sum'),
              'nip': ('num_ip_u', 'sum'), 'dip': ('den_ip_u', 'sum'),
              'gmv': ('gmv', 'sum')}


@st.cache_data(show_spinner='Indexing brands and aggregating event data...', max_entries=12)
def event_summary(event, seg, mtime26, mtime25):
    """Aggregate once per event/segment/year, not once per day or search click."""
    dates26 = event_dates(df_2026, EVENT_WINDOWS[event]['2026'])
    dates25 = event_dates(df_2025, EVENT_WINDOWS[event]['2025'])

    def grouped(df, dates):
        if not dates:
            return pd.DataFrame(columns=['date_', 'brand', *EVENT_AGGS])
        # Work only with the event range and necessary columns.
        sub = filter_segment(df, seg)
        sub = sub.loc[sub['date_'].isin(dates),
                      ['date_', 'brand', 'num_op_u', 'den_op_u',
                       'num_ip_u', 'den_ip_u', 'gmv']]
        sub = sub.dropna(subset=['brand'])
        if sub.empty:
            return pd.DataFrame(columns=['date_', 'brand', *EVENT_AGGS])
        return sub.groupby(['date_', 'brand'], as_index=False,
                           observed=True).agg(**EVENT_AGGS)

    return grouped(df_2026, dates26), grouped(df_2025, dates25)


def render_event(event):
    st.subheader(f'{event} Like-to-Like Comparison')
    dates26 = event_dates(df_2026, EVENT_WINDOWS[event]['2026'])
    dates25 = event_dates(df_2025, EVENT_WINDOWS[event]['2025'])
    grouped26, grouped25 = event_summary(
        event, segment, path_2026.stat().st_mtime_ns, path_2025.stat().st_mtime_ns)

    if grouped26.empty and grouped25.empty:
        st.info(f'No data in the {event} comparison windows.')
        return

    # The brand selector contains EVERY brand in both years, including brands
    # outside the default top-50 view.
    all_brands = sorted({str(b) for g in (grouped26, grouped25)
                         for b in g['brand'].dropna().unique()
                         if str(b).strip().lower() not in EXCLUDED_BRANDS})

    # Rank the default view by the original D1 2026 segment GMV share.
    first_day = dates26[0] if dates26 else None
    day_one = grouped26[grouped26['date_'] == first_day] if first_day else grouped26.iloc[:0]
    ranked = (day_one.loc[~day_one['brand'].astype(str).str.strip().str.lower().isin(EXCLUDED_BRANDS)]
              .sort_values('gmv', ascending=False)['brand'].astype(str).tolist())
    default_brands = ranked[:TOP_N_LTL]
    if len(default_brands) < TOP_N_LTL:
        default_brands.extend(b for b in all_brands
                              if b not in default_brands) 
        default_brands = default_brands[:TOP_N_LTL]

    picked = st.multiselect(
        f'Search any brand (empty = top {TOP_N_LTL} brands by D1 2026 GMV):',
        options=all_brands, placeholder='Type to search any brand...',
        key=f'{event}_brand_search')
    shown = picked if picked else default_brands

    def by_day(grouped, day, year):
        names = [f'I/P ASP {year}', f'O/P ASP {year}', f'BBD GMV {year}']
        if day is None:
            return pd.DataFrame(columns=names)
        daily = grouped.loc[grouped['date_'] == day]
        if daily.empty:
            return pd.DataFrame(columns=names)
        overall = daily[['nop', 'dop', 'nip', 'dip', 'gmv']].sum()
        # Filter to visible brands ONLY after calculating the full segment GMV
        # denominator, so searched brand shares remain mathematically correct.
        visible = daily[daily['brand'].isin(shown)].set_index('brand')
        visible = pd.concat([pd.DataFrame([overall], index=['Overall']),
                             visible[['nop', 'dop', 'nip', 'dip', 'gmv']]])
        return pd.DataFrame({
            names[0]: visible['nip'] / visible['dip'].replace(0, np.nan) * 100,
            names[1]: visible['nop'] / visible['dop'].replace(0, np.nan) * 100,
            names[2]: visible['gmv'] / overall['gmv'] * 100
                      if overall['gmv'] else np.nan,
        })

    # Only materialize the displayed brands (normally 50), instead of
    # thousands of brands x every day x six metrics.
    parts = []
    for i in range(max(len(dates26), len(dates25))):
        d26 = dates26[i] if i < len(dates26) else None
        d25 = dates25[i] if i < len(dates25) else None
        day_table = by_day(grouped26, d26, '26').join(
            by_day(grouped25, d25, '25'), how='outer').reindex(columns=EVENT_METRICS)
        day_table.columns = pd.MultiIndex.from_product([[f'{event} D{i+1}'], day_table.columns])
        parts.append(day_table)

    if not parts:
        st.info('No dates available.')
        return
    full = pd.concat(parts, axis=1)
    t = full.reindex(['Overall'] + list(shown))
    t.index.name = 'Brand'

    # Apply styling only to the small displayed subset.
    disc_cols = [c for c in t.columns if c[1].startswith(('I/P ASP', 'O/P ASP'))]
    gmv_cols = [c for c in t.columns if c[1].startswith('BBD GMV')]

    def color(v):
        try:
            f = float(v)
            return '' if np.isnan(f) else (GREEN if f < 0 else RED)
        except (ValueError, TypeError):
            return ''

    styled = (t.style.format('{:.0f}%', subset=pd.IndexSlice[:, disc_cols], na_rep='-')
                    .format('{:.1f}%', subset=pd.IndexSlice[:, gmv_cols], na_rep='-'))
    fn = styled.map if hasattr(styled, 'map') else styled.applymap
    st.dataframe(fn(color, subset=pd.IndexSlice[:, disc_cols]),
                 use_container_width=True, height=min(700, 80 + 35 * len(t)))
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
               f'showing {len(shown)} {"selected" if picked else "default"} brands '
               f'(search covers all {len(all_brands):,} brands) · '
               '- = no matched sales or unpaired day')


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
