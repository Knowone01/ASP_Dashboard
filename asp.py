import streamlit as st
import pandas as pd
import numpy as np

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
TOP_N_BRANDS     = 100

# BBD like-to-like windows (inclusive, YYYYMMDD strings)
BBD_RANGE_26    = ('20261008', '20261231')
BBD_RANGE_25    = ('20250922', '20251002')
EXCLUDED_BRANDS = {'', 'unknown'}


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

    df['date_']          = df['date_'].str.strip()
    df['alpha_flag']     = df['alpha_flag'].str.strip().str.upper()
    df['branded_flag']   = df['branded_flag'].str.strip()
    df['super_category'] = df['super_category'].str.strip()
    df['vertical']       = df['vertical'].str.strip()
    df['ja_asp_bucket']  = df['ja_asp_bucket'].str.strip()
    df['d1_asp_bucket']  = df['d1_asp_bucket'].str.strip()
    df['brand']          = df['brand'].str.strip() if 'brand' in df.columns else ''

    return df[df['super_category'].isin(TARGET_SUPERCATS)]


def fmt_date(d):
    try:
        return pd.to_datetime(str(d), format='%Y%m%d').strftime('%d %b')
    except Exception:
        return str(d)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
segment = st.sidebar.selectbox("Segment:", options=SEGMENT_OPTIONS)

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


# ── BBD like-to-like helpers ──────────────────────────────────────────────────
def bbd_dates(df, rng):
    return sorted(d for d in df['date_'].unique() if rng[0] <= str(d) <= rng[1])


def bbd_day_table(day_idx, dates_26, dates_25):
    cols = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25', 'BBD GMV 26', 'BBD GMV 25']

    def agg(df, dates, sfx):
        names = [f'I/P ASP {sfx}', f'O/P ASP {sfx}', f'BBD GMV {sfx}']
        if day_idx >= len(dates):
            return pd.DataFrame(columns=names)
        g = df[df['date_'] == dates[day_idx]].groupby('brand').agg(
            nop=('num_op_u', 'sum'), dop=('den_op_u', 'sum'),
            nip=('num_ip_u', 'sum'), dip=('den_ip_u', 'sum'), gmv=('gmv', 'sum'))
        g.loc['Overall'] = g.sum()          # share denominator = all brands
        return pd.DataFrame({
            names[0]: g['nip'] / g['dip'].replace(0, np.nan) * 100,
            names[1]: g['nop'] / g['dop'].replace(0, np.nan) * 100,
            names[2]: g['gmv'] / g.loc['Overall', 'gmv'] * 100,
        })

    t = agg(df26, dates_26, '26').join(agg(df25, dates_25, '25'), how='outer')[cols]
    return t


def bbd_all_days(dates_26, dates_25, max_days):
    parts = []
    for i in range(max_days):
        t = bbd_day_table(i, dates_26, dates_25)
        t.columns = pd.MultiIndex.from_product([[f'BBD D{i+1}'], t.columns])
        parts.append(t)
    full = pd.concat(parts, axis=1)
    first_gmv = full.columns[4]                      # D1 'BBD GMV 26'
    top  = full.loc[['Overall']]
    rest = full.drop('Overall').sort_values(first_gmv, ascending=False, na_position='last')
    return pd.concat([top, rest])


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    'Output ASP 2026', 'Output ASP 2025',
    'Input ASP 2026',  'Input ASP 2025',
    'Output ASP (JA Bucket)', 'Input ASP (JA Bucket)',
    'Output ASP Brand 2026', 'Output ASP Brand 2025',
    'Input ASP Brand 2026',  'Input ASP Brand 2025',
    'BBD Like-to-Like'
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
    st.subheader("BBD Like-to-Like Comparison")

    dates_26 = bbd_dates(df26, BBD_RANGE_26)
    dates_25 = bbd_dates(df25, BBD_RANGE_25)
    max_days = max(len(dates_26), len(dates_25))

    if max_days == 0:
        st.info("No data in the BBD date windows.")
    else:
        all_brands = sorted(
            b for b in (set(df26['brand'].dropna().unique()) | set(df25['brand'].dropna().unique()))
            if b.strip().lower() not in EXCLUDED_BRANDS
        )
        picked = st.multiselect("Search brand (empty = all brands):", all_brands,
                                placeholder="Type to search brand...")

        t = bbd_all_days(dates_26, dates_25, max_days)
        keep = ['Overall'] + (picked if picked else
                              [b for b in t.index if b != 'Overall'
                               and str(b).strip().lower() not in EXCLUDED_BRANDS])
        t = t.loc[[b for b in keep if b in t.index]]
        t.index.name = 'Brand'

        disc_names = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25']
        gmv_names  = ['BBD GMV 26', 'BBD GMV 25']
        disc_cols  = [c for c in t.columns if c[1] in disc_names]
        gmv_cols   = [c for c in t.columns if c[1] in gmv_names]

        def color(v):
            try:
                f = float(v)
                return "" if np.isnan(f) else (GREEN if f < 0 else RED)
            except (TypeError, ValueError):
                return ""

        s = (t.style
               .format('{:.0f}%', subset=pd.IndexSlice[:, disc_cols], na_rep='-')
               .format('{:.1f}%', subset=pd.IndexSlice[:, gmv_cols],  na_rep='-'))
        fn = s.map if hasattr(s, 'map') else s.applymap
        st.dataframe(fn(color, subset=pd.IndexSlice[:, disc_cols]), use_container_width=True)

        st.caption('Day windows: ' + '  ·  '.join(
            f"D{i+1} = {fmt_date(dates_26[i]) if i < len(dates_26) else '-'}"
            f" | {fmt_date(dates_25[i]) if i < len(dates_25) else '-'}"
            for i in range(max_days)))
        st.caption(f'{segment}  ·  BBD GMV = brand share of segment GMV that day  ·  '
                   'sorted by D1 BBD GMV 26  ·  "-" = no matched sales / day outside window')
