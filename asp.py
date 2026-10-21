"""Memory-conscious ASP dashboard. Requires streamlit, pandas, numpy, pyarrow."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='ASP Disc% Dashboard', layout='wide')
st.title('ASP Discount % Dashboard')

GREEN = 'background-color: #d1e7dd; color: #0f5132'
RED = 'background-color: #f8d7da; color: #842029'
TARGET = ['WomenWesternCore', 'WomenWesternGrowth']
SEGMENTS = ['Overall', 'Unbranded Non-Alpha', 'Branded Non-Alpha', 'Alpha']
METRICS = ['num_op_u', 'den_op_u', 'num_ip_u', 'den_ip_u', 'gmv']
BUCKET_ORDER = ['0-200', '200-300', '0-300', '300-500', '500-1000', 'Above 500', 'Above 1000']
EXTRA_BUCKETS = [('0-300', ['0-200', '200-300']), ('Above 500', ['500-1000', 'Above 1000'])]
EVENT_WINDOWS = {
    'BBD': {'2026': ('20261008', '20261018'), '2025': ('20250922', '20251002')},
    'SPL': {'2026': ('20260924', '20261007'), '2025': ('20250908', '20250921')},
    'BDS': {'2026': ('20261019', None), '2025': ('20251004', None)},
}
TOP_N_LTL = 300
TOP_N_BRANDS = 100
EXCLUDED_BRANDS = {'', 'unknown'}
BASE = Path(__file__).resolve().parent
FILES = {'2025': BASE / 'data_2025.parquet', '2026': BASE / 'data.parquet'}
for path in FILES.values():
    if not path.exists():
        st.error(f'Missing {path.name}. Run python convert_to_parquet.py first.')
        st.stop()


def fmt_date(d):
    try:
        return pd.to_datetime(str(d), format='%Y%m%d').strftime('%d %b')
    except (ValueError, TypeError):
        return str(d)


def segment_filters(segment):
    result = [('super_category', 'in', TARGET)]
    if segment == 'Alpha':
        result.append(('alpha_flag', '==', 'TRUE'))
    elif segment == 'Branded Non-Alpha':
        result += [('alpha_flag', '==', 'FALSE'), ('branded_flag', '==', 'Branded')]
    elif segment == 'Unbranded Non-Alpha':
        result += [('alpha_flag', '==', 'FALSE'), ('branded_flag', '==', 'Unbranded')]
    return result


@st.cache_data(show_spinner='Preparing compact summaries...', max_entries=5, ttl=1200)
def grouped(year, segment, group_col, date_from, date_to, file_version):
    """Never cache raw CSV/Parquet rows: only compressed grouped summaries."""
    filters = segment_filters(segment)
    if date_from:
        filters.append(('date_', '>=', date_from))
    if date_to:
        filters.append(('date_', '<=', date_to))
    cols = list(dict.fromkeys(['date_', group_col] + METRICS))
    frame = pd.read_parquet(FILES[year], engine='pyarrow', columns=cols, filters=filters)
    if frame.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=['date_'] + METRICS)
    frame['date_'] = frame['date_'].astype(str)
    frame[group_col] = frame[group_col].fillna('').astype(str).str.strip()
    # Original overall totals exclude null/blank groups, but include unknown brands.
    frame = frame[frame[group_col] != '']
    for col in METRICS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(0)
    totals = frame.groupby('date_', sort=False, observed=True)[METRICS].sum().reset_index()
    by_group = frame.groupby([group_col, 'date_'], sort=False, observed=True)[METRICS].sum().reset_index()
    del frame
    gc.collect()
    return by_group, totals


def get_grouped(year, segment, group_col, start=None, end=None):
    return grouped(year, segment, group_col, start, end, FILES[year].stat().st_mtime_ns)


def discount(df, num_col, den_col):
    return (df[num_col] / df[den_col].replace(0, np.nan)).fillna(0) * 100


def pivot_report(year, segment, group_col, label, num_col, den_col, order=None, top_brands=False, extras=None):
    grouped_df, totals = get_grouped(year, segment, group_col)
    if grouped_df.empty:
        return pd.DataFrame()
    agg = grouped_df[[group_col, 'date_', num_col, den_col]].copy()
    agg['disc'] = discount(agg, num_col, den_col)
    p = agg.pivot(index=group_col, columns='date_', values='disc')
    p = p.reindex(sorted(p.columns), axis=1)
    p.columns = [fmt_date(c) for c in p.columns]
    p.columns.name = None
    p = p.reset_index().rename(columns={group_col: label})
    date_cols = [c for c in p.columns if c != label]
    if extras:
        extra_rows = []
        for name, source in extras:
            subset = grouped_df[grouped_df[group_col].isin(source)]
            if subset.empty:
                continue
            by_day = subset.groupby('date_')[[num_col, den_col]].sum().reset_index()
            by_day['disc'] = discount(by_day, num_col, den_col)
            row = {label: name}
            row.update({fmt_date(r['date_']): r['disc'] for _, r in by_day.iterrows()})
            extra_rows.append(row)
        if extra_rows:
            p = pd.concat([p, pd.DataFrame(extra_rows)], ignore_index=True)
    if order:
        p = pd.DataFrame({label: [b for b in order if b in set(p[label])]}).merge(p, on=label, how='left')
    elif top_brands or group_col == 'vertical':
        gmv_order = grouped_df.groupby(group_col)['gmv'].sum().sort_values(ascending=False).index.tolist()
        if top_brands:
            gmv_order = gmv_order[:TOP_N_BRANDS]
        p = pd.DataFrame({label: gmv_order}).merge(p, on=label, how='left')
    totals = totals.copy()
    totals['disc'] = discount(totals, num_col, den_col)
    overall_row = {label: 'Overall'}
    overall_row.update({fmt_date(r['date_']): r['disc'] for _, r in totals.iterrows() if fmt_date(r['date_']) in date_cols})
    return pd.concat([pd.DataFrame([overall_row]), p], ignore_index=True)


def show_table(df, label):
    if df.empty:
        st.info(f'No {label} data for selected segment.')
        return
    p = df.set_index(label)
    # Native dataframe formatting avoids Pandas Styler generating huge HTML/CSS tables.
    st.dataframe(p, width='stretch', height=min(680, 38 * (len(p) + 1) + 5),
                 column_config={c: st.column_config.NumberColumn(c, format='%.2f%%') for c in p.columns})


def sections(year, segment, num, den, section_list):
    for idx, (group, label, order, top, extras) in enumerate(section_list):
        st.markdown(f'**{label}**')
        show_table(pivot_report(year, segment, group, label, num, den, order, top, extras), label)
        if idx < len(section_list) - 1:
            st.divider()


def calendar_dates(start, end, last_available):
    end = end or last_available
    if not end or end < start:
        return []
    return [d.strftime('%Y%m%d') for d in pd.date_range(start=start, end=end, freq='D')]


def event_summary(year, segment, event):
    start, end = EVENT_WINDOWS[event][year]
    by_brand, overall = get_grouped(year, segment, 'brand', start, end)
    valid = by_brand['date_'] if not by_brand.empty else overall['date_']
    last_date = str(valid.max()) if len(valid) else None
    # For open-ended BDS, use the last date available for this segment.
    dates = calendar_dates(start, end, last_date)
    if not by_brand.empty:
        by_brand = by_brand[~by_brand['brand'].astype(str).str.lower().isin(EXCLUDED_BRANDS)].copy()
    return by_brand, overall, dates


def event_table(event, seg):
    data26, all26, dates26 = event_summary('2026', seg, event)
    data25, all25, dates25 = event_summary('2025', seg, event)
    max_days = max(len(dates26), len(dates25))
    if max_days == 0:
        return pd.DataFrame(), dates26, dates25

    def calc(df, totals, year, dates):
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['ip'] = discount(df, 'num_ip_u', 'den_ip_u').replace(0, 0)
        df['op'] = discount(df, 'num_op_u', 'den_op_u').replace(0, 0)
        gmvs = totals.set_index('date_')['gmv']
        df['share'] = df['gmv'] / df['date_'].map(gmvs).replace(0, np.nan) * 100
        df['day'] = df['date_'].map({date: f'{event} D{i+1}' for i, date in enumerate(dates)})
        values = df.pivot(index='brand', columns='day', values=['ip', 'op', 'share'])
        values.columns = pd.MultiIndex.from_tuples(
            [(day, {'ip': 'I/P ASP', 'op': 'O/P ASP', 'share': 'BBD GMV'}[kind] + f' {year[-2:]}')
             for kind, day in values.columns])
        # Overall is calculated from all source rows, not only displayed brands.
        total = totals.copy()
        total['ip'] = discount(total, 'num_ip_u', 'den_ip_u')
        total['op'] = discount(total, 'num_op_u', 'den_op_u')
        total['share'] = np.where(total['gmv'].ne(0), 100., np.nan)
        for i, date in enumerate(dates):
            row = total[total['date_'] == date]
            if not row.empty:
                day = f'{event} D{i+1}'
                for field, name in [('ip', 'I/P ASP'), ('op', 'O/P ASP'), ('share', 'BBD GMV')]:
                    values.loc['Overall', (day, f'{name} {year[-2:]}')] = row.iloc[0][field]
        return values

    p26 = calc(data26, all26, '2026', dates26)
    p25 = calc(data25, all25, '2025', dates25)
    full = p26.join(p25, how='outer') if not p26.empty and not p25.empty else (p26 if p25.empty else p25)
    if full.empty:
        return full, dates26, dates25
    if 'Overall' not in full.index:
        full.loc['Overall'] = np.nan
    second = ['I/P ASP 26', 'I/P ASP 25', 'O/P ASP 26', 'O/P ASP 25', 'BBD GMV 26', 'BBD GMV 25']
    all_columns = pd.MultiIndex.from_product([[f'{event} D{i+1}' for i in range(max_days)], second])
    full = full.reindex(columns=all_columns)
    rank_column = (f'{event} D1', 'BBD GMV 26')
    rest = full.drop(index='Overall').sort_values(rank_column, ascending=False, na_position='last')
    return pd.concat([full.loc[['Overall']], rest]), dates26, dates25


@st.cache_data(show_spinner='Building event summary...', max_entries=3, ttl=1200)
def cached_event(event, seg, v26, v25):
    return event_table(event, seg)


def show_event(event, seg):
    st.subheader(f'{event} Like-to-Like Comparison')
    full, dates26, dates25 = cached_event(event, seg, FILES['2026'].stat().st_mtime_ns,
                                           FILES['2025'].stat().st_mtime_ns)
    if full.empty:
        st.info('No data in this event window.')
        return
    brands = sorted((x for x in full.index if x != 'Overall'), key=str.casefold)
    selected = st.multiselect('Search any brand (empty = top 300 by Day 1 2026 GMV):',
                              options=brands, key=f'{event}_brand_search',
                              placeholder='Type to find any brand...')
    displayed = full.loc[['Overall'] + selected].copy() if selected else full.head(TOP_N_LTL + 1).copy()
    displayed.index.name = 'Brand'
    # MultiIndex columns stay intact; avoid expensive Styler HTML/CSS generation.
    st.dataframe(displayed.round(1), width='stretch', height=620)
    with st.expander('View exact day-to-day date mapping'):
        for i in range(max(len(dates25), len(dates26))):
            d25 = fmt_date(dates25[i]) if i < len(dates25) else '-'
            d26 = fmt_date(dates26[i]) if i < len(dates26) else '-'
            st.write(f'D{i+1}: {d25} 2025 vs {d26} 2026')
    if len(dates25) != len(dates26):
        st.warning(f'Unequal event lengths: {len(dates25)} days in 2025, {len(dates26)} days in 2026.')
    st.caption(f'{seg} · GMV = brand share of segment GMV on each day · '
               f'{"selected brands" if selected else "top 300 brands"} · ranked by Day 1 2026 GMV')


VIEWS = [
    'Output ASP 2026 (Vertical Level)', 'Output ASP 2025 (Vertical Level)',
    'Input ASP 2026 (Vertical Level)', 'Input ASP 2025 (Vertical Level)',
    'Output ASP (JA Bucket)', 'Input ASP (JA Bucket)',
    'Output ASP 2026 (Brand Level)', 'Output ASP 2025 (Brand Level)',
    'Input ASP 2026 (Brand Level)', 'Input ASP 2025 (Brand Level)',
    'BBD Like-to-Like', 'SPL Like-to-Like', 'BDS Like-to-Like',
]
st.sidebar.header('Filters')
seg = st.sidebar.selectbox('Segment:', SEGMENTS)
# A selector, rather than st.tabs, actually prevents non-selected views from executing.
view = st.selectbox('Dashboard view', VIEWS, key='dashboard_view')
st.caption(f'Segment: **{seg}** · Super categories: {", ".join(TARGET)}')

if 'Like-to-Like' in view:
    show_event(view.split()[0], seg)
elif 'JA Bucket' in view:
    num, den = ('num_op_u', 'den_op_u') if view.startswith('Output') else ('num_ip_u', 'den_ip_u')
    for year in ['2026', '2025']:
        st.subheader(f'Super Category — {year}')
        sections(year, seg, num, den, [('super_category', 'Super Category', None, False, None)])
        st.divider()
    for year in ['2026', '2025']:
        st.subheader(f'JA ASP Bucket — {year}')
        sections(year, seg, num, den, [('ja_asp_bucket', 'JA ASP Bucket', BUCKET_ORDER, False, EXTRA_BUCKETS)])
        st.divider()
    st.caption(f'Disc% = {num} / {den} × 100 · Derived buckets use summed numerators/denominators')
else:
    year = '2026' if '2026' in view else '2025'
    num, den = ('num_op_u', 'den_op_u') if view.startswith('Output') else ('num_ip_u', 'den_ip_u')
    st.subheader(f'{"Output" if view.startswith("Output") else "Input"} ASP Disc% — {year}')
    if 'Brand Level' in view:
        sections(year, seg, num, den, [('brand', 'Brand', None, True, None)])
        st.caption(f'Top {TOP_N_BRANDS} brands by GMV')
    else:
        sections(year, seg, num, den, [('super_category', 'Super Category', None, False, None),
                                        ('vertical', 'Vertical', None, False, None)])
    st.caption(f'Disc% = {num} / {den} × 100 · {seg}')
