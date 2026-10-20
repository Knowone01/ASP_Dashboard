from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
FILES = ('data.csv', 'data_2025.csv')
TARGET_SUPERCATS = {'WomenWesternCore', 'WomenWesternGrowth'}
NUM_COLS = ('num_op_u', 'den_op_u', 'num_ip_u', 'den_ip_u', 'units', 'gmv', 'mrp')
TEXT_COLS = ('date_', 'alpha_flag', 'branded_flag', 'super_category',
             'vertical', 'ja_asp_bucket', 'd1_asp_bucket', 'brand')


def convert(filename):
    source = BASE / filename
    target = source.with_suffix('.parquet')
    if not source.exists():
        raise FileNotFoundError(f'Missing source CSV: {source}')

    print(f'Converting {source.name} ...', flush=True)
    writer = None
    total = 0
    try:
        for chunk in pd.read_csv(source, dtype=str, chunksize=100_000):
            chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
            for col in NUM_COLS:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(
                        chunk[col].str.replace(',', '', regex=False), errors='coerce'
                    ).fillna(0).astype('float32')
            for col in TEXT_COLS:
                if col in chunk.columns:
                    chunk[col] = chunk[col].str.strip()
            chunk['alpha_flag'] = chunk['alpha_flag'].str.upper()
            if 'brand' not in chunk.columns:
                chunk['brand'] = ''
            chunk = chunk[chunk['super_category'].isin(TARGET_SUPERCATS)]
            if chunk.empty:
                continue
            # Fix string/numeric types per chunk to avoid schema changes across batches.
            for col in chunk.columns:
                if col not in NUM_COLS:
                    chunk[col] = chunk[col].astype('string')
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression='zstd')
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total += len(chunk)
        if writer is None:
            raise ValueError(f'{source.name} has no rows in the target super categories')
    except Exception:
        if writer is not None:
            writer.close()
        target.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    print(f'Saved {target.name}: {total:,} rows; '
          f'{target.stat().st_size / 1024**2:.1f} MB', flush=True)


if __name__ == '__main__':
    for name in FILES:
        convert(name)
