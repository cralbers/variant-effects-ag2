#!/usr/bin/env python
# coding: utf-8

# # AlphaGenome Atlas: Variant & Raw Track Scoring Across BED Regions
# 
# This notebook:
# 1. Reads 0-based half-open genomic intervals (`chrom`, `start`, `end`) from a standard 3-column BED file.
# 2. Queries the AlphaGenome Atlas API (`alphagenome.atlas.atlas`) for all SNVs in each BED region:
#    - Composite `AVI_SCORE` (`avi_raw`, `avi_phred`, `avi_quantile`, `top_percentile`)
#    - 18-modality `AVI_SCORE_FEATURE_IMPORTANCE` attribution weights (`fi_<MODALITY>`)
#    - Raw track-level scores across all experimental assay scorers, filtering gene-level scorers (e.g., `RNA_SEQ`, `SPLICE_JUNCTIONS`) specifically to **`LMNA`**.
# 3. Saves the combined variant scores and track metadata catalog to Parquet files.

# # Imports & API Key Setup

# In[1]:


# @title Imports & API Key Setup
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import glob
import itertools
import os
import pickle
import tempfile
from typing import Any

from alphagenome import colab_utils
from alphagenome.atlas import atlas
from alphagenome.data import gene_annotation
from alphagenome.data import gene_expression_data, track_data
from alphagenome.data import genome
from alphagenome.data import transcript
from alphagenome.models.v2 import dna_client
from alphagenome.models.v2 import model_condition
from alphagenome.visualization import plot_components
import anndata
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns

pd.set_option('display.max_rows', None)

load_dotenv(dotenv_path='.env')
load_dotenv(dotenv_path=os.path.expanduser('~/.env'))

API_KEYS = [
    v
    for _, v in sorted(
        (
            (k, os.environ[k].strip())
            for k in os.environ
            if k.startswith('AG_API_KEY_')
            and k.removeprefix('AG_API_KEY_').isdigit()
        ),
        key=lambda kv: int(kv[0].removeprefix('AG_API_KEY_')),
    )
    if v
]
if not API_KEYS and os.environ.get('ALPHAGENOME_API_KEY'):
    API_KEYS = [os.environ['ALPHAGENOME_API_KEY'].strip()]

print(f'Loaded {len(API_KEYS)} AlphaGenome API key(s).')


# #  Configuration & Atlas Client Initialization

# In[2]:


# @title Configuration & Atlas Client Initialization
BED_FILE_PATH = '/users/PAS2905/coraalbers/ag/ag_data/noncoding.bed'  # Path to input 3-column BED file (chrom, start, end)
# Checkpoints follow the script directory, not the directory sbatch was launched from.
# Set ATLAS_OUTPUT_DIR to project space when the files outgrow home.
_script_file = globals().get('__file__')
SCRIPT_DIR = (
    os.path.dirname(os.path.abspath(_script_file)) if _script_file else os.getcwd()
)
OUTPUT_DIR = os.path.abspath(os.environ.get('ATLAS_OUTPUT_DIR', SCRIPT_DIR))
OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, 'atlas_bed_variant_scores.parquet')
TRACK_METADATA_PARQUET_PATH = os.path.join(OUTPUT_DIR, 'atlas_track_metadata.parquet')

# Target gene filter for gene-level scorers (e.g., RNA_SEQ, SPLICE_JUNCTIONS)
TARGET_GENE_SYMBOL = 'LMNA'
TARGET_GENE_ENSEMBL_PREFIX = 'ENSG00000160789'  # GENCODE v46 Ensembl ID prefix for LMNA

# Maximum interval size (bp) per query_interval call (regions larger than this are chunked)
MAX_INTERVAL_WINDOW_BP = 1000
# Number of parallel worker threads when querying per-variant track scores across API_KEYS
MAX_WORKERS = max(1, min(16, len(API_KEYS) * 2))

if not API_KEYS:
    raise ValueError(
        'No API keys found. Set AG_API_KEY_1..N or ALPHAGENOME_API_KEY in .env'
    )

# Create an AtlasClient per API key for round-robin load balancing
atlas_clients = [atlas.create(key) for key in API_KEYS]
primary_client = atlas_clients[0]

# Discover all available experimental track scorers from Atlas metadata
AVI_SCORERS = ['AVI_SCORE', 'AVI_SCORE_FEATURE_IMPORTANCE']
scorer_meta_map = primary_client.scorer_metadata()
EXPERIMENTAL_SCORERS = [
    name
    for name in scorer_meta_map.keys()
    if not name.endswith('_ACTIVE') and name not in AVI_SCORERS
]

print(f'AVI scorers: {AVI_SCORERS}')
print(f'Experimental track scorers ({len(EXPERIMENTAL_SCORERS)}): {EXPERIMENTAL_SCORERS}')


# # Export Track Metadata Catalog to Parquet

# In[3]:


# @title Export Track Metadata Catalog to Parquet
def _atomic_save_parquet(df: pd.DataFrame, output_parquet_path: str) -> None:
    """Atomically writes a DataFrame to Parquet so an interrupted checkpoint cannot corrupt the file.

    The temp file is created in the destination directory, written only by pyarrow, fsynced,
    then renamed into place. pandas.DataFrame.to_parquet is not used: it opens the path
    itself and pyarrow opens it again, and closing the empty first handle last can drop or
    truncate the file on cluster filesystems.
    """
    output_parquet_path = os.path.abspath(output_parquet_path)
    out_dir = os.path.dirname(output_parquet_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(output_parquet_path)}.',
        suffix='.parquet',
        dir=out_dir,
    )
    os.close(fd)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp_path, compression='snappy')
        with open(tmp_path, 'rb') as synced:
            os.fsync(synced.fileno())
        os.replace(tmp_path, output_parquet_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_track_metadata_catalog(scorer_map: dict[str, Any]) -> pd.DataFrame:
    """Compiles track metadata across all experimental scorers into a DataFrame."""
    rows = []
    for scorer_name in EXPERIMENTAL_SCORERS:
        meta = scorer_map[scorer_name]
        df = meta.track_metadata
        if df is None or df.empty:
            continue
        for track_idx, row in df.iterrows():
            t_idx = int(track_idx)
            t_name = str(row.get('name', f'track_{t_idx}'))
            rows.append({
                'scorer': scorer_name,
                'track_index': t_idx,
                'column_prefix': f'{scorer_name}__t{t_idx:04d}',
                'track_name': t_name,
                'biosample_name': str(row.get('biosample_name', '')),
                'biosample_type': str(row.get('biosample_type', '')),
                'ontology_curie': str(row.get('ontology_curie', '')),
                'strand': str(row.get('strand', '')),
                'transcription_factor': str(row.get('transcription_factor', '')),
                'histone_mark': str(row.get('histone_mark', '')),
                'assay': str(row.get('assay', '')),
                'is_signed': bool(meta.is_signed),
            })
    return pd.DataFrame(rows)


track_metadata_df = build_track_metadata_catalog(scorer_meta_map)
_atomic_save_parquet(track_metadata_df, TRACK_METADATA_PARQUET_PATH)
print(
    f'Saved {len(track_metadata_df)} track metadata records to '
    f'{TRACK_METADATA_PARQUET_PATH}'
)
track_metadata_df.head()


# # Load 3-Column BED File (0-based, half-open [start, end))

# In[4]:


# @title Load 3-Column BED File (0-based, half-open [start, end))
def normalize_chrom(chrom: str) -> str:
    """Ensures chromosome string has a 'chr' prefix (e.g., '1' -> 'chr1')."""
    c = str(chrom).strip()
    return c if c.lower().startswith('chr') else f'chr{c}'


def load_bed_regions(bed_path: str) -> pd.DataFrame:
    """Reads a standard 3-column BED file into a DataFrame.

    BED coordinates are 0-based, half-open [start, end), which matches
    `alphagenome.data.genome.Interval` directly.
    """
    if not os.path.exists(bed_path):
        raise FileNotFoundError(f'BED file not found: {bed_path}')

    regions = []
    with open(bed_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(('#', 'track', 'browser')):
                continue
            parts = stripped.split('\t') if '\t' in stripped else stripped.split()
            if len(parts) < 3:
                continue
            # Skip header line if start/end are not integers
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue

            if end <= start:
                raise ValueError(
                    f'Invalid BED interval on line {line_num}: '
                    f'end ({end}) must be > start ({start}).'
                )

            chrom = normalize_chrom(parts[0])
            regions.append({
                'region_id': len(regions),
                'chromosome': chrom,
                'bed_start': start,
                'bed_end': end,
                'region_str': f'{chrom}:{start}-{end}',
                'width_bp': end - start,
            })

    bed_df = pd.DataFrame(regions)
    print(
        f'Loaded {len(bed_df)} region(s) from {bed_path} '
        f'(total {bed_df["width_bp"].sum():,} bp).'
    )
    return bed_df


GTF_FEATHER_URL = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/'
    'hg38/gencode.v46.annotation.gtf.gz.feather'
)


def load_gencode_gtf(custom_path: str | None = None) -> pd.DataFrame:
    """Loads GENCODE v46 GTF annotations from local cache or GCS."""
    cache_path = os.path.expanduser(
        '~/.cache/alphagenome/gencode.v46.annotation.gtf.gz.feather'
    )
    for candidate in (custom_path, os.environ.get('ALPHAGENOME_GTF_PATH'), cache_path):
        if candidate and os.path.exists(candidate):
            return pd.read_feather(candidate)
    df = pd.read_feather(GTF_FEATHER_URL)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_feather(cache_path)
    except Exception:
        pass
    return df


def get_gene_centered_window(
    gene_symbol: str = TARGET_GENE_SYMBOL,
    window_size_bp: int = dna_client.SEQUENCE_LENGTH_1MB,
    gtf_df: pd.DataFrame | None = None,
) -> genome.Interval:
    """Returns a genomic interval of `window_size_bp` (default 1 Mb) centered on `gene_symbol`."""
    if gtf_df is None:
        gtf_df = load_gencode_gtf()
    gene_interval = gene_annotation.get_gene_interval(
        gtf_df, gene_symbol=gene_symbol
    )
    return gene_interval.resize(window_size_bp)


def filter_bed_regions_to_gene_window(
    bed_regions_df: pd.DataFrame,
    gene_symbol: str = TARGET_GENE_SYMBOL,
    window_size_bp: int = dna_client.SEQUENCE_LENGTH_1MB,
    gtf_df: pd.DataFrame | None = None,
    require_fully_within: bool = True,
) -> pd.DataFrame:
    """Filters BED regions to those within a 1 Mb window centered on `gene_symbol` (e.g., LMNA).

    Args:
        bed_regions_df: DataFrame from `load_bed_regions`.
        gene_symbol: Target gene symbol to center the window on (default: 'LMNA').
        window_size_bp: Window width in bp (default: `dna_client.SEQUENCE_LENGTH_1MB` = 1,048,576 bp).
        gtf_df: Optional preloaded GENCODE v46 DataFrame.
        require_fully_within: If True, keeps BED regions completely inside the window
            (`bed_start >= window.start` and `bed_end <= window.end`). If False, keeps any
            overlapping region and clips boundary coordinates to the window.
    """
    window = get_gene_centered_window(
        gene_symbol=gene_symbol,
        window_size_bp=window_size_bp,
        gtf_df=gtf_df,
    )
    win_chrom = normalize_chrom(window.chromosome)
    win_start = int(window.start)
    win_end = int(window.end)

    print(
        f'{gene_symbol} centered window ({window.width:,} bp): '
        f'{win_chrom}:{win_start}-{win_end}'
    )

    if bed_regions_df.empty:
        return bed_regions_df.copy()

    chrom_match = bed_regions_df['chromosome'] == win_chrom
    if require_fully_within:
        pos_match = (
            (bed_regions_df['bed_start'] >= win_start)
            & (bed_regions_df['bed_end'] <= win_end)
        )
        filtered_df = bed_regions_df.loc[chrom_match & pos_match].copy().reset_index(drop=True)
    else:
        pos_match = (
            (bed_regions_df['bed_start'] < win_end)
            & (bed_regions_df['bed_end'] > win_start)
        )
        filtered_df = bed_regions_df.loc[chrom_match & pos_match].copy().reset_index(drop=True)
        filtered_df['bed_start'] = filtered_df['bed_start'].clip(lower=win_start)
        filtered_df['bed_end'] = filtered_df['bed_end'].clip(upper=win_end)
        filtered_df['width_bp'] = filtered_df['bed_end'] - filtered_df['bed_start']
        filtered_df['region_str'] = [
            f'{c}:{s}-{e}'
            for c, s, e in zip(
                filtered_df['chromosome'],
                filtered_df['bed_start'],
                filtered_df['bed_end'],
            )
        ]

    print(
        f'Filtered BED regions: kept {len(filtered_df)} / {len(bed_regions_df)} '
        f'region(s) ({filtered_df["width_bp"].sum():,} bp) within {gene_symbol} 1 Mb window.'
    )
    return filtered_df


bed_df_all = load_bed_regions(BED_FILE_PATH)
bed_df = filter_bed_regions_to_gene_window(
    bed_df_all,
    gene_symbol=TARGET_GENE_SYMBOL,
    window_size_bp=dna_client.SEQUENCE_LENGTH_1MB,
)
bed_df


# # Helper Functions to Pull AVI & Raw Track Scores (Filtered to LMNA for Gene-Level Scorers)

# In[5]:


# @title Helper Functions to Pull AVI & Raw Track Scores (Filtered to LMNA for Gene-Level Scorers)
def _is_gene_level_obs(obs_df: pd.DataFrame) -> bool:
    """Returns True if AnnData.obs contains gene-level annotation columns."""
    if obs_df is None or obs_df.empty:
        return False
    gene_cols = {'gene_name', 'gene_id', 'gene_symbol'}
    if any(col in obs_df.columns for col in gene_cols):
        return True
    # Fallback check if 'name' holds Ensembl IDs or gene symbols
    if 'name' in obs_df.columns:
        sample_vals = obs_df['name'].dropna().astype(str)
        if (
            sample_vals.str.upper().eq(TARGET_GENE_SYMBOL.upper()).any()
            or sample_vals.str.startswith('ENSG').any()
        ):
            return True
    return False


def _filter_obs_to_target_gene(
    obs_df: pd.DataFrame,
    target_gene_symbol: str = TARGET_GENE_SYMBOL,
    target_ensembl_prefix: str = TARGET_GENE_ENSEMBL_PREFIX,
) -> np.ndarray:
    """Returns a boolean mask over AnnData.obs matching the target gene (LMNA)."""
    mask = np.zeros(len(obs_df), dtype=bool)
    target_upper = target_gene_symbol.strip().upper()

    for col in ('gene_name', 'gene_symbol', 'name'):
        if col in obs_df.columns:
            vals = obs_df[col].fillna('').astype(str).str.strip().str.upper()
            mask |= (vals == target_upper).to_numpy()

    for col in ('gene_id', 'name'):
        if col in obs_df.columns and target_ensembl_prefix:
            vals = obs_df[col].fillna('').astype(str).str.strip()
            mask |= vals.str.startswith(target_ensembl_prefix).to_numpy()

    return mask


def _obs_row_suffix(obs_row: pd.Series, row_idx: int, total_matched: int) -> str:
    """Builds a column suffix for gene-level rows (e.g., LMNA or LMNA junction coordinates)."""
    # Check if splice junction coordinates exist in obs_row
    j_start = obs_row.get('junction_start', obs_row.get('start', None))
    j_end = obs_row.get('junction_end', obs_row.get('end', None))
    if pd.notna(j_start) and pd.notna(j_end):
        return f'__{TARGET_GENE_SYMBOL}_{int(j_start)}_{int(j_end)}'
    if total_matched > 1:
        row_id = str(obs_row.get('name', obs_row.name)).replace(':', '_').replace(' ', '_')
        return f'__{TARGET_GENE_SYMBOL}_{row_id}'
    return f'__{TARGET_GENE_SYMBOL}'


def _extract_avi_and_features(scores_dict: dict[str, anndata.AnnData]) -> pd.DataFrame:
    """Extracts AVI_SCORE and AVI_SCORE_FEATURE_IMPORTANCE from query_interval output."""
    if 'AVI_SCORE' not in scores_dict:
        raise ValueError('query_interval response did not contain AVI_SCORE.')

    avi_adata = scores_dict['AVI_SCORE']
    fi_adata = scores_dict.get('AVI_SCORE_FEATURE_IMPORTANCE')

    variant_objs = [
        v if isinstance(v, genome.Variant) else genome.Variant.from_str(str(v))
        for v in avi_adata.obs['variant']
    ]
    raw_scores = np.ravel(avi_adata.X)

    if 'quantiles' in avi_adata.layers:
        cdf_q = np.ravel(avi_adata.layers['quantiles'])
        tail_q = np.maximum(1e-7, 1.0 - cdf_q)
        phred = -10.0 * np.log10(tail_q)
        top_pct = tail_q * 100.0
    else:
        tail_q = np.full(len(variant_objs), np.nan)
        phred = np.full(len(variant_objs), np.nan)
        top_pct = np.full(len(variant_objs), np.nan)

    df = pd.DataFrame({
        'variant': [str(v) for v in variant_objs],
        'chromosome': [v.chromosome for v in variant_objs],
        'position': [int(v.position) for v in variant_objs],
        'ref': [v.reference_bases for v in variant_objs],
        'alt': [v.alternate_bases for v in variant_objs],
        'avi_phred': phred.astype(np.float32),
        'avi_raw': raw_scores.astype(np.float32),
        'avi_quantile': tail_q.astype(np.float32),
        'top_percentile': top_pct.astype(np.float32),
    })

    if fi_adata is not None and fi_adata.X is not None and fi_adata.X.size > 0:
        fi_names = (
            list(fi_adata.var['name'])
            if (fi_adata.var is not None and 'name' in fi_adata.var.columns)
            else list(fi_adata.var_names)
        )
        fi_mat = np.asarray(fi_adata.X, dtype=np.float32)
        max_idx = np.argmax(np.abs(fi_mat), axis=1)
        df['top_modality'] = [str(fi_names[i]) for i in max_idx]
        df['top_feature_importance'] = fi_mat[np.arange(len(df)), max_idx]
        for col_i, feat_name in enumerate(fi_names):
            df[f'fi_{feat_name}'] = fi_mat[:, col_i]

    return df, variant_objs


def _extract_single_variant_track_scores(
    variant_obj: genome.Variant,
    client: atlas.AtlasClient,
    scorers: list[str],
) -> dict[str, Any]:
    """Queries raw track-level scores for a single variant, filtering gene-level scorers to LMNA."""
    row_scores: dict[str, Any] = {'variant': str(variant_obj)}
    if not scorers:
        return row_scores

    assay_scores = client.query_variant(variant_obj, requested_scorers=scorers)
    for scorer_name, adata in assay_scores.items():
        if adata is None or adata.X is None or adata.X.size == 0:
            continue
        mat = np.asarray(adata.X, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        obs_df = adata.obs
        if _is_gene_level_obs(obs_df):
            gene_mask = _filter_obs_to_target_gene(obs_df)
            if not gene_mask.any():
                continue
            matched_indices = np.flatnonzero(gene_mask)
            for m_idx in matched_indices:
                obs_row = obs_df.iloc[m_idx]
                suffix = _obs_row_suffix(obs_row, m_idx, len(matched_indices))
                vec = mat[m_idx]
                for t_idx, val in enumerate(vec):
                    row_scores[f'{scorer_name}__t{t_idx:04d}{suffix}'] = float(val)
        else:
            # Non-gene-level scorer: 1 row per variant x n_tracks
            vec = mat[0]
            for t_idx, val in enumerate(vec):
                row_scores[f'{scorer_name}__t{t_idx:04d}'] = float(val)

    return row_scores


def _extract_interval_track_scores_if_available(
    scores_dict: dict[str, anndata.AnnData],
    scorers: list[str],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Extracts experimental track scores directly from query_interval if returned."""
    extracted_dfs = []
    missing_scorers = []

    for scorer_name in scorers:
        if scorer_name not in scores_dict:
            missing_scorers.append(scorer_name)
            continue

        adata = scores_dict[scorer_name]
        if adata is None or adata.X is None or adata.X.size == 0:
            continue

        obs_df = adata.obs
        if obs_df is None or 'variant' not in obs_df.columns:
            missing_scorers.append(scorer_name)
            continue

        mat = np.asarray(adata.X, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)

        if _is_gene_level_obs(obs_df):
            gene_mask = _filter_obs_to_target_gene(obs_df)
            if not gene_mask.any():
                continue
            sub_obs = obs_df.loc[gene_mask].copy()
            sub_mat = mat[gene_mask]
            sub_obs['_variant_str'] = sub_obs['variant'].astype(str)

            # Group by variant in case multiple junctions exist for LMNA
            if sub_obs['_variant_str'].duplicated().any():
                var_records: dict[str, dict[str, Any]] = {}
                for row_i, (_, obs_row) in enumerate(sub_obs.iterrows()):
                    v_str = obs_row['_variant_str']
                    rec = var_records.setdefault(v_str, {'variant': v_str})
                    suffix = _obs_row_suffix(obs_row, row_i, 2)
                    for t_idx in range(sub_mat.shape[1]):
                        rec[f'{scorer_name}__t{t_idx:04d}{suffix}'] = float(
                            sub_mat[row_i, t_idx]
                        )
                extracted_dfs.append(pd.DataFrame(list(var_records.values())))
            else:
                col_names = [
                    f'{scorer_name}__t{t_idx:04d}__{TARGET_GENE_SYMBOL}'
                    for t_idx in range(sub_mat.shape[1])
                ]
                scorer_df = pd.DataFrame(sub_mat, columns=col_names)
                scorer_df.insert(0, 'variant', sub_obs['_variant_str'].to_numpy())
                extracted_dfs.append(scorer_df)
        else:
            col_names = [
                f'{scorer_name}__t{t_idx:04d}' for t_idx in range(mat.shape[1])
            ]
            scorer_df = pd.DataFrame(mat, columns=col_names)
            scorer_df.insert(0, 'variant', obs_df['variant'].astype(str).to_numpy())
            extracted_dfs.append(scorer_df)

    if not extracted_dfs:
        return None, missing_scorers

    merged = extracted_dfs[0]
    for extra_df in extracted_dfs[1:]:
        merged = merged.merge(extra_df, on='variant', how='outer')
    return merged, missing_scorers


def score_bed_interval_chunk(
    chrom: str,
    chunk_start: int,
    chunk_end: int,
    clients: list[atlas.AtlasClient],
    experimental_scorers: list[str] = EXPERIMENTAL_SCORERS,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """Scores all SNVs in a 0-based half-open [chunk_start, chunk_end) interval."""
    interval = genome.Interval(chromosome=chrom, start=chunk_start, end=chunk_end)
    client0 = clients[0]

    # Try querying AVI + experimental scorers via query_interval first;
    # fall back to AVI on query_interval + experimental scorers via query_variant if needed.
    all_requested = AVI_SCORERS + list(experimental_scorers)
    try:
        scores_dict = client0.query_interval(
            interval, requested_scorers=all_requested
        )
    except Exception:
        scores_dict = client0.query_interval(
            interval, requested_scorers=AVI_SCORERS
        )

    avi_df, variant_objs = _extract_avi_and_features(scores_dict)
    interval_tracks_df, remaining_scorers = _extract_interval_track_scores_if_available(
        scores_dict, experimental_scorers
    )

    if interval_tracks_df is not None:
        avi_df = avi_df.merge(interval_tracks_df, on='variant', how='left')

    if remaining_scorers:
        client_cycle = itertools.cycle(clients)
        tasks = [(v_obj, next(client_cycle)) for v_obj in variant_objs]
        variant_track_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _extract_single_variant_track_scores,
                    v_obj,
                    c,
                    remaining_scorers,
                )
                for v_obj, c in tasks
            ]
            for fut in as_completed(futures):
                variant_track_rows.append(fut.result())

        if variant_track_rows:
            tracks_df = pd.DataFrame(variant_track_rows)
            avi_df = avi_df.merge(tracks_df, on='variant', how='left')

    return avi_df


# # Score All Regions in the BED File & Iteratively Save/Resume Parquet

# In[ ]:


# @title Score All Regions in the BED File & Iteratively Save/Resume Parquet
def score_all_bed_regions(
    bed_regions_df: pd.DataFrame,
    clients: list[atlas.AtlasClient] = atlas_clients,
    experimental_scorers: list[str] = EXPERIMENTAL_SCORERS,
    max_window_bp: int = MAX_INTERVAL_WINDOW_BP,
    output_parquet_path: str | None = OUTPUT_PARQUET_PATH,
    resume: bool = True,
    save_every_chunk: bool = True,
) -> pd.DataFrame:
    """Iterates through BED regions, pulls Atlas variant scores, and iteratively saves to Parquet with auto-resume.

    Args:
        bed_regions_df: DataFrame of BED intervals to score.
        clients: Pool of authenticated AtlasClient instances.
        experimental_scorers: List of experimental track scorers to query.
        max_window_bp: Max chunk size in bp per `query_interval` call (default: 1000).
        output_parquet_path: Path to output Parquet file.
        resume: If True and `output_parquet_path` exists, loads already-scored variants
            and skips completed regions/chunks.
        save_every_chunk: If True, checkpoints to `output_parquet_path` after every chunk
            (useful for multi-kb regions); if False, checkpoints after each completed region.
    """
    all_region_dfs: list[pd.DataFrame] = []
    scored_positions_by_region: dict[str, set[int]] = {}

    if resume and output_parquet_path and os.path.exists(output_parquet_path):
        print(f'Found existing Parquet file at {output_parquet_path}. Loading for resume...')
        existing_df = pd.read_parquet(output_parquet_path)
        if not existing_df.empty and {'region_str', 'position'}.issubset(existing_df.columns):
            all_region_dfs.append(existing_df)
            for reg_str, group in existing_df.groupby('region_str'):
                scored_positions_by_region[str(reg_str)] = set(
                    group['position'].astype(int).tolist()
                )
            print(
                f'Resuming: loaded {len(existing_df):,} previously scored variants across '
                f'{len(scored_positions_by_region):,} region(s).'
            )

    total_regions = len(bed_regions_df)
    for region_idx, (_, region) in enumerate(bed_regions_df.iterrows(), start=1):
        reg_id = int(region['region_id'])
        chrom = str(region['chromosome'])
        bed_start = int(region['bed_start'])
        bed_end = int(region['bed_end'])
        reg_str = str(region['region_str'])
        width_bp = int(region['width_bp'])

        done_positions = scored_positions_by_region.get(reg_str, set())
        expected_chunks = [
            (cs, min(cs + max_window_bp, bed_end))
            for cs in range(bed_start, bed_end, max_window_bp)
        ]
        # A 0-based half-open chunk [cs, ce) covers 1-based positions (cs + 1)..ce
        pending_chunks = [
            (cs, ce)
            for cs, ce in expected_chunks
            if not ((cs + 1) in done_positions and ce in done_positions)
        ]

        if not pending_chunks:
            print(
                f'[{region_idx}/{total_regions}] Skipping BED region {reg_str} '
                f'(region_id={reg_id}, {width_bp:,} bp) — already completed.'
            )
            continue

        print(
            f'[{region_idx}/{total_regions}] Scoring BED region {reg_str} '
            f'(region_id={reg_id}, {width_bp:,} bp, '
            f'{len(pending_chunks)}/{len(expected_chunks)} chunk(s) remaining)...'
        )

        region_new_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(pending_chunks, start=1):
            chunk_df = score_bed_interval_chunk(
                chrom=chrom,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                clients=clients,
                experimental_scorers=experimental_scorers,
            )
            if chunk_df.empty:
                continue

            chunk_df.insert(0, 'region_str', reg_str)
            chunk_df.insert(0, 'bed_end', bed_end)
            chunk_df.insert(0, 'bed_start', bed_start)
            chunk_df.insert(0, 'region_id', reg_id)

            all_region_dfs.append(chunk_df)
            region_new_chunks.append(chunk_df)
            scored_positions_by_region.setdefault(reg_str, set()).update(
                chunk_df['position'].astype(int).tolist()
            )

            if output_parquet_path and save_every_chunk:
                combined_df = pd.concat(all_region_dfs, ignore_index=True)
                all_region_dfs = [combined_df]
                _atomic_save_parquet(combined_df, output_parquet_path)
                print(
                    f'  -> Checkpointed chunk {chunk_idx}/{len(pending_chunks)} '
                    f'({chrom}:{chunk_start}-{chunk_end}, +{len(chunk_df):,} variants; '
                    f'total saved: {len(combined_df):,}) to {output_parquet_path}'
                )

        if not region_new_chunks:
            continue

        new_variants_count = sum(len(c) for c in region_new_chunks)
        if output_parquet_path and not save_every_chunk:
            combined_df = pd.concat(all_region_dfs, ignore_index=True)
            all_region_dfs = [combined_df]
            _atomic_save_parquet(combined_df, output_parquet_path)
            print(
                f'  -> Completed region {reg_str} (+{new_variants_count:,} variants; '
                f'total saved: {len(combined_df):,}) to {output_parquet_path}'
            )
        else:
            print(
                f'  -> Completed region {reg_str} (+{new_variants_count:,} new variants).'
            )
        gc.collect()

    if not all_region_dfs:
        print('No variants scored.')
        return pd.DataFrame()

    combined_df = (
        all_region_dfs[0]
        if len(all_region_dfs) == 1
        else pd.concat(all_region_dfs, ignore_index=True)
    )
    if output_parquet_path:
        _atomic_save_parquet(combined_df, output_parquet_path)
        print(
            f'\nFinished! {len(combined_df):,} total variants '
            f'({combined_df.shape[1]:,} columns) saved in {output_parquet_path}'
        )

    return combined_df


all_variant_scores_df = score_all_bed_regions(bed_df)
all_variant_scores_df.head(20)


# In[ ]:





# In[ ]:




