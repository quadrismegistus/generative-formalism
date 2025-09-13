"""Corpus loading, sampling, and table utilities for the Chadwyck-Healey poetry dataset.

This module provides:
- Loading and normalization of Chadwyck-Healey metadata and poem texts
- Deterministic sampling by attributes (period, subcorpus, rhyme)
- Simple in-memory caching to avoid repeated I/O
- Generation of period×subcorpus summary tables and LaTeX output
"""
from . import *
from generative_formalism.rhyme.rhyme_measurement import get_rhyme_data_for

CORPUS_METADATA = None
CORPUS = None

# === Metadata loading and normalization ===

def get_chadwyck_corpus_metadata(
    fields=CHADWYCK_CORPUS_FIELDS,
    period_by=CORPUS_PERIOD_BY,
    download_if_necessary=True,
    overwrite=False,
    min_num_lines=MIN_NUM_LINES,
    max_num_lines=MAX_NUM_LINES,
    min_author_dob=MIN_AUTHOR_DOB,
    max_author_dob=MAX_AUTHOR_DOB,
) -> pd.DataFrame:
    """Load and normalize Chadwyck-Healey corpus metadata.

    This function reads `PATH_CHADWYCK_HEALEY_METADATA`, downloading and unzipping
    if missing. It coerces numeric fields, derives `id_hash` and binned `period`,
    applies min/max filters, and caches the resulting DataFrame in `CORPUS_METADATA`.

    Parameters
    - fields: Mapping from raw column names to canonical names used downstream.
    - period_by: Size of year bin for `period` derived from `author_dob`.
    - download_if_necessary: If True, download metadata when not present on disk.
    - overwrite: If True, force re-download when files exist.
    - min_num_lines, max_num_lines: Optional poem-length filters.
    - min_author_dob, max_author_dob: Optional birth-year filters.

    Returns
    - pd.DataFrame indexed by `id`, sorted by `id_hash`, including normalized fields
      and derived `period`.
    - Caches the DataFrame in the module-level `CORPUS_METADATA`.
    """
    global CORPUS_METADATA
    if CORPUS_METADATA is not None:
        print('* Loading corpus metadata from memory')
        return CORPUS_METADATA
    
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        download_chadwyck_corpus_metadata(overwrite=overwrite)
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        return pd.DataFrame(columns=fields.keys()).set_index('id')
    
    print(f'* Loading metadata from {PATH_CHADWYCK_HEALEY_METADATA}')
    df = pd.read_csv(PATH_CHADWYCK_HEALEY_METADATA).fillna("")
    df['author_dob'] = pd.to_numeric(df['author_dob'], errors='coerce')
    df['num_lines'] = pd.to_numeric(df['num_lines'], errors='coerce')
    df['id_hash'] = [get_id_hash(x) for x in df['id']]
    print(f'* Loaded {len(df)} rows of metadata')

    def get_attdbase_str(x):
        if not x:
            return ""
        if 'African-American' in x:
            return 'African-American Poetry'
        if 'American' in x:
            return 'American Poetry'
        if 'English' in x:
            return 'English Poetry'
        return x

    def get_attperi_str(x):
        if not x:
            return ""
        x = x.replace('Fifteenth-Century Poetry', 'Fifteenth Century Poetry 1400-1500')
        last_word = x.split()[-1]
        if '-' in last_word and last_word[0].isdigit():
            while len(last_word) < 9:
                last_word = '0' + last_word
            all_but_last = ' '.join(x.split()[:-1])
            if all_but_last.endswith(','):
                all_but_last = all_but_last[:-1]
            return last_word + ' ' + all_but_last
        return x

    if 'attperi' in df.columns:
        df['attperi_str'] = df['attperi'].apply(get_attperi_str)
    if 'attdbase' in df.columns:
        df['attdbase_str'] = df['attdbase'].apply(get_attdbase_str)

    df = df[list(fields.keys())].rename(columns=fields)

    odf = df.fillna("")
    odf = odf[odf.author_dob != ""]
    if min_author_dob is not None:
        odf = odf[odf.author_dob >= min_author_dob]
        print(f'* Filtering: {len(odf):,} rows after author birth year >= {min_author_dob}')
    if max_author_dob is not None:
        odf = odf[odf.author_dob <= max_author_dob]
        print(f'* Filtering: {len(odf):,} rows after author birth year <= {max_author_dob}')

    def get_period_dob(x, ybin=period_by):
        if not x:
            return ""
        n = int(x // ybin * ybin)
        return f'{n}-{n + ybin}'

    odf['period'] = odf.author_dob.apply(get_period_dob)

    if min_num_lines is not None:
        odf = odf[odf.num_lines >= min_num_lines]
        print(f'* Filtering: {len(odf):,} rows after number of lines >= {min_num_lines}')
    if max_num_lines is not None:
        odf = odf[odf.num_lines <= max_num_lines]
        print(f'* Filtering: {len(odf):,} rows after number of lines <= {max_num_lines}')

    odf = odf.drop_duplicates('id').set_index('id').sort_values('id_hash')
    CORPUS_METADATA = odf
    return odf



def download_chadwyck_corpus_metadata(overwrite=False):
    """Download and unzip corpus metadata if the local file is missing.

    Parameters
    - overwrite: If True, re-download even when the file exists.
    """
    if URL_CHADWYCK_HEALEY_METADATA and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA)):
        PATH_CHADWYCK_HEALEY_METADATA_ZIP = PATH_CHADWYCK_HEALEY_METADATA+'.zip'
        print(f"* Downloading metadata to {PATH_CHADWYCK_HEALEY_METADATA_ZIP}")
        download_file(URL_CHADWYCK_HEALEY_METADATA, PATH_CHADWYCK_HEALEY_METADATA_ZIP)
        print(f'* Unzipping metadata to {PATH_CHADWYCK_HEALEY_METADATA}')
        unzip_file(PATH_CHADWYCK_HEALEY_METADATA_ZIP, PATH_CHADWYCK_HEALEY_METADATA)

def download_chadwyck_corpus_txt(overwrite=False):
    """Download and unzip corpus texts if the local directory is missing.

    Parameters
    - overwrite: If True, re-download even when the directory exists.
    """
    if URL_CHADWYCK_HEALEY_TXT and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_TXT)):
        PATH_CHADWYCK_HEALEY_TXT_ZIP = PATH_CHADWYCK_HEALEY_TXT+'.zip'
        print(f"* Downloading corpus text to {PATH_CHADWYCK_HEALEY_TXT_ZIP}")
        download_file(URL_CHADWYCK_HEALEY_TXT, PATH_CHADWYCK_HEALEY_TXT_ZIP)

        print(f'* Unzipping corpus text to {PATH_CHADWYCK_HEALEY_TXT}')
        unzip_file(PATH_CHADWYCK_HEALEY_TXT_ZIP, PATH_CHADWYCK_HEALEY_TXT)




def get_txt(id, clean_poem=True) -> str:
    """Load a poem's raw text by `id` from disk.

    Parameters
    - id: Chadwyck-Healey poem identifier (path-like under the texts root).
    - clean_poem: If True, apply `clean_poem_str` to normalize the text.

    Returns
    - The poem text as a string (possibly cleaned), or an empty string if missing.
    """
    fn = os.path.join(PATH_CHADWYCK_HEALEY_TXT, id) + '.txt'
    if os.path.exists(fn):
        with open(fn) as f:
            out = f.read().strip()
            if out and clean_poem:
                out = clean_poem_str(out)
            return out
    return ""

def get_chadwyck_corpus_texts(df_meta, clean_poem=True) -> list[str]:
    """Vectorized helper to load poem texts for all `id`s in `df_meta`.

    Parameters
    - df_meta: Metadata DataFrame with index `id`.
    - clean_poem: If True, clean each poem via `clean_poem_str`.

    Returns
    - list[str] aligned with `df_meta.index`.
    """
    print(f'* Loading {len(df_meta)} texts')
    return [
        get_txt(id, clean_poem=clean_poem)
        for id in tqdm(df_meta.reset_index().id, desc='  ')
    ]

def get_chadwyck_corpus(df_meta=None, *args, clean_poem=True, force=False, download_if_necessary=True, **kwargs) -> pd.DataFrame:
    """Load metadata and poem texts into a single corpus DataFrame.

    Parameters
    - clean_poem: If True, clean poem texts after reading.
    - force: If True, ignore in-memory cache and rebuild corpus.
    - args/kwargs: Passed to `get_chadwyck_corpus_metadata`.

    Returns
    - pd.DataFrame with metadata plus a `txt` column containing poem text.

    Side Effects
    - Caches the result in the module-level `CORPUS`.
    """
    global CORPUS
    print(f'* Loading Chadwyck-Healey corpus (metadata + txt)')

    if not force and CORPUS is not None:
        print('* Loading corpus from memory')
        return CORPUS

    df_meta = get_chadwyck_corpus_metadata(*args, **kwargs) if df_meta is None else df_meta
    if df_meta is None or not len(df_meta):
        return pd.DataFrame()

    if download_if_necessary and not os.path.exists(PATH_CHADWYCK_HEALEY_TXT):
        download_chadwyck_corpus_txt()
    
    if not os.path.exists(PATH_CHADWYCK_HEALEY_TXT):
        print(f'* Warning: No corpus text files to load')
        return pd.DataFrame()

    df_meta['txt'] = get_chadwyck_corpus_texts(df_meta, clean_poem=clean_poem)

    CORPUS = df_meta
    return df_meta


def sample_chadwyck_corpus(
        df_corpus,
        sample_by,
        min_sample_n=MIN_SAMPLE_N,
        max_sample_n=MAX_SAMPLE_N,
        prefer_min_id_hash=False,
        ) -> pd.DataFrame:
    """Deterministically sample `df_corpus` by one or more grouping keys.

    Rules
    - Keep only groups with at least `min_sample_n` items (if provided).
    - Within each group, sort by `id_hash` and take the first `max_sample_n` rows
      (if provided). This ensures stable sampling across runs.

    Parameters
    - df_corpus: Corpus DataFrame (e.g., from `get_chadwyck_corpus`).
    - sample_by: Column name or list of names to group by.
    - min_sample_n, max_sample_n: Group size constraints.

    Returns
    - pd.DataFrame containing the sampled rows.
    """

    if not len(df_corpus):
        print(f'* Warning: No corpus to sample')
        return pd.DataFrame()

    print(f'* Sampling corpus by {sample_by} (min {min_sample_n}, max {max_sample_n})')
    print(f'* Original sample size: {len(df_corpus)}')
    # sort by id hash
    df = df_corpus.sort_values('id_hash')

    sample_by = [sample_by] if isinstance(sample_by, str) else sample_by
    if min_sample_n:
        df = df.groupby(sample_by).filter(lambda x: len(x) >= min_sample_n)

    if max_sample_n:
        if prefer_min_id_hash:
            df = df.sort_values('id_hash')
        else:
            df = df.sample(frac=1)
        
        df = df.sort_values('id_hash').groupby(sample_by).head(max_sample_n)
    
    print(f'* Final sample size: {len(df)}\n')
    s=df.groupby(sample_by).size()
    s.name = '/'.join(sample_by)
    describe_qual(s, count=False)
    return df



    

# def get_chadwyck
    
def get_chadwyck_corpus_sampled_by_rhyme_as_in_paper() -> pd.DataFrame:
    """Load the rhyme-based sample used in the paper (precomputed)."""
    return pd.read_csv(PATH_SAMPLE_RHYMES_IN_PAPER).fillna('').set_index('id').sort_values('id_hash')

def get_chadwyck_corpus_sampled_by_period_as_in_paper() -> pd.DataFrame:
    """Load the period-based sample used in the paper (precomputed)."""
    return pd.read_csv(PATH_SAMPLE_PERIOD_IN_PAPER).fillna('').set_index('id').sort_values('id_hash')


def get_chadwyck_corpus_sampled_by_period_subcorpus_as_in_paper(display=False) -> pd.DataFrame:
    """Load the period×subcorpus sample used in the paper and optionally display a table."""
    odf = pd.read_csv(PATH_SAMPLE_PERIOD_SUBCORPUS_IN_PAPER).fillna('').set_index('id').sort_values('id_hash')
    if display:
        display_period_subcorpus_tables(odf)
    return odf

def get_chadwyck_corpus_sampled_by_sonnet_period_as_in_paper() -> pd.DataFrame:
    """Load the sonnet-based sample used in the paper (precomputed)."""
    return pd.read_csv(PATH_SAMPLE_SONNET_IN_PAPER).fillna('').set_index('id').sort_values('id_hash')

def display_period_subcorpus_tables(df):
    """Display summary tables for a sampled DataFrame (IPython rich display if available)."""
    try_display(get_period_subcorpus_table(df, return_display=True))




# Samplers

def gen_chadwyck_corpus_sampled_by_rhyme() -> pd.DataFrame:
    """Generate a rhyme-stratified sample from the full corpus."""
    df_corpus = get_chadwyck_corpus()
    df_corpus = df_corpus[df_corpus.rhyme.isin({'y','n'})]
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by='rhyme',
    )
    return df

def gen_chadwyck_corpus_sampled_by_period() -> pd.DataFrame:
    """Generate a period-stratified sample from the full corpus."""
    df_corpus = get_chadwyck_corpus()
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by='period',
    )
    return df

def gen_chadwyck_corpus_sampled_by_period_subcorpus() -> pd.DataFrame:
    """Generate a period×subcorpus-stratified sample from the full corpus."""
    df_corpus = get_chadwyck_corpus()
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by=['period','subcorpus'],
    )
    return df

def gen_chadwyck_corpus_sampled_by_sonnet_period() -> pd.DataFrame:
    """Generate a sonnet-stratified sample from the full corpus."""
    df_corpus = get_chadwyck_corpus()
    # Filter for sonnets based on genre metadata and 14-line poems
    df_sonnets = df_corpus[
        (df_corpus.genre.str.contains('sonnet', case=False, na=False)) |
        (df_corpus.title.str.contains('sonnet', case=False, na=False))
    ]
    # Further filter by 14 lines (traditional sonnet length)
    df_sonnets = df_sonnets[df_sonnets.num_lines == 14]
    df = sample_chadwyck_corpus(
        df_sonnets,
        sample_by='period',  # Sample sonnets by period for diversity
    )
    return df

def get_chadwyck_corpus_sampled_by_rhyme(force=False) -> pd.DataFrame:
    """Load or generate rhyme-stratified sample; cache on disk at `PATH_SAMPLE_RHYMES_REPLICATED`."""
    path = PATH_SAMPLE_RHYMES_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating rhyme sample')
        odf = gen_chadwyck_corpus_sampled_by_rhyme()
        if len(odf):
            save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading rhyme sample from {path}')
        odf = pd.read_csv(path).set_index('id').sort_values('id_hash')
    return odf

def get_chadwyck_corpus_sampled_by_period(force=False) -> pd.DataFrame:
    """Load or generate period-stratified sample; cache on disk at `PATH_SAMPLE_PERIOD_REPLICATED`."""
    path = PATH_SAMPLE_PERIOD_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating period sample')
        odf = gen_chadwyck_corpus_sampled_by_period()
        if len(odf):
            save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading period sample from {path}')
        odf = pd.read_csv(path).set_index('id').sort_values('id_hash')
    return odf

def get_chadwyck_corpus_sampled_by_period_subcorpus(force=False, display=False) -> pd.DataFrame:
    """Load or generate period×subcorpus sample; cache on disk at `PATH_SAMPLE_PERIOD_SUBCORPUS_REPLICATED`."""
    path = PATH_SAMPLE_PERIOD_SUBCORPUS_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating period subcorpus sample')
        odf = gen_chadwyck_corpus_sampled_by_period_subcorpus()
        if len(odf):
            save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading period subcorpus sample from {path}')
        odf = pd.read_csv(path).set_index('id').sort_values('id_hash')
    if display:
        try:
            from IPython.display import display
            img = get_period_subcorpus_table(odf, return_display=True)
            display(img)
        except (NameError, ImportError):
            print(f'* Warning: Could not display image')
            pass
    return odf

def get_chadwyck_corpus_sampled_by_sonnet_period(force=False) -> pd.DataFrame:
    """Load or generate sonnet-stratified sample; cache on disk at `PATH_SAMPLE_SONNET_REPLICATED`."""
    path = PATH_SAMPLE_SONNET_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating sonnet sample')
        odf = gen_chadwyck_corpus_sampled_by_sonnet_period()
        if len(odf):
            save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading sonnet sample from {path}')
        odf = pd.read_csv(path).set_index('id').sort_values('id_hash')
    return odf


def get_chadwyck_corpus_sampled_by_rhyme_as_replicated(overwrite=False) -> pd.DataFrame:
    """Convenience wrapper to compute or load rhyme-stratified sample (replication)."""
    df_smpl = get_chadwyck_corpus_sampled_by_rhyme(force=overwrite)
    return df_smpl

def get_chadwyck_corpus_sampled_by_period_as_replicated(overwrite=False) -> pd.DataFrame:
    """Convenience wrapper to compute or load period-stratified sample (replication)."""
    df_smpl = get_chadwyck_corpus_sampled_by_period(force=overwrite)
    return df_smpl

def get_chadwyck_corpus_sampled_by_period_subcorpus_as_replicated(overwrite=False, display=False) -> pd.DataFrame:
    """Convenience wrapper to compute or load period×subcorpus sample (replication)."""
    df_smpl = get_chadwyck_corpus_sampled_by_period_subcorpus(force=overwrite)
    if display:
        display_period_subcorpus_tables(df_smpl)
    return df_smpl

def get_chadwyck_corpus_sampled_by_sonnet_period_as_replicated(overwrite=False) -> pd.DataFrame:
    """Convenience wrapper to compute or load sonnet-stratified sample (replication)."""
    df_smpl = get_chadwyck_corpus_sampled_by_sonnet_period(force=overwrite)
    return df_smpl

def check_paths():
    """Check if the paths to the Chadwyck-Healey corpus and metadata are set and exist.
    Uses constants from `constants.py`.
    """
    # Get the Chadwyck-Healey corpus path
    print(f"""{"✓" if PATH_CHADWYCK_HEALEY_TXT and os.path.exists(PATH_CHADWYCK_HEALEY_TXT) else "X"} Chadwyck-Healey corpus path: {PATH_CHADWYCK_HEALEY_TXT}""")
    print(f"""{"✓" if PATH_CHADWYCK_HEALEY_METADATA and os.path.exists(PATH_CHADWYCK_HEALEY_METADATA) else "X"} Chadwyck-Healey metadata path: {PATH_CHADWYCK_HEALEY_METADATA}""")

    # Download if necessary?
    print(f"""{"✓" if URL_CHADWYCK_HEALEY_METADATA and URL_CHADWYCK_HEALEY_METADATA else "X"} Metadata file URL set in environment (.env or shell)""")
    print(f"""{"✓" if URL_CHADWYCK_HEALEY_TXT and URL_CHADWYCK_HEALEY_TXT else "X"} Corpus text file URL set in environment (.env or shell)""")


def describe_corpus(dfx: pd.DataFrame) -> pd.DataFrame:
    """Print high-level descriptive stats of a corpus DataFrame and return it."""
    printm(f'----')

    printm(f'#### Historical period breakdown (from author birth year)')
    describe_qual(dfx.period)
    printm(f'----')

    printm(f'#### Subcorpus breakdown')
    describe_qual(dfx.subcorpus)
    printm(f'----')

    # printm(f'#### Historical period breakdown (from metadata)')
    # describe_qual(dfx.period_meta)
    # printm(f'----')
    

    printm(f'#### Historical period + subcorpus breakdown')
    describe_qual_grouped(dfx, ['period', 'subcorpus'])
    printm(f'----')

    # printm(f'#### Author birth year distribution')
    # describe_numeric(dfx.author_dob)
    # printm(f'----')

    # printm(f'#### Number of lines in poems')
    # describe_numeric(dfx.num_lines)
    # printm(f'----')

    printm(f'#### Annotated rhyme distribution')
    describe_qual(dfx.rhyme)
    printm(f'----')

    printm(f'#### Metadata')
    return dfx



def get_period_subcorpus_table(df_smpl, save_latex_to=PATH_TEX_PERIOD_SUBCORPUS_COUNTS, save_latex_to_suffix='tmp',return_display=False, table_num=None):
    """Build a period×subcorpus summary table and optionally save LaTeX.

    Parameters
    - df_smpl: Sampled DataFrame containing `period`, `subcorpus`, `author`, `id`.
    - save_latex_to: Base path for LaTeX/table image output; if falsy, skip saving.
    - save_latex_to_suffix: Filename suffix for differentiation.
    - return_display: If True, return a display object suitable for notebooks.
    - table_num: Optional table number for LaTeX captioning.

    Returns
    - A formatted DataFrame (if not returning display object) or a display/image object.
    """
    df_meta = get_chadwyck_corpus_metadata()
    
    # Build summary table using sample groupings (df_smpl)
    rows = []
    for (period, subcorpus), gdf in df_smpl.groupby(['period', 'subcorpus']):
        meta_q = df_meta.query(f'subcorpus=="{subcorpus}" & period=="{period}"')
        rows.append({
            'period': period,
            'subcorpus': subcorpus,
            'num_poets_total': meta_q.author.nunique(),
            'num_poets': gdf.author.nunique(),
            'num_poems_total': len(meta_q),
            'num_poems': len(gdf),
        })

    df_table = pd.DataFrame(rows).set_index(['period', 'subcorpus']).sort_index()

    # convert numbers to comma'd strings
    def format_number(x):
        x = int(x)
        return f'{x:,.0f}'

    df_formatted = df_table.applymap(format_number)

    df_formatted.rename_axis(['Period', 'Subcorpus'], inplace=True)
    df_formatted.columns = ['# Poets (corpus)', '# Poets (sample)', '# Poems (corpus)', '# Poems (sample)']

    # Build grouped LaTeX tabular matching the requested style (no outer table here)
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')

    periods = sorted({idx[0] for idx in df_formatted.index})

    tabular_lines = []
    tabular_lines.append('\\begin{tabular}{llrrrr}')
    tabular_lines.append('\\toprule')
    tabular_lines.append('& & \\multicolumn{2}{c}{Corpus} & \\multicolumn{2}{c}{Sample} \\\\')
    tabular_lines.append('\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}')
    tabular_lines.append('Period & Subcorpus & \\# Poems & \\# Poets & \\# Poems & \\# Poets \\\\')
    tabular_lines.append('\\midrule')

    for period in periods:
        subdf = df_formatted.xs(period, level=0)
        subcorp_order = sorted(list(subdf.index))
        n = len(subcorp_order)
        for i, subcorpus in enumerate(subcorp_order):
            row = subdf.loc[subcorpus]
            period_disp = _escape_latex_text(period).replace('-', '--') if i == 0 else ''
            sub_disp = _escape_latex_text(subcorpus)
            vals = [
                row['# Poems (corpus)'],
                row['# Poets (corpus)'],
                row['# Poems (sample)'],
                row['# Poets (sample)'],
            ]
            if i == 0:
                tabular_lines.append(f'\\multirow[t]{{{n}}}{{*}}{{{period_disp}}} & {sub_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\')
            else:
                tabular_lines.append(f' & {sub_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\')
        tabular_lines.append('\\cline{1-6}')

    # replace trailing cline with bottomrule
    if tabular_lines[-1] == '\\cline{1-6}':
        tabular_lines[-1] = '\\bottomrule'
    else:
        tabular_lines.append('\\bottomrule')

    tabular_lines.append('\\end{tabular}')

    tabular_str = '\n'.join(tabular_lines)

    # Use generic table wrapper/saver
    if save_latex_to:
        _ = df_to_latex_table(
            inner_latex=tabular_str,
            save_latex_to=save_latex_to,
            save_latex_to_suffix=save_latex_to_suffix,
            table_num=table_num,
            caption='Number of poets and poems in the Chadwyck-Healey corpus and sample.',
            label='tab:num_poems_corpus',
            position='t',
            center=True,
            size='\\small',
            resize_to_textwidth=True,
            return_display=return_display,
        )
        if return_display and _ is not None:
            return _

    return df_formatted


def get_period_subcorpus_table_as_in_paper(df_smpl=None, save_latex=True, return_display=False, table_num=TABLE_NUM_PERIOD_SUBCORPUS_COUNTS):
    """Recreate the period×subcorpus table exactly as in the paper."""
    df_smpl = df_smpl if df_smpl is not None else get_chadwyck_corpus_sampled_by_period_subcorpus_as_in_paper()
    return get_period_subcorpus_table(
        df_smpl,
        save_latex_to_suffix=PAPER_REGENERATED_SUFFIX,
        return_display=return_display,
        table_num=table_num,
    )

def get_period_subcorpus_table_as_replicated(save_latex=None, return_display=False, table_num=TABLE_NUM_PERIOD_SUBCORPUS_COUNTS):
    """Generate the period×subcorpus table for the replication sample."""
    df_smpl = get_chadwyck_corpus_sampled_by_period_subcorpus_as_replicated()
    return get_period_subcorpus_table(
        df_smpl,
        save_latex_to_suffix=REPLICATED_SUFFIX,
        return_display=return_display,
        table_num=table_num,
    )



def get_rhyme_data_for_corpus_sampled_by_rhyme_as_in_paper(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_RHYME
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_rhyme_as_in_paper, output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_rhyme_as_replicated(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_RHYME
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_rhyme_as_replicated, output_path, **kwargs)

    
def get_rhyme_data_for_corpus_sampled_by_period_as_in_paper(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_PERIOD
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_period_as_in_paper, output_path=output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_period_as_replicated(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_PERIOD
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_period_as_replicated, output_path=output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_period_subcorpus_as_in_paper(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_PERIOD_SUBCORPUS
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_period_subcorpus_as_in_paper, output_path=output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_period_subcorpus_as_replicated(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_PERIOD_SUBCORPUS
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_period_subcorpus_as_replicated, output_path=output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_sonnet_period_as_in_paper(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_SONNET_PERIOD
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_sonnet_period_as_in_paper, output_path=output_path, **kwargs)

def get_rhyme_data_for_corpus_sampled_by_sonnet_period_as_replicated(output_path=None, **kwargs):
    output_path = output_path if output_path else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_SONNET_PERIOD
    return get_rhyme_data_for(get_chadwyck_corpus_sampled_by_sonnet_period_as_replicated, output_path=output_path, **kwargs)
