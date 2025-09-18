"""Corpus loading, sampling, and table utilities for the Chadwyck-Healey poetry dataset.

This module provides:
- Loading and normalization of Chadwyck-Healey metadata and poem texts
- Deterministic sampling by attributes (period, subcorpus, rhyme)
- Simple in-memory caching to avoid repeated I/O
- Generation of period×subcorpus summary tables and LaTeX output
"""
from . import *
from generative_formalism.rhyme.rhyme_measurements import get_rhyme_data_for

CORPUS_METADATA = None
CORPUS = None

# === Metadata loading and normalization ===

def get_chadwyck_corpus_metadata(force=False, verbose=DEFAULT_VERBOSE) -> pd.DataFrame:
    global CORPUS_METADATA
    if CORPUS_METADATA is not None:
        return CORPUS_METADATA

    if not force and os.path.exists(PATH_CHADWYCK_HEALEY_METADATA_SMALL):
        if verbose:
            printm(f'* Loading from `{PATH_CHADWYCK_HEALEY_METADATA_SMALL}`')
        odf = pd.read_csv(PATH_CHADWYCK_HEALEY_METADATA_SMALL).set_index('id')
    else:
        odf = preprocess_chadwyck_corpus_metadata(verbose=verbose)
    CORPUS_METADATA = odf
    return odf

def preprocess_chadwyck_corpus_metadata(
    fields=CHADWYCK_CORPUS_FIELDS,
    period_by=CORPUS_PERIOD_BY,
    download_if_necessary=True,
    overwrite=False,
    min_num_lines=MIN_NUM_LINES,
    max_num_lines=MAX_NUM_LINES,
    min_author_dob=MIN_AUTHOR_DOB,
    max_author_dob=MAX_AUTHOR_DOB,
    verbose=DEFAULT_VERBOSE,
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
        if verbose:
            printm('* Loading corpus metadata from memory')
        return CORPUS_METADATA
    
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        download_chadwyck_corpus_metadata(overwrite=overwrite, verbose=verbose)
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        return pd.DataFrame(columns=fields.keys()).set_index('id')
    
    if verbose:
        printm(f'* Loading metadata from `{PATH_CHADWYCK_HEALEY_METADATA}`')

    df = pd.read_csv(PATH_CHADWYCK_HEALEY_METADATA).fillna("")
    df['author_dob'] = pd.to_numeric(df['author_dob'], errors='coerce')
    df['num_lines'] = pd.to_numeric(df['num_lines'], errors='coerce')
    df['id_hash'] = [get_id_hash(x) for x in df['id']]
    
    if verbose:
        printm(f'* Loaded {len(df)} rows of metadata')

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
        if verbose:
            printm(f'*Filtering: {len(odf):,} rows after author birth year >= {min_author_dob}')
    if max_author_dob is not None:
        odf = odf[odf.author_dob <= max_author_dob]
        if verbose:
            printm(f'*Filtering: {len(odf):,} rows after author birth year <= {max_author_dob}')

    def get_period_dob(x, ybin=period_by):
        if not x:
            return ""
        n = int(x // ybin * ybin)
        return f'{n}-{n + ybin}'

    odf['period'] = odf.author_dob.apply(get_period_dob)

    if min_num_lines is not None:
        odf = odf[odf.num_lines >= min_num_lines]
        if verbose:
            printm(f'*Filtering: {len(odf):,} rows after number of lines >= {min_num_lines}')
    if max_num_lines is not None:
        odf = odf[odf.num_lines <= max_num_lines]
        if verbose:
            printm(f'*Filtering: {len(odf):,} rows after number of lines <= {max_num_lines}')

    odf = odf.drop_duplicates('id').set_index('id').sort_values('id_hash')
    if verbose:
        printm(f'*Dropped duplicates and set index')
    CORPUS_METADATA = odf
    odf.to_csv(PATH_CHADWYCK_HEALEY_METADATA_SMALL)
    return odf



def download_chadwyck_corpus_metadata(overwrite=False, verbose=DEFAULT_VERBOSE):
    """Download and unzip the Chadwyck-Healey corpus metadata file.

    Downloads the metadata CSV file from the configured URL if it doesn't exist
    locally or if overwrite=True. The downloaded file is unzipped to the
    expected location for subsequent loading.

    Parameters
    ----------
    overwrite : bool, default=False
        If True, re-download and unzip even if the file already exists.
    verbose : bool, default=False
        If True, print progress messages during download and unzip operations.

    Returns
    -------
    None
        The function modifies files on disk but doesn't return a value.

    Calls
    -----
    - download_file(URL_CHADWYCK_HEALEY_METADATA, PATH_CHADWYCK_HEALEY_METADATA_ZIP)
    - unzip_file(PATH_CHADWYCK_HEALEY_METADATA_ZIP, PATH_CHADWYCK_HEALEY_METADATA)
    """
    if URL_CHADWYCK_HEALEY_METADATA and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA)):
        PATH_CHADWYCK_HEALEY_METADATA_ZIP = PATH_CHADWYCK_HEALEY_METADATA+'.zip'
        if verbose:
            printm(f"* Downloading metadata to `{PATH_CHADWYCK_HEALEY_METADATA_ZIP}`")
        download_file(URL_CHADWYCK_HEALEY_METADATA, PATH_CHADWYCK_HEALEY_METADATA_ZIP)
        if verbose:
            printm(f'*Unzipping metadata to `{PATH_CHADWYCK_HEALEY_METADATA}`')
        unzip_file(PATH_CHADWYCK_HEALEY_METADATA_ZIP, PATH_CHADWYCK_HEALEY_METADATA)

def download_chadwyck_corpus_txt(overwrite=False, verbose=DEFAULT_VERBOSE):
    """Download and unzip the Chadwyck-Healey corpus text files.

    Downloads the corpus text files (individual poem text files) from the
    configured URL if the local directory doesn't exist or if overwrite=True.
    The downloaded ZIP file is unzipped to create the directory structure
    containing individual poem text files.

    Parameters
    ----------
    overwrite : bool, default=False
        If True, re-download and unzip even if the directory already exists.
    verbose : bool, default=False
        If True, print progress messages during download and unzip operations.

    Returns
    -------
    None
        The function modifies files on disk but doesn't return a value.

    Calls
    -----
    - download_file(URL_CHADWYCK_HEALEY_TXT, PATH_CHADWYCK_HEALEY_TXT_ZIP)
    - unzip_file(PATH_CHADWYCK_HEALEY_TXT_ZIP, PATH_CHADWYCK_HEALEY_TXT)
    """
    if URL_CHADWYCK_HEALEY_TXT and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_TXT)):
        PATH_CHADWYCK_HEALEY_TXT_ZIP = PATH_CHADWYCK_HEALEY_TXT+'.zip'
        if verbose:
            printm(f"* Downloading corpus text to `{PATH_CHADWYCK_HEALEY_TXT_ZIP}`")
        download_file(URL_CHADWYCK_HEALEY_TXT, PATH_CHADWYCK_HEALEY_TXT_ZIP)

        if verbose:
            printm(f'*Unzipping corpus text to `{PATH_CHADWYCK_HEALEY_TXT}`')
        unzip_file(PATH_CHADWYCK_HEALEY_TXT_ZIP, PATH_CHADWYCK_HEALEY_TXT)




def get_txt(id, clean_poem=True) -> str:
    """Load a poem's text content from the Chadwyck-Healey corpus by its ID.

    Reads the text file for the specified poem ID from the local corpus directory.
    Optionally applies text cleaning/normalization to the loaded content.

    Parameters
    ----------
    id : str
        Chadwyck-Healey poem identifier used as the filename (without .txt extension).
    clean_poem : bool, default=True
        If True, apply text cleaning/normalization using clean_poem_str().

    Returns
    -------
    str
        The poem's text content as a string. Returns empty string if the file
        doesn't exist or can't be read.

    Calls
    -----
    - clean_poem_str(out) [if clean_poem=True]
    """
    fn = os.path.join(PATH_CHADWYCK_HEALEY_TXT, id) + '.txt'
    if os.path.exists(fn):
        with open(fn) as f:
            out = f.read().strip()
            if out and clean_poem:
                out = clean_poem_str(out)
            return out
    return ""

def get_chadwyck_corpus_texts(df_meta, clean_poem=True, verbose=DEFAULT_VERBOSE) -> list[str]:
    """Load poem text content for all poems in the metadata DataFrame.

    Efficiently loads text content for multiple poems by iterating through
    the metadata DataFrame and calling get_txt() for each poem ID.
    Shows progress using tqdm for large datasets.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame with poem IDs as index. Should contain an 'id' column
        with the Chadwyck-Healey poem identifiers.
    clean_poem : bool, default=True
        If True, apply text cleaning to each poem using clean_poem_str().
    verbose : bool, default=False
        If True, print progress information during loading.

    Returns
    -------
    list[str]
        List of poem text strings in the same order as df_meta.index.
        Empty strings are returned for poems that cannot be loaded.

    Calls
    -----
    - get_txt(id, clean_poem=clean_poem) [for each poem ID in df_meta]
    - tqdm(df_meta.reset_index().id, desc='  ') [for progress display]
    """
    if verbose:
        printm(f'* Loading {len(df_meta)} texts')
    return [
        get_txt(id, clean_poem=clean_poem)
        for id in tqdm(df_meta.reset_index().id, desc='  ')
    ]

def get_chadwyck_corpus(df_meta=None, *args, clean_poem=True, force=False, download_if_necessary=True, verbose=DEFAULT_VERBOSE, **kwargs) -> pd.DataFrame:
    """Load metadata and poem texts into a single corpus DataFrame.

    Combines corpus metadata with poem text content into a single DataFrame.
    Uses in-memory caching to avoid repeated expensive loading operations.

    Parameters
    ----------
    df_meta : pd.DataFrame, optional
        Pre-loaded metadata DataFrame. If None, loads using get_chadwyck_corpus_metadata.
    clean_poem : bool, default=True
        If True, apply text cleaning/normalization to poem texts.
    force : bool, default=False
        If True, ignore in-memory cache and rebuild corpus.
    download_if_necessary : bool, default=True
        If True, download corpus files if not present locally.
    *args, **kwargs
        Additional arguments passed to get_chadwyck_corpus_metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata plus a 'txt' column containing poem text.

    Calls
    -----
    - get_chadwyck_corpus_metadata(*args, **kwargs) [if df_meta is None]
    - download_chadwyck_corpus_txt() [if download_if_necessary=True and corpus text not found]
    - get_chadwyck_corpus_texts(df_meta, clean_poem=clean_poem) [to load poem texts]
    """
    global CORPUS
    if verbose:
        printm(f'* Loading Chadwyck-Healey corpus (metadata + txt)')

    if not force and CORPUS is not None:
        # printm('* Loading corpus from memory')
        return CORPUS

    df_meta = get_chadwyck_corpus_metadata(*args, **kwargs) if df_meta is None else df_meta
    if df_meta is None or not len(df_meta):
        return pd.DataFrame()

    if download_if_necessary and not os.path.exists(PATH_CHADWYCK_HEALEY_TXT):
        download_chadwyck_corpus_txt()
    
    if not os.path.exists(PATH_CHADWYCK_HEALEY_TXT):
        printm(f'* Warning: No corpus text files to load')
        return pd.DataFrame()

    df_meta['txt'] = get_chadwyck_corpus_texts(df_meta, clean_poem=clean_poem)

    CORPUS = df_meta
    return df_meta










def check_chadwyck_healey_paths():
    """Check if the paths to the Chadwyck-Healey corpus and metadata are set and exist.

    Validates the configuration and availability of corpus files and URLs.
    Prints status indicators for each required path and URL.

    Returns
    -------
    None
        Prints status information but doesn't return a value.

    Calls
    -----
    - os.path.exists(PATH_CHADWYCK_HEALEY_TXT)
    - os.path.exists(PATH_CHADWYCK_HEALEY_METADATA)
    """
    # Get the Chadwyck-Healey corpus path
    path_txt = PATH_CHADWYCK_HEALEY_TXT
    path_meta = PATH_CHADWYCK_HEALEY_METADATA_SMALL
    path_txt_exists = path_txt and os.path.exists(path_txt)
    path_metadata_exists = path_meta and os.path.exists(path_meta)
    url_txt_exists = URL_CHADWYCK_HEALEY_TXT and URL_CHADWYCK_HEALEY_TXT
    url_metadata_exists = URL_CHADWYCK_HEALEY_METADATA and URL_CHADWYCK_HEALEY_METADATA
    out = f'''
* {"✓" if path_txt_exists else "X"} Chadwyck-Healey corpus path: `{nice_path(path_txt)}`
* {"✓" if path_metadata_exists else "X"} Chadwyck-Healey metadata path: `{nice_path(path_meta)}`
* {"✓" if url_metadata_exists else "X"} Metadata file URL set in environment (.env or shell)
* {"✓" if url_txt_exists else "X"} Corpus text file URL set in environment (.env or shell)
'''

    printm(out)

    return bool((path_txt_exists and path_metadata_exists) or (url_txt_exists and url_metadata_exists))




def describe_corpus(dfx: pd.DataFrame) -> pd.DataFrame:
    """Print high-level descriptive statistics of a corpus DataFrame and return it.

    Generates a comprehensive summary of the corpus including breakdowns by
    historical period, subcorpus, period×subcorpus combinations, and rhyme
    annotations. Useful for understanding the composition and distribution
    of poems in a sample.

    Parameters
    ----------
    dfx : pd.DataFrame
        Corpus DataFrame to analyze, should contain columns for 'period',
        'subcorpus', and 'rhyme'.

    Returns
    -------
    pd.DataFrame
        The same DataFrame that was passed in (unchanged).

    Calls
    -----
    - describe_qual(dfx.period) [for period breakdown]
    - describe_qual(dfx.subcorpus) [for subcorpus breakdown]
    - describe_qual_grouped(dfx, ['period', 'subcorpus']) [for cross-tabulation]
    - describe_qual(dfx.rhyme) [for rhyme distribution]
    """
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
