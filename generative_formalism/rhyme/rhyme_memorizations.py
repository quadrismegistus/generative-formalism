from . import *



def get_all_memorization_data_with_rhyme_data(overwrite: bool = False, verbose: bool = True):
    """Get comprehensive memorization data with rhyme analysis for all poems.

    This function combines memorization detection results from multiple sources
    (Antoniak et al., Chadwyck completions, Dolma) and enriches them with
    rhyme analysis data.

    Parameters
    ----------
    overwrite : bool, default False
        If True, force reprocessing of all data sources instead of using cached results.
    verbose : bool, default True
        If True, print progress messages during processing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing memorization data with rhyme analysis, indexed by poem ID.
        Includes columns for:
        - Memorization detection flags and sources
        - Poem metadata (title, author, dates, etc.)
        - Rhyme analysis metrics and features
        - Unique ID hash for each poem

    Notes
    -----
    The returned DataFrame combines data from:
    - Antoniak et al. memorization study (closed and open source detection)
    - Chadwyck poetry corpus completions (similarity-based detection)
    - Dolma training corpus (open source detection)

    Rhyme data is computed using get_rhyme_for_sample() and joined with
    left suffix '_from_sample' to avoid column name conflicts.
    """
    df_mem = get_all_memorization_data(overwrite=overwrite, verbose=verbose)
    df_mem['id_hash'] = [get_id_hash(id) for id in df_mem.index]

    df_mem_rhyme_data = get_rhyme_for_sample(df_mem)
    df_mem_rhyme = df_mem.join(df_mem_rhyme_data, how='left', lsuffix='_from_sample')
    return df_mem_rhyme



def get_all_memorization_data(overwrite: bool = False, verbose: bool = True):
    """Aggregate memorization detection results from all available data sources.

    This function combines memorization detection results from three different
    sources and data types: Antoniak et al. study, Chadwyck poetry corpus
    completions, and Dolma training corpus. It handles data integration,
    column ordering, and caching.

    Parameters
    ----------
    overwrite : bool, default False
        If True, force reprocessing of all data sources instead of using cached results.
    verbose : bool, default True
        If True, print progress messages and summary statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated memorization data from all sources, indexed by poem ID.
        Includes columns for:
        - 'found': Boolean indicating if poem was detected as memorized
        - 'found_source': Source type ('closed' or 'open')
        - 'found_corpus': Corpus identifier ('antoniak-et-al' or 'chadwyck')
        - Poem metadata (title, author, dates, text, etc.)
        - 'id_hash': Unique hash identifier for each poem

    Notes
    -----
    Data Sources:
    - Antoniak et al.: Public domain poems with closed/open memorization detection
    - Chadwyck closed: Similarity-based detection in GenAI completions (original paper method)
    - Chadwyck open: Detection in Dolma training corpus

    The function performs column reordering to prioritize shared columns across
    all sources, followed by source-specific columns. Only poems marked as
    'found' (memorized) are returned in the final result.

    Cached results are stored in PATH_ALL_MEMORIZATION_DATA for performance.
    """
    from generative_formalism.corpus.corpus import get_chadwyck_corpus_metadata

    if not overwrite and os.path.exists(PATH_ALL_MEMORIZATION_DATA):
        print(f"* Loading from {PATH_ALL_MEMORIZATION_DATA}")
        return pd.read_csv(PATH_ALL_MEMORIZATION_DATA).fillna("").set_index("id")

    df_mem_antoniak = get_antoniak_et_al_memorization_data(
        overwrite=overwrite, verbose=verbose
    )
    df_mem_chadwyck_closed = get_memorized_poems_in_completions_as_in_paper(
        verbose=verbose
    )
    df_mem_chadwyck_open = get_memorized_poems_in_dolma()

    df_mem = pd.concat(
        [df_mem_antoniak, df_mem_chadwyck_closed, df_mem_chadwyck_open]
    ).fillna("")
    df_mem['found_source_corpus'] = df_mem['found_source'] + '|' + df_mem['found_corpus']

    shared_cols = [
        col
        for col in df_mem.columns
        if col in df_mem_antoniak.columns
        and col in df_mem_chadwyck_closed.columns
        and col in df_mem_chadwyck_open.columns
    ]
    unshared_cols = [col for col in df_mem.columns if col not in shared_cols]

    odf = df_mem[shared_cols + unshared_cols]

    if verbose:
        print(f"* Writing to {PATH_ALL_MEMORIZATION_DATA}")
    odf.to_csv(PATH_ALL_MEMORIZATION_DATA)
    odf = odf.query('found!=""')
    if verbose:
        describe_qual_grouped(
            df_mem,
            groupby=['found_corpus', 'found_source','found'],
            name='poems found by corpus and source'
        )

    odf['id_hash'] = [get_id_hash(id) for id in odf.index]
    return odf







## PREVIOUS DATA


def preprocess_antoniak_et_al_memorization_data(
    data_fldr: str = PATH_ANTONIAK_ET_AL_DIR,
    out_path: str = PATH_ANTONIAK_ET_AL_CSV,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess Antoniak et al. memorization data from raw files.

    This function processes the raw Antoniak et al. dataset by combining:
    1. Public domain poetry metadata and text
    2. Closed-model memorization detection results
    3. Open-model memorization detection from Walsh et al. (wimbd) data

    The function handles data cleaning, ID extraction, date processing,
    and integration of multiple data sources into a unified format.

    Parameters
    ----------
    data_fldr : str, default PATH_ANTONIAK_ET_AL_DIR
        Path to directory containing Antoniak et al. raw data files:
        - poetry-evaluation_public-domain-poems.csv
        - memorization_results.csv
        - wimbd*.csv files (from Walsh et al.)
    out_path : str, default PATH_ANTONIAK_ET_AL_CSV
        Path where processed data will be saved as compressed CSV.
    verbose : bool, default DEFAULT_VERBOSE
        If True, print progress messages during processing.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns:
        - 'id': Poem identifier (extracted from poem_link)
        - 'found': Boolean indicating memorization detection
        - 'found_source': 'closed' or 'open'
        - 'found_corpus': 'antoniak-et-al'
        - 'txt': Full poem text
        - 'title': Poem title
        - 'author_dob_str': Author birth year as string
        - 'author_dob': Author birth year as numeric
        - Additional metadata columns from original dataset

    Notes
    -----
    The preprocessing involves:
    - Extracting poem IDs from URLs in poem_link column
    - Processing author birth/death dates to extract birth years
    - Loading closed-model memorization predictions
    - Processing wimbd CSV files to detect open-model memorization
    - Merging all data sources into unified format

    Results are cached to out_path for subsequent loads.
    """
    if verbose:
        print(f"* Preprocessing Antoniak et al. memorization data from {data_fldr}")
    df_antoniak = pd.read_csv(
        os.path.join(data_fldr, "poetry-evaluation_public-domain-poems.csv")
    ).fillna("")
    df_antoniak["id"] = df_antoniak["poem_link"].apply(
        lambda x: "/".join(str(x).split("/")[-2:])
    )
    df_antoniak = df_antoniak.rename(
        columns={"poem_text": "txt", "poem_title": "title"}
    ).set_index("id")
    df_antoniak["author_dob_str"] = [
        x.split("–")[0].strip() for x in df_antoniak["birth_death_dates"]
    ]
    df_antoniak["author_dob"] = pd.to_numeric(
        df_antoniak["author_dob_str"], errors="coerce"
    )

    df_mem_closed = pd.read_csv(os.path.join(data_fldr, "memorization_results.csv"))
    df_mem_closed = df_mem_closed.drop(
        columns=[c for c in ["Unnamed: 0"] if c in df_mem_closed.columns],
        errors="ignore",
    )
    df_mem_closed = df_mem_closed.rename(columns={"poem": "id", "prediction": "found"})
    df_mem_closed["found"] = df_mem_closed["found"].astype(bool)
    df_mem_closed["found_source"] = "closed"
    df_mem_closed["found_corpus"] = "antoniak-et-al"
    df_mem_closed = df_mem_closed.set_index("id")

    df_mem_open = df_mem_closed.copy()
    df_mem_open["found"] = False
    df_mem_open["found_source"] = "open"
    df_mem_open["found_corpus"] = "antoniak-et-al"

    # open-model training data hits (Walsh et al. wimbd outputs placed in the same folder)
    for fn in os.listdir(data_fldr):
        if "wimbd" in fn and fn.endswith(".csv"):
            try:
                fndf = pd.read_csv(os.path.join(data_fldr, fn))
            except Exception:
                continue
            for poem_id in set(fndf.get("poem_id", [])):
                # print(f'* {poem_id} in df_mem_open.index?')
                if poem_id in df_mem_open.index:
                    df_mem_open.loc[poem_id, "found"] = True

    df_mem = pd.concat(
        [
            df_mem_open,
            df_mem_closed,
        ]
    )

    df_mem_antoniak = df_mem.merge(df_antoniak, on="id", how="left")
    df_mem_antoniak = df_mem_antoniak.fillna("")
    if verbose:
        print(f"* Writing to {out_path}")
    df_mem_antoniak.to_csv(out_path)
    return df_mem_antoniak


_ANTONIAK_ET_AL_MEMORIZATION_DATA = None


def get_antoniak_et_al_memorization_data(
    path=PATH_ANTONIAK_ET_AL_CSV,
    data_fldr=PATH_ANTONIAK_ET_AL_DIR,
    verbose: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load Antoniak et al. memorization data with caching support.
    Convenience function for preprocess_antoniak_et_al_memorization_data() 
    with caching support.

    Returns processed Antoniak et al. dataset containing public domain poems 
    with memorization detection results from both closed and open language models.
    """
    global _ANTONIAK_ET_AL_MEMORIZATION_DATA
    if not overwrite and _ANTONIAK_ET_AL_MEMORIZATION_DATA is not None:
        return _ANTONIAK_ET_AL_MEMORIZATION_DATA

    if not overwrite and os.path.exists(path):
        if verbose:
            print(f"* Loading from {path}")
        odf = pd.read_csv(path).set_index("id")
    else:
        if verbose:
            print(f"* Preprocessing from {data_fldr}")
        odf = preprocess_antoniak_et_al_memorization_data(
            out_path=path,
            data_fldr=data_fldr,
            verbose=verbose,
        )

    _ANTONIAK_ET_AL_MEMORIZATION_DATA = odf
    return odf












## COMPLETIONS

_MEMORIZED_POEMS_IN_COMPLETIONS_AS_IN_PAPER = None
_MEMORIZED_POEMS_IN_COMPLETIONS_AS_REPLICATED = None


def get_memorized_poems_in_completions_as_in_paper(
    threshold: int = 95,
    verbose: bool = True,
    overwrite: bool = False,
):
    """Get memorized poems from GenAI completions using original paper methodology.

    This function identifies poems that were memorized by language models by analyzing
    GenAI rhyme completions data using the methodology from the original paper.
    It uses similarity-based detection to find poems that appear to be directly
    copied from training data.

    Parameters
    ----------
    threshold : int, default 95
        Similarity threshold (0-100) for determining memorization.
        Poems with line_sim > threshold are considered memorized.
    verbose : bool, default True
        If True, print progress messages during processing.
    overwrite : bool, default False
        If True, force reprocessing instead of using cached results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing poems detected as memorized in GenAI completions,
        indexed by poem ID. Includes:
        - Original poem metadata and text
        - Similarity scores and detection results
        - 'found': True for all rows (only memorized poems returned)
        - 'found_source': 'closed'
        - 'found_corpus': 'chadwyck'

    Notes
    -----
    This function uses the original paper's methodology for detecting memorization:
    1. Loads GenAI rhyme completions data (get_genai_rhyme_completions_as_in_paper)
    2. Applies similarity-based detection with specified threshold
    3. Returns only poems that exceed the memorization threshold

    The results are cached in a global variable for performance. This represents
    the "closed-source" detection method using direct model outputs rather than
    searching training corpora.

    See Also
    --------
    get_memorized_poems_in_completions : Core memorization detection logic
    get_genai_rhyme_completions_as_in_paper : Source data for detection
    """
    global _MEMORIZED_POEMS_IN_COMPLETIONS_AS_IN_PAPER

    if not overwrite and _MEMORIZED_POEMS_IN_COMPLETIONS_AS_IN_PAPER is not None:
        return _MEMORIZED_POEMS_IN_COMPLETIONS_AS_IN_PAPER

    df_smpl = get_genai_rhyme_completions_as_in_paper(
        by_line=False,
        filter_recognized=False,
        verbose=verbose,
    )
    out = get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=False
    )
    _MEMORIZED_POEMS_IN_COMPLETIONS_AS_IN_PAPER = out
    return out


def get_unmemorized_poems_in_completions_as_in_paper(
    threshold: int = 95, verbose: bool = True
):
    """Get unmemorized poems from GenAI completions using original paper methodology.

    This function identifies poems that were NOT memorized by language models by analyzing
    GenAI rhyme completions data using the methodology from the original paper.
    It returns poems that fall below the similarity threshold for memorization detection.

    Parameters
    ----------
    threshold : int, default 95
        Similarity threshold (0-100) for determining memorization.
        Poems with line_sim <= threshold are considered unmemorized.
    verbose : bool, default True
        If True, print progress messages during processing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing poems detected as NOT memorized in GenAI completions,
        indexed by poem ID. Includes:
        - Original poem metadata and text
        - Similarity scores below the threshold
        - 'found': False for all rows (only unmemorized poems returned)
        - 'found_source': 'closed'
        - 'found_corpus': 'chadwyck'

    Notes
    -----
    This function complements get_memorized_poems_in_completions_as_in_paper():
    1. Loads the same GenAI rhyme completions data
    2. Applies the same similarity-based detection logic
    3. Returns only poems that fall BELOW the memorization threshold

    This allows for analysis of both memorized and unmemorized poems from the
    same dataset, enabling comparison studies and statistical analysis.

    Unlike the memorized version, this function does not cache results since
    it may be called with different thresholds.

    See Also
    --------
    get_memorized_poems_in_completions_as_in_paper : Complementary function for memorized poems
    get_memorized_poems_in_completions : Core detection logic
    """
    df_smpl = get_genai_rhyme_completions_as_in_paper(
        by_line=True, filter_recognized=False
    )
    return get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True
    )


def get_memorized_poems_in_completions_as_replicated(
    threshold: int = 95, verbose: bool = True, overwrite: bool = False
):
    """Get memorized poems from GenAI completions using replicated methodology.

    This function identifies poems that were memorized by language models by analyzing
    GenAI rhyme completions data using the replicated methodology (not the original
    paper method). It uses similarity-based detection to find poems that appear to
    be directly copied from training data.

    Parameters
    ----------
    threshold : int, default 95
        Similarity threshold (0-100) for determining memorization.
        Poems with line_sim > threshold are considered memorized.
    verbose : bool, default True
        If True, print progress messages during processing.
    overwrite : bool, default False
        If True, force reprocessing instead of using cached results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing poems detected as memorized in GenAI completions
        using replicated methodology, indexed by poem ID. Includes:
        - Original poem metadata and text
        - Similarity scores and detection results
        - 'found': True for all rows (only memorized poems returned)
        - 'found_source': 'closed'
        - 'found_corpus': 'chadwyck'

    Notes
    -----
    This function uses the replicated methodology (not the original paper method):
    1. Loads GenAI rhyme completions data (get_genai_rhyme_completions_as_replicated)
    2. Applies similarity-based detection with specified threshold
    3. Returns only poems that exceed the memorization threshold

    The replicated methodology may differ from the original paper in data processing,
    filtering, or analysis approach. This allows for comparison between different
    methodological approaches to memorization detection.

    The results are cached in a global variable for performance. This represents
    the "closed-source" detection method using direct model outputs.

    See Also
    --------
    get_memorized_poems_in_completions_as_in_paper : Original paper methodology
    get_memorized_poems_in_completions : Core memorization detection logic
    get_genai_rhyme_completions_as_replicated : Source data for detection
    """
    global _MEMORIZED_POEMS_IN_COMPLETIONS_AS_REPLICATED
    if not overwrite and _MEMORIZED_POEMS_IN_COMPLETIONS_AS_REPLICATED is not None:
        return _MEMORIZED_POEMS_IN_COMPLETIONS_AS_REPLICATED
    df_smpl = get_genai_rhyme_completions_as_replicated(
        by_line=True, filter_recognized=False
    )
    out = get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=False
    )
    _MEMORIZED_POEMS_IN_COMPLETIONS_AS_REPLICATED = out
    return out


def get_unmemorized_poems_in_completions_as_replicated(
    threshold: int = 95, verbose: bool = True
):
    """Get unmemorized poems from GenAI completions using replicated methodology.

    This function identifies poems that were NOT memorized by language models by analyzing
    GenAI rhyme completions data using the replicated methodology (not the original
    paper method). It returns poems that fall below the similarity threshold for
    memorization detection.

    Parameters
    ----------
    threshold : int, default 95
        Similarity threshold (0-100) for determining memorization.
        Poems with line_sim <= threshold are considered unmemorized.
    verbose : bool, default True
        If True, print progress messages during processing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing poems detected as NOT memorized in GenAI completions
        using replicated methodology, indexed by poem ID. Includes:
        - Original poem metadata and text
        - Similarity scores below the threshold
        - 'found': False for all rows (only unmemorized poems returned)
        - 'found_source': 'closed'
        - 'found_corpus': 'chadwyck'

    Notes
    -----
    This function complements get_memorized_poems_in_completions_as_replicated():
    1. Loads GenAI rhyme completions data using replicated methodology
    2. Applies the same similarity-based detection logic
    3. Returns only poems that fall BELOW the memorization threshold

    The replicated methodology may differ from the original paper in data processing,
    filtering, or analysis approach. This allows for comparison studies between
    different methodological approaches for both memorized and unmemorized poems.

    Unlike the memorized version, this function does not cache results since
    it may be called with different thresholds.

    See Also
    --------
    get_memorized_poems_in_completions_as_replicated : Complementary function for memorized poems
    get_unmemorized_poems_in_completions_as_in_paper : Original paper methodology
    get_memorized_poems_in_completions : Core detection logic
    """
    df_smpl = get_genai_rhyme_completions_as_replicated(
        by_line=True, filter_recognized=False
    )
    return get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True
    )


def get_memorized_poems_in_completions(
    df_smpl, threshold: int = 95, verbose: bool = True, return_unmemorized=False
):
    """Core function for detecting memorized poems in GenAI completions using similarity.

    This function implements the core logic for memorization detection by analyzing
    similarity scores between generated completions and original poems. It applies
    a threshold-based classification to determine which poems show evidence of
    memorization.

    Parameters
    ----------
    df_smpl : pd.DataFrame
        Input DataFrame containing GenAI completion data with similarity scores.
        Must include columns: 'id_human', 'line_sim', and other metadata.
    threshold : int, default 95
        Similarity threshold (0-100) for memorization detection.
        - If return_unmemorized=False: poems with line_sim > threshold are memorized
        - If return_unmemorized=True: poems with line_sim <= threshold are unmemorized
    verbose : bool, default True
        If True, print progress messages (currently unused in this function).
    return_unmemorized : bool, default False
        If False, return only memorized poems (line_sim > threshold).
        If True, return only unmemorized poems (line_sim <= threshold).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with memorization detection results, indexed by poem ID.
        Includes:
        - 'found': Boolean indicating memorization detection result
        - 'found_source': 'closed' (all rows)
        - 'found_corpus': 'chadwyck' (all rows)
        - 'id_gen': Original generation/completion ID
        - All original columns from df_smpl (except renamed id_human -> id)

    Notes
    -----
    The function performs the following processing:
    1. Removes duplicate entries based on 'id_human' (keeps first occurrence)
    2. Applies similarity threshold to classify poems as memorized/unmemorized
    3. Adds standard metadata columns for corpus integration
    4. Renames columns for consistent naming ('id_human' -> 'id', 'id' -> 'id_gen')
    5. Sets poem ID as the DataFrame index

    This is the core detection logic used by higher-level functions that handle
    different data sources (original paper vs. replicated methodology).

    The similarity score (line_sim) is expected to be a percentage (0-100) where
    higher values indicate stronger evidence of memorization.
    """
    odf = df_smpl.copy().reset_index()
    odf = odf.drop_duplicates(subset="id_human")

    odf["found"] = odf.line_sim > threshold
    odf["found_source"] = "closed"
    odf["found_corpus"] = "chadwyck"
    return odf.rename(
        columns={
            "id_human": "id",
            "id": "id_gen",
        }
    ).set_index("id")







## DOLMA

def preprocess_memorized_poems_in_dolma(*args, overwrite: bool = False, verbose: bool = DEFAULT_VERBOSE, **kwargs) -> pd.DataFrame:
    """Preprocess memorized poems detected in Dolma training corpus.

    This function processes raw data from poems detected as memorized in the Dolma
    training corpus. It combines the memorized poem text with metadata from the
    Chadwyck poetry corpus to create a unified dataset for analysis.

    Parameters
    ----------
    *args
        Unused positional arguments (for compatibility with wrapper functions).
    overwrite : bool, default False
        If True, force reprocessing of raw data instead of using cached results.
    verbose : bool, default DEFAULT_VERBOSE
        If True, print progress messages during processing.
    **kwargs
        Unused keyword arguments (for compatibility with wrapper functions).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing memorized poems from Dolma, indexed by poem ID.
        Includes:
        - 'txt': Full poem text
        - 'found_source': 'open' (all rows - detected in training data)
        - 'found_corpus': 'chadwyck' (all rows)
        - Chadwyck corpus metadata (title, author, dates, etc.)
        - Columns with '_from_corpus' suffix for metadata source disambiguation

    Notes
    -----
    The preprocessing involves:
    1. Loading raw memorized poems from pickled Dolma detection results
    2. Renaming 'poem' column to 'txt' for consistency
    3. Adding standard metadata columns (found_source='open', found_corpus='chadwyck')
    4. Joining with Chadwyck corpus metadata using left join
    5. Saving processed results to CSV for caching

    This represents "open-source" memorization detection, where poems were found
    by searching the actual training corpus rather than analyzing model outputs.

    The function uses PATH_RAW_DOLMA_PKL as input and PATH_DOLMA_CSV as output cache.
    """
    from ..corpus.corpus import get_chadwyck_corpus_metadata

    if not overwrite and os.path.exists(PATH_DOLMA_CSV):
        if verbose:
            print(f"* Loading from {PATH_DOLMA_CSV}")
        return pd.read_csv(PATH_DOLMA_CSV).set_index("id")

    if verbose:
        print(f"* Preprocessing from {PATH_RAW_DOLMA_PKL}")

    df_dolma = (
        pd.read_pickle(PATH_RAW_DOLMA_PKL)
        .rename(columns={"poem": "txt"})
        .assign(found_source="open", found_corpus="chadwyck")
        .set_index("id")
    )

    df_meta = get_chadwyck_corpus_metadata()

    odf = df_dolma.join(df_meta, on="id", how="left", rsuffix='_from_corpus')
    if verbose:
        print(f"* Writing to {PATH_DOLMA_CSV}")
    odf.to_csv(PATH_DOLMA_CSV)
    return odf

def get_memorized_poems_in_dolma(*args, overwrite: bool = False, verbose: bool = True, **kwargs):
    """Get memorized poems detected in the Dolma training corpus.

    This function provides access to poems that were detected as memorized in the
    Dolma training corpus. Note that Dolma is no longer publicly accessible, so
    this function works with pre-computed detection results.

    Parameters
    ----------
    *args
        Positional arguments passed to preprocess_memorized_poems_in_dolma().
    overwrite : bool, default False
        If True, force reprocessing of cached data instead of using existing results.
    verbose : bool, default True
        If True, print progress messages during data loading/preprocessing.
    **kwargs
        Keyword arguments passed to preprocess_memorized_poems_in_dolma().

    Returns
    -------
    pd.DataFrame
        DataFrame containing poems detected as memorized in Dolma training corpus,
        indexed by poem ID. See preprocess_memorized_poems_in_dolma() for details.

    Notes
    -----
    Dolma is no longer publicly accessible, so this function relies on pre-computed
    memorization detection results stored in the repository. The detection was
    performed by searching for poem sequences within the Dolma training corpus.

    This represents "open-source" memorization detection, where poems were found
    by direct corpus search rather than analyzing language model outputs.

    See Also
    --------
    preprocess_memorized_poems_in_dolma : Core preprocessing function
    get_memorized_poems_in_completions_as_in_paper : Closed-source detection method
    """
    ## Dolma no longer accessible
    return preprocess_memorized_poems_in_dolma(*args, overwrite=overwrite, verbose=verbose, **kwargs)













# def find_poem_in_dolma(
#     poem_id: str,
#     df_lines: pd.DataFrame,
#     es_config_path: str,
#     index: str = "docs_v1.7_2024-06-04",
#     stash: 'HashStash' = None,
# ):
#     """Count Dolma docs containing each of the first-N lines for a poem.

#     Expects df_lines to contain line-by-line rows for a single poem id including
#     the original lines (`line_real`) and `line_num`, with `_first_n_lines` column
#     present to specify how many to search.
#     """
#     try:
#         from wimbd.es import es_init, count_documents_containing_phrases
#     except Exception:
#         raise RuntimeError("wimbd is required for Dolma search. Install and configure ES.")

#     if stash is None:
#         stash = HashStash(os.path.join(PATH_STASH, "find_in_dolma"), engine="pairtree", compress=False, b64=False)

#     stash_key = {"poem_id": poem_id, "index": index}
#     if stash_key in stash:
#         return stash[stash_key]

#     es = es_init(os.path.expanduser(es_config_path))
#     dfg = df_lines[df_lines["id"] == poem_id].copy()
#     n = int(dfg.iloc[0]["first_n_lines"]) if "first_n_lines" in dfg.columns else int(dfg.iloc[0].get("_first_n_lines", 5))
#     dfg = dfg[dfg["line_num"] <= n]
#     query_lines = [ln.strip() for ln in dfg.sort_values("line_num").line_real]
#     count = count_documents_containing_phrases(index, query_lines, es=es, all_phrases=True)
#     result = {"id": poem_id, "lines": query_lines, "count": int(count)}
#     stash[stash_key] = result
#     return result


# def get_dolma_hits_for_ids(
#     poem_ids: list[str],
#     df_lines: pd.DataFrame,
#     es_config_path: str,
#     index: str = "docs_v1.7_2024-06-04",
# ):
#     """Batch Dolma search for many poem ids; returns a DataFrame with counts and found flag."""
#     out = []
#     for pid in tqdm(poem_ids, desc="* Searching Dolma"):
#         try:
#             res = find_poem_in_dolma(pid, df_lines=df_lines, es_config_path=es_config_path, index=index)
#             out.append(res)
#         except Exception:
#             continue
#     ddf = pd.DataFrame(out)
#     if not ddf.empty:
#         ddf["found"] = ddf["count"] > 0
#     return ddf


# def annotate_dolma_hits_with_corpus(df_dolma_hits: pd.DataFrame, period_by: int = CORPUS_PERIOD_BY) -> pd.DataFrame:
#     """Attach Chadwyck metadata to Dolma hit rows (expects column `id`)."""
#     from generative_formalism.corpus.corpus import get_chadwyck_corpus_metadata
#
#     if df_dolma_hits.empty:
#         return df_dolma_hits

#     df_meta = get_chadwyck_corpus_metadata(period_by=period_by).reset_index()

#     return df_dolma_hits.merge(df_meta, on="id", how="left")