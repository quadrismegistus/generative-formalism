"""Corpus loading, sampling, and table utilities for the Chadwyck-Healey poetry dataset.

This module provides:
- Loading and normalization of Chadwyck-Healey metadata and poem texts
- Deterministic sampling by attributes (period, subcorpus, rhyme)
- Simple in-memory caching to avoid repeated I/O
- Generation of period×subcorpus summary tables and LaTeX output
"""

from . import *


def sample_chadwyck_corpus(
    df_corpus,
    sample_by,
    min_sample_n=MIN_SAMPLE_N,
    max_sample_n=MAX_SAMPLE_N,
    prefer_min_id_hash=False,
    sort_id_hash=True,
    verbose=False,
) -> pd.DataFrame:
    """Deterministically sample the corpus by one or more grouping criteria.

    Creates a balanced sample from the corpus by grouping on specified criteria,
    filtering groups by size constraints, and taking deterministic subsets within
    each group. Uses id_hash sorting to ensure reproducible results across runs.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus DataFrame to sample from (e.g., from get_chadwyck_corpus()).
        Must contain the columns specified in sample_by plus 'id_hash'.
    sample_by : str or list[str]
        Column name(s) to group by for stratified sampling.
    min_sample_n : int, default=MIN_SAMPLE_N
        Minimum number of items required in a group to be included.
    max_sample_n : int, default=MAX_SAMPLE_N
        Maximum number of items to take from each group.
    prefer_min_id_hash : bool, default=False
        If True, prefer items with smaller id_hash values when sampling.
    sort_id_hash : bool, default=True
        If True, sort the sample by id_hash.
    verbose : bool, default=False
        If True, print progress information.


    Returns
    -------
    pd.DataFrame
        Sampled DataFrame containing the selected rows from df_corpus.

    Calls
    -----
    - describe_qual(s, count=False) [to display group size distribution]
    """

    if not len(df_corpus):
        print(f"* Warning: No corpus to sample")
        return pd.DataFrame()

    if verbose:
        print(
            f"* Sampling corpus by {sample_by} (min {min_sample_n}, max {max_sample_n})"
        )
        print(f"* Original sample size: {len(df_corpus)}")

    # sort by id hash
    if sort_id_hash:
        df = df_corpus.sort_values("id_hash")
    else:
        df = df_corpus

    sample_by = [sample_by] if isinstance(sample_by, str) else sample_by
    if min_sample_n:
        df = df.groupby(sample_by).filter(lambda x: len(x) >= min_sample_n)

    if max_sample_n:
        if sort_id_hash:
            df = df.sort_values("id_hash")
        else:
            df = df.sample(frac=1)

        df = df.groupby(sample_by).head(max_sample_n)

        if sort_id_hash:
            df = df.sort_values("id_hash")

    if verbose:
        print(f"* Final sample size: {len(df)}\n")
    s = df.groupby(sample_by).size()

    if verbose:
        describe_qual(s, count=False, name="/".join(sample_by))
    return df


# def get_chadwyck


def get_chadwyck_corpus_sampled_by(
    sample_by,
    as_in_paper=True,
    as_replicated=False,
    as_regenerated=False,
    display=False,
    verbose=False,
    **kwargs,
) -> pd.DataFrame:
    """Load or generate a sampled corpus by the specified criteria.

    Parameters
    - sample_by: Sampling criteria ('period', 'period_subcorpus', 'rhyme', 'sonnet_period')
    - as_in_paper: If True, load precomputed sample from paper
    - as_replicated: If True, load/generate replicated sample
    - as_regenerated: If True, load/generate regenrated sample
    - **kwargs: Additional arguments passed to generation/display functions

    Returns
    - pd.DataFrame containing the sampled corpus

    Calls
    -----
    - get_path(data_name, as_in_paper=True, as_replicated=False)

    """
    from .tex import display_period_subcorpus_tables

    # Map sample_by to data name
    sample_by_map = {
        "period": DATA_NAME_CORPUS_SAMPLE_BY_PERIOD,
        "period_subcorpus": DATA_NAME_CORPUS_SAMPLE_BY_PERIOD_SUBCORPUS,
        "rhyme": DATA_NAME_CORPUS_SAMPLE_BY_RHYME,
        "sonnet_period": DATA_NAME_CORPUS_SAMPLE_BY_SONNET_PERIOD,
    }

    if sample_by not in sample_by_map:
        raise ValueError(
            f"Invalid sample_by: {sample_by}. Must be one of {list(sample_by_map.keys())}"
        )

    data_name = sample_by_map[sample_by]

    if as_replicated or as_regenerated:
        odf = get_chadwyck_corpus_sampled_by_replicated(
            sample_by,
            as_in_paper=as_in_paper,
            as_replicated=as_replicated,
            as_regenerated=as_regenerated,
            **kwargs,
        )
    elif as_in_paper:
        # Load precomputed sample from paper
        path = get_path(data_name, as_in_paper=True, as_replicated=False)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Precomputed sample not found at {path}")
        odf = pd.read_csv(path).fillna("").set_index("id").sort_values("id_hash")
    else:
        raise ValueError("Must specify as_in_paper, as_replicated, or as_regenerated")

    # Handle display options for specific sample types
    if display:
        gby = (
            "period"
            if sample_by == "sonnet_period"
            else (sample_by.split("_") if isinstance(sample_by, str) else sample_by)
        )
        describe_qual_grouped(
            odf, groupby=gby, sort_index=True, count=False, name=sample_by,
        )

    odf._sample_by = sample_by
    odf._as_in_paper = as_in_paper
    odf._as_replicated = as_replicated
    odf._as_regenerated = as_regenerated

    return odf


# Samplers


def gen_chadwyck_corpus_sampled_by(sample_by, display=False, **kwargs) -> pd.DataFrame:
    """Generate a stratified sample from the full Chadwyck-Healey corpus.

    Creates a balanced sample of poems using the specified stratification criteria.
    Handles different sampling types including rhyme, period, period×subcorpus,
    and sonnet-based sampling.

    Parameters
    ----------
    sample_by : str
        Sampling criteria ('rhyme', 'period', 'period_subcorpus', 'sonnet_period').
    display : bool, default=False
        If True, display summary tables for certain sample types (e.g., period).
    **kwargs: Additional arguments passed to sample_chadwyck_corpus

    Returns
    -------
    pd.DataFrame
        DataFrame containing the stratified sample with balanced representation.

    Calls
    -----
    - get_chadwyck_corpus() [to load the full corpus]
    - sample_chadwyck_corpus(df_corpus, sample_by=...) [to create stratified sample]
    - get_period_subcorpus_table(df, return_display=True) [if display=True for period samples]
    - display(img) [if display=True and IPython available]
    """
    if sample_by not in ["rhyme", "period", "period_subcorpus", "sonnet_period"]:
        raise ValueError(
            f"Invalid sample_by: {sample_by}. Must be one of ['rhyme', 'period', 'period_subcorpus', 'sonnet_period']"
        )

    df_corpus = get_chadwyck_corpus()

    if sample_by == "rhyme":
        # Filter for explicit rhyme annotations
        df_corpus = df_corpus[df_corpus.rhyme.isin({"y", "n"})]
        df = sample_chadwyck_corpus(df_corpus, sample_by="rhyme", **kwargs)

    elif sample_by == "period":
        df = sample_chadwyck_corpus(df_corpus, sample_by="period", **kwargs)

    elif sample_by == "period_subcorpus":
        df = sample_chadwyck_corpus(
            df_corpus, sample_by=["period", "subcorpus"], **kwargs
        )

    elif sample_by == "sonnet_period":
        # Filter for sonnets based on genre metadata and 14-line poems
        df_sonnets = df_corpus[
            (df_corpus.genre.str.contains("sonnet", case=False, na=False))
            | (df_corpus.title.str.contains("sonnet", case=False, na=False))
        ]
        # Further filter by 14 lines (traditional sonnet length)
        df_sonnets = df_sonnets[df_sonnets.num_lines == 14]
        kwargs['max_sample_n'] = 154
        df = sample_chadwyck_corpus(df_sonnets, sample_by="period", **kwargs)

    else:
        sample_by = sample_by.split("_")
        df = sample_chadwyck_corpus(df_corpus, sample_by=sample_by, **kwargs)

    return df


def get_chadwyck_corpus_sampled_by_replicated(
    sample_by,
    force=False,
    display=False,
    verbose=False,
    as_in_paper=True,
    as_replicated=False,
    as_regenerated=False,
    **kwargs,
) -> pd.DataFrame:
    """Load or generate a stratified sample with disk caching.

    Loads a pre-generated stratified sample from disk if available, otherwise
    generates a new sample and caches it. This ensures efficient reuse of
    expensive sampling operations.

    Parameters
    ----------
    sample_by : str
        Sampling criteria ('rhyme', 'period', 'period_subcorpus', 'sonnet_period').
    force : bool, default=False
        If True, regenerate the sample even if a cached version exists.
    display : bool, default=False
        If True, display summary tables for certain sample types.
    verbose : bool, default=False
        If True, print progress information.
    as_in_paper : bool, default=True
        If True, use precomputed sample from paper.
    as_replicated : bool, default=False
        If True, use replicated sample.
    as_regenerated : bool, default=False
        If True, use regenrated sample.
    **kwargs: Additional arguments passed to gen_chadwyck_corpus_sampled_by

    Returns
    -------
    pd.DataFrame
        DataFrame containing the stratified sample.

    Calls
    -----
    - gen_chadwyck_corpus_sampled_by(sample_by, display=display) [if generating new sample]
    - save_sample(odf, path, overwrite=True) [if saving generated sample]
    - pd.read_csv(path).set_index('id').sort_values('id_hash') [if loading cached sample]
    - get_period_subcorpus_table(odf, return_display=True) [if display=True for period_subcorpus]
    - display(img) [if display=True and IPython available]
    """
    if sample_by not in ["rhyme", "period", "period_subcorpus", "sonnet_period"]:
        raise ValueError(
            f"Invalid sample_by: {sample_by}. Must be one of ['rhyme', 'period', 'period_subcorpus', 'sonnet_period']"
        )

    

    sort_id_hash = not as_replicated
    if verbose:
        print(f'-' * 40)
        print(f'* Getting corpus sampled by{sample_by}')
        print(f'* as_in_paper: {as_in_paper}')
        print(f'* as_replicated: {as_replicated}')
        print(f'* as_regenerated: {as_regenerated}')
        print(f'* sort_id_hash: {sort_id_hash}')

    # Map sample_by to the appropriate data name
    data_name_map = {
        "rhyme": DATA_NAME_CORPUS_SAMPLE_BY_RHYME,
        "period": DATA_NAME_CORPUS_SAMPLE_BY_PERIOD,
        "period_subcorpus": DATA_NAME_CORPUS_SAMPLE_BY_PERIOD_SUBCORPUS,
        "sonnet_period": DATA_NAME_CORPUS_SAMPLE_BY_SONNET_PERIOD,
    }

    data_name = data_name_map[sample_by]
    path = get_path(
        data_name,
        as_in_paper=as_in_paper,
        as_replicated=as_replicated,
        as_regenerated=as_regenerated,
    )

    if force or not os.path.exists(path):
        print(f"* Generating {sample_by} sample")
        odf = gen_chadwyck_corpus_sampled_by(
            sample_by,
            display=display,
            sort_id_hash=sort_id_hash,
            verbose=verbose,
            **kwargs,
        )
        if len(odf):
            save_sample(odf, path, overwrite=True)
    else:
        print(f"* Loading {sample_by} sample from {path}")
        odf = pd.read_csv(path).set_index("id")
        if sort_id_hash:
            odf = odf.sort_values("id_hash")

    # # Handle display for period_subcorpus if not already handled during generation
    # if display and sample_by == 'period_subcorpus' and os.path.exists(path):
    #     try:
    #         from IPython.display import display
    #         img = get_period_subcorpus_table(odf, return_display=True)
    #         display(img)
    #     except (NameError, ImportError):
    #         print(f'* Warning: Could not display image')

    return odf
