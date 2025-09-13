from . import *

## PREVIOUS DATA


def preprocess_antoniak_et_al_memorization_data(
    data_fldr: str = PATH_ANTONIAK_ET_AL_DIR,
    out_path: str = PATH_ANTONIAK_ET_AL_CSV,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:

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
    """Load Antoniak et al. public-domain poems and closed/open memorization flags.

    Returns
    - DataFrame indexed by poem id with columns: txt, found_closed (bool), found_open (bool), and metadata.
    Also writes a compressed CSV copy under `data/antoniak_et_al_memorization_results.csv.gz` in this repo.
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
    df_smpl = get_genai_rhyme_completions_as_in_paper(
        by_line=True, filter_recognized=False
    )
    return get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True
    )


def get_memorized_poems_in_completions_as_replicated(
    threshold: int = 95, verbose: bool = True, overwrite: bool = False
):
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
    df_smpl = get_genai_rhyme_completions_as_replicated(
        by_line=True, filter_recognized=False
    )
    return get_memorized_poems_in_completions(
        df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True
    )


def get_memorized_poems_in_completions(
    df_smpl, threshold: int = 95, verbose: bool = True, return_unmemorized=False
):
    odf = df_smpl.copy().reset_index()
    odf = odf.drop_duplicates(subset="id_human")

    odf["found"] = odf.line_sim > threshold
    odf["found_source"] = "closed"
    odf["found_corpus"] = "chadwyck"
    # if not return_unmemorized:
    #     poems = [
    #         (g, gdf)
    #         for g, gdf in df_smpl.groupby("id")
    #         if gdf.line_sim.max() > threshold
    #     ]
    # else:
    #     poems = [
    #         (g, gdf)
    #         for g, gdf in df_smpl.groupby("id")
    #         if gdf.line_sim.max() <= threshold
    #     ]

    # odf = pd.concat([gdf for g, gdf in poems]) if len(poems) else pd.DataFrame()
    return odf.rename(
        columns={
            "id_human": "id",
            "id": "id_gen",
            "poem_title": "title",
        }
    ).set_index("id")


## DOLMA


def get_memorized_poems_in_dolma(*args, verbose: bool = True, **kwargs):
    ## Dolma no longer accessible
    return pd.read_pickle(PATH_RAW_DOLMA_PKL)


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


def get_all_memorization_data(overwrite: bool = False, verbose: bool = True):
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
    df_mem_chadwyck_open = get_poems_found_in_dolma()

    df_mem = pd.concat(
        [df_mem_antoniak, df_mem_chadwyck_closed, df_mem_chadwyck_open]
    ).fillna("")

    shared_cols = [
        col
        for col in df_mem.columns
        if col in df_mem_antoniak.columns
        and col in df_mem_chadwyck_closed.columns
        and col in df_mem_chadwyck_open.columns
        and col not in ["found_source", "found_corpus"]
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


def get_poems_found_in_dolma():
    return (
        pd.read_pickle(PATH_RAW_DOLMA_PKL)
        .assign(found_source="open", found_corpus="chadwyck")
        .set_index("id")
    )
