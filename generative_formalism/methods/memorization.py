from .. import *
from generative_formalism.rhyme.rhyme_completions import (
    get_genai_rhyme_completions_as_in_paper,
    get_genai_rhyme_completions_as_replicated,
    filter_recognized_completions,
)
from generative_formalism.rhyme.rhyme_measurement import (
    get_rhyme_for_sample,
)
from generative_formalism.corpus.corpus import get_chadwyck_corpus_metadata



def get_memorized_poems_in_completions_as_in_paper(threshold: int = 95, verbose: bool = True):
    df_smpl = get_genai_rhyme_completions_as_in_paper(by_line=True, filter_recognized=False)
    return get_memorized_poems_in_completions(df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=False)

def get_unmemorized_poems_in_completions_as_in_paper(threshold: int = 95, verbose: bool = True):
    df_smpl = get_genai_rhyme_completions_as_in_paper(by_line=True, filter_recognized=False)
    return get_memorized_poems_in_completions(df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True)


def get_memorized_poems_in_completions_as_replicated(threshold: int = 95, verbose: bool = True):
    df_smpl = get_genai_rhyme_completions_as_replicated(by_line=True, filter_recognized=False)
    return get_memorized_poems_in_completions(df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=False)

def get_unmemorized_poems_in_completions_as_replicated(threshold: int = 95, verbose: bool = True):
    df_smpl = get_genai_rhyme_completions_as_replicated(by_line=True, filter_recognized=False)
    return get_memorized_poems_in_completions(df_smpl, threshold=threshold, verbose=verbose, return_unmemorized=True)




def get_memorized_poems_in_completions(df_smpl, threshold: int = 95, verbose: bool = True, return_unmemorized = False):
    if not return_unmemorized:
        poems = [(g,gdf) for g, gdf in df_smpl.groupby('id') if gdf.line_sim.max() > threshold]
    else:
        poems = [(g,gdf) for g, gdf in df_smpl.groupby('id') if gdf.line_sim.max() <= threshold]
    
    odf = pd.concat([gdf for g, gdf in poems]) if len(poems) else pd.DataFrame()
    return odf





def find_poem_in_dolma(
    poem_id: str,
    df_lines: pd.DataFrame,
    es_config_path: str,
    index: str = "docs_v1.7_2024-06-04",
    stash: 'HashStash' = None,
):
    """Count Dolma docs containing each of the first-N lines for a poem.

    Expects df_lines to contain line-by-line rows for a single poem id including
    the original lines (`line_real`) and `line_num`, with `_first_n_lines` column
    present to specify how many to search.
    """
    try:
        from wimbd.es import es_init, count_documents_containing_phrases
    except Exception:
        raise RuntimeError("wimbd is required for Dolma search. Install and configure ES.")

    if stash is None:
        stash = HashStash(os.path.join(PATH_STASH, "find_in_dolma"), engine="pairtree", compress=False, b64=False)

    stash_key = {"poem_id": poem_id, "index": index}
    if stash_key in stash:
        return stash[stash_key]

    es = es_init(os.path.expanduser(es_config_path))
    dfg = df_lines[df_lines["id"] == poem_id].copy()
    n = int(dfg.iloc[0]["first_n_lines"]) if "first_n_lines" in dfg.columns else int(dfg.iloc[0].get("_first_n_lines", 5))
    dfg = dfg[dfg["line_num"] <= n]
    query_lines = [ln.strip() for ln in dfg.sort_values("line_num").line_real]
    count = count_documents_containing_phrases(index, query_lines, es=es, all_phrases=True)
    result = {"id": poem_id, "lines": query_lines, "count": int(count)}
    stash[stash_key] = result
    return result


def get_dolma_hits_for_ids(
    poem_ids: list[str],
    df_lines: pd.DataFrame,
    es_config_path: str,
    index: str = "docs_v1.7_2024-06-04",
):
    """Batch Dolma search for many poem ids; returns a DataFrame with counts and found flag."""
    out = []
    for pid in tqdm(poem_ids, desc="* Searching Dolma"):
        try:
            res = find_poem_in_dolma(pid, df_lines=df_lines, es_config_path=es_config_path, index=index)
            out.append(res)
        except Exception:
            continue
    ddf = pd.DataFrame(out)
    if not ddf.empty:
        ddf["found"] = ddf["count"] > 0
    return ddf


def load_antoniak_memorization_data(poetry_eval_repo_dir: str = "../../poetry-eval") -> pd.DataFrame:
    """Load Antoniak et al. public-domain poems and closed/open memorization flags.

    Returns
    - DataFrame indexed by poem id with columns: txt, found_closed (bool), found_open (bool), and metadata.
    Also writes a compressed CSV copy under `data/antoniak_et_al_memorization_results.csv.gz` in this repo.
    """
    data_fldr = os.path.join(poetry_eval_repo_dir, "data")
    df_antoniak = pd.read_csv(os.path.join(data_fldr, "poetry-evaluation_public-domain-poems.csv"))
    df_antoniak["id"] = df_antoniak["poem_link"].apply(lambda x: "/".join(str(x).split("/")[-2:]))
    df_antoniak = df_antoniak.rename(columns={"poem_text": "txt"}).set_index("id")

    df_mem = pd.read_csv(os.path.join(data_fldr, "memorization_results.csv")).drop(columns=[c for c in ["Unnamed: 0"] if c in df_mem.columns], errors="ignore")
    df_mem = df_mem.rename(columns={"poem": "id", "prediction": "found_closed"})
    df_mem["found_closed"] = df_mem["found_closed"].astype(bool)
    df_mem_antoniak = df_antoniak.merge(df_mem, on="id").set_index("id")

    # open-model training data hits (Walsh et al. wimbd outputs placed in the same folder)
    df_mem_antoniak["found_open"] = False
    for fn in os.listdir(data_fldr):
        if "wimbd" in fn and fn.endswith(".csv"):
            try:
                fndf = pd.read_csv(os.path.join(data_fldr, fn))
            except Exception:
                continue
            for poem_id in set(fndf.get("poem_id", [])):
                if poem_id in df_mem_antoniak.index:
                    df_mem_antoniak.loc[poem_id, "found_open"] = True

    out_path = os.path.join(PATH_DATA, "antoniak_et_al_memorization_results.csv.gz")
    df_mem_antoniak.to_csv(out_path)
    return df_mem_antoniak


def annotate_dolma_hits_with_corpus(df_dolma_hits: pd.DataFrame, period_by: int = CORPUS_PERIOD_BY) -> pd.DataFrame:
    """Attach Chadwyck metadata to Dolma hit rows (expects column `id`)."""
    if df_dolma_hits.empty:
        return df_dolma_hits
    df_meta = get_chadwyck_corpus_metadata(period_by=period_by).reset_index()
    return df_dolma_hits.merge(df_meta, on="id", how="left")

