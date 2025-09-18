from . import *


@cache
def get_genai_rhyme_completions(
    *args,
    as_in_paper=True,
    as_replicated=False,
    by_line=True,
    verbose=DEFAULT_VERBOSE,
    min_num_lines=10,
    threshold=95,
    filter_recognized=True,
    path=None,
    line_gen=True,
    **kwargs,
):
    """Unified accessor for genai rhyme completions with source selection.

    Select between data-as-in-paper, replicated (stash-based), or regenerated
    outputs using flags consistent with corpus/sample accessors.

    Exactly one of as_in_paper, as_replicated should be True.
    """
    from ..corpus.corpus import get_chadwyck_corpus_metadata

    flags_true = sum([bool(as_in_paper), bool(as_replicated)])
    if flags_true != 1:
        raise ValueError("Specify exactly one of as_in_paper, as_replicated")

    if as_replicated:
        df_preprocessed = get_stash_df_completions(verbose=verbose)
    else:
        df_preprocessed = preprocess_legacy_genai_rhyme_completions(path=path, overwrite=False)
        
    df_postprocessed = postprocess_genai_rhyme_completions(
        df_preprocessed,
        by_line=by_line,
        verbose=verbose,
        min_num_lines=min_num_lines,
        threshold=threshold,
        filter_recognized=filter_recognized,
        line_gen=line_gen,
    )

    df_meta = get_chadwyck_corpus_metadata()
    df_postprocessed = df_postprocessed.reset_index().merge(df_meta, left_on='id_human', right_on='id', suffixes=['', '_from_meta'], how='left')

    df_postprocessed._data_name = f'genai_rhyme_completions_{"real" if not line_gen else "gen"}'
    df_postprocessed._sample_by = ''  # N/A for this data
    df_postprocessed._as_in_paper = as_in_paper
    df_postprocessed._as_replicated = as_replicated
    return df_postprocessed


def get_rhyme_for_completed_poems_by(
    *args,
    as_in_paper=True,
    as_replicated=False,
    verbose=DEFAULT_VERBOSE,
    **kwargs,
):
    """Unified accessor for rhyme analysis over completed poems.

    Mirrors the "by" pattern used in corpus/sample to select data source.
    """
    flags_true = sum([bool(as_in_paper), bool(as_replicated)])
    if flags_true != 1:
        raise ValueError("Specify exactly one of as_in_paper, as_replicated")

    if as_replicated:
        return get_rhyme_for_completed_poems_as_replicated(
            *args,
            verbose=verbose,
            **kwargs,
        )

    # Default: as_in_paper
    return get_rhyme_for_completed_poems_as_in_paper(
        *args,
        verbose=verbose,
        **kwargs,
    )





def preprocess_legacy_genai_rhyme_completions(
    path=None, overwrite=False, first_n_lines=FIRST_N_LINES, verbose=DEFAULT_VERBOSE
):
    """Preprocess legacy generative AI rhyme completions from raw pickle files.

    This function loads and preprocesses legacy rhyme completion data from multiple
    pickle files (v3-v7), combines them, deduplicates, and saves to CSV format.
    It also generates unique IDs for generated poems and provides statistics
    about the dataset.

    Parameters
    ----------
    path : str, optional
        Path to save the processed CSV file. If None, uses default path
        from get_path(DATA_NAME_GENAI_RHYME_COMPLETIONS).
    overwrite : bool, default=False
        Whether to overwrite existing processed data.
    first_n_lines : int, default=FIRST_N_LINES
        Number of first lines from original poems to consider.
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with MultiIndex containing completion data,
        indexed by GENAI_RHYME_COMPLETIONS_INDEX.

    Calls
    -----
    - get_path(DATA_NAME_GENAI_RHYME_COMPLETIONS) [if path is None]
    - pd.read_csv(path) [to load existing processed data]
    - pd.read_pickle(f"{PATH_RAWDATA}/data.output.gen_poems.v*.pkl") [to load raw data]
    - get_id_hash_str(...) [to generate unique IDs]
    - df.to_csv(path) [to save processed data]
    """
    # Set default path if not provided
    if path is None:
        path = get_path(DATA_NAME_GENAI_RHYME_COMPLETIONS)

    global PREPROCESSED_LEGACY_COMPLETION_DATA

    if not overwrite and PREPROCESSED_LEGACY_COMPLETION_DATA is not None:
        printm(f"* Loading genai rhyme completions from `{nice_path(path)}`")
        return PREPROCESSED_LEGACY_COMPLETION_DATA

    if not overwrite and os.path.exists(path):
        if verbose: printm(
            f"* Loading genai rhyme completions from `{nice_path(path)}`"
        )
        odf = pd.read_csv(path).fillna("")
        odf = odf.set_index([i for i in GENAI_RHYME_COMPLETIONS_INDEX if i in odf.columns])

    else:
        if verbose: printm(f"* Preprocessing genai rhyme completions")
        index = "_id	_first_n_lines	_model	_say_poem	_version	_timestamp".split()
        df3 = (
            pd.read_pickle(f"{PATH_RAWDATA}/rhyme_completions/data.output.gen_poems.v3.pkl.gz")
            .assign(_say_poem=True)
            .reset_index()
            .set_index(index)
        )
        df4 = (
            pd.read_pickle(f"{PATH_RAWDATA}/rhyme_completions/data.output.gen_poems.v4.pkl.gz")
            .assign(_say_poem=True)
            .reset_index()
            .set_index(index)
        )
        df5 = (
            pd.read_pickle(f"{PATH_RAWDATA}/rhyme_completions/data.output.gen_poems.v5.pkl.gz")
            .assign(_say_poem=True)
            .reset_index()
            .set_index(index)
        )
        df6 = pd.read_pickle(f"{PATH_RAWDATA}/rhyme_completions/data.output.gen_poems.v6.pkl.gz")
        df7 = pd.read_pickle(f"{PATH_RAWDATA}/rhyme_completions/data.output.gen_poems.v7.pkl.gz")

        df = pd.concat(
            [
                df3,  # .query('_model=="ollama/llama3.1:8b"'),
                df4,  # .query('_model=="ollama/llama3.1:8b-text-q4_K_M" | _model=="ollama/mistral" | _model=="ollama/mistral:text"'),
                df5,  # .query('_model=="ollama/llama3.1:8b"'),
                df6,
                df7,
            ]
        ).reset_index()
        df.columns = [x[1:] if x and x[0] == "_" else x for x in df.columns]

        def get_id_gen(gdf):
            model = gdf.model.iloc[0]
            id = gdf.id.iloc[0]
            version = str(gdf.version.iloc[0])
            timestamp = str(gdf.timestamp.iloc[0])
            txt = gdf.line_gen.str.cat(sep="\n").strip()
            return get_id_hash_str("__".join([model, id, version, timestamp, txt]))

        df = pd.concat(
            [
                gdf.assign(id_gen=get_id_gen(gdf))
                for g, gdf in df.groupby(["model", "id", "version"])
            ]
        )

        df = df[df.say_poem]
        df.drop(columns=["say_poem"], inplace=True)
        df = df[~df.model.str.contains("poetry")]
        df = df.query(f"first_n_lines == {first_n_lines}")
        # df = df.drop(columns=['first_n_lines', 'version', 'timestamp'])

        # Convert timestamp to date string
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date.astype(str)

        odf = df.rename(columns={"id": "id_human", "id_gen": "id"})
        odf.drop(columns=["timestamp"], inplace=True)
        odf = odf.drop_duplicates(["model", "id_human", "stanza_num", "line_num"])

        odf = odf.set_index(GENAI_RHYME_COMPLETIONS_INDEX)

        if verbose: printm(
            f"* Saving genai rhyme completions to {nice_path(path)}"
        )
        odf.to_csv(path)

    PREPROCESSED_LEGACY_COMPLETION_DATA = odf

    # if verbose:
        # human_ids = odf.reset_index().id_human.unique()
        # printm(f"* Found {len(human_ids)} unique human poems for input to models")
        # gen_ids = odf.reset_index().id.unique()
        # printm(f"* Found {len(gen_ids)} unique generated poems")

        # printm("* Distribution of input poem lengths")
        # describe_numeric(
        #     pd.Series([len(gdf) for g, gdf in odf.groupby("id_human")], name="num_lines")
        # )

        # printm("* Distribution of output poem lengths")
        # describe_numeric(
        #     pd.Series([len(gdf) for g, gdf in odf.groupby("id")], name="num_lines"),
        #     fixed_range=(MIN_NUM_LINES, MAX_NUM_LINES),
        # )

    ## Reattach metadata
    return odf


def postprocess_genai_rhyme_completions(
    odf,
    threshold=95,
    filter_recognized=True,
    min_num_lines=10,
    verbose=DEFAULT_VERBOSE,
    keep_first_n_lines=False,
    by_line=False,
    line_gen=True,
):
    """Postprocess generative AI rhyme completions data.

    This function cleans and filters the rhyme completions data by filling NaN values
    and optionally filtering out recognized/memorized completions based on similarity.

    Args:
        odf (pd.DataFrame): Input DataFrame containing rhyme completion data.
        threshold (int, optional): Similarity threshold for filtering recognized completions.
            Defaults to 95.
        filter_recognized (bool, optional): Whether to filter out recognized completions.
            Defaults to True.
        min_num_lines (int, optional): Minimum number of lines for poem format conversion.
            Defaults to 10. Currently unused.

    Returns:
        pd.DataFrame: Postprocessed DataFrame with cleaned and filtered data.
    """
    odf = reset_index(odf).fillna("").rename(columns={'temp':'temperature'})

    # If we have line-level data, optionally filter and either return by-line or convert to poem format
    has_line_cols = all(col in odf.columns for col in ["line_real", "line_gen"])


    if has_line_cols:
        # Only attempt filtering if needed columns are present
        if all(col in odf.columns for col in ["line_real", "line_gen"]):
            # Test for line similarity
            tqdm.pandas(desc='Computing line similarity')
            odf["line_sim"] = odf.progress_apply(
                lambda row: (
                    fuzz.ratio(row.line_real.strip(), row.line_gen.strip())
                    if isinstance(row.line_gen, str)
                    and isinstance(row.line_real, str)
                    and row.line_gen
                    and row.line_real
                    else np.nan
                ),
                axis=1,
            )
        
        if filter_recognized:
            odf = filter_recognized_completions(odf, threshold=threshold, verbose=verbose)

        if by_line:
            return odf.rename(columns={'id':'id_human'})

        poems_df = to_poem_txt_format(
            odf,
            keep_first_n_lines=keep_first_n_lines,
            verbose=verbose,
            filter_recognized=False,  # already filtered above if requested
            threshold=threshold,
            line_gen=line_gen
        )

        # Compute hashes if possible
        if all(col in poems_df.columns for col in ["model", "temperature", "prompt", "txt"]):
            poems_df["id"] = [
                get_id_hash_str(f"{model}__{temp:.4f}__{prompt}__{txt}")
                for model, temp, prompt, txt in zip(
                    poems_df.model, poems_df.temperature, poems_df.prompt, poems_df.txt
                )
            ]
        # Always ensure id_hash exists
        if "id" in poems_df.columns:
            poems_df["id_hash"] = [get_id_hash(id) for id in poems_df.id]
        return poems_df

    # Otherwise, assume stash rows (prompt/response) and convert directly to poem-level
    poems_df = parse_stash_rows_to_poems(
        odf,
        keep_first_n_lines=keep_first_n_lines,
        verbose=verbose,
    )
    return poems_df


# Filter out recognized completions
def filter_recognized_completions(df, threshold=95, groupby=COMPLETIONS_GROUPBY, verbose=DEFAULT_VERBOSE):
    """Filter out recognized/memorized completions based on line similarity.

    This function computes similarity scores between real and generated lines
    and filters out poems where the maximum similarity exceeds a threshold,
    indicating potential memorization. Currently only computes similarity
    scores without actual filtering.

    Args:
        df (pd.DataFrame): Input DataFrame containing completion data with
            'line_real' and 'line_gen' columns.
        threshold (int, optional): Similarity threshold above which completions
            are considered recognized/memorized. Defaults to 95.
        groupby (list, optional): Columns to group by when filtering.
            Defaults to COMPLETIONS_GROUPBY.
        verbose (bool, optional): Whether to print verbose output.
            Defaults to DEFAULT_VERBOSE.

    Returns:
        pd.DataFrame: DataFrame with added 'line_sim' column containing
            similarity scores between real and generated lines.

    Note:
        The actual filtering logic is commented out in the current implementation.
        This function only adds similarity scores to the DataFrame.
    """
    # Ensure required columns exist
    required_cols = {"line_real", "line_gen"}
    if not required_cols.issubset(set(df.columns)):
        if verbose:
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            printm(f"* Skipping similarity computation (missing columns: {missing})")
        if "line_sim" not in df.columns:
            df["line_sim"] = np.nan
        return df


    
    gby = groupby
    num1 = len(df.groupby(gby))
    grps = []
    for g, gdf in tqdm(df.groupby(gby), desc='Filtering out recognized completions', total=num1):
        if gdf.line_sim.max() < threshold:
            grps.append(gdf)
    if not grps:
        if verbose:
            printm(f"* Filtered out {num1} recognized poems (none passed threshold)")
        return df.iloc[0:0]
    df_safe = pd.concat(grps)
    
    if verbose:
        printm(f"* Filtered out {num1 - len(grps)} recognized poems")
    return df_safe



# To poem format


def to_poem_txt_format(df, keep_first_n_lines=False, verbose=DEFAULT_VERBOSE, filter_recognized=True, threshold=95, line_gen=True):
    """Convert line-by-line completion data to poem text format.

    This function takes a DataFrame with line-by-line completion data and
    converts it to a poem-centric format where each row represents a complete
    poem with concatenated text. It can optionally include the first N lines
    from the original human poem.

    Args:
        df (pd.DataFrame): Input DataFrame with line-by-line completion data,
            containing columns like 'id', 'model', 'line_real', 'line_gen', etc.
        keep_first_n_lines (bool, optional): Whether to include the first N lines
            from the original human poem at the beginning of each generated poem.
            Defaults to True.

    Returns:
        pd.DataFrame: DataFrame in poem text format with columns including:
            - id: Generated poem ID
            - id_human: Original human poem ID
            - id_hash: Hash of the generated poem ID
            - model: Model used for generation
            - txt: Complete poem text
            - num_lines: Number of lines in the poem
            - first_n_lines: Number of first lines kept from original
            - keep_first_n_lines: Boolean flag for keeping first lines
    """
    df = df.reset_index()
    if not 'id_human' in df.columns and 'id' in df.columns:
        df['id_human'] = df['id']

    line_col = 'line_gen' if line_gen else 'line_real'

    if filter_recognized:
        df = filter_recognized_completions(df, threshold=threshold, verbose=verbose)

    num_poems = df.id.nunique()
    if verbose:
        printm(
            f"* Converting to poem txt format"
            + (
                " (keeping first lines from original poem)"
                if keep_first_n_lines
                else " (not keeping first lines from original poem)"
            )
        )

    def get_num_lines(txt):
        return len([x for x in txt.split("\n") if x.strip()])

    def get_row(gdf):
        model = gdf.model.iloc[0]
        # id_gen = gdf.id.iloc[0]
        id_human = gdf.id_human.iloc[0] if "id_human" in gdf else None
        line_num = 0
        first_n_lines = gdf.iloc[0].first_n_lines

        if not keep_first_n_lines:
            lines = list(gdf[line_col][first_n_lines:])
        else:
            lines = list(gdf.line_real[:first_n_lines]) + list(
                gdf[line_col][first_n_lines:]
            )

        txt = "\n".join(lines)
        id_gen = get_id_hash_str(f"{model}__{txt}")
        id_hash = get_id_hash(id_gen)


        row_out = {
            "id": id_gen,
            "id_human": id_human,
            "id_hash": id_hash,
            "model": model,
            "txt": txt,
            "num_lines": len(lines),
            "first_n_lines": first_n_lines,
            "keep_first_n_lines": keep_first_n_lines,
        }
        if 'line_sim' in gdf:
            row_out['line_sim'] = gdf.line_sim.max()
        # Propagate known meta columns if present
        for meta_col in ["temperature", "prompt", "system_prompt", "date"]:
            if meta_col in gdf:
                row_out[meta_col] = gdf[meta_col].iloc[0]
        return row_out

    df = pd.concat([pd.DataFrame([get_row(gdf) for g, gdf in df.groupby("id")])])
    return df.set_index(
        [
            x
            for x in GENAI_RHYME_COMPLETIONS_INDEX + ["keep_first_n_lines"]
            if x in df.columns
        ]
    )


def parse_stash_rows_to_poems(df, keep_first_n_lines=False, verbose=DEFAULT_VERBOSE):
    """Convert stash rows with prompts/responses to poem-level DataFrame.

    Processes cached completion results from the hash stash, parsing the structured
    prompt/response format to extract complete poems. Combines original poem lines
    with AI-generated completions to create full poem texts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stash rows with columns: 'model', 'prompt',
        'system_prompt', 'temperature', 'response'.
    keep_first_n_lines : bool, default=True
        Whether to include the original first N lines in the final poem.
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to print progress information (currently unused).

    Returns
    -------
    pd.DataFrame
        DataFrame with poem-level data indexed by GENAI_RHYME_COMPLETIONS_INDEX,
        containing complete poems with metadata.

    Calls
    -----
    - get_id_hash_str(f"{model}__{temperature:.4f}__{prompt}__{txt}") [to generate stable IDs]
    - get_id_hash(poem_id) [to generate hash IDs]
    """
    import re as _re

    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        prompt = str(row.get("prompt", ""))
        system_prompt = str(row.get("system_prompt", ""))
        response = str(row.get("response", ""))
        model = row.get("model", "")
        try:
            temperature = float(row.get("temperature", 0.0))
        except Exception:
            temperature = 0.0

        # Parse number of lines
        m_total = _re.search(r"NUMBER OF LINES:\s*(\d+)", prompt)
        num_lines_declared = int(m_total.group(1)) if m_total else None

        # Parse first_n_lines from system prompt (preferred)
        m_fnl = _re.search(r"first\s+(\d+)\s+lines", system_prompt, flags=_re.IGNORECASE)
        first_n = int(m_fnl.group(1)) if m_fnl else None

        # Parse enumerated first lines from prompt
        first_lines = {}
        for ln in prompt.splitlines():
            if "\t" in ln:
                a, b = ln.split("\t", 1)
                a = a.strip()
                if a.isdigit():
                    first_lines[int(a)] = b
        if first_n is None and first_lines:
            first_n = max(first_lines.keys())
        if first_n is None:
            first_n = 0

        # Parse generated lines from response
        gen_lines = {}
        for ln in str(response).splitlines():
            if "\t" in ln:
                a, b = ln.split("\t", 1)
                a = a.strip()
                if a.isdigit():
                    gen_lines[int(a)] = b

        # Determine total lines
        if num_lines_declared is None:
            max_gen = max(gen_lines.keys(), default=0)
            num_lines_declared = max(max_gen, first_n)

        # Build poem text
        out_lines = []
        for i in range(1, int(num_lines_declared) + 1):
            if i <= first_n and keep_first_n_lines:
                out_lines.append(first_lines.get(i, ""))
            elif i > first_n:
                out_lines.append(gen_lines.get(i, ""))
            else:
                # i <= first_n but keep_first_n_lines is False -> skip
                continue

        txt = "\n".join(out_lines)

        # Stable ID based on model, temperature, prompt, and poem text
        try:
            poem_id = get_id_hash_str(f"{model}__{temperature:.4f}__{prompt}__{txt}")
        except Exception:
            poem_id = get_id_hash_str(f"{model}__{prompt}__{txt}")
        poem_id_hash = get_id_hash(poem_id)

        rows.append(
            {
                "id": poem_id,
                "id_hash": poem_id_hash,
                "model": model,
                "txt": txt,
                "num_lines": len(out_lines),
                "first_n_lines": first_n,
                "keep_first_n_lines": keep_first_n_lines,
                "temperature": temperature,
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        )

    out = pd.DataFrame(rows)
    # Drop duplicates by id if any
    if "id" in out.columns:
        out = out.drop_duplicates("id")
    return out.set_index(
        [
            x
            for x in GENAI_RHYME_COMPLETIONS_INDEX + ["keep_first_n_lines"]
            if x in out.columns
        ]
    )


#### GENERATING


def get_rhyme_completion_system_prompt(first_n_lines=2, say_poem=True):
    ttype = "poem" if say_poem else "text"
    system_prompt = f"""
The following is the first {first_n_lines} lines from a {ttype} given in the user prompt, whose true number of lines is stated there.

Complete the {ttype} – do this from memory if you know it; if not, imitate its style and theme for the same number of lines as in the original.

Return lines in tab-separated form, starting from line {first_n_lines+1} up to the stated number of lines:

    line#\tline

Do not return any other text besides these tab-separated lines.
""".strip()
    return system_prompt


def get_rhyme_completion_user_prompt(txt, first_n_lines=2, **meta):
    prompt = []
    for k in meta:
        prompt.append(f"{k.upper()}: {meta[k]}")

    num_lines = len([x for x in txt.split("\n") if x.strip()])
    prompt.append(f"NUMBER OF LINES: {num_lines}")
    prompt.append("")
    prompt.append(get_first_n_lines(txt, first_n_lines))
    return "\n".join(prompt)


def complete_poem(
    txt,
    first_n_lines=5,
    force=False,
    model="deepseek/deepseek-chat",
    verbose=DEFAULT_VERBOSE,
    use_system_prompt=True,
    say_poem=True,
    temperature=DEFAULT_TEMPERATURE,
    stash=STASH_GENAI_RHYME_COMPLETIONS,
    **meta,
):
    """Complete a poem using language models with streaming interface.

    This function takes the first N lines of a poem and generates the remaining lines
    using the specified language model. It uses the streaming LLM interface from llms.py
    for efficient text generation with caching support.

    Args:
        txt (str): The complete original poem text to use as input
        first_n_lines (int, optional): Number of initial lines to provide as context.
            Defaults to 5.
        force (bool, optional): Whether to bypass cache and force new generation.
            Defaults to False.
        model (str, optional): The model identifier to use for generation.
            Defaults to 'deepseek/deepseek-chat'.
        verbose (bool, optional): Whether to print verbose output during generation.
            Defaults to False.
        use_system_prompt (bool, optional): Whether to use a system prompt for the model.
            Automatically disabled for text-only models. Defaults to True.
        say_poem (bool, optional): Whether to instruct the model to generate a "poem"
            vs generic "text". Affects system prompt. Defaults to True.
        temperature (float, optional): Sampling temperature for text generation (0.0-1.0).
            Defaults to DEFAULT_TEMPERATURE.
        stash (BaseHashStash, optional): Cache storage backend for results.
            Defaults to STASH_GENAI if None.
        **meta: Additional metadata to include in the user prompt (e.g., AUTHOR, TITLE).

    Returns:
        pd.DataFrame: DataFrame containing line-by-line completion data with columns:
            - stanza_num: Stanza number
            - line_num: Line number within the poem
            - line_real: Original line from input poem
            - line_gen: Generated line from the model
            Returns empty DataFrame if generation fails or line count mismatch occurs.

    Note:
        The function expects the model to return lines in tab-separated format:
        "line_number\tline_text". If the generated response doesn't match the
        expected number of lines, an empty DataFrame is returned.
    """
    all_lines = get_first_n_lines(txt, None)
    num_lines = int(all_lines.split("\n")[-1].split("\t")[0])
    user_prompt = get_rhyme_completion_user_prompt(txt, first_n_lines=first_n_lines)
    system_prompt = get_rhyme_completion_system_prompt(
        first_n_lines=first_n_lines, say_poem=say_poem
    )
    # if verbose: print(user_prompt)
    if model.endswith(":text") or "-text-" in model:
        use_system_prompt = False

    # Use default stash if none provided
    if stash is None:
        stash = STASH_GENAI

    stash_key = {
        "model": model,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        **meta,
    }
    if not force and stash is not None and stash_key in stash:
        response = stash[stash_key]
    else:
        response = generate_text(
            model=model,
            prompt=user_prompt,
            system_prompt=system_prompt if use_system_prompt else None,
            temperature=temperature,
            verbose=verbose,
            force=force,
            stash=stash,
        )
        stash[stash_key] = response
    response_id = get_id_hash_str(
        "__".join([model, user_prompt, system_prompt, str(temperature), response])
    )

    # newlnd={}
    newlnd = {str(i): "" for i in range(first_n_lines + 1, num_lines + 1)}
    for ln in response.strip().split("\n"):
        if "\t" in ln:
            a, b = ln.split("\t", 1)
            if a in newlnd and not newlnd[a]:
                newlnd[a] = b

    need_num_lines = num_lines - first_n_lines
    got_num_lines = len([v for v in newlnd.values() if v.strip()])
    if got_num_lines != need_num_lines:
        # logger.warning(f'Line length mismatch: {len(o)} output to needed {num_lines - first_n_lines}')
        return pd.DataFrame()

    oldlnd = {}
    snum = 1
    for ln in all_lines.split("\n"):
        if not ln.strip():
            snum += 1
        else:
            lnum, line = ln.split("\t", 1)
            line_gen = newlnd.get(lnum, "")
            oldlnd[lnum] = {
                "id": response_id,
                "stanza_num": snum,
                "line_num": int(lnum),
                "line_real": line,
                "line_gen": line_gen,
                "line_sim": fuzz.ratio(line, line_gen) if line and line_gen else "",
            }

    return pd.DataFrame(oldlnd.values()).set_index(["id", "stanza_num", "line_num"])


def get_all_rhyme_completions(*args, by_line=False, verbose=DEFAULT_VERBOSE, **kwargs):
    """Load and combine both original and replicated rhyme completion datasets.

    Retrieves rhyme completion data from both the original paper dataset and
    the newly generated replicated dataset, then combines them into a single
    DataFrame for comprehensive analysis.

    Parameters
    ----------
    *args
        Positional arguments passed to both get_genai_rhyme_completions()
        and get_genai_rhyme_completions_as_replicated().
    by_line : bool, default=False
        If True, returns line-by-line data. If False, converts to poem text format.
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to print progress information.
    **kwargs
        Additional keyword arguments passed to both functions.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing rhyme completion data from both sources.

    Calls
    -----
    - get_genai_rhyme_completions(*args, by_line=by_line, verbose=verbose, **kwargs)
    - get_genai_rhyme_completions_as_replicated(*args, by_line=by_line, verbose=verbose, **kwargs)
    - pd.concat([df1, df2]) [to combine the datasets]
    """
    df1 = get_genai_rhyme_completions_by(
        *args, as_in_paper=True, as_replicated=False,
        by_line=by_line, verbose=verbose, **kwargs
    )
    df2 = get_genai_rhyme_completions_by(
        *args, as_in_paper=False, as_replicated=True,
        by_line=by_line, verbose=verbose, **kwargs
    )
    if verbose:
        printm(f"* Loaded {len(df1)} existing completions")
        printm(f"* Loaded {len(df2)} replicated completions")
    return pd.concat([df1, df2])


def generate_more_completions(
    n=3,
    df_sofar=None,
    models=MODEL_LIST,
    first_n_lines=FIRST_N_LINES,
    temperatures=[DEFAULT_TEMPERATURE],
    verbose=DEFAULT_VERBOSE,
    force=REPLICATE_OVERWRITE,
    max_n_combo=None,
    source_poems_sample="period",
):
    """
    Generate more poem completions using various models and source poems.

    This function generates additional poem completions by sampling from available models
    and source poems from the Chadwyck corpus, with intelligent prioritization of
    underrepresented combinations to ensure balanced data collection across different
    model-poem pairs.

    Parameters
    ----------
    n : int, default=3
        Number of completions to generate.
    df_sofar : pd.DataFrame, optional
        Existing dataframe of generated completions to build upon.
        If None, loads all existing rhyme completions.
    models : list, default=MODEL_LIST
        List of model identifiers to use for generation.
    first_n_lines : int, default=FIRST_N_LINES
        Number of first lines to provide as context for completion.
    temperatures : list, default=[DEFAULT_TEMPERATURE]
        List of temperature values for generation.
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to print progress and status information.
    force : bool, default=REPLICATE_OVERWRITE
        Whether to force regeneration even if cached results exist.
    max_n_combo : int, optional
        Maximum number of entries allowed per model-poem combination.
        If provided, overrepresented combinations will be excluded from selection.
    source_poems_sample : str, default='period'
        Which corpus sample to use for source poems.
        Options: 'period', 'rhyme', 'period_subcorpus'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing generated completion data with columns:
        id, model, id_human, first_n_lines, temperature, response.

    Calls
    -----
    - get_chadwyck_corpus_sampled_by_*() [to load source poems based on source_poems_sample]
    - complete_poem() [to generate individual completions]
    - random.choices() [for weighted random selection of models/poems]
    """

    from ..corpus.sample import get_chadwyck_corpus_sampled_by

    # Load source poems from corpus
    source_poems = get_chadwyck_corpus_sampled_by(source_poems_sample)

    if source_poems.empty:
        if verbose:
            print(f"!!! No source poems found for sample type: {source_poems_sample}")
        return pd.DataFrame()

    # Get existing completions data
    if df_sofar is None:
        try:
            df = get_genai_rhyme_completions(by_line=True)
            if verbose:
                printm(f"* Loaded {len(df)} existing completions")
        except:
            df = pd.DataFrame()
            if verbose:
                printm("* No existing completions found, starting fresh")
    else:
        df = df_sofar

    # Create copies to modify
    models = list(models)
    source_poem_ids = list(source_poems.index)
    first_n_lines_options = [first_n_lines]

    # Filter out combinations that exceed max_n_combo
    if (
        not df.empty
        and max_n_combo is not None
        and "model" in df.columns
        and "id_human" in df.columns
    ):
        combo_counts = df.groupby(["model", "id_human", "first_n_lines"]).size()
        overrepresented_combos = combo_counts[combo_counts >= max_n_combo].index

        # Filter available options
        valid_models = set(models)
        valid_poem_ids = set(source_poem_ids)
        valid_first_n_lines = set(first_n_lines_options)

        for model, poem_id, fnl in overrepresented_combos:
            if (model, poem_id, fnl) in [
                (m, p, f)
                for m in models
                for p in source_poem_ids
                for f in first_n_lines_options
            ]:
                # This specific combination is overrepresented
                continue

        if verbose and len(overrepresented_combos) > 0:
            printm(
                f"  * Found {len(overrepresented_combos)} overrepresented model-poem-first_n_lines combinations"
            )

    # Set up probability weights if we have existing data
    if not df.empty and "model" in df.columns and "id_human" in df.columns:
        model_counts = df["model"].value_counts().reindex(models, fill_value=0)
        poem_counts = (
            df["id_human"].value_counts().reindex(source_poem_ids, fill_value=0)
        )

        # Inverse probability weighting
        model_weights = 1 / (model_counts.reindex(models, fill_value=0) + 1)
        poem_weights = 1 / (poem_counts.reindex(source_poem_ids, fill_value=0) + 1)
    else:
        model_weights = None
        poem_weights = None

    iterr = tqdm(total=n, position=0)
    bad_models = set()
    outld = []

    for _ in range(n):
        good_models = [m for m in models if m not in bad_models]
        if not good_models:
            if verbose:
                print("!!! No more models available.")
            break
        if not source_poem_ids:
            if verbose:
                print("!!! No more source poems available.")
            break

        # Use weighted random choice if weights are available
        if model_weights is not None and not model_weights.empty:
            current_model_weights = model_weights.reindex(good_models).dropna()
            if not current_model_weights.empty:
                model = random.choices(
                    population=current_model_weights.index.tolist(),
                    weights=current_model_weights.values.tolist(),
                    k=1,
                )[0]
            else:
                model = random.choice(good_models)
        else:
            model = random.choice(good_models)

        if poem_weights is not None and not poem_weights.empty:
            poem_id = random.choices(
                population=poem_weights.index.tolist(),
                weights=poem_weights.values.tolist(),
                k=1,
            )[0]
        else:
            poem_id = random.choice(source_poem_ids)

        first_n_lines = random.choice(first_n_lines_options)

        temperature = round(
            (random.choice(temperatures) if temperatures else random.uniform(0.0, 1.0)),
            4,
        )

        # Count current usage for display
        session_model_count = sum(1 for item in outld if item["model"] == model)
        session_poem_count = sum(1 for item in outld if item["id_human"] == poem_id)
        session_combo_count = sum(
            1
            for item in outld
            if item["model"] == model
            and item["id_human"] == poem_id
            and item["first_n_lines"] == first_n_lines
        )

        # Count from overall stash (including historical data)
        if not df.empty:
            stash_model_count = (
                len(df[df["model"] == model]) if "model" in df.columns else 0
            )
            stash_poem_count = (
                len(df[df["id_human"] == poem_id]) if "id_human" in df.columns else 0
            )
            stash_combo_count = (
                len(
                    df[
                        (df["model"] == model)
                        & (df["id_human"] == poem_id)
                        & (df["first_n_lines"] == first_n_lines)
                    ]
                )
                if all(
                    col in df.columns for col in ["model", "id_human", "first_n_lines"]
                )
                else 0
            )
        else:
            stash_model_count = stash_poem_count = stash_combo_count = 0

        # Check if this combination would exceed max_n_combo
        total_combo_count = stash_combo_count + session_combo_count
        if max_n_combo is not None and total_combo_count >= max_n_combo:
            if verbose:
                print(
                    f"!!! Skipping {model} + {poem_id} + first_n_lines={first_n_lines} (would exceed max_n_combo={max_n_combo})"
                )
            continue

        # Get source poem text
        source_poem = source_poems.loc[poem_id]
        poem_txt = source_poem["txt"]
        poem_meta = {
            # "AUTHOR": source_poem.get("author", ""),
            # "TITLE": source_poem.get("title", ""),
            # "YEAR": source_poem.get("year", ""),
            "id_human": poem_id,
        }

        iterr.set_description(
            f">>> {model} (n_model={stash_model_count + session_model_count:0,}, n_poem={stash_poem_count + session_poem_count:0,}, n_combo={stash_combo_count + session_combo_count:0,}): poem {poem_id} (first_{first_n_lines})"
        )
        try:
            if verbose:
                printm("----")
            completion_result = complete_poem(
                txt=poem_txt,
                first_n_lines=first_n_lines,
                model=model,
                temperature=temperature,
                verbose=verbose,
                force=force,
                **poem_meta,
            ).reset_index()

            if not completion_result.empty:
                outld.append(
                    {
                        "id": completion_result.id.iloc[0],
                        "model": model,
                        "id_human": poem_id,
                        "first_n_lines": first_n_lines,
                        "temperature": temperature,
                        "response": completion_result,
                    }
                )
                if verbose:
                    print(
                        f"\n✓ Generated completion with {len(completion_result)} lines"
                    )
            else:
                if verbose:
                    print(f"✗ Empty completion result for {model} + {poem_id}")

            if verbose:
                printm("----")
        except Exception as e:
            if verbose:
                print(f"!!! Error on model: {model} ({e})")
            bad_models.add(model)
        iterr.update(1)

    iterr.close()
    return pd.DataFrame(outld)


def get_first_n_lines(txt, n=5):
    lines = []
    nline = 0
    for ln in txt.split("\n"):
        if n and nline >= n:
            break
        if ln.strip():
            nline += 1
            lines.append(f"{nline}\t{ln}")
        else:
            lines.append(ln)
    return "\n".join(ln for i, ln in enumerate(lines)).strip()


def get_stash_df_completions(
    stash=STASH_GENAI_RHYME_COMPLETIONS, verbose=DEFAULT_VERBOSE
):
    """Extract rhyme completion data from the hash stash cache.

    Retrieves all cached rhyme completion results from the hash stash
    and converts them into a pandas DataFrame for analysis.

    Parameters
    ----------
    stash : BaseHashStash, default=STASH_GENAI_RHYME_COMPLETIONS
        Hash stash object containing cached completion results.
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing completion data with 'response' column
        and all stash metadata columns.

    Calls
    -----
    - stash.df.rename(columns={"_value": "response"}).reset_index() [to extract data]
    """
    if verbose:
        printm(f"* Collecting from `{stash.path}`")
    odf = stash.df.rename(columns={"_value": "response"}).reset_index()
    if verbose:
        printm(f"  * {len(odf)} generated completions")
    return odf


def get_rhyme_for_genai_human_completions():
    df_genai_rhyme_completions_genai = get_genai_rhyme_completions(by_line=False, line_gen=True)
    df_genai_rhyme_completions_genai_rhyme = get_rhyme_for_sample(
        df_genai_rhyme_completions_genai,
        force=False,
        with_sample=True,
    )

    df_genai_rhyme_completions_human = get_genai_rhyme_completions(by_line=False, line_gen=False)
    df_genai_rhyme_completions_human_rhyme = get_rhyme_for_sample(
        df_genai_rhyme_completions_human, 
        force=False, 
        with_sample=True
    )

    df = pd.concat([
        df_genai_rhyme_completions_genai_rhyme, 
        df_genai_rhyme_completions_human_rhyme.assign(model=HIST)
    ])
    df = df[df.num_lines_prosodic>=10]

    df._data_name = 'genai_rhyme_completions_human_genai'
    df._as_in_paper = df_genai_rhyme_completions_genai_rhyme._as_in_paper
    df._as_replicated = df_genai_rhyme_completions_genai_rhyme._as_replicated
    return df

def get_text_vs_instruct_completions():
    df_genai_rhyme_completions = get_genai_rhyme_completions(by_line=False)
    df = df_genai_rhyme_completions[df_genai_rhyme_completions.model.isin({'ollama/llama3.1:8b', 'ollama/llama3.1:8b-text-q4_K_M'})]
    def rename_model(model):
        return 'llama3.1:instruct' if not 'text' in model else 'llama3.1:text'

    df['model'] = df.model.apply(rename_model)
    df._data_name = 'genai_rhyme_completions_text_vs_instruct'
    df._as_in_paper = df_genai_rhyme_completions._as_in_paper
    df._as_replicated = df_genai_rhyme_completions._as_replicated
    return df