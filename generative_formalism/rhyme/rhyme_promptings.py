from . import *


def generate_rhyme_prompt_text(*args, stash=STASH_GENAI_RHYME_PROMPTS, verbose=True, **kwargs):
    """
    Convenience function for generate_text using rhyme stash.

    Args:
        args: Arguments for generate_text
        stash: Stash to use for caching. Defaults to STASH_GENAI_RHYME_PROMPTS.
        verbose: Whether to print verbose output. Defaults to True.
        kwargs: Keyword arguments for generate_text

    Returns:
        str: The generated text
    """
    return generate_text(*args, stash=stash, verbose=verbose, **kwargs)


def generate_more_poems_from_rhyme_prompts(
    n=3,
    df_sofar=None,
    models=MODEL_LIST,
    prompts=PROMPT_LIST,
    temperatures=None,
    verbose=True,
    force=False,
    max_n_combo=None,
):
    """
    Generate more poems from rhyme prompts using various models and configurations.
    
    This function generates additional poems by sampling from available models and prompts,
    with intelligent prioritization of underrepresented combinations to ensure balanced
    data collection across different model-prompt pairs.
    
    Args:
        n (int, optional): Number of poems to generate. Defaults to 3.
        df_sofar (pd.DataFrame, optional): Existing dataframe of generated poems to build upon.
            If None, loads all existing rhyme promptings. Defaults to None.
        models (list, optional): List of model identifiers to use for generation.
            Defaults to MODEL_LIST from constants.
        prompts (list, optional): List of prompt templates to use for generation.
            Defaults to PROMPT_LIST from constants.
        temperatures (list, optional): List of temperature values for generation.
            If None, uses default temperature. Defaults to None.
        verbose (bool, optional): Whether to print progress and status information.
            Defaults to True.
        force (bool, optional): Whether to force regeneration even if cached results exist.
            Defaults to False.
        max_n_combo (int, optional): Maximum number of entries allowed per model-prompt
            combination. If provided, model-prompt pairs that already have this many
            or more entries will be excluded from selection. Defaults to None (no limit).
    
    Returns:
        list: List of dictionaries containing generated poem data, including model,
            prompt, temperature, generated text, and metadata.
    
    Note:
        The function uses inverse probability weighting to prioritize model-prompt
        combinations that have been used less frequently, ensuring balanced sampling
        across the available options. Models that consistently fail are temporarily
        excluded from further attempts.
    """

    # Prioritize underrepresented models and prompts
    df = get_all_genai_rhyme_promptings(verbose=verbose) if df_sofar is None else df_sofar

    # Create copies to modify
    models = list(models)
    prompts = list(prompts)

    if not df.empty and "model" in df.columns and "prompt" in df.columns:
        model_counts = df["model"].value_counts().reindex(models, fill_value=0)
        prompt_counts = df["prompt"].value_counts().reindex(prompts, fill_value=0)

        # Filter out model-prompt combinations that exceed max_n_combo
        if max_n_combo is not None:
            combo_counts = df.groupby(['model', 'prompt']).size()
            overrepresented_combos = combo_counts[combo_counts >= max_n_combo].index
            
            # Remove models and prompts that would result in overrepresented combinations
            valid_models = []
            valid_prompts = []
            
            for model in models:
                for prompt in prompts:
                    if (model, prompt) not in overrepresented_combos:
                        if model not in valid_models:
                            valid_models.append(model)
                        if prompt not in valid_prompts:
                            valid_prompts.append(prompt)
            
            # Update the lists to only include valid options
            models = valid_models
            prompts = valid_prompts
            
            if verbose and (len(valid_models) < len(model_counts) or len(valid_prompts) < len(prompt_counts)):
                n_filtered_combos = len(overrepresented_combos)
                print(f"  * Filtered out {n_filtered_combos} model-prompt combinations with >= {max_n_combo} entries")
                print(f"  * Using {len(valid_models)} models and {len(valid_prompts)} prompts")

        # Inverse probability weighting, adding 1 to avoid division by zero and give unseen items a chance
        model_weights = 1 / (model_counts.reindex(models, fill_value=0) + 1)
        prompt_weights = 1 / (prompt_counts.reindex(prompts, fill_value=0) + 1)
    else:
        model_weights = None
        prompt_weights = None

    iterr = tqdm(total=n, position=0)
    bad_models = set()
    outld = []
    for _ in range(n):
        good_models = [m for m in models if m not in bad_models]
        if not good_models:
            if verbose:
                print("!!! No more models available.")
            break
        if not prompts:
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

        if prompt_weights is not None and not prompt_weights.empty:
            prompt = random.choices(
                population=prompt_weights.index.tolist(),
                weights=prompt_weights.values.tolist(),
                k=1,
            )[0]
        else:
            prompt = random.choice(prompts)

        temperature = round(
            (random.choice(temperatures) if temperatures else random.uniform(0.0, 1.0)),
            4,
        )
        
        # Count current usage for display
        session_model_count = sum(1 for item in outld if item['model'] == model)
        session_prompt_count = sum(1 for item in outld if item['prompt'] == prompt)
        session_combo_count = sum(1 for item in outld if item['model'] == model and item['prompt'] == prompt)
        
        # Count from overall stash (including historical data)
        if not df.empty:
            stash_model_count = len(df[df['model'] == model]) if 'model' in df.columns else 0
            stash_prompt_count = len(df[df['prompt'] == prompt]) if 'prompt' in df.columns else 0
            stash_combo_count = len(df[(df['model'] == model) & (df['prompt'] == prompt)]) if 'model' in df.columns and 'prompt' in df.columns else 0
        else:
            stash_model_count = stash_prompt_count = stash_combo_count = 0
        
        # Check if this combination would exceed max_n_combo
        total_combo_count = stash_combo_count + session_combo_count
        if max_n_combo is not None and total_combo_count >= max_n_combo:
            if verbose:
                print(f"!!! Skipping {model} + '{prompt[:30]}...' (would exceed max_n_combo={max_n_combo})")
            continue
        
        iterr.set_description(f'>>> {model} (n_model={stash_model_count + session_model_count:0,}, n_prompt={stash_prompt_count + session_prompt_count:0,}, n_combo={stash_combo_count + session_combo_count:0,}): "{prompt[:50]}"')
        try:
            if verbose:
                printm("----")
            response = generate_rhyme_prompt_text(
                model,
                prompt,
                temperature=temperature,
                verbose=verbose,
                force=force,
            )
            outld.append({
                'model': model,
                'prompt': prompt,
                'temperature': temperature,
                'response': response,
            })
            if verbose:
                printm("----")
        except Exception as e:
            if verbose:
                print(f"!!! Error on model: {model} ({e})")
            bad_models.add(model)
        iterr.update(1)
    return pd.DataFrame(outld)


## Collection of previous genai promptings


def get_legacy_df_poems1(path_pkl=PATH_RAW_PKL, verbose=DEFAULT_VERBOSE):
    if verbose:
        print(f"  * Collecting from {path_pkl}")
    if path_pkl and os.path.exists(path_pkl):
        df_poems1 = (
            pd.read_pickle(path_pkl)
            .fillna("")
            .query('prompt!=""')
            .rename(columns={"poem": "response", "temp": "temperature"})
        )
        if verbose:
            print(f"  * {len(df_poems1)} generated poems")
    else:
        df_poems1 = pd.DataFrame()
    return df_poems1


def get_legacy_df_poems2(path_json=PATH_RAW_JSON, verbose=DEFAULT_VERBOSE):
    if path_json and os.path.exists(path_json):
        if verbose:
            print(f"  * Collecting from {path_json}")
        newdata = []
        with gzip.open(path_json, "rt") as f:
            ld = json.loads(f.read())
            for d in ld:
                prompt = d["prompt"]["user_prompt"]
                model = d["prompt"]["model"]
                temp = d["prompt"]["temperature"]
                txt = d["response"].split("</think>")[-1].strip()
                newdata.append(
                    {
                        "model": model,
                        "temperature": temp,
                        "prompt": prompt,
                        "response": txt,
                    }
                )

        if verbose:
            print(f"  * {len(newdata)} generated poems")
        df2 = pd.DataFrame(newdata)
        return df2
    else:
        return pd.DataFrame()


def get_stash_df_poems(verbose=DEFAULT_VERBOSE):
    if verbose:
        print(f"  * Collecting from {STASH_GENAI_RHYME_PROMPTS.path}")
    odf = STASH_GENAI_RHYME_PROMPTS.df.rename(columns={"_value": "response"})
    if verbose:
        print(f"  * {len(odf)} generated poems")
    return odf


def preprocess_rhyme_promptings(overwrite=False, save_to=PATH_GENAI_PROMPTS_IN_PAPER, verbose=DEFAULT_VERBOSE, **kwargs):
    """Preprocess rhyme promptings data.

    This function preprocesses rhyme promptings data from legacy pickle and JSON files,
    combines them, and saves to CSV format.

    Args:
        overwrite (bool, optional): Whether to overwrite existing processed data.
            Defaults to False.
        save_to (str, optional): Path to save the processed data.
            Defaults to PATH_GENAI_PROMPTS_IN_PAPER.

    Returns:
        pd.DataFrame: Preprocessed data as a dataframe.
    """

    if not overwrite and os.path.exists(save_to):
        return pd.read_csv(save_to)
    else:
        df1 = get_legacy_df_poems1(verbose=verbose)
        df2 = get_legacy_df_poems2(verbose=verbose)
        df_prompts = pd.concat([df1, df2])

        if verbose:
            print(f"* Saving to {save_to}")
        df_prompts.to_csv(save_to)

    return df_prompts


def get_genai_rhyme_promptings_as_in_paper(
    *args, overwrite=False, save_to=PATH_GENAI_PROMPTS_IN_PAPER, verbose=DEFAULT_VERBOSE, **kwargs
):
    """
    Convenience function calling `preprocess_rhyme_promptings` and `postprocess_rhyme_promptings`.

    Args:
        overwrite (bool, optional): Whether to overwrite existing processed data.
            Defaults to False.
        save_to (str, optional): Path to save the processed data.
            Defaults to PATH_GENAI_PROMPTS_IN_PAPER.

    Returns:
        pd.DataFrame: Postprocessed data as a dataframe.
    """
    if verbose:
        print("* Collecting genai rhyme promptings as used in paper")
        print(f'  * Collecting from {save_to}')
    
    df_prompts = preprocess_rhyme_promptings(
        overwrite=overwrite, save_to=save_to, verbose=verbose, **kwargs,
    )

    kwargs["save_latex_to_suffix"] = PAPER_REGENERATED_SUFFIX
    return postprocess_rhyme_promptings(df_prompts, *args, verbose=verbose, **kwargs)


def get_genai_rhyme_promptings_as_replicated(*args, verbose=DEFAULT_VERBOSE, **kwargs):
    """
    Get genai rhyme promptings as replicated in this implementation.
    
    This function retrieves the rhyme promptings data that was generated
    and replicated in this codebase, as opposed to the original data
    used in the paper. It processes the data through the same postprocessing
    pipeline but uses the replicated suffix for output files.
    
    Args:
        *args: Variable length argument list passed to postprocess_rhyme_promptings.
        verbose (bool, optional): Whether to print verbose output during processing.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to postprocess_rhyme_promptings.
            Common kwargs include:
            - prompts: List of prompts to process (defaults to PROMPT_LIST)
            - models: List of models to process (defaults to MODEL_LIST)
            - min_lines: Minimum number of lines per poem (defaults to MIN_NUM_LINES)
            - max_lines: Maximum number of lines per poem (defaults to MAX_NUM_LINES)
            - save_to: Path to save processed data
            - overwrite: Whether to overwrite existing files
    
    Returns:
        pd.DataFrame: Processed rhyme promptings data with replicated suffix
        applied to output files. Contains the same structure as the paper
        data but generated from the current implementation's stash.
        
    Note:
        This function uses REPLICATED_SUFFIX for output file naming to
        distinguish it from the original paper data. The underlying data
        comes from get_stash_df_poems() which contains the replicated
        generation results.
    """
    if verbose:
        print("\n* Collecting genai rhyme promptings as replicated here")
    df_prompts = get_stash_df_poems(verbose=verbose)

    kwargs["save_latex_to_suffix"] = REPLICATED_SUFFIX
    return postprocess_rhyme_promptings(df_prompts, *args, verbose=verbose, **kwargs)


def get_genai_rhyme_promptings(df_prompts, *args, verbose=True, **kwargs):
    odf = postprocess_rhyme_promptings(df_prompts, *args, verbose=verbose, **kwargs)
    return odf


def get_all_genai_rhyme_promptings(*args, display=False, verbose=True, **kwargs):
    """
    Collect all genai rhyme promptings from both the paper and replicated here.

    Args:
        args: Arguments for get_genai_rhyme_promptings_as_in_paper and get_genai_rhyme_promptings_as_replicated
        display: Whether to display the promptings
        verbose: Whether to print verbose output. Defaults to True.
        kwargs: Keyword arguments for get_genai_rhyme_promptings_as_in_paper and get_genai_rhyme_promptings_as_replicated

    Returns:
        pd.DataFrame: All genai rhyme promptings
    """
    df1 = get_genai_rhyme_promptings_as_in_paper(*args, display=False, verbose=verbose, **kwargs)
    df2 = get_genai_rhyme_promptings_as_replicated(*args, display=False, verbose=verbose, **kwargs)
    odf = pd.concat([df1, df2])
    if display:
        display_rhyme_promptings(odf, **kwargs)
    return odf


def display_rhyme_promptings(df_prompts, **kwargs):
    kwargs["return_display"] = True
    img1 = get_rhyme_promptings_table(df_prompts, **kwargs)
    img2 = get_num_poems_per_model_table(df_prompts, **kwargs)
    try_display(img1)
    try_display(img2)


def postprocess_rhyme_promptings(
    df_prompts,
    prompts=PROMPT_LIST,
    models=MODEL_LIST,
    min_lines=MIN_NUM_LINES,
    max_lines=MAX_NUM_LINES,
    save_to=None,
    overwrite=False,
    display=False,
    verbose=True,
    **display_kwargs,
):
    """Postprocess rhyme promptings data.

    This function postprocesses rhyme promptings data by cleaning the text,
    setting the prompt type, and filtering the data by prompt and model.

    Args:
        df_prompts (pd.DataFrame): Input DataFrame containing rhyme promptings data.
        prompts (list, optional): List of prompts to include.
            Defaults to PROMPT_LIST.
        models (list, optional): List of models to include.
            Defaults to MODEL_LIST.
        min_lines (int, optional): Minimum number of lines for filtering.
            Defaults to MIN_NUM_LINES.
        max_lines (int, optional): Maximum number of lines for filtering.
            Defaults to MAX_NUM_LINES.
        save_to (str, optional): Path to save the processed data.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing processed data.
            Defaults to False.
        display (bool, optional): Whether to display the processed data.
            Defaults to False.
        verbose (bool, optional): Whether to print verbose output.
            Defaults to True.
        **display_kwargs: Additional keyword arguments passed to display_rhyme_promptings.

    Returns:
        pd.DataFrame: Postprocessed data as a dataframe.
    """

    # Set other cols

    df_prompts = df_prompts.fillna("")
    df_prompts["txt"] = df_prompts.response.apply(clean_poem_str)
    df_prompts["num_lines"] = df_prompts.txt.apply(get_num_lines)
    df_prompts["prompt_type"] = df_prompts.prompt.apply(
        lambda x: PROMPT_TO_TYPE.get(x, "Unknown")
    )
    df_prompts["temperature"] = pd.to_numeric(df_prompts.temperature, errors="coerce")

    # print(f"* Aggregated and filtered")
    df_prompts = df_prompts[df_prompts.prompt.isin(prompts)]
    df_prompts = df_prompts[df_prompts.model.isin(models)]

    if verbose:
        print(f"  * {len(df_prompts):,} generated responses")
        print(f"  * {df_prompts.response.nunique():,} unique responses")
        print(f"  * {df_prompts.txt.nunique():,} unique poems")
        print(f"  * {df_prompts.prompt.nunique():,} unique prompts")
        print(f"  * {df_prompts.prompt_type.nunique():,} unique prompt types")

    cols = ["id", "id_hash","prompt_type", "prompt", "model", "temperature", "txt", "num_lines"]

    df_prompts["id"] = [
        get_id_hash_str(f"{model}__{temp:.4f}__{prompt}__{txt}")
        for model, temp, prompt, txt in zip(
            df_prompts.model, df_prompts.temperature, df_prompts.prompt, df_prompts.txt
        )
    ]
    df_prompts["id_hash"] = [get_id_hash(id) for id in df_prompts.id]
    df_prompts["txt"] = df_prompts.txt.apply(clean_poem_str)
    df_prompts["num_lines"] = df_prompts.txt.apply(get_num_lines)
    df_prompts = df_prompts.query(
        f"num_lines >= {min_lines} and num_lines <= {max_lines}"
    )

    odf = df_prompts.drop_duplicates("id").set_index("id").sort_values("id_hash")
    odf = odf[[c for c in cols if c in odf.columns]]

    if save_to:
        save_sample(odf, save_to, overwrite=overwrite)

    if display:
        display_rhyme_promptings(odf, **display_kwargs)

    return odf


def get_rhyme_promptings_table(df_prompts, return_display=False, **kwargs):
    df_prompts = df_prompts.copy().query('prompt!=""')
    df_prompts["model9"] = df_prompts.model.apply(get_model_cleaned)
    df_prompts["model"] = df_prompts.model.apply(rename_model)

    df_prompts_stats = pd.DataFrame(
        [
            {
                "prompt_type": get_nice_prompt_type(prompt_type),
                "prompt": prompt,
                "num_poems": len(gdf),
                "num_poems_per_model": int(round(len(gdf) / gdf.model9.nunique())),
            }
            for (prompt_type, prompt), gdf in df_prompts.groupby(
                ["prompt_type", "prompt"]
            )
        ]
    )
    df_prompts_stats["prompt_type"] = pd.Categorical(
        df_prompts_stats["prompt_type"],
        categories=["Rhymed", "Unrhymed", "Rhyme unspecified"],
    )
    df_prompts_stats = (
        df_prompts_stats.set_index(["prompt_type", "prompt"])
        .sort_index()
        .rename_axis(["Prompt type", "Prompt"])[["num_poems", "num_poems_per_model"]]
    )
    df_prompts_stats.columns = ["# Poems", "Avg. # poems per model"]

    # Build custom tabular with multirow groups by prompt_type
    def _escape_latex_text(s, fix_typos=True):
        s = str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        if fix_typos:
            s = s.replace("an rhym", "a rhym").replace("an ryhm", "a rhym")
        return s

    tabular_lines = []
    tabular_lines.append("\\begin{tabular}{llrr}")
    tabular_lines.append("\\toprule")
    tabular_lines.append(" &  & \\# Poems & \\# per model (avg.) \\\\")
    tabular_lines.append("Prompt type & Prompt &  &  \\\\")
    tabular_lines.append("\\midrule")

    present_types = list(df_prompts_stats.index.get_level_values(0).unique())
    for prompt_type in present_types:
        try:
            subdf = df_prompts_stats.xs(prompt_type, level=0, drop_level=True)
        except KeyError:
            continue
        n = len(subdf)
        rows = list(subdf.reset_index().itertuples(index=False, name=None))
        for i, (prompt, num_poems, num_per_model) in enumerate(rows):
            ptype_disp = _escape_latex_text(prompt_type) if i == 0 else ""
            prompt_disp = _escape_latex_text(prompt)
            if i == 0:
                tabular_lines.append(
                    f"\\multirow[t]{{{n}}}{{*}}{{{ptype_disp}}} & {prompt_disp} & {num_poems} & {num_per_model} \\\\"
                )
            else:
                tabular_lines.append(
                    f" & {prompt_disp} & {num_poems} & {num_per_model} \\\\"
                )
        tabular_lines.append("\\cline{1-4}")

    if tabular_lines[-1] == "\\cline{1-4}":
        tabular_lines[-1] = "\\bottomrule"
    else:
        tabular_lines.append("\\bottomrule")
    tabular_lines.append("\\end{tabular}")

    tabular_str = "\n".join(tabular_lines)

    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=f"{PATH_TEX}/table_rhyme_promptings.tex",
        caption="Number of poems generated for each prompt.",
        label="tab:num_poems_rhyme_promptings",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


### METER


def get_num_poems_per_model_table(df_prompts, return_display=False, **kwargs):
    df = df_prompts.copy().query('prompt!=""')
    # Normalize model names
    df["model9"] = df.model.apply(get_model_cleaned)
    df["model"] = df.model.apply(rename_model)
    df = df[df.model != ""]
    # Map prompt types to display buckets
    _TYPE_DISP = {
        "DO_rhyme": "Rhymed",
        "do_NOT_rhyme": "Unrhymed",
        "MAYBE_rhyme": "Rhyme unspecified",
        "Unknown": "Unknown",
    }
    df["prompt_type_disp"] = df.prompt.apply(lambda x: PROMPT_TO_TYPE.get(x, "Unknown"))
    df["prompt_type_disp"] = df["prompt_type_disp"].map(_TYPE_DISP).fillna("Unknown")

    # Aggregate counts per display model name and category
    df_counts = (
        df.groupby(["model", "model9", "prompt_type_disp"])
        .size()
        .reset_index(name="num_poems")
    )
    # Pivot to columns in desired order
    cat_order = ["Rhymed", "Unrhymed", "Rhyme unspecified"]
    pivot = (
        df_counts.pivot_table(
            index=["model", "model9"],
            columns="prompt_type_disp",
            values="num_poems",
            fill_value=0,
        )
        .reindex(columns=cat_order, fill_value=0)
        .reset_index()
    )
    # Sort models alphabetically by cleaned name for stability
    pivot = pivot.sort_values("model9")

    # Build LaTeX tabular
    def _esc(s):
        return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    lines = []
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append(" & \\# Rhymed & \\# Unrhymed & \\# Rhyme unspecified \\\\")
    lines.append("Model &  &  &  \\\\")
    lines.append("\\midrule")

    for _, row in pivot.iterrows():
        model_disp = _esc(row["model9"])
        rh = int(row.get("Rhymed", 0))
        ur = int(row.get("Unrhymed", 0))
        mu = int(row.get("Rhyme unspecified", 0))
        lines.append(f"{model_disp} & {rh} & {ur} & {mu} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tabular_str = "\n".join(lines)

    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=f"{PATH_TEX}/table_num_poems_models.tex",
        caption="Number of poems generated for each model and prompt category.",
        label="tab:num_poems_models",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


# Get demo
def get_demo_model_prompt(demo_model=DEMO_MODEL, demo_prompt=DEMO_PROMPT):
    """
    Return demo model and prompt, defaults to DEMO_MODEL and DEMO_PROMPT.
    """
    #     print(f'''* Demo model: {demo_model}
    # * Demo prompt: {demo_prompt}
    # ''')

    return demo_model, demo_prompt


def get_rhyme_for_prompted_poems_as_in_paper(**kwargs):
    df_smpl = get_genai_rhyme_promptings_as_in_paper(by_line=False, **kwargs)
    df_smpl_w_rhyme_data = get_rhyme_for_sample(df_smpl, **kwargs)
    return df_smpl_w_rhyme_data

def get_rhyme_for_prompted_poems_as_replicated(**kwargs):
    df_smpl = get_genai_rhyme_promptings_as_replicated(by_line=False, **kwargs)
    df_smpl_w_rhyme_data = get_rhyme_for_sample(df_smpl, **kwargs)
    return df_smpl_w_rhyme_data