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
    # df = get_all_genai_rhyme_promptings(verbose=verbose) if df_sofar is None else df_sofar
    df = pd.DataFrame() if df_sofar is None else df_sofar

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
                printm(f"  * Filtered out {n_filtered_combos} model-prompt combinations with >= {max_n_combo} entries")
                printm(f"  * Using {len(valid_models)} models and {len(valid_prompts)} prompts")

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
        printm(f"  * Collecting from {path_pkl}")
    if path_pkl and os.path.exists(path_pkl):
        df_poems1 = (
            pd.read_pickle(path_pkl)
            .fillna("")
            .query('prompt!=""')
            .rename(columns={"poem": "response", "temp": "temperature"})
        )
        if verbose:
            printm(f"  * {len(df_poems1)} generated poems")
    else:
        df_poems1 = pd.DataFrame()
    return df_poems1


def get_legacy_df_poems2(path_json=PATH_RAW_JSON, verbose=DEFAULT_VERBOSE):
    if path_json and os.path.exists(path_json):
        if verbose:
            printm(f"  * Collecting from {path_json}")
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
            printm(f"  * {len(newdata)} generated poems")
        df2 = pd.DataFrame(newdata)
        return df2
    else:
        return pd.DataFrame()


def get_stash_df_poems(verbose=DEFAULT_VERBOSE):
    if verbose:
        printm(f"  * Collecting from {STASH_GENAI_RHYME_PROMPTS.path}")
    ld = STASH_GENAI_RHYME_PROMPTS.ld
    df = pd.DataFrame(ld).rename(columns={"_value": "txt", "temp": "temperature"})
    df['prompt_type'] = df.prompt.apply(lambda x: PROMPT_TO_TYPE.get(x, "Unknown"))
    df['id'] = [get_id_hash_str("__".join(vals)) for vals in df.applymap(str).values]
    df['id_hash'] = [get_id_hash(id) for id in df.id]
    if verbose:
        printm(f"  * {len(df)} generated poems")
    return df.set_index('id')


def preprocess_rhyme_promptings(overwrite=False, verbose=DEFAULT_VERBOSE, **kwargs):
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

    path = get_path(DATA_NAME_GENAI_RHYME_PROMPTINGS)
    if not overwrite and os.path.exists(path):
        df_prompts = pd.read_csv(path).set_index('id')
    else:
        df1 = get_legacy_df_poems1(verbose=verbose)
        df2 = get_legacy_df_poems2(verbose=verbose)
        df_prompts = pd.concat([df1, df2])

        if verbose:
            printm(f"* Saving to `{nice_path(path)}`")
        df_prompts.to_csv(path)

    return df_prompts.rename(columns={'response': 'txt', 'temp':'temperature'})


def get_all_genai_rhyme_promptings(*args, **kwargs):
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
    kwargs["as_in_paper"] = True
    kwargs["as_replicated"] = False
    return get_genai_rhyme_promptings(*args, **kwargs)


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
    as_in_paper=True,
    as_replicated=False,
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

    df_prompts = df_prompts.fillna("").rename(columns={'response': 'txt', 'temp':'temperature'})
    df_prompts["txt"] = df_prompts.txt.apply(clean_poem_str)
    df_prompts["num_lines"] = df_prompts.txt.apply(get_num_lines)
    df_prompts["prompt_type"] = df_prompts.prompt.apply(
        lambda x: PROMPT_TO_TYPE.get(x, "Unknown")
    )
    df_prompts["temperature"] = pd.to_numeric(df_prompts.temperature, errors="coerce")

    # printm(f"* Aggregated and filtered")
    df_prompts = df_prompts[df_prompts.prompt.isin(prompts)]
    df_prompts = df_prompts[df_prompts.model.isin(models)]

    if verbose:
        printm(f"""
* {len(df_prompts):,} generated responses
* {df_prompts.txt.nunique():,} unique poems
* {df_prompts.prompt.nunique():,} unique prompts
* {df_prompts.prompt_type.nunique():,} unique prompt types
""")

    cols = ["id", "data_source", "id_hash","prompt_type", "prompt", "model", "temperature", "txt", "num_lines"]

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
        display_rhyme_promptings(odf, as_in_paper=as_in_paper, as_replicated=as_replicated, **display_kwargs)

    return odf


# Get demo

def get_rhyme_for_prompted_poems_as_in_paper(**kwargs):
    raise NotImplementedError("Use get_rhyme_for_prompted_poems_by(..., as_in_paper=True) instead")

def get_rhyme_for_prompted_poems_as_replicated(**kwargs):
    raise NotImplementedError("Use get_rhyme_for_prompted_poems_by(..., as_replicated=True) instead")


def get_genai_rhyme_promptings(
    as_in_paper=True,
    as_replicated=False,
    verbose=DEFAULT_VERBOSE,
    display=False,
    **kwargs,
):
    """Unified accessor for genai rhyme promptings with source selection.

    Mirrors the corpus/sample "by" pattern. Exactly one of the flags should be True.
    """
    ld = []
    if as_in_paper:
        df = preprocess_rhyme_promptings(overwrite=False).reset_index().assign(data_source='in_paper')
        ld.extend(df.to_dict(orient='records'))
    
    if as_replicated:
        if verbose:
            printm("* Collecting genai rhyme promptings as replicated here")
        df = get_stash_df_poems(verbose=verbose).reset_index().assign(data_source='replicated')
        ld.extend(df.to_dict(orient='records'))

    if len(ld) == 0:
        raise ValueError("No data sources selected")

    df_prompts = pd.DataFrame(ld).fillna("").set_index('id')
    odf = postprocess_rhyme_promptings(
        df_prompts, 
        display=display, 
        verbose=verbose, 
        **kwargs
    )
    odf['model_type'] = odf.model.apply(rename_model)
    odf._data_name = f'genai_rhyme_promptings'
    odf._sample_by = '' # N/A for this data
    odf._as_in_paper = as_in_paper
    odf._as_replicated = as_replicated
    return odf



def get_rhyme_for_prompted_poems_by(
    *args,
    as_in_paper=True,
    as_replicated=False,
    verbose=DEFAULT_VERBOSE,
    **kwargs,
):
    """Unified accessor for rhyme analysis over prompted poems.

    Selects between in-paper, replicated, or regenerated sources.
    """
    flags_true = sum([bool(as_in_paper), bool(as_replicated)])
    if flags_true != 1:
        raise ValueError("Specify exactly one of as_in_paper, as_replicated")

    if as_replicated:
        return get_rhyme_for_prompted_poems_as_replicated(
            *args,
            verbose=verbose,
            **kwargs,
        )


    # Default: as_in_paper
    return get_rhyme_for_prompted_poems_as_in_paper(
        *args,
        verbose=verbose,
        **kwargs,
    )


def get_human_genai_rhyme_data(sample_by='period', as_in_paper=True, as_replicated=False):
    from ..corpus.sample import get_chadwyck_corpus_sampled_by
    df_human = get_chadwyck_corpus_sampled_by(sample_by)
    df_human_rhyme = get_rhyme_for_sample(df_human, with_sample=True)
    df_human_rhyme['xcol'] = df_human_rhyme['period']

    df_genai = get_genai_rhyme_promptings()
    df_genai_rhyme = get_rhyme_for_sample(df_genai, with_sample=True)
    df_genai_rhyme['xcol'] = df_genai_rhyme['model_type']

    df_both = pd.concat([
        df_human_rhyme.assign(source='Historical poems', prompt_type=''),
        df_genai_rhyme.assign(source='Generative poems')
    ]).fillna('')

    subcorpus_names = {
        '': '(n/a)',
        'DO_rhyme': 'Rhymed',
        'do_NOT_rhyme': 'Unrhymed',
        'MAYBE_rhyme': 'Rhyme unspecified',
    }
    df_both['prompt_type'] = df_both.prompt_type.apply(lambda x: subcorpus_names.get(x,x))

    df_both._data_name = 'human_genai_rhyme_data'
    df_both._as_in_paper = as_in_paper
    df_both._as_replicated = as_replicated
    return df_both

