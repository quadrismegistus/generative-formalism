from . import *

def generate_rhyme_prompt_text(*args, stash=STASH_GENAI_RHYME_PROMPTS, **kwargs):
    return generate_text(*args, stash=stash, **kwargs)

def generate_more_poems_from_rhyme_prompts(n=3, models=MODEL_LIST, prompts=PROMPT_LIST, temperatures=None, verbose=True):
    iterr = tqdm(total=n, position=0)
    bad_models = set()
    for n in range(n):
        if not models: break
        if not prompts: break
        model = random.choice(models)
        prompt = random.choice(prompts)
        temperature = round((random.choice(temperatures) if temperatures else random.uniform(0.0, 1.0)), 4)
        iterr.set_description(f'>>> {model} ({temperature}): "{prompt}"')
        try:
            if verbose:
                printm('----')
            response = generate_rhyme_prompt_text(
                model,
                prompt,
                temperature=temperature,
                verbose=verbose
            )
            if verbose:
                printm('----')
        except Exception as e:
            print(f'!!! Error on model: {model}')
            models = [m for m in models if m != model]
        iterr.update(1)


## Collection of previous genai promptings

def get_legacy_df_poems1(path_pkl=PATH_RAW_PKL):
    print(f'* Collecting from {path_pkl}')
    if path_pkl and os.path.exists(path_pkl):
        df_poems1 = pd.read_pickle(path_pkl).fillna('').query('prompt!=""').rename(columns={'poem':'response', 'temp':'temperature'})
        print(f'  * {len(df_poems1)} generated poems')
    else:
        df_poems1 = pd.DataFrame()
    return df_poems1

def get_legacy_df_poems2(path_json=PATH_RAW_JSON):
    if path_json and os.path.exists(path_json):
        print(f'* Collecting from {path_json}')
        newdata = []
        with gzip.open(path_json, 'rt') as f:
            ld = json.loads(f.read())
            for d in ld:
                prompt = d['prompt']['user_prompt']
                model = d['prompt']['model']
                temp = d['prompt']['temperature']
                txt = d['response'].split('</think>')[-1].strip()
                newdata.append({
                    'model':model,
                    'temperature':temp,
                    'prompt':prompt,
                    'response':txt,
                })
        
        print(f'  * {len(newdata)} generated poems')
        df2=pd.DataFrame(newdata)
        return df2
    else:
        return pd.DataFrame()

def get_stash_df_poems():
    print(f'* Collecting from {STASH_GENAI_RHYME_PROMPTS.path}')
    odf = STASH_GENAI_RHYME_PROMPTS.df.rename(columns={'_value':'response'}).drop(columns=['system_prompt'])
    print(f'  * {len(odf)} generated poems')
    return odf

def preprocess_rhyme_promptings(overwrite=False, save_to=PATH_GENAI_PROMPTS_IN_PAPER):
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
        df1 = get_legacy_df_poems1()
        df2 = get_legacy_df_poems2()
        df_prompts = pd.concat([df1, df2])

        print(f'* Saving to {save_to}')
        df_prompts.to_csv(save_to)

    return df_prompts


def get_genai_rhyme_promptings_as_in_paper(*args, overwrite=False, save_to=PATH_GENAI_PROMPTS_IN_PAPER, **kwargs):
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
    df_prompts = preprocess_rhyme_promptings(overwrite=overwrite, save_to=save_to, **kwargs)

    kwargs['save_latex_to_suffix'] = PAPER_REGENERATED_SUFFIX
    return postprocess_rhyme_promptings(df_prompts, *args, **kwargs)

def get_genai_rhyme_promptings_as_replicated(*args, **kwargs):
    printm('#### Collecting genai rhyme promptings used in paper')
    df_prompts = get_stash_df_poems()

    kwargs['save_latex_to_suffix'] = REPLICATED_SUFFIX
    return postprocess_rhyme_promptings(df_prompts, *args, **kwargs)

def get_genai_rhyme_promptings(df_prompts, *args, **kwargs):
    odf = postprocess_rhyme_promptings(df_prompts, *args, **kwargs)
    return odf

def get_all_genai_rhyme_promptings(*args, display=False, **kwargs):
    df1 = get_genai_rhyme_promptings_as_in_paper(*args, display=False, **kwargs)
    df2 = get_genai_rhyme_promptings_as_replicated(*args, display=False, **kwargs)
    odf = pd.concat([df1, df2])
    if display:
        display_rhyme_promptings(odf,**kwargs)
    return odf

def display_rhyme_promptings(df_prompts, **kwargs):
    kwargs['return_display'] = True
    img1 = get_rhyme_promptings_table(df_prompts,**kwargs)
    img2 = get_num_poems_per_model_table(df_prompts,**kwargs)
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
    **display_kwargs
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
        **display_kwargs: Additional keyword arguments passed to display_rhyme_promptings.
    
    Returns:
        pd.DataFrame: Postprocessed data as a dataframe.
    """
    
    # Set other cols

    df_prompts = df_prompts.fillna('')
    df_prompts['txt'] = df_prompts.response.apply(clean_poem_str)
    df_prompts['num_lines'] = df_prompts.txt.apply(get_num_lines)
    df_prompts['prompt_type'] = df_prompts.prompt.apply(lambda x: PROMPT_TO_TYPE.get(x, 'Unknown'))
    df_prompts['temperature'] = pd.to_numeric(df_prompts.temperature, errors='coerce')


    printm(f'##### Aggregated and filtered')
    df_prompts = df_prompts[df_prompts.prompt.isin(prompts)]
    df_prompts = df_prompts[df_prompts.model.isin(models)]
    
    print(f'* {len(df_prompts):,} generated responses')
    print(f'* {df_prompts.response.nunique():,} unique responses')
    print(f'* {df_prompts.txt.nunique():,} unique poems')
    print(f'* {df_prompts.prompt.nunique():,} unique prompts')
    print(f'* {df_prompts.prompt_type.nunique():,} unique prompt types')

    

    cols = ['prompt_type','prompt','model','temperature','txt','num_lines']
    cols = [c for c in cols if c in df_prompts.columns]

    df_prompts['id'] = [get_id_hash_str(f'{model}__{temp:.4f}__{prompt}__{txt}') for model,temp,prompt,txt in zip(df_prompts.model,df_prompts.temperature,df_prompts.prompt,df_prompts.txt)]
    df_prompts['id_hash'] = [get_id_hash(id) for id in df_prompts.id]
    df_prompts['txt'] = df_prompts.txt.apply(clean_poem_str)
    df_prompts['num_lines'] = df_prompts.txt.apply(get_num_lines)
    df_prompts = df_prompts.query(f'num_lines >= {min_lines} and num_lines <= {max_lines}')
    
    odf = df_prompts.drop_duplicates('id').set_index('id').sort_values('id_hash')
    odf=odf[cols]

    if save_to:
        save_sample(odf, save_to, overwrite=overwrite)

    if display:
        display_rhyme_promptings(odf,**display_kwargs)
    
    return odf



def get_rhyme_promptings_table(df_prompts, return_display=False, **kwargs):
    df_prompts = df_prompts.copy().query('prompt!=""')
    df_prompts['model9'] = df_prompts.model.apply(get_model_cleaned)
    df_prompts['model'] = df_prompts.model.apply(rename_model)

    df_prompts_stats = pd.DataFrame([
        {
            'prompt_type':get_nice_prompt_type(prompt_type),
            'prompt':prompt,
            'num_poems': len(gdf),
            'num_poems_per_model': int(round(len(gdf) / gdf.model9.nunique())),
            }
        for (prompt_type,prompt),gdf in df_prompts.groupby(['prompt_type','prompt'])
    ])
    df_prompts_stats['prompt_type'] = pd.Categorical(df_prompts_stats['prompt_type'], categories=['Rhymed','Unrhymed','Rhyme unspecified'])
    df_prompts_stats = df_prompts_stats.set_index(['prompt_type','prompt']).sort_index().rename_axis(['Prompt type', 'Prompt'])[['num_poems', 'num_poems_per_model']]
    df_prompts_stats.columns = ['# Poems', 'Avg. # poems per model']

    # Build custom tabular with multirow groups by prompt_type
    def _escape_latex_text(s, fix_typos=True):
        s = str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
        if fix_typos:
            s=s.replace('an rhym', 'a rhym').replace('an ryhm', 'a rhym')
        return s


    tabular_lines = []
    tabular_lines.append('\\begin{tabular}{llrr}')
    tabular_lines.append('\\toprule')
    tabular_lines.append(' &  & \\# Poems & \\# per model (avg.) \\\\')
    tabular_lines.append('Prompt type & Prompt &  &  \\\\')
    tabular_lines.append('\\midrule')

    present_types = list(df_prompts_stats.index.get_level_values(0).unique())
    for prompt_type in present_types:
        try:
            subdf = df_prompts_stats.xs(prompt_type, level=0, drop_level=True)
        except KeyError:
            continue
        n = len(subdf)
        rows = list(subdf.reset_index().itertuples(index=False, name=None))
        for i, (prompt, num_poems, num_per_model) in enumerate(rows):
            ptype_disp = _escape_latex_text(prompt_type) if i == 0 else ''
            prompt_disp = _escape_latex_text(prompt)
            if i == 0:
                tabular_lines.append(f'\\multirow[t]{{{n}}}{{*}}{{{ptype_disp}}} & {prompt_disp} & {num_poems} & {num_per_model} \\\\')
            else:
                tabular_lines.append(f' & {prompt_disp} & {num_poems} & {num_per_model} \\\\')
        tabular_lines.append('\\cline{1-4}')

    if tabular_lines[-1] == '\\cline{1-4}':
        tabular_lines[-1] = '\\bottomrule'
    else:
        tabular_lines.append('\\bottomrule')
    tabular_lines.append('\\end{tabular}')

    tabular_str = '\n'.join(tabular_lines)

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
    df['model9'] = df.model.apply(get_model_cleaned)
    df['model'] = df.model.apply(rename_model)
    df = df[df.model != '']
    # Map prompt types to display buckets
    _TYPE_DISP = {
        'DO_rhyme': 'Rhymed',
        'do_NOT_rhyme': 'Unrhymed',
        'MAYBE_rhyme': 'Rhyme unspecified',
        'Unknown': 'Unknown',
    }
    df['prompt_type_disp'] = df.prompt.apply(lambda x: PROMPT_TO_TYPE.get(x, 'Unknown'))
    df['prompt_type_disp'] = df['prompt_type_disp'].map(_TYPE_DISP).fillna('Unknown')

    # Aggregate counts per display model name and category
    df_counts = (
        df.groupby(['model', 'model9', 'prompt_type_disp'])
          .size()
          .reset_index(name='num_poems')
    )
    # Pivot to columns in desired order
    cat_order = ['Rhymed', 'Unrhymed', 'Rhyme unspecified']
    pivot = (
        df_counts.pivot_table(index=['model', 'model9'], columns='prompt_type_disp', values='num_poems', fill_value=0)
        .reindex(columns=cat_order, fill_value=0)
        .reset_index()
    )
    # Sort models alphabetically by cleaned name for stability
    pivot = pivot.sort_values('model9')

    # Build LaTeX tabular
    def _esc(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')

    lines = []
    lines.append('\\begin{tabular}{lrrr}')
    lines.append('\\toprule')
    lines.append(' & \\# Rhymed & \\# Unrhymed & \\# Rhyme unspecified \\\\')
    lines.append('Model &  &  &  \\\\')
    lines.append('\\midrule')

    for _, row in pivot.iterrows():
        model_disp = _esc(row['model9'])
        rh = int(row.get('Rhymed', 0))
        ur = int(row.get('Unrhymed', 0))
        mu = int(row.get('Rhyme unspecified', 0))
        lines.append(f'{model_disp} & {rh} & {ur} & {mu} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    tabular_str = '\n'.join(lines)

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

