from . import *






### RHYME


@stashed_result(engine='pairtree')
def get_rhyme_for_txt(txt, max_dist=1):
    try:
        txt = limit_lines(txt)
        text = prosodic.Text(txt)
        rhyme_d = text.get_rhyming_lines(max_dist)
        all_rhyming_lines = set()
        all_perfectly_rhyming_lines = set()
        for l1, (score, l2) in rhyme_d.items():
            all_rhyming_lines.update({l1, l2})
            if not score:
                all_perfectly_rhyming_lines.update({l1, l2})
        num_rhyming_lines = len(all_rhyming_lines)
        num_perfectly_rhyming_lines = len(all_perfectly_rhyming_lines)
        num_lines = text.num_lines
        return {
            'num_rhyming_lines': num_rhyming_lines,
            'num_perfectly_rhyming_lines': num_perfectly_rhyming_lines,
            'num_lines': num_lines,
        }
    except Exception:
        return {}


def get_rhyme_for_sample(path_sample, force=False):
    df = pd.read_csv(path_sample).fillna("").set_index('id')
    ofn = path_sample.replace('.csv.gz', '.csv').replace('.csv', '.rhyme_data.csv')
    if not force and os.path.exists(ofn):
        df_rhymes = pd.read_csv(ofn).fillna("").set_index('id')
    else:
        df_rhymes = (
            pd.DataFrame((get_rhyme_for_txt(txt) for txt in tqdm(df.txt)), index=df.index)
            .dropna()
            .applymap(int)
        )
        if 'line_sim' in df.columns:
            line_sims = dict(zip(df.index, df.line_sim))
            df_rhymes['line_sim'] = df_rhymes.index.map(line_sims)
        df_rhymes.to_csv(ofn)
    return postprocess_rhyme_sample(df, df_rhymes)


def postprocess_rhyme_sample(df_poems, df_rhymes, rhyme_threshold=4):
    df = df_poems.join(df_rhymes, rsuffix='_prosodic', how='inner')
    num_lines = df.num_lines_prosodic if 'num_lines_prosodic' in df.columns else df.num_lines
    df['num_lines_prosodic'] = pd.to_numeric(num_lines, errors='coerce')
    df['num_rhyming_lines'] = pd.to_numeric(df.num_rhyming_lines, errors='coerce')
    df['num_perfectly_rhyming_lines'] = pd.to_numeric(df.num_perfectly_rhyming_lines, errors='coerce')
    df['perc_rhyming_lines'] = df.num_rhyming_lines / df.num_lines_prosodic * 100
    df['perc_perfectly_rhyming_lines'] = df.num_perfectly_rhyming_lines / df.num_lines_prosodic * 100
    df['num_rhyming_lines_per10l'] = (df.num_rhyming_lines / df.num_lines_prosodic * 10).fillna(0).round(0).astype(int)
    df['num_perfectly_rhyming_lines_per10l'] = (df.num_perfectly_rhyming_lines / df.num_lines_prosodic * 10).fillna(0).round(0).astype(int)
    if 'rhyme' in df.columns:
        df['rhyme'] = ['?' if not x else x for x in df.rhyme]
        df['rhyme_bool'] = df.rhyme.apply(lambda x: (True if x == 'y' else (False if x == 'n' else None)))
    df['rhyme_pred'] = df.num_perfectly_rhyming_lines_per10l.apply(lambda x: x >= rhyme_threshold)
    df['rhyme_pred_perc'] = df.rhyme_pred * 100
    return df


def get_rhyme_for_completed_poems(period_by=50, filter_line_sim=True, rename_models=True):
    df = get_rhyme_for_sample(PATH_GENAI_COMPLETIONS, force=True).reset_index()
    if 'line_sim' in df.columns:
        df['line_sim'] = pd.to_numeric(df.line_sim, errors='coerce')
        if filter_line_sim:
            df = df[(df.model == HIST) | (df.model == '') | (df.line_sim < 95)]
    df = df.groupby(['id_human', 'id', 'model']).mean(numeric_only=True).reset_index()
    df_meta = get_chadwyck_corpus(period_by=period_by)
    df = df.merge(df_meta, left_on='id_human', right_on='id', suffixes=['', '_meta'], how='left')
    if rename_models:
        df['model9'] = df.model.apply(get_model_cleaned)
        df['model'] = df.model.apply(rename_model)
        df = df[df.model != '']
    return df

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


def collect_genai_rhyme_promptings(
        collect_legacy=True,
        collect_stash=True,
        save=False,
        path_pkl=PATH_RAW_PKL, 
        path_json=PATH_RAW_JSON, 
        prompts=PROMPTS, 
        min_lines=10, 
        max_lines=100, 
        overwrite=False
    ):
    printm('#### Collecting genai rhyme promptings')
    valid_prompts = set(PROMPT_TO_TYPE.keys())
    valid_models = set(MODEL_TO_TYPE.keys())
    
    dfs = []
    if collect_legacy:
        printm('##### Collecting legacy data')
        df1 = get_legacy_df_poems1()
        df2 = get_legacy_df_poems2()
        dfs.extend([df1, df2])
    if collect_stash:
        printm('##### Collecting stash data')
        df3 = get_stash_df_poems()
        dfs.append(df3)

    # Concat
    df_prompts = pd.concat(dfs)

    # Set other cols
    df_prompts['txt'] = df_prompts.response.apply(clean_genai_poem)
    df_prompts['num_lines'] = df_prompts.txt.apply(get_num_lines)
    df_prompts['prompt_type'] = df_prompts.prompt.apply(lambda x: PROMPT_TO_TYPE.get(x, 'Unknown'))
    df_prompts['temperature'] = pd.to_numeric(df_prompts.temperature, errors='coerce')


    printm(f'##### Aggregated and filtered')
    df_prompts = df_prompts[df_prompts.prompt.isin(valid_prompts)]
    df_prompts = df_prompts[df_prompts.model.isin(valid_models)]
    
    print(f'* {len(df_prompts):,} generated responses')
    print(f'* {df_prompts.response.nunique():,} unique responses')
    print(f'* {df_prompts.txt.nunique():,} unique poems')
    print(f'* {df_prompts.prompt.nunique():,} unique prompts')
    print(f'* {df_prompts.prompt_type.nunique():,} unique prompt types')

    

    cols = ['prompt_type','prompt','model','temperature','txt','num_lines']
    cols = [c for c in cols if c in df_prompts.columns]

    id_list = [get_id_hash_str(f'{model}__{temp:.4f}__{prompt}__{txt}') for model,temp,prompt,txt in zip(df_prompts.model,df_prompts.temperature,df_prompts.prompt,df_prompts.txt)]
    df_prompts['id_hash'] = [get_id_hash(id) for id in id_list]
    df_prompts = df_prompts.sort_values('id_hash')
    df_prompts['txt'] = df_prompts.txt.apply(clean_genai_poem)
    df_prompts['num_lines'] = df_prompts.txt.apply(get_num_lines)
    
    df_prompts = df_prompts.query(f'num_lines >= {min_lines} and num_lines <= {max_lines}')
    odf = df_prompts.drop_duplicates('id_hash').set_index('id_hash').sort_index()
    odf=odf[cols]
    if save:
        save_sample(odf, PATH_SAMPLE_RHYMES, overwrite=overwrite)
    return odf

### METER
