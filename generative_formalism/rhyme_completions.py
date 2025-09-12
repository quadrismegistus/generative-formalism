from . import *

PATH_GENAI_RHYME_COMPLETIONS = f'{PATH_DATA}/corpus_genai_rhyme_completions.csv.gz'
PATH_GENAI_RHYME_COMPLETIONS_LEGACY = f'{PATH_DATA}/corpus_genai_rhyme_completions.legacy.csv.gz'
COMPLETIONS_GROUPBY = ['model','id','first_n_lines','stanza_num','line_num']


def get_legacy_genai_rhyme_completions(path=PATH_GENAI_RHYME_COMPLETIONS_LEGACY,**kwargs):
    """Load and postprocess legacy generative AI rhyme completions data.
    
    This function loads legacy rhyme completion data from a CSV file, or creates it
    if it doesn't exist. The data is then postprocessed to filter and clean it.
    
    Args:
        path (str, optional): Path to the legacy rhyme completions CSV file.
            Defaults to PATH_GENAI_RHYME_COMPLETIONS_LEGACY.
        **kwargs: Additional keyword arguments passed to postprocess_genai_rhyme_completions.
    
    Returns:
        pd.DataFrame: Postprocessed DataFrame containing legacy rhyme completion data.
    """
    if not os.path.exists(path):
        odf = _collect_prev_genai_completions(path=path)
    else:
        odf = pd.read_csv(path)
    
    odf = postprocess_genai_rhyme_completions(odf, **kwargs)
    return odf

def postprocess_genai_rhyme_completions(odf, threshold=95, filter_recognized=True, min_num_lines=10):
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
    odf = odf.fillna('')

    if filter_recognized:
        odf = filter_recognized_completions(odf, threshold=threshold)
    
    # odf = _to_poem_txt_format(odf, min_num_lines=min_num_lines)
    return odf


# Filter out recognized completions
def filter_recognized_completions(df, threshold=95, groupby=COMPLETIONS_GROUPBY):
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
    
    Returns:
        pd.DataFrame: DataFrame with added 'line_sim' column containing
            similarity scores between real and generated lines.
    
    Note:
        The actual filtering logic is commented out in the current implementation.
        This function only adds similarity scores to the DataFrame.
    """
    # Test for line similarity
    df['line_sim'] = df.progress_apply(
        lambda row: fuzz.ratio(
            row.line_real.strip(), 
            row.line_gen.strip()
        ) if row.line_gen and row.line_real else np.nan, 
        axis=1
    )
    return df
    
    gby=groupby
    num1=len(df.groupby(gby))
    grps=[]
    grps_unsafe=[]
    for g,gdf in df.groupby(gby): 
        if gdf.line_sim.max()<threshold:
            grps.append(gdf)
        else:
            grps_unsafe.append(gdf)
    gdf = random.choice(grps)
    df_safe = pd.concat(grps)
    df_unsafe = pd.concat(grps_unsafe)
    print(f'* Filtered out {num1 - len(grps)} recognized poems')
    return df_safe



## Legacy
GENAI_RHYME_COMPLETIONS_INDEX = ['id_human','model','first_n_lines', 'version','date','id','stanza_num','line_num']
FIRST_N_LINES = 5
PREPROCESSED_LEGACY_COMPLETION_DATA = None
def preprocess_legacy_genai_rhyme_completions(path=PATH_GENAI_RHYME_COMPLETIONS, overwrite=False, first_n_lines=FIRST_N_LINES):
    """Preprocess legacy generative AI rhyme completions from raw pickle files.
    
    This function loads and preprocesses legacy rhyme completion data from multiple
    pickle files (v3-v7), combines them, deduplicates, and saves to CSV format.
    It also generates unique IDs for generated poems and provides statistics
    about the dataset.
    
    Args:
        path (str, optional): Path to save the processed CSV file.
            Defaults to PATH_GENAI_RHYME_COMPLETIONS.
        overwrite (bool, optional): Whether to overwrite existing processed data.
            Defaults to False.
        first_n_lines (int, optional): Number of first lines from original poems
            to consider. Defaults to FIRST_N_LINES.
    
    Returns:
        pd.DataFrame: Processed DataFrame with MultiIndex containing completion data,
            indexed by GENAI_RHYME_COMPLETIONS_INDEX.
    
    Note:
        This function uses a global cache (PREPROCESSED_LEGACY_COMPLETION_DATA)
        to avoid reprocessing data within the same session.
    """

    global PREPROCESSED_LEGACY_COMPLETION_DATA

    if not overwrite and PREPROCESSED_LEGACY_COMPLETION_DATA is not None:
        return PREPROCESSED_LEGACY_COMPLETION_DATA

    if not overwrite and os.path.exists(PATH_GENAI_RHYME_COMPLETIONS):
        print(f'* Loading legacy genai rhyme completions from {nice_path(PATH_GENAI_RHYME_COMPLETIONS)}')
        odf = pd.read_csv(PATH_GENAI_RHYME_COMPLETIONS).fillna('').set_index(GENAI_RHYME_COMPLETIONS_INDEX)
    else:
        print(f'* Preprocessing legacy genai rhyme completions')
        index='_id	_first_n_lines	_model	_say_poem	_version	_timestamp'.split()
        df3 = pd.read_pickle(f'{PATH_RAWDATA}/data.output.gen_poems.v3.pkl').assign(_say_poem=True).reset_index().set_index(index)
        df4=pd.read_pickle(f'{PATH_RAWDATA}/data.output.gen_poems.v4.pkl').assign(_say_poem=True).reset_index().set_index(index)
        df5=pd.read_pickle(f'{PATH_RAWDATA}/data.output.gen_poems.v5.pkl').assign(_say_poem=True).reset_index().set_index(index)
        df6=pd.read_pickle(f'{PATH_RAWDATA}/data.output.gen_poems.v6.pkl')
        df7=pd.read_pickle(f'{PATH_RAWDATA}/data.output.gen_poems.v7.pkl')

        df = pd.concat([
            df3,#.query('_model=="ollama/llama3.1:8b"'),
            df4,#.query('_model=="ollama/llama3.1:8b-text-q4_K_M" | _model=="ollama/mistral" | _model=="ollama/mistral:text"'),
            df5,#.query('_model=="ollama/llama3.1:8b"'),
            df6,
            df7
        ]).reset_index()
        df.columns = [x[1:] if x and x[0]=='_' else x for x in df.columns]

        

        def get_id_gen(gdf):
            model = gdf.model.iloc[0]
            id = gdf.id.iloc[0]
            version = str(gdf.version.iloc[0])
            timestamp = str(gdf.timestamp.iloc[0])
            txt = gdf.line_gen.str.cat(sep='\n').strip()
            return get_id_hash_str('__'.join([model, id, version, timestamp, txt]))

        df = pd.concat([gdf.assign(id_gen=get_id_gen(gdf)) for g,gdf in df.groupby(['model','id','version'])])

        df = df.drop_duplicates(['model','id','id_gen','stanza_num','line_num'])
        df = df[df.say_poem]
        df.drop(columns=['say_poem'], inplace=True)
        df = df[~df.model.str.contains('poetry')]
        df = df.query(f'first_n_lines == {first_n_lines}')
        # df = df.drop(columns=['first_n_lines', 'version', 'timestamp'])

        # Convert timestamp to date string
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date.astype(str)


        odf = df.rename(columns={'id': 'id_human', 'id_gen':'id'})
        odf.drop(columns=['timestamp'], inplace=True)



        odf = odf.set_index(GENAI_RHYME_COMPLETIONS_INDEX)
    
        print(f'* Saving legacy genai rhyme completions to {nice_path(PATH_GENAI_RHYME_COMPLETIONS)}')
        odf.to_csv(PATH_GENAI_RHYME_COMPLETIONS)

    PREPROCESSED_LEGACY_COMPLETION_DATA = odf

    human_ids = odf.reset_index().id_human.unique()
    print(f'* Found {len(human_ids)} unique human poems for input to models')
    gen_ids = odf.reset_index().id.unique()
    print(f'* Found {len(gen_ids)} unique generated poems')

    print('* Distribution of input poem lengths')
    describe_numeric(pd.Series([len(gdf) for g,gdf in odf.groupby('id_human')], name='num_lines'))

    print('* Distribution of output poem lengths')
    describe_numeric(
        pd.Series([len(gdf) for g,gdf in odf.groupby('id')], name='num_lines'),
        fixed_range=(MIN_NUM_LINES, MAX_NUM_LINES)
    )

    return odf


def get_genai_rhyme_completions_as_in_paper(by_line=True, keep_first_n_lines=True):
    """Get generative AI rhyme completions data as used in the paper.
    
    This function retrieves preprocessed legacy rhyme completion data and
    optionally converts it to poem text format for analysis. It provides
    statistics about the dataset including line counts and poem length distributions.
    
    Args:
        by_line (bool, optional): If True, returns line-by-line data.
            If False, converts to poem text format. Defaults to True.
        keep_first_n_lines (bool, optional): Whether to keep the first N lines
            from original poems when converting to poem format. Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame containing rhyme completion data, either in
            line-by-line format or poem text format depending on by_line parameter.
    """
    df = preprocess_legacy_genai_rhyme_completions(overwrite=False)

    if by_line:
        return df
    

    odf = to_poem_txt_format(df, keep_first_n_lines=keep_first_n_lines)
    print(f'* Total lines: {odf.num_lines.sum().astype(int):,}')
    print('* Distribution of output poem lengths')
    describe_numeric(odf.num_lines, fixed_range=(MIN_NUM_LINES, MAX_NUM_LINES))
    return odf

# To poem format

def to_poem_txt_format(df, keep_first_n_lines=True):
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
    num_poems = df.id.nunique()
    print(f'* Converting to poem txt format' + (' (keeping first lines from original poem)' if keep_first_n_lines else ' (not keeping first lines from original poem)'))
    
    def get_num_lines(txt):
        return len([x for x in txt.split('\n') if x.strip()])


    def get_row(gdf):
        model = gdf.model.iloc[0]
        id_gen = gdf.id.iloc[0]
        id_hash = get_id_hash(id_gen)
        id_human = gdf.id_human.iloc[0]
        line_num = 0
        first_n_lines = gdf.iloc[0].first_n_lines

        if not keep_first_n_lines:
            lines = list(gdf.line_gen[first_n_lines:])
        else:
            lines = list(gdf.line_real[:first_n_lines]) + list(gdf.line_gen[first_n_lines:])
        
        txt = '\n'.join(lines)

        return {
            'id':id_gen,
            'id_human':id_human,
            'id_hash': id_hash,
            'model':model,
            'txt':txt,
            'num_lines':len(lines),
            'first_n_lines':first_n_lines,
            'keep_first_n_lines':keep_first_n_lines,
        }
    df = pd.concat([
        pd.DataFrame([get_row(gdf) for g,gdf in df.groupby('id')])
    ])
    return df.set_index([x for x in GENAI_RHYME_COMPLETIONS_INDEX+['keep_first_n_lines'] if x in df.columns])