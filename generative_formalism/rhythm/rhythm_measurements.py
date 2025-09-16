from . import *
from collections import defaultdict

# @timeout(3)
def parse_text(txt):
    """Parse poem text using Prosodic for meter and stress analysis.

    Uses the Prosodic library to analyze the metrical structure of the input text,
    parsing at the line level with single-threaded processing.

    Parameters
    ----------
    txt : str
        The poem text to parse for metrical analysis.

    Returns
    -------
    prosodic.Text
        Parsed Prosodic Text object containing meter and stress information.

    Calls
    -----
    - prosodic.Text(txt).parse(parse_unit='line', num_proc=1).best
    """
    return prosodic.Text(txt).parse(parse_unit='line', num_proc=1)

def get_parses_for_txt(txt, stash=STASH_RHYTHM, force=False):
    """Get prosodic parses for a text, with caching and optional postprocessing.

    Retrieves cached prosodic parse data if available and not forced to regenerate.
    If no cached data exists or force=True, parses the text using Prosodic and caches
    the result. Optionally postprocesses the parse data into rhythm measurements.

    Parameters
    ----------
    txt : str
        The poem text to parse.
    stash : HashStash, default=STASH_RHYTHM
        Cache storage for parsed data.
    force : bool, default=False
        If True, re-parse even if cached data exists.
    
    Returns
    -------
    pd.DataFrame or dict
        Raw parse DataFrame if postprocess=False, or processed rhythm measurements
        dict if postprocess=True. Returns empty DataFrame if parsing fails.

    Calls
    -----
    - parse_text(txt) [if no cached data or force=True]
    """
    stash_key = txt
    odf = None
    if not force and stash is not None and stash_key in stash:
        odf = stash[stash_key]
    
    if odf is None:
        odf = pd.DataFrame()
        try:
            odf = parse_text(txt).scansions.df.query('parse_rank==1')
        except:
            # print(f'! {e}')
            # raise e
            pass
        
        if type(odf) == pd.DataFrame and len(odf):
            stash[stash_key] = odf
    
    return odf

def gen_parses_for_sample(df_smpl, stash=STASH_RHYTHM, force=False, verbose=DEFAULT_VERBOSE, **kwargs):
    """Generate and cache prosodic parses for a sample of poems.

    Iterates through a sample of poems and ensures their prosodic parses are
    generated and stored in the cache. Useful for pre-computing parses before
    analysis.

    Parameters
    ----------
    df_smpl : pd.DataFrame
        DataFrame containing poem texts in a 'txt' column.
    stash : HashStash, default=STASH_RHYTHM
        Cache storage for parsed data.
    force : bool, default=False
        If True, re-parse even if cached data exists.
    verbose : bool, default=DEFAULT_VERBOSE
        If True, show progress information.
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    None
        Parses are stored in the cache but nothing is returned.

    Calls
    -----
    - _clean_df(df_smpl)
    - get_parses_for_txt(txt, stash=stash, force=force, postprocess=False) [for each text]
    """
    df = _clean_df(df_smpl)
    for txt in tqdm(df.txt,desc='* Getting parses for sample', total=len(df)):
        get_parses_for_txt(txt, stash=stash, force=force, postprocess=False)


def get_parses_for_sample(df_smpl, stash=STASH_RHYTHM, force=False, gen=True, verbose=DEFAULT_VERBOSE, **kwargs):
    """Collect prosodic parses for a sample of poems into a single DataFrame.

    Retrieves or generates prosodic parse data for each poem in the sample,
    combining results into a single DataFrame indexed by poem ID.

    Parameters
    ----------
    df_smpl : pd.DataFrame
        DataFrame containing poem texts in a 'txt' column, indexed by poem IDs.
    stash : HashStash, default=STASH_RHYTHM
        Cache storage for parsed data.
    force : bool, default=False
        If True, re-parse even if cached data exists.
    gen : bool, default=True
        If True, generate new parses; if False, only use cached data.
    verbose : bool, default=DEFAULT_VERBOSE
        If True, show progress information.
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of parse data with multi-index ['id', original_index],
        or empty DataFrame if no valid parses found.

    Calls
    -----
    - _clean_df(df_smpl)
    - get_parses_for_txt(txt, stash=stash, force=force) [if gen=True]
    """
    df_smpl = _clean_df(df_smpl)
    l = []
    
    def get_res(id,txt,_gen=gen):
        try:
            if gen:
                res = get_parses_for_txt(txt, stash=stash, force=force)
            else:
                res = stash.get(txt)
        except KeyboardInterrupt:
            return pd.DataFrame()
        
        if not type(res) == pd.DataFrame or not len(res):
            return pd.DataFrame()
        

        return res.assign(id=id) if 'id' not in res.columns else res


    for (id,txt) in tqdm(zip(df_smpl.index, df_smpl.txt),desc='* Getting parses for sample', total=len(df_smpl)):
        l.append(get_res(id,txt))
    if not len(l):
        return pd.DataFrame()
    
    odf = pd.concat(l)
    odf = odf.reset_index().set_index(['id'] + odf.index.names)
    return odf


def get_rhythm_for_sample(df_smpl, stash=STASH_RHYTHM, force=False, gen=True, verbose=DEFAULT_VERBOSE, with_sample=False, **kwargs):
    """Extract rhythm measurements for a sample of poems.

    Computes rhythm measurements (meter, stress patterns, etc.) for each poem
    in the sample, returning a DataFrame with one row per poem. Results are cached
    to disk based on the sample's data name for efficient reuse.

    Parameters
    ----------
    df_smpl : pd.DataFrame
        DataFrame containing poem texts in a 'txt' column, indexed by poem IDs.
    stash : HashStash, default=STASH_RHYTHM
        Cache storage for parsed data.
    force : bool, default=False
        If True, re-parse even if cached data exists.
    gen : bool, default=True
        If True, generate new parses; if False, only use cached data.
    verbose : bool, default=DEFAULT_VERBOSE
        If True, show progress information.
    with_sample : bool, default=False
        If True, join results with original sample data.
    **kwargs
        Additional keyword arguments (unused).

    Returns
    -------
    pd.DataFrame
        DataFrame with rhythm measurements, indexed by poem ID, or empty
        DataFrame if no valid measurements found.

    Calls
    -----
    - _clean_df(df_smpl)
    - get_rhythm_for_txt(txt, stash=stash, force=force) [if gen=True]
    - postprocess_parses_data(stash.get(txt)) [if gen=False]
    """
    df_smpl = _clean_df(df_smpl)
    
    # Check for cached rhythm data based on sample data name
    data_name = getattr(df_smpl, '_data_name', None)
    path = get_path('rhythm_data_for_'+data_name, as_in_paper=df_smpl._as_in_paper, as_replicated=df_smpl._as_replicated) if data_name else None
    if path and not force and os.path.exists(path):
        if verbose:
            print(f"* Loading rhythm data for {data_name} from {path}")
        df_rhythm = pd.read_csv(path).fillna("").set_index('id')
    
    else:
        # Generate rhythm data if not cached or forced
        l = []
        
        def get_res(id, txt):
            if gen:
                res_df = get_rhythm_for_txt(txt, stash=stash, force=force)
            else:
                res_df = postprocess_parses_data(stash.get(txt))
            
            if not type(res_df) == pd.DataFrame or not len(res_df):
                return pd.DataFrame()
            
            return res_df.assign(id=id) if 'id' not in res_df.columns else res_df
        
        for (id, txt) in tqdm(zip(df_smpl.index, df_smpl.txt), desc='* Getting rhythm for sample', total=len(df_smpl)):
            l.append(get_res(id, txt))
        
        if not len(l):
            df_rhythm = pd.DataFrame()
        else:
            df_rhythm = pd.concat(l).set_index('id') if len(l) else pd.DataFrame()
        
        # Save rhythm data to cache if path is available
        print(path, len(df_rhythm))
        if path and len(df_rhythm):
            if verbose:
                print(f"* Saving rhythm data for {data_name} to {path}")
            df_rhythm.to_csv(path)
    
    # Set metadata attributes on output dataframe
    if len(df_rhythm):
        df_rhythm._data_name = data_name
        df_rhythm._sample_by = getattr(df_smpl, '_sample_by', None)
        df_rhythm._as_in_paper = getattr(df_smpl, '_as_in_paper', False)
        df_rhythm._as_replicated = getattr(df_smpl, '_as_replicated', False)
        
        # Join with sample data if requested
        if with_sample:
            df_rhythm = df_rhythm.join(df_smpl, how='left', rsuffix='_from_sample')
    
    return df_rhythm




def postprocess_parses_data(df_parses):
    """Extract rhythm measurements from prosodic parse data.

    Processes raw prosodic parse DataFrame to compute aggregated rhythm metrics
    including iambic pentameter detection, stress patterns, and foot-level
    measurements.

    Parameters
    ----------
    df_parses : pd.DataFrame
        Raw prosodic parse data from Prosodic analysis.

    Returns
    -------
    dict
        Dictionary containing aggregated rhythm measurements including:
        - iambic_pentameter metrics
        - syllable-level stress patterns (up to 10 syllables)
        - foot stress patterns
        - meter composition statistics

    Calls
    -----
    - np.mean() [for averaging measurements]
    - sum() [for summing counts]
    - prosodic.split_scansion() [for foot analysis]
    """
    if df_parses is None or not len(df_parses):
        return pd.DataFrame()
    df_parses = df_parses.reset_index()
    row_d = {}

    data_to_average = defaultdict(list)
    data_to_sum = defaultdict(list)

    ld = []
    for i,row in df_parses.iterrows():
        # get the parse
        stress_profile = row['parse_stress']
        if len(stress_profile) < 4:
            continue

        meter_profile = row['parse_meter']
        parse_ambiguity = row['parse_ambig']
        
        d = {'stanza_num': row['stanza_num'], 'line_num': row['line_num'], 'line_txt': row['line_txt'], 'linepart_num': row['linepart_num'], 'parse_rank': row['parse_rank'], 'parse_txt': row['parse_txt']}
        d['is_iambic_pentameter'] = int(meter_profile == "-+"*5)
        d['is_unambigously_iambic_pentameter'] = int(meter_profile == "-+"*5 and parse_ambiguity == 1)

        # get the stress profile

        num_sylls_for_data = len(stress_profile) if len(stress_profile) < 10 else 10
        for syll_num in range(1,num_sylls_for_data+1):
            syll_stress = stress_profile[syll_num-1]
            d[f'syll{syll_num:02d}_stress'] = int(syll_stress=='+')

        # get foot profile
        d['forth_syllable_stressed'] = int(stress_profile[3]=='+')

        scansion = prosodic.split_scansion(meter_profile)
        d['num_pos_ww'] = len([x for x in scansion if x == 'ww'])
        d['num_pos'] = len(scansion)
        d['num_sylls'] = len(stress_profile)
        d['perc_ww_in_meter'] = d['num_pos_ww']/d['num_pos']

        ld.append(d)

    df_new_parses = pd.DataFrame(ld)
    return df_new_parses

    # data_to_average = {k:float(np.mean(v)) for k,v in data_to_average.items()}
    # data_to_sum = {k:sum(v) for k,v in data_to_sum.items()}
    # data_to_average['perc_ww_in_meter'] = data_to_sum['num_ww_in_meter']/data_to_sum['num_feet_in_meter']
    # return {**row_d, **data_to_sum, **data_to_average}






def get_rhythm_for_txt(txt, **kwargs):
    """Get rhythm measurements for a single poem text.

    Convenience function that retrieves processed rhythm measurements for a text,
    equivalent to calling get_parses_for_txt with postprocess=True.

    Parameters
    ----------
    txt : str
        The poem text to analyze.
    **kwargs
        Additional keyword arguments passed to get_parses_for_txt.

    Returns
    -------
    dict
        Dictionary of rhythm measurements, or empty dict if parsing fails.

    Calls
    -----
    - get_parses_for_txt(txt, **kwargs)
    """
    parses = get_parses_for_txt(txt, **kwargs)
    return postprocess_parses_data(parses)



    
df_attrs = ['_data_name', '_sample_by', '_as_in_paper', '_as_replicated']

def _clean_df(df):
    """Clean and prepare a poem DataFrame for rhythm analysis.

    Fills missing values, sets appropriate index based on available columns,
    and sorts by id_hash for deterministic processing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing poem data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with filled NAs and proper indexing.

    Calls
    -----
    - pd.DataFrame.fillna("")
    - pd.DataFrame.set_index() [if 'id' or 'id_hash' columns exist]
    - pd.DataFrame.sort_values('id_hash') [if 'id_hash' column exists]
    """
    d = {x:getattr(df, x,None) for x in df_attrs}
    
    df = df.fillna("")
    if 'id' in df.columns:
        df = df.set_index('id')
    if 'id_hash' in df.columns:
        df = df.sort_values('id_hash')
    for k,v in d.items():
        setattr(df, k, v)
    return df





PATH_CORPUS_RAW = os.path.join(PATH_RAWDATA, 'corpus')
PATH_RAW_SHAKSONNETS = os.path.join(PATH_CORPUS_RAW, 'shakespeare_sonnets.txt')


def get_rhythm_for_shakespeare_sonnets(force=False):
    """Load and analyze rhythm in Shakespeare's sonnets.

    Reads the complete text of Shakespeare's sonnets, splits into individual poems,
    and computes rhythm measurements for each sonnet.

    Parameters
    ----------
    force : bool, default=False
        If True, re-parse sonnets even if cached data exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with rhythm measurements for each of the 154 sonnets,
        indexed by sonnet ID (e.g., 'shakespeare_sonnet_001').

    Calls
    -----
    - get_id_hash(s) [for each sonnet text]
    - get_rhythm_for_txt(txt, force=force) [for each sonnet]
    """
    assert os.path.exists(PATH_RAW_SHAKSONNETS)
    with open(PATH_RAW_SHAKSONNETS, 'r') as f:
        txt = f.read()
    
    sonnets = [s.strip() for s in txt.strip().split('\n\n') if len(s.strip())]
    ids = [f'shakespeare_sonnet_{i:03d}' for i in range(1,len(sonnets)+1)]
    id_hash = [get_id_hash(s) for s in sonnets]
    assert len(sonnets) == 154

    
    ldf = []
    for i,txt in tqdm(list(enumerate(sonnets)),desc='* Getting rhythm for shakespeare sonnets', total=len(sonnets),position=0):
        id = f'shakespeare_sonnet_{i+1:03d}'
        id_hash = get_id_hash(txt)

        odf = get_rhythm_for_txt(txt, force=force).assign(
            id=id,
            id_hash=id_hash,
            txt=txt
        )
        ldf.append(odf)
    
    if not len(ldf):
        return pd.DataFrame()
    
    df_sonnets = pd.concat(ldf)
    return df_sonnets.set_index('id')





