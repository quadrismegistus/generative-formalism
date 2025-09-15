from . import *

@timeout(3)
def get_rhyme_for_txt(txt, max_dist=RHYME_MAX_DIST, stash=STASH_RHYME, force=False):
    stash_key = (txt, max_dist)
    if not force and stash is not None and stash_key in stash:
        return stash[stash_key]

    out = {}    
    try:
        txt = limit_lines(txt)
        text = prosodic.Text(txt)
        rhyme_d = text.get_rhyming_lines(max_dist=max_dist)
        all_rhyming_lines = set()
        all_perfectly_rhyming_lines = set()
        for l1, (score, l2) in rhyme_d.items():
            all_rhyming_lines.update({l1, l2})
            if not score:
                all_perfectly_rhyming_lines.update({l1, l2})
        num_rhyming_lines = len(all_rhyming_lines)
        num_perfectly_rhyming_lines = len(all_perfectly_rhyming_lines)
        num_lines = text.num_lines
        out = {
            'num_rhyming_lines': num_rhyming_lines,
            'num_perfectly_rhyming_lines': num_perfectly_rhyming_lines,
            'num_lines': num_lines,
            'rhyming_line_pairs': [
                (l2.txt.strip(), l1.txt.strip(), score)
                for l1, (score, l2) in rhyme_d.items()
            ],
        }
    except Exception:
        pass
    
    stash[stash_key] = out
    return out


def get_rhyme_for_sample(df_smpl, max_dist=RHYME_MAX_DIST, stash=STASH_RHYME, force=False, verbose=DEFAULT_VERBOSE, **kwargs):
    df = df_smpl.fillna("")
    if 'id' in df.columns:
        df = df.set_index('id')
    df = df.sort_values('id_hash')

    # cache = dict(stash.items())
    # cache = stash

    def get_res(txt):
        # if not force and txt in cache:
            # return cache[txt]
        res = get_rhyme_for_txt(txt, max_dist=max_dist, stash=stash, force=force)
        return res

    df_rhymes = pd.DataFrame((get_res(txt) for txt in tqdm(df.txt,desc='* Getting rhymes for sample')), index=df.index)
    return postprocess_rhyme_sample(df, df_rhymes)


def postprocess_rhyme_sample(df_poems, df_rhymes, rhyme_threshold=4, with_sample=False):
    # df = df_poems.join(df_rhymes, rsuffix='_prosodic', how='left')
    df = df_rhymes.reset_index()
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

    if 'id' in df.columns:
        df = df.drop_duplicates(subset='id')
        odf = df.set_index('id')
    
    if with_sample:
        odf = odf.join(df_poems, how='left', rsuffix='_from_sample')
    return odf

def load_rhyme_data(path=None):
    if path and os.path.exists(path):
        return pd.read_csv(path).fillna("").set_index('id')
    else:
        return None


def get_rhyme_data_for(get_func, output_path=None, overwrite=False, with_sample=False, **kwargs):
    df_smpl = get_func(**kwargs)

    if not overwrite and output_path and os.path.exists(output_path):
        df_rhyme_data = load_rhyme_data(output_path)
    else:
        df_rhyme_data = get_rhyme_for_sample(df_smpl, **kwargs)
        if output_path:
            save_sample(df_rhyme_data, output_path, overwrite=True)

    if with_sample:
        print(f"* Joining sample and rhyme data")
        assert df_smpl.index.name == 'id', "Sample dataframe must have an 'id' index"
        assert df_rhyme_data.index.name == 'id', "Rhyme data dataframe must have an 'id' index"
        df_smpl_w_rhyme_data = df_rhyme_data.join(df_smpl, how='left', rsuffix='_from_sample').fillna("")
        return df_smpl_w_rhyme_data
    else:
        return df_rhyme_data







def get_rhyme_data_for_corpus_sampled_by(sample_by, as_in_paper=True, as_replicated=False, output_path=None, **kwargs):
    """Get rhyme data for corpus sampled by specified criteria.

    Loads a corpus sample using the specified criteria and computes rhyme
    measurements for all poems in the sample. Results can be cached to disk
    for efficient reuse.

    Parameters
    ----------
    sample_by : str
        Sampling criteria ('period', 'period_subcorpus', 'rhyme', 'sonnet_period').
    as_in_paper : bool, default=True
        If True, use precomputed sample from paper.
    as_replicated : bool, default=False
        If True, use replicated sample.
    output_path : str, optional
        Path to save rhyme data. If None, uses default path based on sample_by.
    **kwargs
        Additional arguments passed to get_rhyme_data_for().

    Returns
    -------
    Rhyme analysis data for the sampled corpus.

    Calls
    -----
    - get_chadwyck_corpus_sampled_by(...) [to load the sample]
    - get_rhyme_data_for(get_sample_func, output_path, **kwargs) [to compute rhyme data]
    """
    # Map sample_by to output path if not provided
    output_path_map = {
        'rhyme': PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_RHYME if as_in_paper else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_RHYME,
        'period': PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_PERIOD if as_in_paper else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_PERIOD,
        'period_subcorpus': PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_PERIOD_SUBCORPUS if as_in_paper else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_PERIOD_SUBCORPUS,
        'sonnet_period': PATH_RHYME_DATA_FOR_PAPER_SAMPLE_BY_SONNET_PERIOD if as_in_paper else PATH_RHYME_DATA_FOR_REPLICATED_SAMPLE_BY_SONNET_PERIOD,
    }

    output_path = output_path if output_path else output_path_map.get(sample_by, None)

    def get_sample_func():
        return get_chadwyck_corpus_sampled_by(sample_by, as_in_paper=as_in_paper, as_replicated=as_replicated)

    return get_rhyme_data_for(get_sample_func, output_path, **kwargs)











def compare_rhyme_data_by_group(
    df, 
    groupby=['period'], 
    valname='rhyme_pred_perc', 
    min_group_size=100,
    verbose=DEFAULT_VERBOSE,
):
    return compare_data_by_group(df, groupby=groupby,valname=valname,min_group_size=min_group_size,verbose=verbose)


def plot_predicted_rhyme_avgs(
    df, 
    y='rhyme_pred_perc', 
    x='period', 
    gby=['period'], 
    color=None, 
    limits=[0,100], 
    min_size=10,
    title=None,
    xlabel=None,
    ylabel=None
):
    """
    Plot predicted rhyme averages with error bars using stderr from statistical analysis.
    
    Uses get_avgs_df from stats.py to compute means and standard errors, then creates
    a plot with error bars representing the stderr whiskers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame containing rhyme prediction data
    y : str
        Column name for y-axis values (default: 'rhyme_pred_perc')
    x : str  
        Column name for x-axis values (default: 'period')
    gby : list
        List of columns to group by for aggregation (default: ['period'])
    color : str or None
        Column name for color grouping (default: None)
    limits : list
        Y-axis limits as [min, max] (default: [0,100])
    min_size : int
        Minimum group size for inclusion (default: 10)
    title : str or None
        Plot title (default: auto-generated)
    xlabel : str or None
        X-axis label (default: auto-generated)
    ylabel : str or None
        Y-axis label (default: auto-generated)
        
    Returns:
    --------
    plotnine plot object
    """

    p9.options.figure_size = (10, 6)
    p9.options.dpi=300
    
    # Get aggregated data with means and stderr using stats function
    figdf = get_avgs_df(df, gby=gby, y=y)
    
    # Filter by minimum size
    figdf = figdf[figdf['count'] >= min_size]
    
    # Set up default labels
    if xlabel is None:
        xlabel = 'Half-century of poet\'s birth' if x == 'period' else x.replace('_', ' ').title()
    if ylabel is None:
        ylabel = 'Predicted percentage of poems with rhyme' if y == 'rhyme_pred_perc' else y.replace('_', ' ').title()
    if title is None:
        title = f'Rhyme Prediction Averages by {x.replace("_", " ").title()}'
    
    # Create the base plot
    plot_aes = p9.aes(x=x, y='mean')
    if color:
        plot_aes = p9.aes(x=x, y='mean', color=color, group=color)
    
    plot = (
        p9.ggplot(figdf, plot_aes)
        # + p9.geom_point(p9.aes(size='count'), alpha=0.25)
        + p9.geom_errorbar(
            p9.aes(ymin='mean - stderr', ymax='mean + stderr'),
            width=0.2,
            size=1,
            alpha=1.0
        )
    )
    
    # Add line if color grouping is used
    if color:
        plot += p9.geom_line(alpha=1)
    
    # Add styling and reference line
    plot += p9.geom_hline(yintercept=50, color='black', linetype='dashed', size=0.5, alpha=0.5)
    
    plot += p9.theme(
        panel_background=p9.element_rect(fill='white'), 
        plot_background=p9.element_rect(fill='white'),
        legend_position='bottom',
        axis_text_x=p9.element_text(angle=45)
    )
    plot += p9.scale_y_continuous(limits=limits)
    plot += p9.labs(
        x=xlabel,
        y=ylabel,
        title=title,
        size='Sample Size',
        color=color.replace('_', ' ').title() if color else None,
    )
    
    return plot



    