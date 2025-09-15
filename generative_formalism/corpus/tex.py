from . import *

def get_period_subcorpus_table(df_smpl, save_latex_to=None, save_latex_to_suffix='tmp',return_display=False, table_num=None, verbose=False, as_in_paper=True, as_replicated=False, **kwargs):
    """Build a period×subcorpus summary table and optionally save LaTeX.

    Creates a formatted table showing the distribution of poems and poets
    across historical periods and literary subcorpora. Can generate LaTeX
    output for academic papers and display formatted tables in notebooks.

    Parameters
    ----------
    df_smpl : pd.DataFrame
        Sampled DataFrame containing 'period', 'subcorpus', 'author', 'id' columns.
    save_latex_to : str, optional
        Base path for LaTeX/table image output. If falsy, skip saving.
    save_latex_to_suffix : str, default='tmp'
        Filename suffix for differentiation between different table versions.
    return_display : bool, default=False
        If True, return a display object suitable for Jupyter notebooks.
    table_num : int, optional
        Table number for LaTeX captioning.
    verbose : bool, default=False
        If True, print progress information during processing.
    **kwargs
        Additional arguments passed to df_to_latex_table().

    Returns
    -------
    pd.DataFrame or display object
        A formatted DataFrame with period×subcorpus statistics, or a display
        object if return_display=True.

    Calls
    -----
    - get_chadwyck_corpus_metadata(verbose=verbose) [to get full corpus stats]
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    from .corpus import get_chadwyck_corpus_metadata
    df_meta = get_chadwyck_corpus_metadata(verbose=verbose)
    
    # Build summary table using sample groupings (df_smpl)
    rows = []
    for (period, subcorpus), gdf in df_smpl.groupby(['period', 'subcorpus']):
        meta_q = df_meta.query(f'subcorpus=="{subcorpus}" & period=="{period}"')
        rows.append({
            'period': period,
            'subcorpus': subcorpus,
            'num_poets_total': meta_q.author.nunique(),
            'num_poets': gdf.author.nunique(),
            'num_poems_total': len(meta_q),
            'num_poems': len(gdf),
        })

    df_table = pd.DataFrame(rows).set_index(['period', 'subcorpus']).sort_index()

    # convert numbers to comma'd strings
    def format_number(x):
        x = int(x)
        return f'{x:,.0f}'

    df_formatted = df_table.applymap(format_number)

    df_formatted.rename_axis(['Period', 'Subcorpus'], inplace=True)
    df_formatted.columns = ['# Poets (corpus)', '# Poets (sample)', '# Poems (corpus)', '# Poems (sample)']

    # Build grouped LaTeX tabular matching the requested style (no outer table here)
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')

    periods = sorted({idx[0] for idx in df_formatted.index})

    tabular_lines = []
    tabular_lines.append('\\begin{tabular}{llrrrr}')
    tabular_lines.append('\\toprule')
    tabular_lines.append('& & \\multicolumn{2}{c}{Corpus} & \\multicolumn{2}{c}{Sample} \\\\')
    tabular_lines.append('\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}')
    tabular_lines.append('Period & Subcorpus & \\# Poems & \\# Poets & \\# Poems & \\# Poets \\\\')
    tabular_lines.append('\\midrule')

    for period in periods:
        subdf = df_formatted.xs(period, level=0)
        subcorp_order = sorted(list(subdf.index))
        n = len(subcorp_order)
        for i, subcorpus in enumerate(subcorp_order):
            row = subdf.loc[subcorpus]
            period_disp = _escape_latex_text(period).replace('-', '--') if i == 0 else ''
            sub_disp = _escape_latex_text(subcorpus)
            vals = [
                row['# Poems (corpus)'],
                row['# Poets (corpus)'],
                row['# Poems (sample)'],
                row['# Poets (sample)'],
            ]
            if i == 0:
                tabular_lines.append(f'\\multirow[t]{{{n}}}{{*}}{{{period_disp}}} & {sub_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\')
            else:
                tabular_lines.append(f' & {sub_disp} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\')
        tabular_lines.append('\\cline{1-6}')

    # replace trailing cline with bottomrule
    if tabular_lines[-1] == '\\cline{1-6}':
        tabular_lines[-1] = '\\bottomrule'
    else:
        tabular_lines.append('\\bottomrule')

    tabular_lines.append('\\end{tabular}')

    tabular_str = '\n'.join(tabular_lines)

    if save_latex_to:
        # Map sample type to appropriate table data name
        data_name_map = {
            'period': DATA_NAME_TABLE_PERIOD_COUNTS,
            'period_subcorpus': DATA_NAME_TABLE_PERIOD_SUBCORPUS_COUNTS,
            'sonnet_period': DATA_NAME_TABLE_SONNET_PERIOD_COUNTS,
        }
        
        # If save_latex_to is a sample type, get the corresponding data name
        if save_latex_to in data_name_map:
            data_name = data_name_map[save_latex_to]
            save_path = get_path(data_name, as_in_paper=as_in_paper, as_replicated=as_replicated)
        else:
            # If save_latex_to is a custom path, use it directly
            save_path = save_latex_to
            
        _ = df_to_latex_table(
            inner_latex=tabular_str,
            save_latex_to=save_path,
            save_latex_to_suffix=None,
            table_num=table_num,
            caption='Number of poets and poems in the Chadwyck-Healey corpus and sample.',
            label='tab:num_poems_corpus',
            position='t',
            center=True,
            size='\\small',
            resize_to_textwidth=True,
            return_display=return_display,
            verbose=verbose,
        )
        if return_display and _ is not None:
            return _

    return df_formatted


# def get_period_subcorpus_table_by_sample(sample_by='period_subcorpus', as_in_paper=True, as_replicated=False, df_smpl=None, save_latex=True, return_display=False, table_num=None, save_latex_to=None, **kwargs):
#     """Generate period×subcorpus table for the specified sample criteria.

#     Convenience function that loads a sample (if not provided) and generates
#     a period×subcorpus summary table with appropriate suffixes for different
#     sample types (paper vs. replicated).

#     Parameters
#     ----------
#     sample_by : str, default='period_subcorpus'
#         Sampling criteria ('period', 'period_subcorpus', 'rhyme', 'sonnet_period').
#     as_in_paper : bool, default=True
#         If True, use precomputed sample from paper.
#     as_replicated : bool, default=False
#         If True, use replicated sample.
#     df_smpl : pd.DataFrame, optional
#         Pre-loaded sample DataFrame. If None, loads using sample criteria.
#     save_latex : bool, default=True
#         If True, save LaTeX output.
#     return_display : bool, default=False
#         If True, return display object for notebooks.
#     table_num : int, optional
#         Table number for LaTeX captioning.
#     **kwargs
#         Additional arguments passed to get_period_subcorpus_table().

#     Returns
#     -------
#     pd.DataFrame or display object
#         Formatted table with period×subcorpus statistics.

#     Calls
#     -----
#     - get_chadwyck_corpus_sampled_by(...) [if df_smpl is None]
#     - get_period_subcorpus_table(...) [to generate the actual table]
#     """
#     if df_smpl is None:
#         df_smpl = get_chadwyck_corpus_sampled_by(sample_by, as_in_paper=as_in_paper, as_replicated=as_replicated)

#     suffix = PAPER_REGENERATED_SUFFIX if as_in_paper else REPLICATED_SUFFIX
#     if table_num is None:
#         table_num = TABLE_NUM_PERIOD_SUBCORPUS_COUNTS

#     # Set default save path based on sample type if not provided
#     if save_latex_to is None:
#         data_name_map = {
#             'period': DATA_NAME_TABLE_PERIOD_COUNTS,
#             'period_subcorpus': DATA_NAME_TABLE_PERIOD_SUBCORPUS_COUNTS,
#             'sonnet_period': DATA_NAME_TABLE_SONNET_PERIOD_COUNTS,
#         }
#         data_name = data_name_map.get(sample_by, DATA_NAME_TABLE_PERIOD_SUBCORPUS_COUNTS)
#         save_latex_to = get_path(data_name, as_in_paper=as_in_paper, as_replicated=as_replicated)

#     return get_period_subcorpus_table(
#         df_smpl,
#         save_latex_to=save_latex_to,
#         save_latex_to_suffix=suffix,
#         return_display=return_display,
#         table_num=table_num,
#         **kwargs
#     )




def display_period_subcorpus_tables(df, **kwargs):
    """Display period×subcorpus summary tables for a sampled DataFrame.

    Creates and displays formatted tables showing the breakdown of poems
    and poets by period and subcorpus in the provided sample DataFrame.
    Uses IPython display if available, otherwise prints to console.

    Parameters
    ----------
    df : pd.DataFrame
        Sampled DataFrame containing 'period' and 'subcorpus' columns.
    **kwargs
        Additional arguments passed to get_period_subcorpus_table().

    Returns
    -------
    None
        Displays the table but doesn't return a value.

    Calls
    -----
    - get_period_subcorpus_table(df, return_display=True, **kwargs)
    - try_display(result) [to display the table in IPython or console]
    """
    kwargs['return_display'] = True
    try_display(get_period_subcorpus_table(df, **kwargs))

