from . import *

def _format_number_with_commas(x):
    """Format number with comma separators for thousands."""
    if isinstance(x, (int, float)):
        return f'{int(x):,}'
    else:
        # If it's already a string, try to convert to int first
        try:
            return f'{int(x):,}'
        except (ValueError, TypeError):
            return str(x)

def get_period_subcorpus_table(df_smpl, save_latex_to=None, save_latex_to_suffix='tmp',return_display=False, table_num=None, verbose=DEFAULT_VERBOSE, as_in_paper=True, as_replicated=False, **kwargs):
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
        return f'{x:,}'

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




def get_rhyme_promptings_table(df_prompts, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
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
            num_poems_fmt = _format_number_with_commas(num_poems)
            num_per_model_fmt = _format_number_with_commas(num_per_model)
            if i == 0:
                tabular_lines.append(
                    f"\\multirow[t]{{{n}}}{{*}}{{{ptype_disp}}} & {prompt_disp} & {num_poems_fmt} & {num_per_model_fmt} \\\\"
                )
            else:
                tabular_lines.append(
                    f" & {prompt_disp} & {num_poems_fmt} & {num_per_model_fmt} \\\\"
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
        save_latex_to=get_path(DATA_NAME_TABLE_RHYME_PROMPTINGS, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of poems generated for each prompt.",
        label="tab:num_poems_rhyme_promptings",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


### METER


def get_num_poems_per_model_table(df_prompts, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
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
        rh = _format_number_with_commas(row.get("Rhymed", 0))
        ur = _format_number_with_commas(row.get("Unrhymed", 0))
        mu = _format_number_with_commas(row.get("Rhyme unspecified", 0))
        lines.append(f"{model_disp} & {rh} & {ur} & {mu} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tabular_str = "\n".join(lines)

    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_NUM_POEMS_MODELS, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of poems generated for each model and prompt category.",
        label="tab:num_poems_models",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


def get_num_poems_completed_per_model_table(df_counts, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
    """Generate LaTeX table for number of poem completions per model and period.
    
    Creates a formatted table showing the distribution of poem completions
    across different AI models and historical periods. The input DataFrame
    should have periods as the index and model names as columns.
    
    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with periods as index and model names as columns, containing
        counts of poem completions. Expected to be output of 
        df.groupby(['model2','period']).size().unstack().T
    return_display : bool, default=False
        If True, return display object for notebooks.
    as_in_paper : bool, default=True
        If True, use precomputed data paths from paper.
    as_replicated : bool, default=False
        If True, use replicated data paths.
    **kwargs
        Additional arguments passed to df_to_latex_table().
        
    Returns
    -------
    str or display object
        LaTeX table string or display object if return_display=True.
        
    Calls
    -----
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    df = df_counts.copy()
    
    # Expected model order from the paper table
    model_order = ['Claude', 'DeepSeek', 'ChatGPT', 'Llama', 'Olmo']
    
    # Map display names to LaTeX-safe versions
    model_latex_names = {
        'Claude': 'Claude-3-Sonnet',
        'DeepSeek': 'DeepSeek', 
        'ChatGPT': 'ChatGPT-3.5-Turbo',
        'Llama': 'Llama3.1',
        'Olmo': 'Olmo2'
    }
    
    # Reorder columns based on model_order, filling missing with 0
    available_models = [m for m in model_order if m in df.columns]
    df = df.reindex(columns=available_models, fill_value=0)
    
    # Sort periods 
    df = df.sort_index()
    
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('-', '--')
    
    # Build LaTeX tabular
    lines = []
    lines.append('\\begin{tabular}{l' + 'r' * len(df.columns) + '}')
    lines.append('\\toprule')
    
    # Header row with model names
    header_models = [model_latex_names.get(col, col) for col in df.columns]
    header_line = ' & ' + ' & '.join(header_models) + ' \\\\'
    lines.append(header_line)
    lines.append('Period &  ' + '&  ' * (len(df.columns) - 1) + ' \\\\')
    lines.append('\\midrule')
    
    # Data rows
    for period, row in df.iterrows():
        period_disp = _escape_latex_text(str(period))
        values = [_format_number_with_commas(row[col]) for col in df.columns]
        data_line = f'{period_disp} & ' + ' & '.join(values) + ' \\\\'
        lines.append(data_line)
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    
    tabular_str = '\n'.join(lines)
    
    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_NUM_POEMS_COMPLETED_MODELS, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of poem ``completions'' generated per model and the historical period of the completed poem.",
        label="tab:num_poems_completed_models",
        position="H",
        size="\\small", 
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


def get_rhyme_accuracy_table(df_preds_tbl, return_display=False, as_in_paper=True, as_replicated=False, optimal_threshold=4, **kwargs):
    """Generate LaTeX table for rhyme detection accuracy by threshold.
    
    Creates a formatted table showing precision, recall, and F1 scores
    for rhyme detection at different thresholds. The optimal threshold
    row will be bolded.
    
    Parameters
    ----------
    df_preds_tbl : pd.DataFrame
        DataFrame with thresholds as index and metrics as columns.
        Expected columns: ['Precision', 'Recall', 'F1 score']
        Index should be "# Rhymes per 10 lines" with threshold values.
        Values should already be formatted as percentages (e.g., '75%').
    return_display : bool, default=False
        If True, return display object for notebooks.
    as_in_paper : bool, default=True
        If True, use precomputed data paths from paper.
    as_replicated : bool, default=False
        If True, use replicated data paths.
    optimal_threshold : int, default=4
        The threshold value to bold in the table.
    **kwargs
        Additional arguments passed to df_to_latex_table().
        
    Returns
    -------
    str or display object
        LaTeX table string or display object if return_display=True.
        
    Calls
    -----
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    df = df_preds_tbl.copy()
    
    # Filter to only include thresholds 1-10 (exclude 0)
    df = df.loc[df.index >= 1]
    
    # Build LaTeX tabular
    lines = []
    lines.append('\\begin{tabular}{llll}')
    lines.append('\\toprule')
    lines.append(' & Precision & Recall & F1 score \\\\')
    lines.append('\\# Rhymes per 10 lines &  &  &  \\\\')
    lines.append('\\midrule')
    
    # Data rows
    for threshold, row in df.iterrows():
        threshold_str = str(threshold)
        precision = str(row['Precision']).replace('%', '\\%')
        recall = str(row['Recall']).replace('%', '\\%')
        f1_score = str(row['F1 score']).replace('%', '\\%')
        
        # Bold the optimal threshold row
        if threshold == optimal_threshold:
            threshold_str = f'\\textbf{{{threshold_str}}}'
            precision = f'\\textbf{{{precision}}}'
            recall = f'\\textbf{{{recall}}}'
            f1_score = f'\\textbf{{{f1_score}}}'
        
        line = f'{threshold_str} & {precision} & {recall} & {f1_score} \\\\'
        lines.append(line)
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    
    tabular_str = '\n'.join(lines)
    
    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_RHYME_ACCURACY, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Precision, recall, and F1 score for rhyme detection in 1,000 poems sampled from those marked as ``rhyming'' and ``unrhyming'' in the metadata of the Chadwyck-Healey poetry collections.",
        label="tab:rhyme_validation",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


def get_memorization_table(df_counts, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
    """Generate LaTeX table for memorization detection results by period.
    
    Creates a formatted table showing the number of poems found/not found
    in closed model output and open training data across historical periods.
    
    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with periods as index and multi-level columns:
        (source, found_status) where source is 'Closed model output' or 
        'Open training data' and found_status is 'Found' or 'Not found'.
        Expected to be output of df_mem.groupby(['found_source', 'found', 'Period']).size().unstack().T
    return_display : bool, default=False
        If True, return display object for notebooks.
    as_in_paper : bool, default=True
        If True, use precomputed data paths from paper.
    as_replicated : bool, default=False
        If True, use replicated data paths.
    **kwargs
        Additional arguments passed to df_to_latex_table().
        
    Returns
    -------
    str or display object
        LaTeX table string or display object if return_display=True.
        
    Calls
    -----
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    df = df_counts.copy()
    
    # Sort periods 
    df = df.sort_index()
    
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('-', '--')
    
    # Build LaTeX tabular - 5 columns (period + 4 data columns)
    lines = []
    lines.append('\\begin{tabular}{lrrrr}')
    lines.append('\\toprule')
    
    # Multi-column header
    lines.append(' & \\multicolumn{2}{r}{Closed model output} & \\multicolumn{2}{r}{Open training data} \\\\')
    lines.append('& Found & Not found & Found & Not found \\\\')
    lines.append('Period &  &  &  &  \\\\')
    lines.append('\\midrule')
    
    # Data rows
    for period, row in df.iterrows():
        period_disp = _escape_latex_text(str(period))
        
        # Get values in the expected order: Closed (Found, Not found), Open (Found, Not found)
        closed_found = _format_number_with_commas(row[('Closed model output', 'Found')])
        closed_not_found = _format_number_with_commas(row[('Closed model output', 'Not found')])
        open_found = _format_number_with_commas(row[('Open training data', 'Found')])
        open_not_found = _format_number_with_commas(row[('Open training data', 'Not found')])
        
        line = f'{period_disp} & {closed_found} & {closed_not_found} & {open_found} & {open_not_found} \\\\'
        lines.append(line)
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    
    tabular_str = '\n'.join(lines)
    
    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_MEMORIZATION, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of poems found in open model training data (using WIMBD) and closed model output (using ``memorization'' detection).",
        label="tab:num_poems_found_not_found",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


def get_sonnets_table(odf, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
    """Generate LaTeX table for sonnets counts by source and model/period.
    
    Creates a formatted table showing the number of sonnets from historical
    sources and LLMs, with grouped rows using multirow for sources.
    
    Parameters
    ----------
    odf : pd.DataFrame
        DataFrame with multi-level index ['Source', ''] and one column '# Sonnets'.
        Expected to be output of df_sonnets.groupby(['source2', 'model2']).size()
        formatted as shown in the notebook.
    return_display : bool, default=False
        If True, return display object for notebooks.
    as_in_paper : bool, default=True
        If True, use precomputed data paths from paper.
    as_replicated : bool, default=False
        If True, use replicated data paths.
    **kwargs
        Additional arguments passed to df_to_latex_table().
        
    Returns
    -------
    str or display object
        LaTeX table string or display object if return_display=True.
        
    Calls
    -----
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    df = odf.copy()
    
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')
    
    # Build LaTeX tabular
    lines = []
    lines.append('\\begin{tabular}{llr}')
    lines.append('\\toprule')
    lines.append(' &  &  \\\\')
    lines.append('Source &  & \\# Sonnets  \\\\')
    lines.append('\\midrule')
    
    # Get unique sources and process each group
    sources = df.index.get_level_values(0).unique()
    
    for source in sources:
        # Get all entries for this source
        source_df = df.xs(source, level=0)
        n_rows = len(source_df)
        
        # Sort the entries - for Historical: Shakespeare first, then centuries
        # For LLM: alphabetical by model name
        if source == 'Historical':
            # Custom sort: Shakespeare first, then C17, C18, C19, C20
            model_order = ['Shakespeare', 'C17', 'C18', 'C19', 'C20']
            # Reindex to get the desired order, filling missing with 0
            available_models = [m for m in model_order if m in source_df.index]
            source_df = source_df.reindex(available_models, fill_value=0)
        else:
            # For LLM, sort alphabetically
            source_df = source_df.sort_index()
        
        # Generate rows for this source
        for i, (model, row) in enumerate(source_df.iterrows()):
            source_disp = _escape_latex_text(source) if i == 0 else ''
            model_disp = _escape_latex_text(str(model))
            sonnet_count = _format_number_with_commas(row['# Sonnets'])
            
            if i == 0:
                lines.append(f'\\multirow[t]{{{n_rows}}}{{*}}{{{source_disp}}} & {model_disp} & {sonnet_count} \\\\')
            else:
                lines.append(f' & {model_disp} & {sonnet_count} \\\\')
        
        # Add midrule between sources (except after the last one)
        if source != sources[-1]:
            lines.append('\\midrule')
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    
    tabular_str = '\n'.join(lines)
    
    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_SONNETS, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of sonnets sampled from the Chadwyck-Healey corpus and generated by LLMs.",
        label="tab:num_sonnets_corpus",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )


def get_text_vs_instruct_table(df_counts, return_display=False, as_in_paper=True, as_replicated=False, **kwargs):
    """Generate LaTeX table for text vs instruct completions by period.
    
    Creates a formatted table showing the number of poem completions
    generated by llama3.1:instruct vs llama3.1:text models across
    historical periods.
    
    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with periods as index and model types as columns.
        Expected columns: ['llama3.1:instruct', 'llama3.1:text']
        Expected to be output of df.groupby(['model','period']).size().unstack().T
    return_display : bool, default=False
        If True, return display object for notebooks.
    as_in_paper : bool, default=True
        If True, use precomputed data paths from paper.
    as_replicated : bool, default=False
        If True, use replicated data paths.
    **kwargs
        Additional arguments passed to df_to_latex_table().
        
    Returns
    -------
    str or display object
        LaTeX table string or display object if return_display=True.
        
    Calls
    -----
    - df_to_latex_table(...) [to generate LaTeX table output]
    """
    df = df_counts.copy()
    
    # Sort periods 
    df = df.sort_index()
    
    def _escape_latex_text(s):
        return str(s).replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('-', '--')
    
    # Build LaTeX tabular - 3 columns (period + 2 model columns)
    lines = []
    lines.append('\\begin{tabular}{lrr}')
    lines.append('\\toprule')
    lines.append(' & llama3.1:instruct & llama3.1:text \\\\')
    lines.append(' &  &  \\\\')
    lines.append('\\midrule')
    
    # Data rows
    for period, row in df.iterrows():
        period_disp = _escape_latex_text(str(period))
        instruct_count = _format_number_with_commas(row['llama3.1:instruct'])
        text_count = _format_number_with_commas(row['llama3.1:text'])
        
        line = f'{period_disp} & {instruct_count} & {text_count} \\\\'
        lines.append(line)
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    
    tabular_str = '\n'.join(lines)
    
    return df_to_latex_table(
        inner_latex=tabular_str,
        save_latex_to=get_path(DATA_NAME_TABLE_TEXT_VS_INSTRUCT, as_in_paper=as_in_paper, as_replicated=as_replicated),
        caption="Number of poem ``completions,'' per historical period of original poem, generated by llama3.1:instruct and llama3.1:text.",
        label="tab:num_poems_instruct_text",
        position="H",
        size="\\small",
        singlespacing=True,
        return_display=return_display,
        **kwargs,
    )
