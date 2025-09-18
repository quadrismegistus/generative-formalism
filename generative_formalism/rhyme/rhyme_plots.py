from . import *

def plot_predicted_rhyme_avgs(
    df, 
    y='rhyme_pred_perc', 
    x='period', 
    gby=None, 
    color=None, 
    limits=[0,100], 
    min_size=10,
    title=None,
    xlabel=None,
    ylabel=None,
    force=False,
    verbose=DEFAULT_VERBOSE
):
    """
    Plot predicted rhyme averages with error bars using stderr from statistical analysis.
    
    Uses get_avgs_df from stats.py to compute means and standard errors, then creates
    a plot with error bars representing the stderr whiskers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rhyme prediction data.
    y : str, default='rhyme_pred_perc'
        Column name for y-axis values.
    x : str, default='period'
        Column name for x-axis values.
    gby : list, default=['period']
        List of columns to group by for aggregation.
    color : str, optional
        Column name for color grouping.
    limits : list, default=[0,100]
        Y-axis limits as [min, max].
    min_size : int, default=10
        Minimum group size for inclusion.
    title : str, optional
        Plot title (auto-generated if None).
    xlabel : str, optional
        X-axis label (auto-generated if None).
    ylabel : str, optional
        Y-axis label (auto-generated if None).

    Returns
    -------
    plotnine.ggplot
        Plot object showing rhyme prediction averages with error bars.

    Calls
    -----
    - get_avgs_df(df, gby=gby, y=y) [to compute aggregated statistics]
    """
    p9.options.figure_size = (10, 6)
    p9.options.dpi=300


    df_smpl = df
    data_name = getattr(df_smpl, '_data_name', None)
    path = get_path(data_name, as_in_paper=df_smpl._as_in_paper, as_replicated=df_smpl._as_replicated, is_figure=True) if data_name else None
    if path and not force and os.path.exists(path):
        if verbose:
            printm(f"* Loading rhyme data for `{data_name}` from `{nice_path(path)}`")
        return try_display(path)
    
    if not gby:
        if data_name:
            if 'period_subcorpus' in data_name:
                gby=['period','subcorpus']
                color = 'subcorpus'
            elif 'period' in data_name:
                gby = ['period']
        else:
            gby=['period']

    

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
            size=.5,
            alpha=1.0
        )
    )

    plot += p9.geom_point(p9.aes(size='count'), shape='o',alpha=.333)
    plot += p9.geom_line(alpha=1)

    # Add styling and reference line
    plot += p9.geom_hline(yintercept=50, color='black', linetype='dashed', size=0.5, alpha=0.5)
    plot += p9.theme_minimal()
    plot += p9.theme(
        panel_background=p9.element_rect(fill='white'),
        plot_background=p9.element_rect(fill='white'),
        # legend_position='bottom',
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

    if path:
        if verbose:
            printm(f'* Saving figure to `{nice_path(path)}`')
            plot.save(path)

    return plot



def plot_human_genai_rhyme_data(df_both, y='rhyme_pred_perc', gby=['period', 'prompt_type', 'source', 'xcol'], color='prompt_type', force=False, verbose=DEFAULT_VERBOSE):
    p9.options.figure_size=(10,5)

    path = get_path_for_df(df_both, is_figure=True)
    if path and not force and os.path.exists(path):
        if verbose:
            printm(f'* Loading figure from `{nice_path(path)}`')
        return try_display(path)

    figdf = df_both.copy()
    
    figdf['source'] = pd.Categorical(figdf['source'], categories=['Historical poems', 'Generative poems'], ordered=True)

    fig = plot_predicted_rhyme_avgs(
        figdf,
        x='xcol',
        y=y,
        gby=gby,
        color=color,
        force=True
    )
    
    fig += p9.theme(axis_text_x=p9.element_text(angle=45))
    fig += p9.facet_wrap("source", scales="free_x")
    fig += p9.labs(
        color='Prompt type',
        y='Predicted percentage of rhyming poems',
        x='Historical period / Generative model',
        title='Predicted percentage of rhyming poems by historical period and generative model',
        size='Number of poems'
    )
    # fig.save(f'../figures/predicted_rhyme_avgs_{y}_std.png')
    fig.save(path)
    return fig



def plot_fig_avgs_prompts(df_genai_rhyme_promptings_rhyme, x='prompt', color='model', model='ChatGPT-3.5 (OpenAI)', force=False, y='rhyme_pred_perc', verbose=DEFAULT_VERBOSE):

    assert 'rhyme_data' in df_genai_rhyme_promptings_rhyme._data_name
    figdf = df_genai_rhyme_promptings_rhyme

    path = get_path_for_df(figdf, is_figure=True).replace('.csv','')+'.by_prompt.png'
    if path and not force and os.path.exists(path):
        if verbose:
            printm(f'* Loading figure from `{nice_path(path)}`')
        return try_display(path)
    
    figdf = figdf.copy()
    p9.options.figure_size=(16,10)

    # Convert mixed type columns to string to avoid comparison errors
    figdf[x] = figdf[x].astype(str)
    if color in figdf.columns:
        figdf[color] = figdf[color].astype(str)
    figdf['model'] = figdf['model'].apply(rename_model)
    
    stats_df = get_avgs_df(figdf, gby=['prompt', 'prompt_type', 'model'], y=y)
    
    # prompts = list(figdf.sort_values('prompt_type').prompt.drop_duplicates())
    prompts = stats_df.groupby(x)['mean'].median().sort_values().index
    stats_df['prompt'] = pd.Categorical(stats_df['prompt'], categories=prompts)
    
    # Get unique prompt types and assign colors
    if 'prompt_type' in figdf.columns:
        prompt_types = figdf['prompt_type'].unique()
        Set1 = [
        # '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#984ea3',  # Purple
        '#4daf4a',  # Green
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999'   # Gray
        ]
        prompt_type_colors = dict(zip(prompt_types, Set1))
        
        # Map prompts to their types and colors
        prompt_to_type = figdf.set_index('prompt')['prompt_type'].to_dict()
        prompt_colors = {prompt: prompt_type_colors[prompt_to_type[prompt]] 
                        for prompt in prompts if prompt in prompt_to_type}
    
    # Create the plot with error bars
    fig = (p9.ggplot(stats_df, p9.aes(x=x, y='mean'))
        + p9.geom_line(p9.aes(group=color), linetype='dashed', alpha=.5)
        + p9.geom_errorbar(
            p9.aes(
                ymin='mean - stderr',
                ymax='mean + stderr',
                # color=color,
                # group=color
            ),
            width=0.5,
            # position=p9.position_dodge(width=0.5)
        )
        + p9.labs(
            x='Model',
            y='Percentage of poems rhyming',
            title='Likelihood of rhyming poems across prompts'
        )
        + p9.theme_minimal()
        + p9.coord_flip()
        + p9.scale_y_continuous(limits=[0,100])
        + p9.geom_hline(yintercept=50, linetype='dashed', color='gray', alpha=.5)
        + p9.facet_wrap('model', nrow=2)
        + p9.theme(plot_background=p9.element_rect(fill='white'), panel_background=p9.element_rect(fill='white'))
    )
    
    # Customize axis text colors based on prompt_type
    if 'prompt_type' in figdf.columns:
        # Create a theme element to color the axis text
        colored_theme = p9.theme(
            axis_text_y=p9.element_text(color=[prompt_colors.get(tick, 'black') for tick in prompts])
        )
        fig = fig + colored_theme
    fig.save(path)
    return fig


def plot_rhyme_for_genai_human_completions(
    df=None,
    force=False,
    verbose=DEFAULT_VERBOSE,
):
    p9.options.figure_size = (10, 6)
    p9.options.dpi = 300
    
    if df is None:
        df = get_rhyme_for_genai_human_completions()

    path = get_path_for_df(df, is_figure=True)
    if not force and os.path.exists(path):
        if verbose: 
            printm(f'* Loading figure for genai human completions from {nice_path(path)}')
        return try_display(path)


    figdf_avg = get_avgs_df(df, gby=['period','model'])
    bad_models = [m for m in figdf_avg.model.unique() if not rename_model(m)]
    printm(f'* Excluding models: {", ".join(["`"+m+"`" for m in bad_models])}')
    figdf_avg = figdf_avg[~figdf_avg.model.isin(bad_models)]
    figdf_avg['model'] = figdf_avg['model'].apply(rename_model)
    models = [HIST] + [x for x in sorted(figdf_avg.model.unique()) if x!=HIST]
    figdf_avg['model'] = pd.Categorical(figdf_avg.model, categories=models)


    fig = p9.ggplot(figdf_avg, p9.aes(x='period', y='mean', color='model', group='model'))
    fig += p9.geom_point(p9.aes(size='count'), shape='o', alpha=.3)
    fig += p9.geom_line()
    fig += p9.geom_errorbar(
        p9.aes(ymin='mean - stderr', ymax='mean + stderr'),
        width=0.25,
        # alpha=0.6
    )
    fig += p9.theme_minimal()

    fig += p9.geom_text(
        p9.aes(label='model'),
        data=figdf_avg[figdf_avg.period=='1950-2000'],
        adjust_text={'x':0.05, 'y':0.05, 'arrowprops': dict(arrowstyle='-', alpha=0)},
        
        show_legend=False,
    )

    fig += p9.theme(panel_background=p9.element_rect(fill='white'), plot_background=p9.element_rect(fill='white'))
    fig += p9.geom_hline(yintercept=50, color='gray', linetype='dashed')
    fig += p9.scale_y_continuous(limits=(0,100))
    fig += p9.labs(
        x='Half-century of poet\'s birth',
        y='Predicted percentage of rhyming poems',
        color='Model',
        size='Number of poems',
        title='Predicted percentage of rhyming poems in generative completions of poems by their historical period',

    )
    fig.save(path)
    return fig




def plot_rhyme_in_memorized_poems(df=None, force=False, verbose=True):
    if df is None:
        df_mem = get_all_memorization_data()
        df = get_rhyme_for_sample(df_mem, with_sample=True)
    
    path = get_path_for_df(df, is_figure=True)
    if path and not force and os.path.exists(path):
        if verbose:
            printm(f'* Loading plot from `{nice_path(path)}`')
        return try_display(path)


    LAB_CLOSED = 'Memorized by closed models\n(Chadwyck-Healey + ChatGPT, Claude, DeepSeek, Llama)'
    LAB_OPEN = 'Found in open model training data\n(Chadwyck-Healey + Dolma)'
    LAB_ANTONIAK_CLOSED = '[Walsh, Preus, and Antoniak 2024] Memorized by closed models\n(Poetry Foundation/Academy of American Poets + ChatGPT)'
    LAB_ANTONIAK_OPEN = '[Walsh, Preus, and Antoniak 2024] Found in open model training data\n(Poetry Foundation/Academy of American Poets + Dolma)'

    def get_found_label(found_source_corpus):
        if found_source_corpus=='open|chadwyck':
            return LAB_OPEN
        elif found_source_corpus=='closed|chadwyck':
            return LAB_CLOSED
        elif found_source_corpus=='open|antoniak-et-al':
            return LAB_ANTONIAK_OPEN
        elif found_source_corpus=='closed|antoniak-et-al':
            return LAB_ANTONIAK_CLOSED
    
    figdf = get_avgs_df(df, ['found_source', 'found','found_source_corpus'])
    figdf.to_csv(path+'.csv')
    figdf['label'] = figdf['found_source_corpus'].apply(get_found_label)
    figdf['label'] = pd.Categorical(figdf['label'], categories=[LAB_OPEN, LAB_CLOSED, LAB_ANTONIAK_OPEN, LAB_ANTONIAK_CLOSED])
    figdf['found'] = figdf['found'].map({True:'Found', False:'Not found'})
    figdf['found'] = pd.Categorical(figdf['found'], categories=['Not found','Found'])
    p9.options.figure_size=10,5
    p9.options.dpi=300
    fig = p9.ggplot(figdf, p9.aes(x='found', y='mean', color='found', fill='found'))
    fig += p9.geom_point(p9.aes(size='count'), alpha=.4)
    fig += p9.geom_errorbar(
        p9.aes(ymin='mean-stderr', ymax='mean+stderr'),
        size=1
    )
    fig+= p9.theme_minimal()
    fig += p9.facet_wrap('label', ncol=1)
    fig+= p9.theme(
        plot_background=p9.element_rect(fill='white', color=None),  # Add white background
        panel_background=p9.element_rect(fill='white', color=None)  # Add white background to panels
    )
    fig += p9.scale_y_continuous(limits=(0,100))
    fig+= p9.labs(
        y='Predicted percentage of rhyming poems',
        x='Found in LLM training data',
        color='Found in LLM training data',
        fill='Found in LLM training data',
        title='Poems in LLM training data are not disproportionately rhyming',
        size='Number of poems',
    )
    fig += p9.guides(color=False, fill=False)
    fig += p9.coord_flip()
    fig.save(path)
    return fig

def plot_rhyme_by_text_vs_instruct(df_rhyme_text_vs_instruct=None, force=False, verbose=DEFAULT_VERBOSE):
    if df_rhyme_text_vs_instruct is None:
        df_rhyme_text_vs_instruct = get_rhyme_for_sample(
            get_text_vs_instruct_completions(), 
            with_sample=True
        )
    df = df_rhyme_text_vs_instruct
    path = get_path(
        f"rhyme_in_text_vs_instruct.png",
        as_in_paper=df._as_in_paper,
        as_replicated=df._as_replicated,
        is_figure=True,
    )
    if path and not force and os.path.exists(path):
        if verbose:
            printm(f"* Loading plot from `{nice_path(path)}`")
        return try_display(path)


    p9.options.figure_size = (10, 6)
    p9.options.dpi = 300

    figdf_avg = get_avgs_df(df_rhyme_text_vs_instruct, gby=['period','model'])
    figdf_avg

    models = [x for x in sorted(figdf_avg.model.unique()) if x!=HIST]
    figdf_avg['model'] = pd.Categorical(figdf_avg.model, categories=models)

    fig = p9.ggplot(figdf_avg, p9.aes(x='period', y='mean', color='model', group='model'))
    fig += p9.geom_point(p9.aes(size='count'), shape='o', alpha=.4)
    fig += p9.geom_line()
    fig += p9.geom_errorbar(
        p9.aes(ymin='mean - stderr', ymax='mean + stderr'),
        width=0.25,
        size=.75,
        # alpha=0.6
    )
    fig += p9.theme_minimal()

    fig += p9.geom_text(
        p9.aes(label='model'),
        data=figdf_avg[figdf_avg.period=='1950-2000'],
        adjust_text={'x':100, 'y':0.05, 'arrowprops': dict(arrowstyle='-', alpha=0)},
        
        show_legend=False,
    )

    fig += p9.theme(panel_background=p9.element_rect(fill='white'), plot_background=p9.element_rect(fill='white'))
    fig += p9.geom_hline(yintercept=50, color='gray', linetype='dashed')
    fig += p9.scale_y_continuous(limits=(0,100))
    fig += p9.labs(
        x='Half-century of poet\'s birth',
        y='Predicted percentage of rhyming poems',
        color='Model',
        size='Number of poems',
        title='Predicted percentage of rhyming poems in generative completions of poems\nby their historical period (text vs. instruction models)',

    )
    if verbose:
        printm(f"* Saving plot to `{nice_path(path)}`")
    fig.save(path)
    return fig
