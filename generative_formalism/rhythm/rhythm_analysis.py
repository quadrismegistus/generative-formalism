from . import *


def get_sonnet_rhythm_data(
    as_in_paper=True,
    as_replicated=False,
    sample_by="sonnet_period",
    force=False,
    collapse_C17_19=True,
    verbose=DEFAULT_VERBOSE,
):
    """Get combined rhythm analysis data for sonnets from multiple sources.

    Collects and combines rhythm measurements for sonnets from three main sources:
    1. Chadwyck corpus sonnets (historical periods C17-19 and C20)
    2. Shakespeare's 154 sonnets
    3. GenAI-generated sonnets

    The function retrieves rhythm measurements for each source, adds period
    labels, and combines them into a single DataFrame for comparative analysis.
    Results are filtered to sonnets with 10-12 syllables per line.

    Returns
    -------
    pd.DataFrame
        Combined rhythm data with columns including:
        - Rhythm measurements (from get_rhythm_for_* functions)
        - group : str, period label ("C17-19", "C20", "Shakespeare", "GenAI")
        - source : str, data source ("chadwyck", "shakespeare", "genai")
        - Additional metadata from original datasets
        Filtered to sonnets with 10-12 syllables per line.

    Calls
    -----
    - get_chadwyck_corpus_sampled_by(sample_by)
    - get_rhythm_for_sample(df_smpl)
    - get_rhythm_for_shakespeare_sonnets()
    - get_genai_rhyme_promptings_as_in_paper()
    - get_rhythm_for_sample(df_genai_sonnets, gen=False)
    """
    path = get_path(
        f"sonnet_rhythm_data_by_{sample_by}",
        as_in_paper=as_in_paper,
        as_replicated=as_replicated,
    )
    if not force and os.path.exists(path):
        if verbose:
            print(f"* Loading sonnet rhythm data from {nice_path(path)}")
        odf = pd.read_csv(path).set_index("id")
    else:
        df_smpl = get_chadwyck_corpus_sampled_by(
            sample_by, as_in_paper=as_in_paper, as_replicated=as_replicated
        )
        df_smpl_rhythm = get_rhythm_for_sample(df_smpl).merge(df_smpl, on="id")

        df_shak_rhythm = get_rhythm_for_shakespeare_sonnets()

        df_smpl_rhythm["group"] = "Sample"
        df_shak_rhythm["group"] = "Shakespeare"

        df_genai_sonnets = get_genai_rhyme_promptings_as_in_paper()
        df_genai_sonnets = df_genai_sonnets[
            df_genai_sonnets.prompt.str.contains("sonnet")
        ].query("num_lines==14")
        df_genai_sonnets_rhythm = get_rhythm_for_sample(df_genai_sonnets, gen=False)
        df_genai_sonnets_rhythm["group"] = "GenAI"

        odf = pd.concat(
            [
                df_smpl_rhythm.assign(source="chadwyck"),
                df_shak_rhythm.assign(source="shakespeare"),
                df_genai_sonnets_rhythm.assign(source="genai"),
            ]
        )
        odf = odf.query("10<=num_sylls<=12")
        if verbose:
            print(f"* Writing sonnet rhythm data to {nice_path(path)}")
        odf.to_csv(path)

    def get_group(dob, group):
        if group in {'Shakespeare', 'GenAI'}:
            return group
        
        if collapse_C17_19:
            if dob >= 1600 and dob < 1900:
                return "C17-19"
            else:
                return "C20"
        else:
            cent = int(dob // 100) + 1
            return f"C{cent}"

    odf["group"] = [get_group(dob, group) for dob, group in zip(odf["author_dob"], odf["group"])]
    odf._sample_by = sample_by
    odf._as_in_paper = as_in_paper
    odf._as_replicated = as_replicated
    return odf


def get_rhythm_data_by_syll(df_rhythm):
    """Transform rhythm data from wide to long format for syllable-level analysis.

    Takes rhythm measurement data with syllable columns (e.g., syll_1_stress, syll_2_stress)
    and melts them into a long-format DataFrame suitable for plotting stress patterns
    across syllable positions.

    Parameters
    ----------
    df_rhythm : pd.DataFrame
        Rhythm data with syllable stress columns and metadata (id, source, group)

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - id : str, poem identifier
        - id_hash : str, hashed identifier
        - source : str, data source ("chadwyck", "shakespeare", "genai")
        - group : str, period/source group
        - syll_num : int, syllable position (1-10)
        - stress : float, stress likelihood (0-100%)
    """
    syll_cols = [c for c in df_rhythm.columns if "syll" in c and c[4].isdigit()]
    df_syll = df_rhythm.reset_index().melt(
        id_vars=["id", "id_hash", "source", "group"],
        value_vars=syll_cols,
        var_name="syll_num",
        value_name="stress",
    )
    df_syll["syll_num"] = df_syll["syll_num"].apply(lambda x: int(x[4:].split("_")[0]))
    df_syll["stress"] *= 100
    df_syll._sample_by = df_rhythm._sample_by
    df_syll._as_in_paper = df_rhythm._as_in_paper
    df_syll._as_replicated = df_rhythm._as_replicated
    return df_syll


def plot_stress_by_syll(
    df_rhythm,
    force=False,
    verbose=DEFAULT_VERBOSE,
):
    """Create a plot showing stress likelihood across syllable positions by group.

    Generates a line plot with error bars showing the average likelihood of stress
    at each syllable position (1-10) for different sonnet sources/groups. The plot
    helps visualize metrical patterns and differences between historical periods
    and generative models.

    Parameters
    ----------
    df_rhythm : pd.DataFrame
        Rhythm data from get_sonnet_rhythm_data() with syllable stress measurements
    force : bool, default False
        Whether to regenerate the plot even if it already exists
    verbose : bool, default DEFAULT_VERBOSE
        Whether to print progress messages

    Returns
    -------
    plotnine.ggplot
        Line plot with points and error bars showing stress patterns by syllable position
    """
    path = get_path(
        f"stress_by_syll_{df_rhythm._sample_by}.png",
        as_in_paper=df_rhythm._as_in_paper,
        as_replicated=df_rhythm._as_replicated,
        is_figure=True,
    )
    if not force and os.path.exists(path):
        if verbose:
            print(f"* Loading stress by syllable plot from {nice_path(path)}")
        return display_img(path)

    p9.options.figure_size = (10, 6)
    df_syll = get_rhythm_data_by_syll(df_rhythm)

    figdf = get_avgs_df(df_syll, ["group", "syll_num"], "stress")

    fig = p9.ggplot(
        figdf,
        p9.aes(
            x="syll_num",
            y="mean",
            color="group",
            shape="group",
        ),
    )
    fig += p9.geom_line(p9.aes(group="group"), size=0.5, alpha=0.5)
    fig += p9.scale_y_continuous(limits=(0, 100))
    fig += p9.theme_classic()
    fig += p9.geom_errorbar(
        p9.aes(ymin="mean-stderr", ymax="mean+stderr"),
        data=figdf,
        width=0.35,
        size=0.8,
        alpha=0.5,
    )
    fig += p9.geom_text(
        p9.aes(label="group"),
        size=9,
        data=figdf,
        position=p9.position_nudge(x=0.25),
        ha="left",
    )
    fig += p9.geom_point(p9.aes(shape="group"), data=figdf, alpha=0.5)
    fig += p9.scale_size_continuous(range=[1, 3])
    fig += p9.scale_y_continuous(limits=(0, 100))
    fig += p9.labs(
        x="Syllable Position",
        y=f"Likelihood of stress (%)",
        title=f"Likelihood of particular syllable in ten-syllable line being stressed",
        color="Sonnet source",
        shape="Sonnet source",
    )
    fig += p9.scale_x_continuous(breaks=range(1, 11))

    # fig = fig + p9.theme(legend_position='bottom')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if verbose:
        print(f"* Saving stress by syllable plot to {nice_path(path)}")
    fig.save(path)
    return fig


def plot_perfect_pentameter(df_rhythm, metric='is_unambigously_iambic_pentameter', force=False, verbose=DEFAULT_VERBOSE):
    """Create a horizontal bar plot showing iambic pentameter adherence by group.

    Generates a horizontal bar plot with error bars showing the percentage of
    sonnet lines that adhere to strict iambic pentameter patterns across different
    sonnet sources. This provides a quantitative measure of metrical regularity.

    Parameters
    ----------
    df_rhythm : pd.DataFrame
        Rhythm data from get_sonnet_rhythm_data() with iambic pentameter metrics
    metric : str, default 'is_unambigously_iambic_pentameter'
        Column name for the iambic pentameter metric to plot

    Returns
    -------
    plotnine.ggplot
        Horizontal bar plot showing percentage of lines meeting iambic pentameter criteria
        by sonnet source, ordered from highest to lowest adherence
    """
    path = get_path(
        f"fig.{metric}.{df_rhythm._sample_by}.png",
        as_in_paper=df_rhythm._as_in_paper,
        as_replicated=df_rhythm._as_replicated,
        is_figure=True,
    )
    if not force and os.path.exists(path):
        if verbose:
            print(f"* Loading perfect pentameter plot from {nice_path(path)}")
        return display_img(path)

    p9.options.figure_size = (8, 6)
    p9.options.dpi=300
    df_rhythm = df_rhythm.copy()
    df_rhythm[metric]*=100
    figdf = get_avgs_df(df_rhythm, ['group'], metric)
    # figdf['mean'] = figdf['has_trochaic_substitution']
    figdf['label'] = [f'{round(xmean,1)}%' for xmean,xstd in zip(figdf['mean'], figdf['stderr'])]

    l = list(reversed(figdf.sort_values('mean')['group'].tolist()))
    figdf['group'] = pd.Categorical(figdf['group'], categories=l, ordered=True)

    fig = p9.ggplot(figdf, p9.aes(x='group', y='mean', label='label'))
    fig += p9.geom_errorbar(p9.aes(ymin='mean - stderr', ymax='mean + stderr'), data=figdf, width=.2, size=1, alpha=.5)
    fig+=p9.geom_point()
    fig+=p9.geom_text(size=9, position=p9.position_nudge(x=.2))
    fig+= p9.coord_flip()
    fig+=p9.theme_minimal()
    fig+=p9.labs(
        y=f'% 10-12 syllable lines{" unambiguously" if metric=="is_unambiguously_iambic_pentameter" else " most easily"} parsed as iambic pentameter',
        x='Sonnet source',
        title='''Inhumanly strict metrical observance of iambic pentameter\nin LLM sonnets'''
    )
    fig+=p9.geom_hline(yintercept=50, linetype='dashed', alpha=.5)

    fig+=p9.scale_y_continuous(limits=(0,100))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if verbose:
        print(f"* Saving perfect pentameter plot to {nice_path(path)}")
    fig.save(path)
    return fig

