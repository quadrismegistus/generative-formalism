from . import *
from itertools import combinations


def cohen_d(x, y):
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d is a measure of the standardized difference between two means.
    It represents how many standard deviations the means are apart.

    Parameters
    ----------
    x : array-like
        First group of values
    y : array-like
        Second group of values

    Returns
    -------
    float
        Cohen's d effect size. Positive values indicate x has higher mean than y.
        Effect sizes are typically interpreted as:
        - 0.2: small effect
        - 0.5: medium effect
        - 0.8: large effect

    Calls
    -----
    None
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )
    return (np.mean(x) - np.mean(y)) / pooled_std


def permutation_test(x, y, n_permutations=10000):
    """
    Perform a permutation test to assess the statistical significance of the difference between two groups.

    This non-parametric test randomly shuffles the combined data and recalculates the difference
    many times to create a null distribution, then calculates the p-value based on where the
    observed difference falls in this distribution.

    Parameters
    ----------
    x : array-like
        First group of values
    y : array-like
        Second group of values
    n_permutations : int, default=10000
        Number of permutations to perform. Higher values give more precise p-values
        but take longer to compute.

    Returns
    -------
    float
        Two-tailed p-value representing the probability of observing a difference
        as extreme as the one observed, assuming the null hypothesis is true.
        Values below 0.05 typically indicate statistical significance.

    Calls
    -----
    None
    """
    observed_diff = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    n1 = len(x)
    diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        diffs.append(diff)
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_value


def compute_stat_signif(
    df,
    groupby="model",
    valname="rhyme_pred_perc",
    verbose=DEFAULT_VERBOSE,
    min_group_size=10,
    group_name="group",
    force=False,
):
    """
    Compute statistical significance tests between all pairs of groups in a DataFrame.

    Performs pairwise comparisons between groups using Cohen's d effect size and permutation tests.
    Returns a DataFrame with comparison results including p-values, effect sizes, and means.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze
    groupby : str or list of str, default='model'
        Column name(s) to group by for comparisons. If string, will be converted to list.
    valname : str, default='rhyme_pred_perc'
        Column name containing the values to compare between groups
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to show progress bar during computation
    min_group_size : int, default=10
        Minimum number of samples required in each group for comparison
    group_name : str, default='group'
        Name of temporary column created to store combined group identifiers

    Returns
    -------
    pandas.DataFrame
        DataFrame with comparison results containing:
        - comparison: String describing the comparison (e.g., "group1 vs group2")
        - n1, n2: Sample sizes of each group
        - p_value: Statistical significance from permutation test
        - effect_size: Absolute Cohen's d effect size
        - effect_size_str: Categorical effect size ('', 'small', 'medium', 'large')
        - mean1, mean2: Means of each group
        - significant: Boolean indicating if p < 0.05
        - Additional columns for each grouping variable

    Calls
    -----
    - cohen_d()
    - permutation_test()
    """
    path = get_path_for_df(df, suffix=f".stats.{valname}_by_{groupby}.csv")
    if path and not force and os.path.exists(path):
        if verbose:
            print(f"* Loading statistics data from {path}")
            return pd.read_csv(path)

    if isinstance(groupby, str):
        groupby = [groupby]

    df[group_name] = df[groupby].applymap(str).apply(lambda x: "|".join(x), axis=1)

    variables = df[group_name].unique()
    results = []
    iterr = list(combinations(variables, 2))
    if verbose:
        iterr = tqdm(iterr, desc="Computing comparisons")
    for var1, var2 in iterr:
        if verbose:
            iterr.set_description(f"Computing comparisons for {var1} vs {var2}")
        group1 = df[df[group_name] == var1][valname]
        group2 = df[df[group_name] == var2][valname]
        if len(group1) < min_group_size or len(group2) < min_group_size:
            continue
        d = cohen_d(group1, group2)
        p = permutation_test(group1.values, group2.values)

        def char_effect_size(x):
            if x < 0.2:
                return ""
            if x < 0.5:
                return "small"
            if x < 0.8:
                return "medium"
            return "large"

        groups_d = {
            **dict(zip(groupby, var1.split("|"))),
            **dict(zip(groupby, var2.split("|"))),
        }

        results.append(
            {
                "comparison": f"{var1} vs {var2}",
                "n1": len(group1),
                "n2": len(group2),
                "p_value": p,
                "effect_size": abs(d),
                "effect_size_str": char_effect_size(abs(d)),
                "mean1": group1.mean(),
                "mean2": group2.mean(),
                "significant": p < 0.05,
                **groups_d,
            }
        )
    results_df = pd.DataFrame(results)
    odf = (
        results_df.sort_values("effect_size", ascending=False)
        if len(results_df) > 0
        else pd.DataFrame()
    )
    if path:
        if verbose:
            print(f'* Saving statistics to {path}')
        odf.to_csv(path,index=False)
    return odf


def compute_all_stat_signif(
    df,
    groupby="period",
    groupby_stat="model",
    valname="rhyme_pred_perc",
    verbose=DEFAULT_VERBOSE,
):
    """
    Compute statistical significance tests for all subgroups within a DataFrame.

    Groups the DataFrame by the specified grouping variable, then runs pairwise statistical
    comparisons within each subgroup using compute_stat_signif. Useful for analyzing
    differences across different categories (e.g., periods, sources).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze
    groupby : str, default='period'
        Column name to group by before running statistical comparisons
    groupby_stat : str or list of str, default='model'
        Column name(s) to use for within-group statistical comparisons
    valname : str, default='rhyme_pred_perc'
        Column name containing the values to compare between groups
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to show progress bar during computation

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with statistical comparison results from all subgroups.
        Each row represents a pairwise comparison within a subgroup, with an additional
        'groupby' column indicating which subgroup the comparison belongs to.

    Calls
    -----
    - compute_stat_signif()
    """
    o = []
    iterr = tqdm(
        list(df.groupby(groupby)), desc="Computing all statistical significance tests"
    )
    for g, gdf in iterr:
        iterr.set_description(
            f"Computing statistical significance tests for {groupby}={g}"
        )
        ogdf = compute_stat_signif(
            gdf, groupby_stat, valname, verbose=DEFAULT_VERBOSE
        ).assign(groupby=g)
        o.append(ogdf)
    return pd.concat(
        o
    )  # .sort_values(['groupby', 'effect_size'], ascending=False).set_index(['groupby', 'comparison']) if len(o) > 0 else pd.DataFrame()


def get_avgs_df(df, gby=["period", "source", "prompt_type"], y="rhyme_pred_perc"):
    """
    Calculate summary statistics (mean, standard error, count) for groups in a DataFrame.

    Groups the DataFrame by specified columns and computes descriptive statistics
    for the target variable, including mean, standard error of the mean, and sample count.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to summarize
    gby : list of str, default=['period', 'source', 'prompt_type']
        Column names to group by for computing statistics
    y : str, default='rhyme_pred_perc'
        Column name containing the values to compute statistics for

    Returns
    -------
    pandas.DataFrame
        DataFrame with summary statistics containing:
        - Grouping columns (from gby parameter)
        - mean: Mean value of y for each group
        - stderr: Standard error of the mean for each group
        - count: Number of observations in each group

    Calls
    -----
    None
    """
    stats_df = (
        df.groupby(gby)[y]
        .agg(
            mean=np.mean,
            stderr=lambda x: x.std() / np.sqrt(len(x)),
            count=len,
        )
        .reset_index()
    )
    return stats_df


def get_pred_stats(predictions, ground_truth, return_counts=False):
    """
    Calculate binary classification performance metrics from predictions and ground truth.

    Computes standard classification metrics including precision, recall, F1-score,
    and accuracy based on the confusion matrix elements.

    Parameters
    ----------
    predictions : array-like of bool
        Predicted binary values (True/False or 1/0)
    ground_truth : array-like of bool
        True binary values (True/False or 1/0)
    return_counts : bool, default=False
        If True, includes confusion matrix counts in return dictionary.
        Currently unused but kept for API compatibility.

    Returns
    -------
    dict
        Dictionary containing classification metrics:
        - f1_score: F1-score (harmonic mean of precision and recall)
        - precision: True Positives / (True Positives + False Positives)
        - recall: True Positives / (True Positives + False Negatives)
        - accuracy: (True Positives + True Negatives) / Total Samples
        - true_positives: Number of true positive predictions
        - false_positives: Number of false positive predictions
        - true_negatives: Number of true negative predictions
        - false_negatives: Number of false negative predictions

    Raises
    ------
    ValueError
        If predictions and ground_truth have different lengths

    Calls
    -----
    None
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def compare_data_by_group(
    df,
    groupby=[],
    valname="",
    min_group_size=100,
    verbose=DEFAULT_VERBOSE,
):
    """
    Compare data between groups with statistical significance testing.

    Wrapper function that performs statistical comparisons between groups in a DataFrame.
    Validates inputs and calls compute_stat_signif with appropriate parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze
    groupby : list of str, default=[]
        Column name(s) to group by for comparisons. Must be provided.
    valname : str, default=''
        Column name containing the values to compare between groups. Must be provided.
    min_group_size : int, default=100
        Minimum number of samples required in each group for comparison
    verbose : bool, default=DEFAULT_VERBOSE
        Whether to show progress bar during computation

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with comparison results if inputs are valid, None otherwise.
        See compute_stat_signif() for details on the return format.

    Calls
    -----
    - compute_stat_signif()
    """
    if not groupby or not valname:
        print(f"* Warning: no groupby or valname provided")
        return

    return compute_stat_signif(
        df,
        groupby=groupby,
        valname=valname,
        min_group_size=min_group_size,
        verbose=verbose,
    )
