from . import *
from itertools import combinations

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def permutation_test(x, y, n_permutations=10000):
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


def compute_stat_signif(df, groupby='model', valname='rhyme_pred_perc',verbose=DEFAULT_VERBOSE,min_group_size=10,group_name='group'):
    if isinstance(groupby, str):
        groupby = [groupby]
    
    df[group_name] = df[groupby].applymap(str).apply(lambda x: '|'.join(x), axis=1)
    
    variables = df[group_name].unique()
    results = []
    iterr = list(combinations(variables, 2))
    if verbose:
        iterr = tqdm(iterr, desc='Computing comparisons')
    for var1, var2 in iterr:
        group1 = df[df[group_name] == var1][valname]
        group2 = df[df[group_name] == var2][valname]
        if len(group1) < min_group_size or len(group2) < min_group_size:
            continue
        d = cohen_d(group1, group2)
        p = permutation_test(group1.values, group2.values)

        def char_effect_size(x):
            if x < .2:
                return ''
            if x < .5:
                return 'small'
            if x < .8:
                return 'medium'
            return 'large'

        groups_d = {
            **dict(zip(groupby, var1.split('|'))),
            **dict(zip(groupby, var2.split('|'))),
        }

        results.append({
            'comparison': f"{var1} vs {var2}",
            "n1": len(group1),
            "n2": len(group2),
            'p_value': p,
            'effect_size': abs(d),
            'effect_size_str': char_effect_size(abs(d)),
            'mean1': group1.mean(),
            'mean2': group2.mean(),
            'significant': p < 0.05,
            **groups_d
        })
    results_df = pd.DataFrame(results)
    return results_df.sort_values('effect_size', ascending=False) if len(results_df) > 0 else pd.DataFrame()

def compute_all_stat_signif(df, groupby='period', groupby_stat='model', valname='rhyme_pred_perc',verbose=DEFAULT_VERBOSE):
    o = []
    for g, gdf in df.groupby(groupby):
        ogdf = compute_stat_signif(gdf, groupby_stat, valname, verbose=verbose).assign(groupby=g)
        o.append(ogdf)
    return pd.concat(o)#.sort_values(['groupby', 'effect_size'], ascending=False).set_index(['groupby', 'comparison']) if len(o) > 0 else pd.DataFrame()


def get_avgs_df(df, gby=['period', 'source', 'prompt_type'], y='rhyme_pred_perc'):
    stats_df = df.groupby(gby)[y].agg(
        mean=np.mean,
        stderr=lambda x: x.std() / np.sqrt(len(x)),
        count=len,
    ).reset_index()
    return stats_df



def get_pred_stats(predictions, ground_truth, return_counts=False):
    if len(predictions) != len(ground_truth):
        raise ValueError('Predictions and ground truth must have the same length')
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
    }

def compare_data_by_group(
    df, 
    groupby=[], 
    valname='', 
    min_group_size=100,
    verbose=DEFAULT_VERBOSE,
):
    if not groupby or not valname:
        print(f'* Warning: no groupby or valname provided')
        return
    
    return compute_stat_signif(df, groupby=groupby,valname=valname,min_group_size=min_group_size,verbose=verbose)