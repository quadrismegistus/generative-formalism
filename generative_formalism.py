import random
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import plotnine as p9
from datetime import datetime
import prosodic
from hashstash import stashed_result
tqdm.pandas()
from rapidfuzz import fuzz
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from functools import lru_cache
cache = lru_cache(maxsize=1000)

load_dotenv()
PATH_CHADWYCK_HEALEY_TXT = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_TXT'))
PATH_CHADWYCK_HEALEY_METADATA = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_METADATA'))


pd.set_option('display.max_rows', 25)
p9.options.figure_size=(10,5)
p9.options.dpi=300

prosodic.USE_CACHE = False
prosodic.LOG_LEVEL = 'CRITICAL'

PATH_SAMPLE = f'../data/corpus_sample.csv.gz'
PATH_SAMPLE_RHYMES = f'../data/corpus_sample_by_rhyme.csv'


EXCLUDE_PROMPTS = [
    'Write an unryhmed poem in the style of Shakespeare\'s dramatic monologues.',
    'Write a poem in the style of Shakespeare\'s dramatic monologues.',
    'Write a poem in the style of e.e. cummings',
    # 'Write a poem in the style of Walt Whitman.',
    # "Write an ryhmed poem in the style of Shakespeare's sonnets.",
    # "Write a poem in the style of Emily Dickinson.",
    # "Write a poem in the style of Walt Whitman.",
    'Write a poem in the style of Wallace Stevens.',
    # "Write a poem in blank verse.",
    'Continue the following poem:\n\nTyping, typing, fingers on the keyboard\nThe keys crack and bend under sweat and weight,\n'
]



CHADWYCK_CORPUS_FIELDS = {
    # 'id':'id', 
    'id_hash':'id_hash',
    'attperi_str':'period_meta', 
    'attdbase_str':'subcorpus',
    'author':'author', 
    'author_dob':'author_dob', 
    'title':'title', 
    'year':'year', 
    'num_lines':'num_lines', 
    'volhead':'volume', 
    'l':'line', 
    'attrhyme':'rhyme', 
    'attgenre':'genre', 
}

@cache
def get_chadwyck_corpus(
        fields=CHADWYCK_CORPUS_FIELDS,
        period_by=50,
        ):
    df = pd.read_csv(PATH_CHADWYCK_HEALEY_METADATA).fillna("").set_index('id')
    df['author_dob'] = pd.to_numeric(df['author_dob'], errors='coerce')    
    df['id_hash'] = [get_id_hash(x) for x in df.index]

    def get_attdbase_str(x):
        if not x:
            return ""
        if 'African-American' in x:
            return 'African-American Poetry'
        if 'American' in x:
            return 'American Poetry'
        if 'English' in x:
            return 'English Poetry'
        return x
    
    def get_attperi_str(x):
        if not x:
            return ""
        x = x.replace('Fifteenth-Century Poetry', 'Fifteenth Century Poetry 1400-1500')
        last_word = x.split()[-1]
        if '-' in last_word and last_word[0].isdigit():
            while len(last_word)<9:
                last_word='0'+last_word
            
            all_but_last = ' '.join(x.split()[:-1])
            if all_but_last.endswith(','):
                all_but_last = all_but_last[:-1]
            return last_word + ' ' + all_but_last
        return x

    # rename cols


    # reformatting period by putting year range first if it exists (for sorting)
    if 'attperi' in df.columns:
        df['attperi_str'] = df['attperi'].apply(get_attperi_str)

    if 'attdbase' in df.columns:
        df['attdbase_str'] = df['attdbase'].apply(get_attdbase_str)

    # rename keys (take cols) to values (renamed cols) in fields dict
    df = df[list(fields.keys())].rename(columns=fields)

    odf=df.fillna("")
    odf=odf[odf.author_dob!=""]
    odf=odf.query('1600<=author_dob<2000')

    def get_period_dob(x, ybin=period_by):
        if not x:
            return ""
        n=int(x//ybin*ybin)
        return f'{n}-{n+ybin}'

    odf['period'] = odf.author_dob.apply(get_period_dob)
    return odf




def get_pred_stats(predictions, ground_truth, return_counts=False):
    """
    Calculate F1 score and related statistics from two lists of booleans.
    
    Args:
        predictions (list): List of boolean predictions
        ground_truth (list): List of boolean ground truth values
    
    Returns:
        dict: Dictionary containing F1 score, precision, recall, and counts
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    # Count true positives, false positives, false negatives
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

PATH_PKL = '/Users/ryan/Dropbox/Prof/Data/data.allpoems.pkl'
PATH_JSON = '/Users/ryan/Dropbox/Prof/Data/data.newpoems2.json'

import json

    
def get_id_hash(id, seed=42, max_val=1000000):
    random.seed(hash(id) + seed)
    return random.randint(0, max_val - 1)



def save_sample(df, path_sample=PATH_SAMPLE, overwrite=False):
    if overwrite or not os.path.exists(path_sample):
        df.to_csv(path_sample)
        print(f'Saved sample to {path_sample}')
    else:
        path_sample_now = f'{os.path.splitext(path_sample)[0]}_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'
        df.to_csv(path_sample_now)
        print(f'Saved new sample to {path_sample_now}')

def get_id_hash_str(id):
    from hashlib import sha256
    return sha256(id.encode()).hexdigest()[:8]



@stashed_result(engine='pairtree')
def get_rhyme_for_txt(txt, max_dist=1):
    try:
        text = prosodic.Text(txt)
        rhyme_d = text.get_rhyming_lines(max_dist)
        all_rhyming_lines = set()
        all_perfectly_rhyming_lines = set()
        for l1,(score,l2) in rhyme_d.items():
            all_rhyming_lines.update({l1,l2})
            if not score: all_perfectly_rhyming_lines.update({l1,l2})
            
            
        num_rhyming_lines = len(all_rhyming_lines)
        num_perfectly_rhyming_lines = len(all_perfectly_rhyming_lines)
        num_lines = text.num_lines
        return {
            'num_rhyming_lines':num_rhyming_lines,
            'num_perfectly_rhyming_lines':num_perfectly_rhyming_lines,
            'num_lines':num_lines,
        }
    except Exception as e:
        return {}

def get_rhyme_for_sample(path_sample, force=False):
    df = pd.read_csv(path_sample).fillna("").set_index('id')

    ofn = path_sample.replace('.csv.gz','.csv').replace('.csv', '.rhyme_data.csv')
    if not force and os.path.exists(ofn):
        df_rhymes = pd.read_csv(ofn).fillna("").set_index('id')
    else:    
        df_rhymes = pd.DataFrame((get_rhyme_for_txt(txt) for txt in tqdm(df.txt)), index=df.index).dropna().applymap(int)
        df_rhymes.to_csv(ofn)
    return postprocess_rhyme_sample(df, df_rhymes)


def postprocess_rhyme_sample(df_poems, df_rhymes, rhyme_threshold=4):
    df = df_poems.join(df_rhymes, rsuffix='_prosodic', how='inner')

    df['num_lines_prosodic'] = pd.to_numeric(df.num_lines_prosodic, errors='coerce')
    df['num_rhyming_lines'] = pd.to_numeric(df.num_rhyming_lines, errors='coerce')
    df['num_perfectly_rhyming_lines'] = pd.to_numeric(df.num_perfectly_rhyming_lines, errors='coerce')
    df['perc_rhyming_lines'] = df.num_rhyming_lines / df.num_lines_prosodic * 100
    df['perc_perfectly_rhyming_lines'] = df.num_perfectly_rhyming_lines / df.num_lines_prosodic * 100
    df['num_rhyming_lines_per10l'] = (df.num_rhyming_lines / df.num_lines_prosodic * 10).fillna(0).round(0).astype(int)
    df['num_perfectly_rhyming_lines_per10l'] = (df.num_perfectly_rhyming_lines / df.num_lines_prosodic * 10).fillna(0).round(0).astype(int)
    if 'rhyme' in df.columns:
        df['rhyme'] = ['?' if not x else x for x in df.rhyme]
        df['rhyme_bool'] = df.rhyme.apply(lambda x: (True if x=="y" else (False if x=="n" else None)))
    df['rhyme_pred'] = df.num_perfectly_rhyming_lines_per10l.apply(lambda x: x>=rhyme_threshold)
    df['rhyme_pred_perc'] = df.rhyme_pred * 100

    return df




### stats

from scipy import stats
import numpy as np
import pandas as pd
from itertools import combinations

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    # Pooled standard deviation
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
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

def compute_stat_signif(df, varname='model', valname='rhyme_pred_perc'):
    variables = df[varname].unique()
    results = []
    
    for var1, var2 in combinations(variables, 2):
        group1 = df[df[varname] == var1][valname]
        group2 = df[df[varname] == var2][valname]
        
        # Calculate effect size
        d = cohen_d(group1, group2)
        # Run permutation test
        p = permutation_test(group1.values, group2.values)

        def char_effect_size(x):
            if x<.2: return ''
            if x<.5: return 'small'
            if x<.8: return 'medium'
            return 'large'

        
        results.append({
            'comparison': f"{var1} vs {var2}",
            'p_value': p,
            'effect_size': abs(d),  # absolute effect size for sorting
            'effect_size_str': char_effect_size(abs(d)),  # absolute effect size for sorting
            'mean1': group1.mean(),
            'mean2': group2.mean(),
            'significant': p < 0.05
        })
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values('effect_size', ascending=False)

def compute_all_stat_signif(df, groupby='period'):
    o=[]
    for g,gdf in df.groupby(groupby):
        ogdf = compute_stat_signif(gdf).assign(groupby=g)
        o.append(ogdf)
    return pd.concat(o).sort_values(['groupby', 'effect_size'],ascending=False).set_index(['groupby','comparison',])
        