import random
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import plotnine as p9
import asyncio
import nest_asyncio
from datetime import datetime
import prosodic
from hashstash import stashed_result
tqdm.pandas()
from rapidfuzz import fuzz
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from functools import lru_cache
from hashstash.engines.jsonl import JSONLHashStash
from tqdm import tqdm
cache = lru_cache(maxsize=1000)
HIST = '(Historical)'


load_dotenv()
PATH_CHADWYCK_HEALEY_TXT = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_TXT',''))
PATH_CHADWYCK_HEALEY_METADATA = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_METADATA',''))


pd.set_option('display.max_rows', 25)
p9.options.figure_size=(10,5)
p9.options.dpi=300

prosodic.USE_CACHE = False
prosodic.LOG_LEVEL = 'CRITICAL'

PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = f'{PATH_HERE}/data'
PATH_STASH = f'{PATH_DATA}/stash'
PATH_STASH_GENAI_RHYME_PROMPTS = f'{PATH_STASH}/genai_rhyme_prompts.jsonl'
PATH_STASH_GENAI_RHYME_COMPLETIONS = f'{PATH_STASH}/genai_rhyme_completions.jsonl'

STASH_GENAI_RHYME_PROMPTS = JSONLHashStash(PATH_STASH_GENAI_RHYME_PROMPTS)
STASH_GENAI_RHYME_COMPLETIONS = JSONLHashStash(PATH_STASH_GENAI_RHYME_COMPLETIONS)

PATH_RAWDATA = f'{PATH_DATA}/raw'
PATH_RAW_PKL = f'{PATH_RAWDATA}/data.allpoems.pkl.gz'
PATH_RAW_JSON = f'{PATH_RAWDATA}/data.newpoems2.json.gz'

PATH_SAMPLE = f'{PATH_DATA}/corpus_sample.csv.gz'
PATH_SAMPLE_RHYMES = f'{PATH_DATA}/corpus_sample_by_rhyme.csv.gz'
PATH_GENAI_PROMPTS = f'{PATH_DATA}/corpus_genai_promptings.csv.gz'
PATH_GENAI_COMPLETIONS = f'{PATH_DATA}/corpus_genai_completions.csv.gz'


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

def get_model_cleaned(x):
    return x.split('/')[-1].strip().title().split('-20')[0].split(':')[0].replace('Gpt-','GPT-').split('-Chat')[0]

def get_model_renamed(x):
    return rename_model(x)
    # if 'gpt-3' in x:
    #     return 'ChatGPT'
    # if 'gpt-4' in x:
    #     return 'ChatGPT'
    # elif 'claude-3' in x:
    #     return 'Claude'
    # elif 'llama3' in x:
    #     return 'Llama'
    # elif 'olmo2' in x:
    #     return 'Olmo'
    # elif 'deepseek' in x:
    #     return 'DeepSeek'
    # elif 'gemini' in x:
    #     return 'Gemini'
    # else:
    #     return x

def rename_model(x):
    if x=='' or x==HIST:
        return HIST
    if 'text' in x:
        return ''
    if 'gpt-3' in x:
        return 'ChatGPT'
    if 'gpt-4' in x:
        return 'ChatGPT'
    elif 'claude-3' in x:
        return 'Claude'
    elif 'llama3' in x:
        return 'Llama'
    elif 'olmo2' in x:
        return 'Olmo'
    elif 'deepseek' in x:
        return 'DeepSeek'
    elif 'gemini' in x:
        return 'Gemini'
    else:
        return ''

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


import json
import sys
import inspect
from typing import AsyncGenerator

try:
    from litellm import acompletion
except Exception:
    acompletion = None

    
def get_id_hash(id, seed=42, max_val=1000000):
    random.seed(hash(id) + seed)
    return random.randint(0, max_val - 1)



def save_sample(df, path_sample=PATH_SAMPLE, overwrite=False):
    if overwrite or not os.path.exists(path_sample):
        df.to_csv(path_sample)
        print(f'  * Saved sample to {path_sample}')
    else:
        path_sample_now = f'{os.path.splitext(path_sample.replace(".gz",""))[0]}_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'
        if path_sample.endswith('.csv.gz'):
            path_sample_now +='.gz'
        df.to_csv(path_sample_now)
        print(f'  * Saved sample to {path_sample_now}')

def get_id_hash_str(id):
    from hashlib import sha256
    return sha256(id.encode()).hexdigest()[:8]

def limit_lines(txt, n=100):
    l=[]
    n0=0
    for line in txt.strip().split('\n'):
        if line.strip():
            n0+=1
        l.append(line)
        if n0>=n:
            break
    return '\n'.join(l).strip()

@stashed_result(engine='pairtree')
def get_rhyme_for_txt(txt, max_dist=1):
    try:
        txt = limit_lines(txt)
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
        if 'line_sim' in df.columns:
            line_sims = dict(zip(df.index, df.line_sim))
            df_rhymes['line_sim'] = df_rhymes.index.map(line_sims)
        df_rhymes.to_csv(ofn)
    return postprocess_rhyme_sample(df, df_rhymes)


def postprocess_rhyme_sample(df_poems, df_rhymes, rhyme_threshold=4):
    df = df_poems.join(df_rhymes, rsuffix='_prosodic', how='inner')
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

def compute_all_stat_signif(df, groupby='period', varname='model', valname='rhyme_pred_perc'):
    o=[]
    for g,gdf in df.groupby(groupby):
        ogdf = compute_stat_signif(gdf, varname, valname).assign(groupby=g)
        o.append(ogdf)
    return pd.concat(o).sort_values(['groupby', 'effect_size'],ascending=False).set_index(['groupby','comparison',])
        


def get_avgs_df(df, gby=['period','source','prompt_type'], y='rhyme_pred_perc'):
    stats_df = df.groupby(gby)[y].agg(
        mean=np.mean,
        stderr=lambda x: x.std() / np.sqrt(len(x)),
        count=len
    ).reset_index()
    return stats_df


def get_rhyme_for_completed_poems(period_by=50, filter_line_sim=True, rename_models=True):
    df=get_rhyme_for_sample('../data/corpus_genai_completions.csv.gz', force=True).reset_index()
    if 'line_sim' in df.columns:
        df['line_sim'] = pd.to_numeric(df.line_sim, errors='coerce')
        if filter_line_sim:
            df = df[(df.model==HIST) | (df.model=='') | (df.line_sim<95)]

    df = df.groupby(['id_human','id','model']).mean(numeric_only=True).reset_index()
    df_meta = get_chadwyck_corpus(period_by=period_by)
    df = df.merge(df_meta, left_on='id_human', right_on='id', suffixes=['','_meta'], how='left')

    
    if rename_models:
        df['model9'] = df.model.apply(get_model_cleaned)
        df['model']=df.model.apply(rename_model)
        df = df[df.model!='']
    return df 

def clean_genai_poem(txt):
    stanzas = txt.split('\n\n')
    stanzas = [st.strip() for st in stanzas if st.strip().count('\n')>0]
    return '\n\n'.join(stanzas)

def get_num_lines(txt):
    return len([x for x in txt.split('\n') if x.strip()])


def printm(text, *args, **kwargs):
    """Print markdown if in Jupyter environment, otherwise normal print"""
    try:
        # Check if we're in a Jupyter environment
        from IPython.display import display, Markdown
        get_ipython()  # This will raise NameError if not in IPython/Jupyter
        
        # If we have additional args or certain kwargs, fall back to regular print
        if args or any(k in kwargs for k in ['file', 'flush']):
            print(text, *args, **kwargs)
        else:
            # Display as markdown
            display(Markdown(str(text)))
    except (NameError, ImportError):
        # Not in Jupyter or IPython not available, use regular print
        print(text, *args, **kwargs)


async def stream_llm_litellm(model: str, prompt: str, temperature: float = 0.7, system_prompt: str = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    """Stream tokens using LiteLLM for non-Gemini providers.

    Args:
        model: Provider-specific model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20240620").
        prompt: The user prompt to send.
        temperature: Sampling temperature.

    Yields:
        Token strings as they arrive.
    """
    if acompletion is None:
        raise RuntimeError("litellm is not installed.  `pip install litellm`")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            timeout=300,
            max_tokens=1024,
        )
        async for chunk in response:
            try:
                delta = None
                if hasattr(chunk, "choices") and chunk.choices:
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None)
                    if delta is not None:
                        token = getattr(delta, "content", None)
                    else:
                        # Fallback some providers use message.content even in stream
                        message = getattr(choice0, "message", None)
                        token = None
                        if message is not None:
                            token = getattr(message, "content", None)
                else:
                    token = None
                if token:
                    yield token
                    if verbose:
                        print(token,end="",flush=True)
            except Exception:
                # Ignore malformed chunks and continue
                continue
    except Exception as e:
        print(f"LiteLLM streaming error: {e}", file=sys.stderr)
        raise


def _extract_google_model_name(model: str) -> str | None:
    """Return inner model name if model starts with the required 'google/' prefix."""
    if not isinstance(model, str):
        return None
    if model.lower().startswith("google/"):
        return model.split("/", 1)[1]
    return None


async def stream_llm_genai(model: str, prompt: str, temperature: float = 0.7, system_prompt: str = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    """Stream tokens from Google Gemini using API key via google-generativeai.

    The dispatcher expects `model` like 'google/gemini-1.5-pro-latest'.
    """
    inner_model = _extract_google_model_name(model)
    if not inner_model:
        raise ValueError("stream_llm_genai expects model starting with 'google/'")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "google-generativeai is not installed. `pip install google-generativeai`"
        ) from e

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")

    genai.configure(api_key=api_key)

    # Attach system instruction if provided
    model_kwargs = {}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt

    gmodel = genai.GenerativeModel(model_name=inner_model, **model_kwargs)

    generation_config = {
        "temperature": float(temperature),
        "max_output_tokens": 1024,
    }

    try:
        response = gmodel.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True,
        )
        for chunk in response:
            try:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
                    if verbose:
                        print(text, end="", flush=True)
            except Exception:
                continue
    except Exception as e:
        print(f"Gemini streaming error: {e}", file=sys.stderr)
        raise


async def stream_llm(model: str, prompt: str, temperature: float = 0.7, system_prompt: str = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    """Dispatcher: route to google-generativeai or LiteLLM based on model prefix.

    - If model starts with 'google/', use Gemini via API key.
    - Otherwise, fall back to LiteLLM.
    """
    if _extract_google_model_name(model):
        async for token in stream_llm_genai(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
            yield token
        return
    async for token in stream_llm_litellm(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
        yield token


async def generate_text_async(model: str, prompt: str, temperature: float = 0.7, system_prompt: str = None, verbose: bool = False) -> str:
    """Convenience wrapper that returns the full text by consuming the stream."""
    output_parts = []
    async for token in stream_llm(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
        output_parts.append(token)
    return "".join(output_parts)

async def collect_async_generator(async_generator):
    result = []
    async for item in async_generator:
        result.append(item)
    return result


def run_async(async_func, *args, **kwargs):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    obj = async_func(*args, **kwargs)
    if inspect.isasyncgen(obj):
        awaitable = collect_async_generator(obj)
    elif inspect.iscoroutine(obj) or isinstance(obj, asyncio.Future):
        awaitable = obj
    else:
        raise TypeError("run_async expected coroutine or async generator")

    if loop.is_running():
        nest_asyncio.apply()
        return loop.run_until_complete(awaitable)
    else:
        return loop.run_until_complete(awaitable)



def generate_text(model: str, prompt: str, temperature: float = 0.7, system_prompt: str = None, verbose: bool = False, force: bool = False) -> str:
    key = {
        'model':model,
        'prompt':prompt,
        'temperature':temperature,
        'system_prompt':system_prompt,
    }
    if not force and key in STASH_GENAI_RHYME_PROMPTS:
        out = STASH_GENAI_RHYME_PROMPTS[key]
        if verbose:
            print(out)
        return out
    
    response = run_async(
        generate_text_async, 
        model=model, 
        prompt=prompt, 
        temperature=temperature, 
        system_prompt=system_prompt, 
        verbose=verbose
    )
    
    STASH_GENAI_RHYME_PROMPTS[key] = response
    
    return response