from . import *


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


def get_rhyme_for_sample(df_smpl, max_dist=RHYME_MAX_DIST, stash=STASH_RHYME, force=False):
    df = df_smpl.fillna("")
    if 'id' in df.columns:
        df = df.set_index('id')
    df = df.sort_values('id_hash')

    cache = dict(stash.items())

    def get_res(txt):
        if not force and txt in cache:
            return cache[txt]
        res = get_rhyme_for_txt(txt, max_dist=max_dist, stash=stash, force=force)
        return res

    df_rhymes = pd.DataFrame((get_res(txt) for txt in tqdm(df.txt)), index=df.index)
    return postprocess_rhyme_sample(df, df_rhymes)


def postprocess_rhyme_sample(df_poems, df_rhymes, rhyme_threshold=4):
    df = df_poems.join(df_rhymes, rsuffix='_prosodic', how='left')
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
    return df


def get_rhyme_for_completed_poems(period_by=50, filter_line_sim=True, rename_models=True):
    df = get_rhyme_for_sample(PATH_GENAI_COMPLETIONS, force=True).reset_index()
    if 'line_sim' in df.columns:
        df['line_sim'] = pd.to_numeric(df.line_sim, errors='coerce')
        if filter_line_sim:
            df = df[(df.model == HIST) | (df.model == '') | (df.line_sim < 95)]
    df = df.groupby(['id_human', 'id', 'model']).mean(numeric_only=True).reset_index()
    df_meta = get_chadwyck_corpus(period_by=period_by)
    df = df.merge(df_meta, left_on='id_human', right_on='id', suffixes=['', '_meta'], how='left')
    if rename_models:
        df['model9'] = df.model.apply(get_model_cleaned)
        df['model'] = df.model.apply(rename_model)
        df = df[df.model != '']
    return df
