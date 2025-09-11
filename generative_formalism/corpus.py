from . import *

CORPUS_METADATA = None
CORPUS = None

def get_chadwyck_corpus_metadata(
    fields=CHADWYCK_CORPUS_FIELDS,
    period_by=CORPUS_PERIOD_BY,
    download_if_necessary=True,
    overwrite=False,
    min_num_lines=MIN_NUM_LINES,
    max_num_lines=MAX_NUM_LINES,
    min_author_dob=MIN_AUTHOR_DOB,
    max_author_dob=MAX_AUTHOR_DOB,
):
    global CORPUS_METADATA
    if CORPUS_METADATA is not None:
        print('* Loading corpus metadata from memory')
        return CORPUS_METADATA
    
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        download_chadwyck_corpus_metadata(overwrite=overwrite)
    if not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA):
        return pd.DataFrame(columns=fields.keys()).set_index('id')
    
    printm(f'#### Getting Chadwyck-Healey corpus metadata')
    print(f'* Loading metadata from {PATH_CHADWYCK_HEALEY_METADATA}')
    df = pd.read_csv(PATH_CHADWYCK_HEALEY_METADATA).fillna("")
    df['author_dob'] = pd.to_numeric(df['author_dob'], errors='coerce')
    df['num_lines'] = pd.to_numeric(df['num_lines'], errors='coerce')
    df['id_hash'] = [get_id_hash(x) for x in df['id']]
    print(f'* Loaded {len(df)} rows of metadata')

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
            while len(last_word) < 9:
                last_word = '0' + last_word
            all_but_last = ' '.join(x.split()[:-1])
            if all_but_last.endswith(','):
                all_but_last = all_but_last[:-1]
            return last_word + ' ' + all_but_last
        return x

    if 'attperi' in df.columns:
        df['attperi_str'] = df['attperi'].apply(get_attperi_str)
    if 'attdbase' in df.columns:
        df['attdbase_str'] = df['attdbase'].apply(get_attdbase_str)

    df = df[list(fields.keys())].rename(columns=fields)

    odf = df.fillna("")
    odf = odf[odf.author_dob != ""]
    if min_author_dob is not None:
        odf = odf[odf.author_dob >= min_author_dob]
        print(f'* Filtering: {len(odf):,} rows after author birth year >= {min_author_dob}')
    if max_author_dob is not None:
        odf = odf[odf.author_dob <= max_author_dob]
        print(f'* Filtering: {len(odf):,} rows after author birth year <= {max_author_dob}')

    def get_period_dob(x, ybin=period_by):
        if not x:
            return ""
        n = int(x // ybin * ybin)
        return f'{n}-{n + ybin}'

    odf['period'] = odf.author_dob.apply(get_period_dob)

    if min_num_lines is not None:
        odf = odf[odf.num_lines >= min_num_lines]
        print(f'* Filtering: {len(odf):,} rows after number of lines >= {min_num_lines}')
    if max_num_lines is not None:
        odf = odf[odf.num_lines <= max_num_lines]
        print(f'* Filtering: {len(odf):,} rows after number of lines <= {max_num_lines}')

    odf = odf.set_index('id_hash')
    CORPUS_METADATA = odf
    return odf



def download_chadwyck_corpus_metadata(overwrite=False):
    if URL_CHADWYCK_HEALEY_METADATA and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_METADATA)):
        PATH_CHADWYCK_HEALEY_METADATA_ZIP = PATH_CHADWYCK_HEALEY_METADATA+'.zip'
        print(f"* Downloading metadata to {PATH_CHADWYCK_HEALEY_METADATA_ZIP}")
        download_file(URL_CHADWYCK_HEALEY_METADATA, PATH_CHADWYCK_HEALEY_METADATA_ZIP)
        print(f'* Unzipping metadata to {PATH_CHADWYCK_HEALEY_METADATA}')
        unzip_file(PATH_CHADWYCK_HEALEY_METADATA_ZIP, PATH_CHADWYCK_HEALEY_METADATA)

def download_chadwyck_corpus_txt(overwrite=False):
    if URL_CHADWYCK_HEALEY_TXT and (overwrite or not os.path.exists(PATH_CHADWYCK_HEALEY_TXT)):
        PATH_CHADWYCK_HEALEY_TXT_ZIP = PATH_CHADWYCK_HEALEY_TXT+'.zip'
        print(f"* Downloading corpus text to {PATH_CHADWYCK_HEALEY_TXT_ZIP}")
        download_file(URL_CHADWYCK_HEALEY_TXT, PATH_CHADWYCK_HEALEY_TXT_ZIP)

        print(f'* Unzipping corpus text to {PATH_CHADWYCK_HEALEY_TXT}')
        unzip_file(PATH_CHADWYCK_HEALEY_TXT_ZIP, PATH_CHADWYCK_HEALEY_TXT)




def get_txt(id, clean_poem=True):
    fn = os.path.join(PATH_CHADWYCK_HEALEY_TXT, id) + '.txt'
    if os.path.exists(fn):
        with open(fn) as f:
            out = f.read().strip()
            if out and clean_poem:
                out = clean_poem_str(out)
            return out
    return ""

def get_chadwyck_corpus_texts(df_meta, clean_poem=True):
    print(f'* Loading {len(df_meta)} texts')
    return [
        get_txt(id, clean_poem=clean_poem)
        for id in tqdm(df_meta.id, desc='  ')
    ]

def get_chadwyck_corpus(*args, clean_poem=True, force=False, **kwargs):
    global CORPUS
    printm(f'##### Loading Chadwyck-Healey corpus (metadata + txt)')

    if not force and CORPUS is not None:
        print('* Loading corpus from memory')
        return CORPUS
    
    df_meta = get_chadwyck_corpus_metadata(*args, **kwargs)

    df_meta['txt'] = get_chadwyck_corpus_texts(df_meta, clean_poem=clean_poem)
    CORPUS = df_meta
    return df_meta


def sample_chadwyck_corpus(
        df_corpus,
        sample_by,
        min_sample_n=MIN_SAMPLE_N,
        max_sample_n=MAX_SAMPLE_N,
        ):

    printm(f'#### Sampling corpus by {sample_by} (min {min_sample_n}, max {max_sample_n})')
    # sort by id hash
    df = df_corpus.sort_index()

    sample_by = [sample_by] if isinstance(sample_by, str) else sample_by
    if min_sample_n:
        df = df.groupby(sample_by).filter(lambda x: len(x) >= min_sample_n).sort_index()
    if max_sample_n:
        df = df.groupby(sample_by).head(max_sample_n).sort_index()

    print(df.groupby(sample_by).size())
    return df



    

# def get_chadwyck
    
def get_chadwyck_corpus_sampled_by_rhyme_as_in_paper():
    return pd.read_csv(PATH_SAMPLE_RHYMES_IN_PAPER).set_index('id_hash')

def get_chadwyck_corpus_sampled_by_period_as_in_paper():
    return pd.read_csv(PATH_SAMPLE_PERIOD_IN_PAPER).set_index('id_hash')

def get_chadwyck_corpus_sampled_by_period_subcorpus_as_in_paper():
    return pd.read_csv(PATH_SAMPLE_PERIOD_SUBCORPUS_IN_PAPER).set_index('id_hash')






# Samplers

def gen_chadwyck_corpus_sampled_by_rhyme():
    df_corpus = get_chadwyck_corpus()
    df_corpus = df_corpus[df_corpus.rhyme.isin({'y','n'})]
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by='rhyme',
    )
    return df

def gen_chadwyck_corpus_sampled_by_period():
    df_corpus = get_chadwyck_corpus()
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by='period',
    )
    return df

def gen_chadwyck_corpus_sampled_by_period_subcorpus():
    df_corpus = get_chadwyck_corpus()
    df = sample_chadwyck_corpus(
        df_corpus,
        sample_by=['period','subcorpus'],
    )
    return df

def get_chadwyck_corpus_sampled_by_rhyme(force=False):
    path = PATH_SAMPLE_RHYMES_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating rhyme sample')
        odf = gen_chadwyck_corpus_sampled_by_rhyme()
        save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading rhyme sample from {path}')
        odf = pd.read_csv(path).set_index('id_hash')
    return odf

def get_chadwyck_corpus_sampled_by_period(force=False):
    path = PATH_SAMPLE_PERIOD_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating period sample')
        odf = gen_chadwyck_corpus_sampled_by_period()
        save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading period sample from {path}')
        odf = pd.read_csv(path).set_index('id_hash')
    return odf

def get_chadwyck_corpus_sampled_by_period_subcorpus(force=False):
    path = PATH_SAMPLE_PERIOD_SUBCORPUS_REPLICATED
    if force or not os.path.exists(path):
        print(f'* Generating period subcorpus sample')
        odf = gen_chadwyck_corpus_sampled_by_period_subcorpus()
        save_sample(odf, path, overwrite=True)
    else:
        print(f'* Loading period subcorpus sample from {path}')
        odf = pd.read_csv(path).set_index('id_hash')
    return odf


def get_chadwyck_corpus_sampled_by_rhyme_as_replicated(overwrite=False):
    df_smpl = get_chadwyck_corpus_sampled_by_rhyme(force=overwrite)
    return df_smpl

def get_chadwyck_corpus_sampled_by_period_as_replicated(overwrite=False):
    df_smpl = get_chadwyck_corpus_sampled_by_period(force=overwrite)
    return df_smpl

def get_chadwyck_corpus_sampled_by_period_subcorpus_as_replicated(overwrite=False):
    df_smpl = get_chadwyck_corpus_sampled_by_period_subcorpus(force=overwrite)
    return df_smpl




def describe_corpus(dfx):
    printm(f'----')
    printm(f'#### Subcorpus breakdown')
    describe_qual(dfx.subcorpus)
    printm(f'----')

    printm(f'#### Historical period breakdown (from metadata)')
    describe_qual(dfx.period_meta)
    printm(f'----')
    
    printm(f'#### Historical period breakdown (from author birth year)')
    describe_qual(dfx.period)
    printm(f'----')

    printm(f'#### Historical period + subcorpus breakdown')
    describe_qual_grouped(dfx, ['period', 'subcorpus'])
    printm(f'----')

    printm(f'#### Author birth year distribution')
    describe_numeric(dfx.author_dob)
    printm(f'----')

    printm(f'#### Number of lines in poems')
    describe_numeric(dfx.num_lines)
    printm(f'----')

    printm(f'#### Annotated rhyme distribution')
    describe_qual(dfx.rhyme)
    printm(f'----')

    printm(f'#### Metadata')
    return dfx
