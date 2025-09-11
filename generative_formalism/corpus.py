from . import *

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
    odf = odf.query('1600<=author_dob<2000')

    def get_period_dob(x, ybin=period_by):
        if not x:
            return ""
        n = int(x // ybin * ybin)
        return f'{n}-{n + ybin}'

    odf['period'] = odf.author_dob.apply(get_period_dob)
    return odf



