from . import *

# Basic constants
HIST = '(Historical)'


# Repository/package paths
PATH_PACKAGE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_PACKAGE)
PATH_DATA = f'{PATH_REPO}/data'
PATH_STASH = f'{PATH_DATA}/stash'

PATH_STASH_GENAI = f'{PATH_STASH}/genform.jsonl'
PATH_STASH_GENAI_RHYME_PROMPTS = f'{PATH_STASH}/genai_rhyme_prompts.jsonl'
PATH_STASH_GENAI_RHYME_COMPLETIONS = f'{PATH_STASH}/genai_rhyme_completions.jsonl'

STASH_GENAI_RHYME_PROMPTS = JSONLHashStash(PATH_STASH_GENAI_RHYME_PROMPTS)
STASH_GENAI_RHYME_COMPLETIONS = JSONLHashStash(PATH_STASH_GENAI_RHYME_COMPLETIONS)
STASH_GENAI = JSONLHashStash(PATH_STASH_GENAI)


REPLICATE_OVERWRITE = True


PATH_STASH_RHYME = f'{PATH_STASH}/rhyme_for_txt.jsonl'
STASH_RHYME = JSONLHashStash(PATH_STASH_RHYME)



# Raw data paths
PATH_RAWDATA = f'{PATH_DATA}/raw'
PATH_RAW_PKL = f'{PATH_RAWDATA}/data.allpoems.pkl.gz'
PATH_RAW_JSON = f'{PATH_RAWDATA}/data.newpoems2.json.gz'

PATH_TEX = f'{PATH_DATA}/tex'

# Samples
PATH_SAMPLE_PERIOD_IN_PAPER = f'{PATH_DATA}/corpus_sample_by_period.csv.gz'
PATH_SAMPLE_PERIOD_SUBCORPUS_IN_PAPER = f'{PATH_DATA}/corpus_sample_by_period_subcorpus.csv.gz'
PATH_SAMPLE_RHYMES_IN_PAPER = f'{PATH_DATA}/corpus_sample_by_rhyme.csv.gz'
PATH_GENAI_PROMPTS_IN_PAPER = f'{PATH_DATA}/corpus_genai_promptings.csv.gz'
PATH_GENAI_COMPLETIONS_IN_PAPER = f'{PATH_DATA}/corpus_genai_completions.csv.gz'

REPLICATED_SUFFIX = 'replicated'
PATH_SAMPLE_PERIOD_REPLICATED = PATH_SAMPLE_PERIOD_IN_PAPER.replace('.csv', f'.{REPLICATED_SUFFIX}.csv')
PATH_SAMPLE_PERIOD_SUBCORPUS_REPLICATED = PATH_SAMPLE_PERIOD_SUBCORPUS_IN_PAPER.replace('.csv', f'.{REPLICATED_SUFFIX}.csv')
PATH_SAMPLE_RHYMES_REPLICATED = PATH_SAMPLE_RHYMES_IN_PAPER.replace('.csv', f'.{REPLICATED_SUFFIX}.csv')
PATH_GENAI_PROMPTS_REPLICATED = PATH_GENAI_PROMPTS_IN_PAPER.replace('.csv', f'.{REPLICATED_SUFFIX}.csv')
PATH_GENAI_COMPLETIONS_REPLICATED = PATH_GENAI_COMPLETIONS_IN_PAPER.replace('.csv', f'.{REPLICATED_SUFFIX}.csv')

USE_SAMPLE_PERIOD_IN_PAPER = True
USE_SAMPLE_PERIOD_REPLICATED = True

USE_SAMPLE_PERIOD_SUBCORPUS_IN_PAPER = True
USE_SAMPLE_PERIOD_SUBCORPUS_REPLICATED = True

USE_SAMPLE_RHYMES_IN_PAPER = True
USE_SAMPLE_RHYMES_REPLICATED = True


USE_GENAI_PROMPTS_IN_PAPER = True
USE_GENAI_PROMPTS_REPLICATED = True

USE_GENAI_COMPLETIONS_IN_PAPER = True
USE_GENAI_COMPLETIONS_REPLICATED = True

# PATH_SAMPLE = PATH_SAMPLE_REPLICATED if USE_SAMPLE_REPLICATED else PATH_SAMPLE_IN_PAPER
# PATH_SAMPLE_RHYMES = PATH_SAMPLE_RHYMES_REPLICATED if USE_SAMPLE_RHYMES_REPLICATED else PATH_SAMPLE_RHYMES_IN_PAPER
# PATH_GENAI_PROMPTS = PATH_GENAI_PROMPTS_REPLICATED if USE_GENAI_PROMPTS_REPLICATED else PATH_GENAI_PROMPTS_IN_PAPER
# PATH_GENAI_COMPLETIONS = PATH_GENAI_COMPLETIONS_REPLICATED if USE_GENAI_COMPLETIONS_REPLICATED else PATH_GENAI_COMPLETIONS_IN_PAPER


# External data env vars
PATH_CORPUS = f'{PATH_DATA}/chadwyck_poetry'
PATH_CHADWYCK_HEALEY_TXT = f'{PATH_CORPUS}/txt'
PATH_CHADWYCK_HEALEY_METADATA = f'{PATH_CORPUS}/metadata.csv'


# Set in .env or here
URL_CHADWYCK_HEALEY_TXT = os.getenv('URL_CHADWYCK_HEALEY_TXT', '')
URL_CHADWYCK_HEALEY_METADATA = os.getenv('URL_CHADWYCK_HEALEY_METADATA', '')



# PROMPTS

PROMPTS = {
    'DO_rhyme': [
        'Write a poem in ballad stanzas.',
        "Write an ryhmed poem in the style of Shakespeare's sonnets.",
        'Write a long poem that does rhyme.',
        'Write a poem in the style of Emily Dickinson.',
        'Write a poem in heroic couplets.',
        'Write an rhyming poem.',
        'Write a poem (with 20+ lines) that rhymes.',
        'Write a poem that does rhyme.',
        'Write a short poem that does rhyme.'
    ],
    
    'do_NOT_rhyme': [
        'Write a poem that does NOT rhyme.',
        'Write a poem (with 20+ lines) that does NOT rhyme.',
        'Write a long poem that does NOT rhyme.',
        'Write a poem in the style of Walt Whitman.',
        'Write a poem in free verse.',
        'Write a poem in blank verse.',
        'Write an unrhymed poem.',
        'Write a short poem that does NOT rhyme.'],
    'MAYBE_rhyme': [
        'Write a poem (with 20+ lines).',
        'Write a long poem.',
        'Write a poem in groups of two lines.',
        'Write a poem.',
        'Write a poem in stanzas of 4 lines each.',
        'Write a short poem.'
    ]
}




# Models
MODEL_LIST = [
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'deepseek/deepseek-chat',
    'gemini-pro',
    'gpt-3.5-turbo',
    'gpt-4-turbo',
    'ollama/llama3.1:70b',
    'ollama/llama3.1:8b',
    'ollama/olmo2',
    'ollama/olmo2:13b'
]


# Demos
DEMO_MODEL = MODEL_LIST[0]
DEMO_PROMPT = PROMPTS['do_NOT_rhyme'][0]


# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')




# Metadata fields mapping
CHADWYCK_CORPUS_FIELDS = {
    'id': 'id',
    'id_hash': 'id_hash',
    'attperi_str': 'period_meta',
    'attdbase_str': 'subcorpus',
    'author': 'author',
    'author_dob': 'author_dob',
    'title': 'title',
    'year': 'year',
    'num_lines': 'num_lines',
    'volhead': 'volume',
    'l': 'line',
    'attrhyme': 'rhyme',
    'attgenre': 'genre',
}
ALL_CHADWYCK_CORPUS_FIELDS = [
#  'Unnamed: 0',
 '_llp_',
 '_path',
 'a1',
 'alias',
 'aliasinv',
 'anote',
 'argument',
 'attauth',
 'attautid',
 'attbytes',
 'attdbase',
 'attgend',
 'attgenre',
 'attidref',
 'attnatn',
 'attperi',
 'attpoet',
 'attpubl',
 'attpubn1',
 'attpubn2',
 'attrhyme',
 'attsize',
 'attview',
 'audclip',
 'audio',
 'authdtls',
 'author',
 'author_dob',
 'author_dod',
 'author_gender',
 'bnote',
 'bo',
 'break',
 'bytes',
 'caesura',
 'caption',
 'cell',
 'chid',
 'collection',
 'conclude',
 'corpus',
 'dedicat',
 'engcorp2',
 'epigraph',
 'epilogue',
 'figure',
 'firstl',
 'gap',
 'greek',
 'hi',
 'hideinft',
 'id',
 'idref',
 'idz',
 'img',
 'it',
 'item',
 'l',
 'label',
 'lacuna',
 'lb',
 'litpack',
 'mainhead',
 'note',
 'num_lines',
 'p',
 'pb',
 'pbl',
 'pndfig',
 'poemcopy',
 'posthumous',
 'preface',
 'prologue',
 'publish',
 'reflink',
 'removed',
 'signed',
 'sl',
 'somauth',
 'sombiog',
 'sompoet',
 'speaker',
 'stage',
 'sub',
 'subhead',
 'sup',
 't1',
 't2',
 't3',
 'target',
 'title',
 'title_volume',
 'trailer',
 'ty',
 'u',
 'usonly',
 'video',
 'volhead',
 'xref',
 'y1',
 'year',
 'year_new',
 'year_old']


MIN_NUM_LINES = 10
MAX_NUM_LINES = 100
MIN_AUTHOR_DOB = 1600
MAX_AUTHOR_DOB = 2000


RHYME_MAX_DIST = 1
RHYME_MUST_BE_PERFECT = True
RHYME_PRED_FEATURE = 'num_perfectly_rhyming_lines_per10l'
RHYME_PRED_THRESHOLD = 4

MIN_SAMPLE_N = 10
MAX_SAMPLE_N = 1000

CORPUS_PERIOD_BY = 50


TABLE_NUM_PERIOD_SUBCORPUS_COUNTS = 5
TABLE_NUM_RHYME_PROMPTINGS = 2

PATH_TEX_PERIOD_SUBCORPUS_COUNTS = os.path.join(PATH_TEX, f'table_{TABLE_NUM_PERIOD_SUBCORPUS_COUNTS}.period_subcorpus_counts.tex')

PAPER_REGENERATED_SUFFIX = 'paper_regenerated'


NICE_PROMPT_TYPE = {
    'DO_rhyme': 'Rhymed',
    'do_NOT_rhyme': 'Unrhymed',
    'MAYBE_rhyme': 'Rhyme unspecified',
}

