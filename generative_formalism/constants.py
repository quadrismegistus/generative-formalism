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

# External data env vars
PATH_CHADWYCK_HEALEY_TXT = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_TXT', ''))
PATH_CHADWYCK_HEALEY_METADATA = os.path.expanduser(os.getenv('PATH_CHADWYCK_HEALEY_METADATA', ''))


# Raw data paths
PATH_RAWDATA = f'{PATH_DATA}/raw'
PATH_RAW_PKL = f'{PATH_RAWDATA}/data.allpoems.pkl.gz'
PATH_RAW_JSON = f'{PATH_RAWDATA}/data.newpoems2.json.gz'


# Samples
PATH_SAMPLE = f'{PATH_DATA}/corpus_sample.csv.gz'
PATH_SAMPLE_RHYMES = f'{PATH_DATA}/corpus_sample_by_rhyme.csv.gz'
PATH_GENAI_PROMPTS = f'{PATH_DATA}/corpus_genai_promptings.csv.gz'
PATH_GENAI_COMPLETIONS = f'{PATH_DATA}/corpus_genai_completions.csv.gz'


# Metadata fields mapping
CHADWYCK_CORPUS_FIELDS = {
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
