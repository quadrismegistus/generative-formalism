import os
import sys
from pprint import pformat, pprint
import json
import random
import asyncio
import inspect
from datetime import datetime
from functools import lru_cache
from typing import AsyncGenerator
import gzip
from warnings import filterwarnings
import requests
import zipfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
import plotnine as p9
import nest_asyncio
import prosodic
from hashstash import stashed_result, JSONLHashStash, HashStash
from hashstash.engines.jsonl import JSONLHashStash
from rapidfuzz import fuzz
from functools import lru_cache
cache = lru_cache(maxsize=None)

try:
    from litellm import acompletion
except Exception:
    acompletion = None


# Global library configuration
tqdm.pandas()
filterwarnings('ignore')
load_dotenv()

p9.options.figure_size = (10, 5)
p9.options.dpi = 300
pd.options.display.max_rows = 5
pd.options.display.max_columns = None

prosodic.USE_CACHE = False
prosodic.LOG_LEVEL = 'CRITICAL'

# Shared cache decorator
cache = lru_cache(maxsize=1000)


# Load constants first so paths and shared objects are available to modules
from .constants import *  # noqa: E402,F401,F403

# Load rest of modules (each module begins with `from . import *` to reuse imports)
from .utils import *  # noqa: E402,F401,F403
from .corpus import *  # noqa: E402,F401,F403
from .rhyme import *  # noqa: E402,F401,F403
from .rhythm import *  # noqa: E402,F401,F403