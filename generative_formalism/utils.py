from . import *


def get_id_hash(id, seed=42, max_val=1000000):
    random.seed(hash(id) + seed)
    return random.randint(0, max_val - 1)


def get_id_hash_str(id):
    from hashlib import sha256
    return sha256(id.encode()).hexdigest()[:8]

def save_sample(df, path_sample=PATH_SAMPLE, overwrite=False):
    if overwrite or not os.path.exists(path_sample):
        df.to_csv(path_sample)
        print(f'* Saved sample to {path_sample}')
    else:
        path_sample_now = f"{os.path.splitext(path_sample.replace('.gz', ''))[0]}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
        if path_sample.endswith('.csv.gz'):
            path_sample_now += '.gz'
        df.to_csv(path_sample_now)
        print(f'* Saved sample to {path_sample_now}')


def printm(text, *args, **kwargs):
    try:
        from IPython.display import display, Markdown
        get_ipython()
        if args or any(k in kwargs for k in ['file', 'flush']):
            print(text, *args, **kwargs)
        else:
            display(Markdown(str(text)))
    except (NameError, ImportError):
        print(text, *args, **kwargs)



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
        raise TypeError('run_async expected coroutine or async generator')
    if loop.is_running():
        nest_asyncio.apply()
        return loop.run_until_complete(awaitable)
    else:
        return loop.run_until_complete(awaitable)




def limit_lines(txt, n=100):
    lines: list[str] = []
    nonempty_count = 0
    for line in txt.strip().split('\n'):
        if line.strip():
            nonempty_count += 1
        lines.append(line)
        if nonempty_count >= n:
            break
    return '\n'.join(lines).strip()


def clean_genai_poem(txt: str) -> str:
    stanzas = txt.split('\n\n')
    stanzas = [st.strip() for st in stanzas if st.strip().count('\n') > 0]
    return '\n\n'.join(stanzas)


def get_num_lines(txt: str) -> int:
    return len([x for x in txt.split('\n') if x.strip()])
