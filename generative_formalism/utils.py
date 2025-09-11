from . import *


def get_id_hash(id, seed=42, max_val=1000000):
    random.seed(hash(id) + seed)
    return random.randint(0, max_val - 1)


def get_id_hash_str(id):
    from hashlib import sha256
    return sha256(id.encode()).hexdigest()[:8]

def save_sample(df, path_sample, overwrite=False):
    os.makedirs(os.path.dirname(path_sample), exist_ok=True)
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


def clean_poem_str(txt: str) -> str:
    stanzas = txt.split('\n\n')
    stanzas = [st.strip() for st in stanzas if st.strip().count('\n') > 0]
    return '\n\n'.join(stanzas)


def get_num_lines(txt: str) -> int:
    return len([x for x in txt.split('\n') if x.strip()])

def download_file(url: str, filepath: str) -> None:
    """Download a file from URL with progress bar, without displaying the URL."""
    import urllib.request
    import urllib.parse
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Get file size first
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
    except:
        total_size = 0
    
    # Set up tqdm progress bar
    progress_bar = None
    if total_size > 0:
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc='  ')
    
    def progress_hook(block_num, block_size, total_size):
        if progress_bar and total_size > 0:
            downloaded = block_num * block_size
            if block_num == 0:
                progress_bar.reset(total=total_size)
            else:
                progress_bar.update(block_size)
    
    try:
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        if progress_bar:
            progress_bar.close()
    except Exception as e:
        if progress_bar:
            progress_bar.close()
        print(f'\nDownload failed: {e}')
        raise

def unzip_file(filepath: str, extract_to: str, remove_zip=True, use_parent_dir=True) -> None:
    """Extract a zip file with progress bar."""
    if use_parent_dir:
        extract_to = os.path.dirname(extract_to)
    
    os.makedirs(extract_to, exist_ok=True)
    
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        # Get list of files to extract
        file_list = zip_ref.infolist()
        
        # Extract with progress bar
        with tqdm(total=len(file_list), unit='file', desc='  ') as pbar:
            for file_info in file_list:
                zip_ref.extract(file_info, extract_to)
                pbar.update(1)
    if remove_zip:
        try:
            os.remove(filepath)
            print(f'* Removed zip file: {filepath}')
        except:
            pass
        

def describe_numeric(s, as_int=True):
    q0=s.quantile(0.0)
    q1=s.quantile(0.25)
    q2=s.quantile(0.5)
    q3=s.quantile(0.75)
    q4=s.quantile(1.0)
    if as_int:
        q0=int(q0)
        q1=int(q1)
        q2=int(q2)
        q3=int(q3)
        q4=int(q4)
    print(f'{s.name}\n{q0} ------- [ {q1}   | {q2} |   {q3} ] -------- {q4}\n')



def describe_qual(s,sort_index=False, count=True):
    if count:
        s = s.value_counts()
        if sort_index:
            s = s.sort_index()
    print(s)
    print()

def describe_qual_grouped(s, groupby, sort_index=False, count=True):
    odf = s.groupby(groupby).size().reset_index().rename(columns={0: 'count'})
    odf.set_index(groupby, inplace=True)
    if sort_index:
        odf.sort_index(inplace=True)
    print(odf)
    print()

