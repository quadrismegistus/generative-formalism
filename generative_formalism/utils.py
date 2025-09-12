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

def try_display(obj):
    try:
        from IPython.display import display
        display(obj)
    except (NameError, ImportError):
        pass

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
        

def describe_numeric(s, as_int=True, fixed_range=None, max_width=60):
    describe_numeric_ascii_boxplot(s, as_int=as_int, fixed_range=fixed_range, max_width=max_width)



def describe_numeric_ascii_boxplot(s, max_width=60, as_int=True, fixed_range=None):
    """Print an ASCII boxplot scaled to the numeric values.

    Parameters
    - s: pandas Series of numeric values
    - max_width: maximum character width of the plot area (inclusive of whiskers)
    - as_int: if True, display summary values as integers; otherwise show two decimals
    - fixed_range: tuple (min_val, max_val) to use fixed scale instead of data range
    """
    if max_width is None or max_width < 5:
        max_width = 5
    s_clean = s.dropna()
    if len(s_clean) == 0:
        print(f'{getattr(s, "name", "series")}\n<no data>')
        return

    q0 = float(s_clean.quantile(0.0))
    q05 = float(s_clean.quantile(0.25 / 2))
    q1 = float(s_clean.quantile(0.25))
    q2 = float(s_clean.quantile(0.5))
    q3 = float(s_clean.quantile(0.75))
    q35 = float(s_clean.quantile(0.75 + (.25 / 2)))
    q4 = float(s_clean.quantile(1.0))

    # Use fixed range if provided, otherwise use data range
    if fixed_range is not None:
        range_min, range_max = fixed_range
        range_min = float(range_min)
        range_max = float(range_max)
    else:
        range_min, range_max = q0, q4

    def fmt(v):
        return str(int(round(v))) if as_int else f'{v:.2f}'

    width = int(max_width)
    if range_max <= range_min or not np.isfinite(range_min + range_max):
        # Degenerate range; render a centered marker
        line = [' '] * width
        mid = width // 2
        line[mid] = '|'
        plot = ''.join(line)
        print(f'{getattr(s, "name", "series")}\n{fmt(range_min)} {plot} {fmt(range_max)}')
        return

    # Map value to position in [0, width-1] using the range
    def pos(v):
        r = (v - range_min) / (range_max - range_min)
        return int(round(r * (width - 1)))

    p0 = 0
    p05 = max(0, min(width - 1, pos(q05)))
    p1 = max(0, min(width - 1, pos(q1)))
    p2 = max(0, min(width - 1, pos(q2)))
    p3 = max(0, min(width - 1, pos(q3)))
    p35 = max(0, min(width - 1, pos(q35)))
    p4 = width - 1

    # Ensure ordering
    p05 = max(p0, min(p05, p4))
    p1 = max(p05, min(p1, p4))
    p2 = max(p1, min(p2, p4))
    p3 = max(p2, min(p3, p4))
    p35 = max(p3, min(p35, p4))

    chars = [' '] * width

    # Draw extended whiskers from Q0.05 to Q0.95 as '-'
    for i in range(p05, p35 + 1):
        chars[i] = '-'

    # Draw box [===] for Q1..Q3
    chars[p1] = f'[ {int(q1) if as_int else q1}'
    chars[p3] = f'{int(q3) if as_int else q3} ]'
    for i in range(p1 + 1, p3):
        chars[i] = ' '

    # Draw median as '|'
    chars[p2] = f' |'

    plot = ''.join(chars)
    title = getattr(s, 'name', None)
    if not title:
        title = 'series'
    print(f'  {fmt(range_min)} {plot} {fmt(range_max)}')


def describe_qual(s,sort_index=False, count=True):
    if count:
        s = s.value_counts()
    if sort_index:
        s = s.sort_index()
    x=repr(s)
    x='\n'.join(x.split('\n')[1:-1])
    print('* Breakdown for', getattr(s, 'name', 'series'))
    print(x)
    print()

def describe_qual_grouped(s, groupby, sort_index=False, count=True):
    odf = s.groupby(groupby).size().reset_index().rename(columns={0: 'count'})
    odf.set_index(groupby, inplace=True)
    if sort_index:
        odf.sort_index(inplace=True)
    print(odf)
    print()





# Attempt to render LaTeX -> PNG using local TeX toolchain; fallback to matplotlib image
def render_latex_to_png(tex_body, out_png_path):
    import tempfile
    import subprocess
    import shutil
    import os as _os

    # If TeX is available, prefer a full LaTeX compile (best fidelity)
    pdflatex = shutil.which('pdflatex')
    pdftocairo = shutil.which('pdftocairo')
    convert = shutil.which('convert')  # ImageMagick

    if pdflatex and (pdftocairo or convert):
        with tempfile.TemporaryDirectory() as td:
            tex_path = _os.path.join(td, 'table.tex')
            pdf_path = _os.path.join(td, 'table.pdf')
            tmp_png = _os.path.join(td, 'table.png')
            # Minimal standalone doc for reliable cropping
            doc = (
                '\\documentclass[preview]{standalone}\n'
                '\\usepackage{booktabs}\n'
                '\\usepackage{multirow}\n'
                '\\usepackage{array}\n'
                '\\usepackage{graphicx}\n'
                '\\usepackage{float}\n'
                '\\usepackage{setspace}\n'
                '\\usepackage[margin=0.5cm]{geometry}\n'
                '\\begin{document}\n'
                f'{tex_body}\n'
                '\\end{document}\n'
            )
            with open(tex_path, 'w') as f:
                f.write(doc)
            # Compile to PDF
            try:
                subprocess.run(
                    [pdflatex, '-interaction=nonstopmode', '-halt-on-error', 'table.tex'],
                    cwd=td,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except Exception as e:
                print(f'* LaTeX compile failed: {e}')
                return False
            # Convert PDF -> PNG (300 DPI)
            try:
                if pdftocairo:
                    subprocess.run(
                        [pdftocairo, '-singlefile', '-png', '-r', '300', pdf_path, _os.path.join(td, 'table')],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
                else:
                    subprocess.run(
                        [convert, '-density', '300', pdf_path, '-quality', '92', tmp_png],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
                # Ensure tmp_png exists (pdftocairo case writes to table.png)
                if not _os.path.exists(tmp_png):
                    tmp_png = _os.path.join(td, 'table.png')
                shutil.copyfile(tmp_png, out_png_path)
                return True
            except Exception as e:
                print(f'* PDF->PNG conversion failed: {e}')
                return False


def df_to_latex_table(
    df=None,
    save_latex_to: str | None = None,
    table_num: int | None = None,
    caption: str | None = None,
    label: str | None = None,
    position: str = 'H',
    center: bool = True,
    size: str | None = '\\small',
    singlespacing: bool = False,
    resize_to_textwidth: bool = False,
    column_format: str | None = None,
    escape: bool = True,
    index: bool = False,
    float_format: str | None = None,
    na_rep: str = '',
    save_latex_to_suffix: str | None = None,
    return_display: bool = False,
    inner_latex: str | None = None,
    longtable: bool = False,
    header: list[str] | None = None,
):
    """Render a DataFrame (or provided LaTeX tabular) into a LaTeX table.

    Parameters
    - df: pandas.DataFrame or None if providing `inner_latex`
    - save_latex_to: where to write the .tex (if None, only return string)
    - table_num: optional table number prefix for caption (e.g., "Table 5: ...")
    - caption, label: LaTeX caption and label
    - position: table float position (e.g., 't', 'H')
    - center: add \\centering
    - size: e.g., '\\small' (set None to skip)
    - singlespacing: add \\singlespacing line
    - resize_to_textwidth: wrap tabular in \\resizebox{\\textwidth}{!}{% ... }
    - column_format, escape, index, float_format, na_rep, header: passed to DataFrame.to_latex
    - save_latex_to_suffix: if provided, insert before .tex (e.g., '.paper_regenerated')
    - return_display: if True and path provided, return IPython Image of compiled PNG
    - inner_latex: full tabular environment string to embed instead of DataFrame.to_latex
    - longtable: pass through to DataFrame.to_latex (no outer table env when True)

    Returns
    - The LaTeX string if not saving to file, otherwise the path or display object when requested.
    """
    # Build tabular (or longtable) body
    if inner_latex is not None:
        tabular_str = inner_latex.strip()
    else:
        if df is None:
            raise ValueError('Either df or inner_latex must be provided')
        # Import pandas lazily
        try:
            import pandas as _pd  # noqa: F401
        except Exception as _e:
            raise RuntimeError('pandas is required for df_to_latex_table when df is provided') from _e

        to_latex_kwargs: dict = {
            'index': index,
            'escape': escape,
            'na_rep': na_rep,
            'longtable': longtable,
            'bold_rows': False,
        }
        if header is not None:
            to_latex_kwargs['header'] = header
        if column_format is not None:
            to_latex_kwargs['column_format'] = column_format
        if float_format is not None:
            # pandas to_latex accepts a function or format string; we pass through strings
            to_latex_kwargs['float_format'] = float_format
        tabular_str = df.to_latex(**to_latex_kwargs).strip()

    # If this is a longtable, do not wrap in a floating table environment
    if longtable:
        latex_str = tabular_str
    else:
        lines: list[str] = []
        lines.append(f'\\begin{{table}}[{position}]')
        if center:
            lines.append('  \\centering')
        if size:
            lines.append(f'  {size}')
        if singlespacing:
            lines.append('  \\singlespacing')
        if resize_to_textwidth:
            lines.append('  \\resizebox{\\textwidth}{!}{%')
        # Ensure the tabular begins on its own line with correct indentation
        lines.append('  ' + tabular_str.replace('\n', '\n  '))
        if resize_to_textwidth:
            lines.append('  }')
        _caption_prefix = f'Table {table_num}: ' if table_num is not None else ''
        if caption is not None:
            lines.append(f'  \\caption{{{_caption_prefix}{caption}}}')
        if label is not None:
            lines.append(f'  \\label{{{label}}}')
        lines.append('\\end{table}')
        latex_str = '\n'.join(lines)

    # Save to file if requested
    if save_latex_to:
        if save_latex_to_suffix:
            if save_latex_to.endswith('.tex'):
                save_path = save_latex_to.replace('.tex', f'.{save_latex_to_suffix}.tex')
            else:
                save_path = f'{save_latex_to}.{save_latex_to_suffix}'
        else:
            save_path = save_latex_to
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            print(f'* Writing LaTeX to {save_path}')
            f.write(latex_str)

        if return_display:
            try:
                from IPython.display import Image  # type: ignore
                png_path = save_path[:-4] + '.png' if save_path.endswith('.tex') else save_path + '.png'
                print(f'* Rendering PNG to {png_path}')
                ok = render_latex_to_png(latex_str, png_path)
                if ok:
                    return Image(png_path)
                else:
                    print(f"* Warning: Could not render PNG")
            except (NameError, ImportError):
                pass
        return save_path

    return latex_str


def get_nice_prompt_type(prompt_type):
    return NICE_PROMPT_TYPE.get(prompt_type, prompt_type)

def nice_path(path):
    return path.replace(PATH_REPO, '{PATH_REPO}')


def documentation(func, docstring=True, signature=True, source=False):
    if source:
        signature = False
    try:
        from IPython.display import Markdown, display
        import inspect
        sig = inspect.signature(func)
        params = []
        for name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                params.append(name)
            else:
                params.append(f"{name}={pformat(param.default, indent=4)}")
        params = '\n    '.join(params)
        pretty_sig = f'{func.__name__}(\n    {params}\n)'
        short_sig = f'{func.__name__}()'
        bad_sig = f'''{func.__name__}(
    
)'''
        if pretty_sig == bad_sig:
            pretty_sig = short_sig
        
        markdown_content = f"**Documentation for `{func.__name__}`**"
        
        if docstring:
            markdown_content += f"""

*Description*

```md
{func.__doc__}
```"""
        
        if signature:
            markdown_content += f"""

*Call signature*

```md
{pretty_sig}
```"""
        
        if source:
            source_code = inspect.getsource(func)
            markdown_content += f"""

*Source code*

```py
{source_code}
```"""
        
        display(Markdown(markdown_content))
    except:
        pass
