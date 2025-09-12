from . import *


async def stream_llm_litellm(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    if acompletion is None:
        raise RuntimeError('litellm is not installed.  `pip install litellm`')
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
                token = None
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, 'delta', None)
                    if delta is not None:
                        token = getattr(delta, 'content', None)
                    else:
                        message = getattr(choice0, 'message', None)
                        if message is not None:
                            token = getattr(message, 'content', None)
                if token:
                    yield token
                    if verbose:
                        print(token, end='', flush=True)
            except Exception:
                continue
    except Exception as e:
        print(f'LiteLLM streaming error: {e}', file=sys.stderr)
        raise


async def stream_llm_genai(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    inner_model = _extract_google_model_name(model)
    if not inner_model:
        raise ValueError("stream_llm_genai expects model starting with 'google/'")
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError('google-generativeai is not installed. `pip install google-generativeai`') from e
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment')
    genai.configure(api_key=api_key)
    model_kwargs = {}
    if system_prompt:
        model_kwargs['system_instruction'] = system_prompt
    gmodel = genai.GenerativeModel(model_name=inner_model, **model_kwargs)
    generation_config = {"temperature": float(temperature), "max_output_tokens": 1024}
    try:
        response = gmodel.generate_content(prompt, generation_config=generation_config, stream=True)
        for chunk in response:
            try:
                text = getattr(chunk, 'text', None)
                if text:
                    yield text
                    if verbose:
                        print(text, end='', flush=True)
            except Exception:
                continue
    except Exception as e:
        print(f'Gemini streaming error: {e}', file=sys.stderr)
        raise


async def stream_llm(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    if _extract_google_model_name(model):
        async for token in stream_llm_genai(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
            yield token
        return
    async for token in stream_llm_litellm(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
        yield token


async def generate_text_async(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> str:
    output_parts: list[str] = []
    # print(f'* model: {model}')
    # print(f'* prompt: {prompt}')
    # print(f'* temperature: {temperature}')
    # print(f'* system_prompt: {system_prompt}')
    # print(f'* verbose: {verbose}')
    async for token in stream_llm(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
        output_parts.append(token)
    return ''.join(output_parts)



def generate_text(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False, force: bool = False, stash: 'BaseHashStash' = STASH_GENAI) -> str:
    key = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature,
        'system_prompt': system_prompt,
    }
    if not force and key in stash:
        out = stash[key]
        if verbose:
            print(out)
        return out
    response = run_async(
        generate_text_async,
        model=model,
        prompt=prompt,
        temperature=temperature,
        system_prompt=system_prompt,
        verbose=verbose,
    )
    stash[key] = response
    return response






def get_model_cleaned(x: str) -> str:
    return (
        x.split('/')[-1].strip().title().split('-20')[0].split(':')[0]
        .replace('Gpt-', 'GPT-').split('-Chat')[0]
    )


def rename_model(x: str) -> str:
    if x == '' or x == HIST:
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


def get_model_renamed(x: str) -> str:
    return rename_model(x)


def _extract_google_model_name(model: str, prefix:str='google/') -> str | None:
    if not is_google_model(model):
        return None
    model = model.replace('gemini-pro', 'gemini-1.5-pro')
    return model.replace(prefix, '')

def is_google_model(model:str):
    if not isinstance(model, str) or not model:
        return None
    return 'gemini' in model






## Constants manipulation
PROMPT_TO_TYPE = {}
for prompt_type, prompt_list in PROMPTS.items():
    for prompt in prompt_list:
        PROMPT_TO_TYPE[prompt] = prompt_type
PROMPT_SET = set(PROMPT_TO_TYPE.keys())
PROMPT_LIST = list(PROMPT_TO_TYPE.keys())


MODEL_TO_TYPE = {m:get_model_renamed(m) for m in MODEL_LIST}
MODEL_TO_NAME = {m:get_model_cleaned(m) for m in MODEL_LIST}



def describe_prompts(prompts=PROMPT_LIST, prompt_to_type=PROMPT_TO_TYPE):
    """Print a description of the prompts with statistics and details.
    
    Args:
        prompts: List of prompt strings to describe. Defaults to PROMPT_LIST.
        prompt_to_type: Dictionary mapping prompts to their types. Defaults to PROMPT_TO_TYPE.
    """
    type_to_prompts = {v:[] for v in set(prompt_to_type.values())}
    for prompt, type in prompt_to_type.items():
        type_to_prompts[type].append(prompt)
    print(f'''* {len(set(prompts))} unique prompts
* {len(set([prompt_to_type.get(p, p) for p in prompts]))} prompt types

* List of prompts:
  {pformat(prompts)}

* List of prompt types:
  {pformat(type_to_prompts)}
''')

def describe_models(models=MODEL_LIST, model_to_type=MODEL_TO_TYPE):
    """Print a description of the models with statistics and details.
    
    Args:
        models: List of model strings to describe. Defaults to MODEL_LIST.
        model_to_type: Dictionary mapping models to their types. Defaults to MODEL_TO_TYPE.
        model_to_name: Dictionary mapping models to their names. Defaults to MODEL_TO_NAME.
    """
    type_to_models = {v:[] for v in set(model_to_type.values())}
    for model, type in model_to_type.items():
        type_to_models[type].append(model)
    # Models
    print(f'''* {len(models)} models (counting parameter changes)
  * {len(type_to_models)} model types ({", ".join(sorted(type_to_models.keys()))})
  * Using models:
  {pformat(type_to_models, indent=4)}
  ''')