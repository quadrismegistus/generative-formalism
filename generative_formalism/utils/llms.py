"""Language model utilities for text generation using various LLM providers.

This module provides unified interfaces for working with different language models
through LiteLLM and Google's Generative AI, including streaming and caching capabilities.
"""

from . import *


async def stream_llm_litellm(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> AsyncGenerator[str, None]:
    """Stream text generation from language models using LiteLLM.
    
    Args:
        model: The model identifier (e.g., 'gpt-4', 'claude-3-opus-20240229')
        prompt: The user prompt/input text
        temperature: Sampling temperature for text generation (0.0-1.0)
        system_prompt: Optional system prompt to set model behavior
        verbose: If True, print tokens to stdout as they're generated
        
    Yields:
        str: Individual tokens/text chunks from the model response
        
    Raises:
        RuntimeError: If litellm is not installed
        Exception: If the LLM request fails
    """
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
    """Stream text generation from Google Generative AI models.
    
    Args:
        model: The Google model identifier (must start with 'google/')
        prompt: The user prompt/input text
        temperature: Sampling temperature for text generation (0.0-1.0)
        system_prompt: Optional system instruction for the model
        verbose: If True, print tokens to stdout as they're generated
        
    Yields:
        str: Individual text chunks from the model response
        
    Raises:
        ValueError: If model doesn't start with 'google/'
        RuntimeError: If google-generativeai is not installed or API key is missing
        Exception: If the Gemini API request fails
    """
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
    """Universal streaming interface for language models.
    
    Automatically routes to the appropriate streaming function based on the model name.
    Google models (containing 'gemini') use the Google Generative AI API,
    all others use LiteLLM.
    
    Args:
        model: The model identifier
        prompt: The user prompt/input text
        temperature: Sampling temperature for text generation (0.0-1.0)
        system_prompt: Optional system prompt/instruction
        verbose: If True, print tokens to stdout as they're generated
        
    Yields:
        str: Individual tokens/text chunks from the model response
    """
    stream_func = stream_llm_genai if _extract_google_model_name(model) else stream_llm_litellm

    try:    
        async for token in stream_func(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
            yield token
    except Exception as e:
        print(f'Stream error: {e}', file=sys.stderr)


async def generate_text_async(model: str, prompt: str, temperature: float = 0.7, system_prompt: str | None = None, verbose: bool = False) -> str:
    """Generate complete text response asynchronously using streaming.
    
    Collects all streamed tokens and returns the complete response as a single string.
    
    Args:
        model: The model identifier
        prompt: The user prompt/input text
        temperature: Sampling temperature for text generation (0.0-1.0)
        system_prompt: Optional system prompt/instruction
        verbose: If True, print tokens to stdout as they're generated
        
    Returns:
        str: The complete generated text response
    """
    output_parts: list[str] = []
    # print(f'* model: {model}')
    # print(f'* prompt: {prompt}')
    # print(f'* temperature: {temperature}')
    # print(f'* system_prompt: {system_prompt}')
    # print(f'* verbose: {verbose}')
    async for token in stream_llm(model=model, prompt=prompt, temperature=temperature, system_prompt=system_prompt, verbose=verbose):
        output_parts.append(token)
    return ''.join(output_parts)

def check_api_keys(verbose: bool = False):
    """
    Check if the API keys are set in the environment.

    Defaults to the environment variables set in the .env file.

    Variables used:
    - GEMINI_API_KEY: Google Gemini API key
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic (Claude) API key
    - DEEPSEEK_API_KEY: DeepSeek API key
    """
    if verbose:
        print(f'{"✓" if GEMINI_API_KEY else "X"} Gemini API key')
        print(f'{"✓" if OPENAI_API_KEY else "X"} OpenAI API key')
        print(f'{"✓" if ANTHROPIC_API_KEY else "X"} Anthropic API key')
        print(f'{"✓" if DEEPSEEK_API_KEY else "X"} DeepSeek API key')
    
    out = []
    if GEMINI_API_KEY:
        out.append("GEMINI_API_KEY")
    if OPENAI_API_KEY:
        out.append("OPENAI_API_KEY")
    if ANTHROPIC_API_KEY:
        out.append("ANTHROPIC_API_KEY")
    if DEEPSEEK_API_KEY:
        out.append("DEEPSEEK_API_KEY")
    return out

def generate_text(model: str, prompt: str, temperature: float = DEFAULT_TEMPERATURE, system_prompt: str | None = None, verbose: bool = False, force: bool = False, stash: 'BaseHashStash' = STASH_GENAI) -> str:
    """Generate text with caching support (synchronous interface).
    
    This is the main text generation function that includes caching capabilities.
    Results are cached based on the combination of model, prompt, temperature, and system_prompt.
    
    Args:
        model: The model identifier
        prompt: The user prompt/input text
        temperature: Sampling temperature for text generation (0.0-1.0)
        system_prompt: Optional system prompt/instruction
        verbose: If True, print the complete response to stdout
        force: If True, bypass cache and force new generation
        stash: Cache storage backend for results
        
    Returns:
        str: The complete generated text response (from cache or new generation)
    """
    if verbose:
        print(f'* Generating text')
        print(f'  * model: {model}')
        prompt_preview = prompt.replace("\n", " ").strip()[:100]
        print(f'  * prompt: {prompt_preview}')
        print(f'  * temperature: {temperature}')
        if system_prompt:
            system_preview = system_prompt.replace("\n", " ").strip()[:100]
            print(f'  * system_prompt: {system_preview}')
        print(f'  * force: {force}')
        print(f'  * stash: {stash}')
    key = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature,
        'system_prompt': system_prompt,
    }
    if not force and key in stash:
        if verbose:
            print(f'  * from_cache: True')
        response = stash[key]
        if verbose:
            for word in response.split(' '):
                print(word, end=' ', flush=True)
                time.sleep(random.uniform(0.01, 0.03))
    else:
        if verbose:
            print(f'  * from_cache: False\n')
        # print(f'\n* Generating new text:')
        response = run_async(
            generate_text_async,
            model=model,
            prompt=prompt,
            temperature=temperature,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        # if verbose:
            # print(f'\n> Response: {response.replace("\n", " ").strip()[:100]}...')
        stash[key] = response
    return response






def get_model_cleaned(x: str) -> str:
    """Clean and format model name for display purposes.
    
    Extracts the core model name, removes version suffixes, and applies
    title case formatting with special handling for GPT models.
    
    Args:
        x: Raw model identifier string
        
    Returns:
        str: Cleaned and formatted model name
        
    Examples:
        >>> get_model_cleaned('openai/gpt-4-turbo-preview')
        'GPT-4-Turbo-Preview'
        >>> get_model_cleaned('anthropic/claude-3-opus-20240229')
        'Claude-3-Opus'
    """
    return (
        x.split('/')[-1].strip().title().split('-20')[0].split(':')[0]
        .replace('Gpt-', 'GPT-').split('-Chat')[0]
    )


def rename_model(x: str) -> str:
    """Map model identifiers to standardized family names.
    
    Converts various model identifiers to consistent family names
    for grouping and analysis purposes.
    
    Args:
        x: Model identifier string
        
    Returns:
        str: Standardized model family name (e.g., 'ChatGPT', 'Claude', 'Llama')
             Returns empty string for text models or unrecognized models
             Returns HIST constant for historical/empty models
    """
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
    """Alias for rename_model function.
    
    Args:
        x: Model identifier string
        
    Returns:
        str: Standardized model family name
    """
    return rename_model(x)


def _extract_google_model_name(model: str, prefix:str='google/') -> str | None:
    """Extract the actual Google model name from a prefixed identifier.
    
    Removes the provider prefix and applies any necessary model name transformations
    for compatibility with the Google Generative AI API.
    
    Args:
        model: Full model identifier (e.g., 'google/gemini-pro')
        prefix: Provider prefix to remove (default: 'google/')
        
    Returns:
        str | None: The extracted model name, or None if not a Google model
                   'gemini-pro' is automatically converted to 'gemini-1.5-pro'
    """
    if not is_google_model(model):
        return None
    model = model.replace('gemini-pro', 'gemini-1.5-pro')
    return model.replace(prefix, '')

def is_google_model(model: str) -> bool | None:
    """Check if a model identifier refers to a Google/Gemini model.
    
    Args:
        model: Model identifier string to check
        
    Returns:
        bool | None: True if model contains 'gemini', None if model is invalid/empty
    """
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
    print(f'''* {len(set(prompts))} unique prompts\n* {len(set([prompt_to_type.get(p, p) for p in prompts]))} prompt types''')
    for type, prompts in type_to_prompts.items():
        print(f'  * {type}:')
        for prompt in prompts:
            print(f'    - {prompt}')
        print()

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
    print(f'''* {len(models)} models (counting parameter changes)\n* {len(type_to_models)} model types ({", ".join(sorted(type_to_models.keys()))})''')
    for type, models in type_to_models.items():
        print(f'  * {type}: {", ".join(models)}')
    print()

def filter_available_models(models=MODEL_LIST, verbose=False):
    """
    Filter the models to only include those with API keys.
    """
    api_keys = check_api_keys(verbose=verbose)
    models2 = []
    for model in models:
        if get_model_api_key_required(model) in api_keys:
            if verbose:
                print(f'  ✓ {model}')
            models2.append(model)
        else:
            if verbose:
                print(f'  ✗ {model}')
    return models2


def get_demo_model_prompt(demo_model=None, demo_prompt=None, verbose=False):
    """
    Return demo model and prompt, defaults to DEMO_MODEL and DEMO_PROMPT.
    """
    #     print(f'''* Demo model: {demo_model}
    # * Demo prompt: {demo_prompt}
    # ''')

    if demo_model is None:
        available_models = filter_available_models(models=MODEL_LIST, verbose=verbose)
        demo_model = available_models[0] if available_models else MODEL_LIST[0]
    if demo_prompt is None:
        # demo_prompt = random.choice(PROMPT_LIST)
        demo_prompt = DEMO_PROMPT

    return demo_model, demo_prompt

def get_working_model():
    models = filter_available_models(models=MODEL_LIST, verbose=False)
    return models[0] if models else MODEL_LIST[0]