# Language Model Interface: `llms.py`

The LLM module provides a unified interface for generating poetry using various language models, supporting different prompting strategies and systematic experimentation.

## Overview

This module handles:
- **Model abstraction** - Unified interface for different LLM providers
- **Prompt engineering** - Systematic approaches to poetry generation prompts
- **Batch generation** - Efficient processing of multiple poetry requests
- **Response processing** - Cleaning and structuring generated poetry

## Core Functions

### Model Interface

#### `generate_poem(model, prompt, **kwargs)`
Generate a single poem using specified model and prompt.

```python
poem = generate_poem(
    model="gpt-4",
    prompt="Write a sonnet about artificial intelligence",
    temperature=0.8,
    max_tokens=200
)
```

#### `batch_generate_poems(model, prompts, **kwargs)`
Generate multiple poems efficiently.

```python
poems = batch_generate_poems(
    model="claude-3",
    prompts=sonnet_prompts,
    batch_size=10,
    temperature=0.7
)
```

### Supported Models

#### OpenAI Models
- **GPT-4** - Latest flagship model
- **GPT-3.5-turbo** - Efficient alternative
- **Text-davinci-003** - Legacy completion model

#### Anthropic Models
- **Claude-3** - Latest Anthropic model
- **Claude-2** - Previous generation
- **Claude-instant** - Faster variant

#### Other Providers
- **PaLM** - Google's model
- **Cohere** - Cohere's generation models
- **Local models** - Support for locally hosted models

### Prompting Strategies

#### Direct Generation
```python
prompt = "Write a haiku about spring."
poem = generate_poem(model, prompt)
```

#### Few-Shot Learning
```python
prompt = f"""
Here are some example haikus:
{example_haikus}

Now write a haiku about winter:
"""
poem = generate_poem(model, prompt)
```

#### Instruction Following
```python
prompt = """
Please write a Shakespearean sonnet with the following constraints:
- Topic: technology and humanity
- Rhyme scheme: ABAB CDCD EFEF GG
- Meter: iambic pentameter
- Include at least one metaphor
"""
poem = generate_poem(model, prompt)
```

#### Chain-of-Thought Prompting
```python
prompt = """
I need to write a sonnet. Let me think through this step by step:
1. Choose a topic: love and loss
2. Plan the rhyme scheme: ABAB CDCD EFEF GG
3. Think about the meter: iambic pentameter
4. Consider the volta: turn at line 9

Now I'll write the sonnet:
"""
poem = generate_poem(model, prompt)
```

## Systematic Experimentation

### Prompt Variations
#### `generate_prompt_variations(base_prompt, variations)`
Create systematic prompt variations for testing.

```python
variations = generate_prompt_variations(
    base_prompt="Write a sonnet about {topic}",
    variations={
        'topic': ['love', 'nature', 'technology'],
        'style': ['in the style of Shakespeare', 'in modern language', ''],
        'constraints': ['with perfect rhyme', 'with slant rhyme', '']
    }
)
```

### Parameter Sweeps
#### `parameter_sweep_generation(model, prompt, param_ranges)`
Test different generation parameters systematically.

```python
results = parameter_sweep_generation(
    model="gpt-4",
    prompt=base_prompt,
    param_ranges={
        'temperature': [0.3, 0.5, 0.7, 0.9],
        'top_p': [0.8, 0.9, 0.95, 1.0],
        'max_tokens': [100, 150, 200, 250]
    }
)
```

### A/B Testing
#### `compare_prompting_strategies(strategies, test_prompts)`
Compare different prompting approaches.

```python
comparison = compare_prompting_strategies(
    strategies=['direct', 'few_shot', 'instruction', 'chain_of_thought'],
    test_prompts=sonnet_topics,
    models=['gpt-4', 'claude-3']
)
```

## Response Processing

### Text Cleaning
#### `clean_generated_poem(raw_text)`
Standardize and clean generated poetry.

```python
clean_poem = clean_generated_poem(raw_response)
# Removes metadata, normalizes formatting, extracts poem text
```

#### `extract_poem_from_response(response)`
Extract pure poem text from model responses.

```python
poem_text = extract_poem_from_response(model_response)
# Handles different response formats across models
```

### Quality Filtering
#### `filter_valid_poems(generated_poems, criteria)`
Filter generated poems based on quality criteria.

```python
valid_poems = filter_valid_poems(
    poems,
    criteria={
        'min_lines': 4,
        'max_lines': 20,
        'has_rhyme': True,
        'coherent_topic': True
    }
)
```

### Metadata Extraction
#### `extract_generation_metadata(response, prompt, model_params)`
Extract metadata about the generation process.

```python
metadata = extract_generation_metadata(response, prompt, params)
# Returns dict with model, prompt, parameters, timing, etc.
```

## Async and Batch Processing

### Asynchronous Generation
```python
async def generate_poems_async(prompts, model="gpt-4"):
    """Generate multiple poems asynchronously."""
    tasks = [acompletion(model=model, messages=[{"role": "user", "content": p}]) 
             for p in prompts]
    responses = await asyncio.gather(*tasks)
    return [extract_poem_from_response(r) for r in responses]
```

### Rate Limiting
```python
def rate_limited_generation(prompts, model, requests_per_minute=60):
    """Generate poems with rate limiting."""
    # Implement rate limiting logic
    # Track API usage
    # Handle retries and errors
```

### Progress Tracking
```python
def generate_with_progress(prompts, model):
    """Generate poems with progress bar."""
    results = []
    for prompt in tqdm(prompts, desc="Generating poems"):
        poem = generate_poem(model, prompt)
        results.append(poem)
    return results
```

## Error Handling and Robustness

### API Error Management
```python
def robust_generation(prompt, model, max_retries=3):
    """Generate poem with retry logic for API errors."""
    for attempt in range(max_retries):
        try:
            return generate_poem(model, prompt)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Content Filtering
```python
def filter_inappropriate_content(poem_text):
    """Filter out inappropriate or problematic content."""
    # Check for various content issues
    # Return filtered text or None if unusable
```

### Validation
```python
def validate_poem_structure(poem_text, expected_form):
    """Validate that generated poem matches expected structure."""
    # Check line counts, stanza structure, etc.
    # Return validation results
```

## Caching and Persistence

### Result Caching
```python
@stashed_result(STASH_GENAI)
def cached_poem_generation(prompt, model, **params):
    """Cache expensive poem generation calls."""
    return generate_poem(model, prompt, **params)
```

### Data Persistence
```python
def save_generation_batch(poems, metadata, filename):
    """Save generation results with full metadata."""
    # Save to JSONL format with full provenance
    # Include model, prompts, parameters, timestamps
```

This module provides the foundation for systematic, reproducible poetry generation experiments, enabling controlled comparison between different models and prompting strategies.
