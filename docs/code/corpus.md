# Corpus Management: `corpus.py`

The corpus module provides comprehensive tools for loading, processing, and managing poetic corpora, including both historical collections and AI-generated poetry datasets.

## Overview

This module handles:
- **Historical corpus loading** - Access to Chadwyck-Healey and other poetry collections
- **Data sampling and filtering** - Creating representative subsets for analysis
- **Metadata management** - Handling author, date, genre, and other contextual information
- **AI corpus integration** - Managing generated poetry alongside historical data

## Core Functions

### Historical Corpus Access

#### `get_chadwyck_corpus_metadata()`
Loads metadata for the Chadwyck-Healey poetry collection.

```python
df_meta = get_chadwyck_corpus_metadata()
# Returns DataFrame with columns: author, title, date, genre, etc.
```

#### `get_chadwyck_corpus_sampled_by_period()`
Creates period-based samples from the historical corpus.

```python
df_period_sample = get_chadwyck_corpus_sampled_by_period(
    n_per_period=1000,    # Poems per time period
    random_state=42       # For reproducibility
)
```

#### `get_chadwyck_corpus_sampled_by_rhyme_as_in_paper()`
Generates the exact rhyme-based sample used in the paper.

```python
df_rhyme_sample = get_chadwyck_corpus_sampled_by_rhyme_as_in_paper()
# Returns the canonical sample for reproducibility
```

### Sampling and Filtering

#### `sample_corpus_by_period(df, n_per_period=1000)`
Creates stratified samples across historical periods.

```python
sample = sample_corpus_by_period(
    df_poems, 
    n_per_period=500,
    balance_authors=True    # Ensure author diversity
)
```

#### `filter_by_metadata(df, **criteria)`
Filters corpus based on metadata criteria.

```python
sonnets = filter_by_metadata(
    df_poems,
    genre="sonnet",
    period=["renaissance", "romantic"],
    min_lines=14,
    max_lines=14
)
```

### Data Processing

#### `normalize_poem_text(text)`
Standardizes poem formatting and encoding.

```python
clean_text = normalize_poem_text(raw_poem_text)
# Handles encoding, line breaks, spacing
```

#### `extract_poem_metadata(df)`
Extracts and standardizes metadata fields.

```python
standardized_meta = extract_poem_metadata(raw_df)
# Creates consistent author, date, genre fields
```

### AI Corpus Integration

#### `load_ai_generated_corpus(model_name)`
Loads AI-generated poetry for specific models.

```python
gpt4_poems = load_ai_generated_corpus("gpt-4")
claude_poems = load_ai_generated_corpus("claude-3")
```

#### `merge_human_ai_corpora(human_df, ai_df)`
Combines human and AI poetry for comparative analysis.

```python
combined = merge_human_ai_corpora(
    historical_poems, 
    ai_poems,
    add_source_labels=True
)
```

## Data Structures

### Poem DataFrame Schema
Standard structure for all poem data:

```python
columns = [
    'text',          # Full poem text
    'author',        # Poet name
    'title',         # Poem title  
    'date',          # Composition/publication date
    'period',        # Historical period
    'genre',         # Poetic form/genre
    'source',        # human/ai indicator
    'model',         # AI model name (if applicable)
    'num_lines',     # Line count
    'num_stanzas',   # Stanza count
]
```

### Metadata Integration
Flexible metadata handling supporting:
- **Historical context** - Period, movement, cultural information
- **Technical details** - Line counts, stanza structure, formal classification
- **AI generation info** - Model, prompt, generation parameters
- **Quality metrics** - Formal adherence scores, human ratings

## Corpus Statistics

### `describe_corpus(df)`
Comprehensive corpus description and statistics.

```python
stats = describe_corpus(df_poems)
# Returns counts by period, author, genre, etc.
```

### `compare_corpus_distributions(df1, df2)`
Statistical comparison between corpora.

```python
comparison = compare_corpus_distributions(
    historical_corpus,
    ai_corpus,
    features=['num_lines', 'num_stanzas', 'period']
)
```

## File I/O and Persistence

### Supported Formats
- **CSV** - For tabular data with metadata
- **JSONL** - For structured text data
- **Compressed formats** - Gzip compression for large files
- **Pickle** - For complex Python objects

### Caching and Performance
- **Result caching** - Expensive operations are cached
- **Lazy loading** - Data loaded only when needed
- **Memory optimization** - Efficient handling of large corpora
- **Progress tracking** - Visual feedback for long operations

## Quality Control

### Data Validation
- **Text integrity** - Ensuring complete, uncorrupted texts
- **Metadata consistency** - Standardized field formats
- **Duplicate detection** - Identifying and handling duplicate poems
- **Encoding verification** - Proper Unicode handling

### Error Handling
- **Missing data** - Graceful handling of incomplete records
- **Format variations** - Flexible parsing of different text formats
- **Path management** - Robust file and directory handling
- **Network resilience** - Handling download and access errors

This module provides the foundation for all corpus-based analysis in the research, ensuring data quality, consistency, and accessibility across different types of poetic texts.
