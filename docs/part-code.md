# Part IV: Code Documentation

This section provides comprehensive documentation of the `generative_formalism` Python package, which implements all the computational methods used in this research.

## Overview

The `generative_formalism` package is designed to support reproducible research in computational poetics. This part covers:

1. **Module Architecture** - How the code is organized and structured
2. **Core Functionality** - Key classes and functions for poetic analysis
3. **Usage Examples** - Practical demonstrations of how to use the tools
4. **Extension Guidelines** - How to adapt and extend the code for new research

## Package Structure

The package is organized into several focused modules:

- **`corpus.py`** - Tools for loading, processing, and managing poetic corpora
- **`prosody.py`** - Computational methods for prosodic analysis
- **`rhyme.py`** - Rhyme detection and classification algorithms  
- **`llms.py`** - Interface for generating poetry with language models
- **`stats.py`** - Statistical analysis and comparison methods
- **`utils.py`** - Utility functions and helper methods
- **`constants.py`** - Configuration and path management

## Design Principles

### Modularity
Each module focuses on a specific aspect of poetic analysis, making the code:
- **Easy to understand** - Clear separation of concerns
- **Simple to test** - Individual components can be tested independently
- **Flexible to extend** - New functionality can be added without affecting existing code

### Reproducibility
All methods are designed to support reproducible research:
- **Deterministic outputs** - Same inputs always produce same results
- **Version control** - All parameters and configurations are tracked
- **Documentation** - Comprehensive docstrings and usage examples

### Performance
The code is optimized for working with large-scale poetic corpora:
- **Efficient algorithms** - Optimized for speed and memory usage
- **Parallel processing** - Support for multi-core analysis
- **Caching mechanisms** - Avoiding redundant computations

## Key APIs

### Corpus Management
```python
# Load historical poetry corpus
df_poems = get_chadwyck_corpus_sampled_by_period()

# Generate AI poetry
ai_poems = generate_poems_with_prompts(prompts, model="gpt-4")
```

### Prosodic Analysis
```python
# Analyze rhyme patterns
rhyme_data = get_rhyme_for_sample(poem_sample)

# Measure metrical patterns  
meter_analysis = analyze_meter(poem_text)
```

### Statistical Comparison
```python
# Compare corpora
comparison_stats = compare_formal_features(human_poems, ai_poems)

# Test for significance
test_results = statistical_significance_test(feature_distributions)
```

## Data Structures

The package uses consistent data structures throughout:

- **Pandas DataFrames** - For tabular data like poem metadata and analysis results
- **Dictionary objects** - For structured analysis results and configuration
- **JSON serialization** - For data persistence and sharing
- **Standard formats** - CSV, JSONL, and other common formats for interoperability

## Configuration and Customization

### Environment Variables
The package uses environment variables for:
- **API keys** - For language model access
- **Data paths** - Configurable locations for corpora and outputs
- **Model settings** - Default parameters for analysis methods

### Custom Extensions
The modular design makes it easy to:
- **Add new metrics** - Extend prosodic analysis with custom measurements
- **Support new models** - Add interfaces for additional language models
- **Implement new methods** - Build on existing infrastructure for novel approaches

## Testing and Validation

The package includes comprehensive testing:
- **Unit tests** - Individual function and method validation
- **Integration tests** - End-to-end workflow verification
- **Performance benchmarks** - Ensuring scalability and efficiency
- **Regression tests** - Protecting against unintended changes

This documentation provides both reference material for developers and practical guidance for researchers who want to use or extend the tools for their own work.
