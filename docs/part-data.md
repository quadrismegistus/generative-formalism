# Part I: Data Collection and Preparation

This section covers the comprehensive data collection and preparation process for analyzing both historical and AI-generated poetry.

## Overview

To understand the formal characteristics of AI-generated verse, we need robust datasets for comparison and analysis. This part documents:

1. **Historical Poetry Corpora** - Processing and analyzing large-scale collections of human poetry
2. **AI Poetry Generation** - Systematic approaches to generating poetry using various language models and prompting strategies
3. **Data Processing** - Methods for cleaning, organizing, and preparing textual data for analysis

## Data Sources

### Historical Corpora
- **Chadwyck-Healey Poetry Collections** - Comprehensive digital archive of English-language poetry
- **Period-specific subcorpora** - Organized by historical periods to track evolution of poetic forms
- **Metadata integration** - Author, date, genre, and other contextual information

### AI-Generated Poetry
- **Multiple language models** - Testing across different model architectures and sizes
- **Varied prompting strategies** - Comparing direct generation, few-shot learning, and instruction-following approaches
- **Systematic sampling** - Ensuring representative coverage across different poetic forms and themes

## Key Challenges

Working with poetic text data presents unique challenges:

- **Format preservation** - Maintaining line breaks, stanza structure, and spacing
- **Encoding issues** - Handling historical texts with varied character encodings
- **Scale considerations** - Processing large volumes of text while preserving granular details
- **Quality control** - Identifying and filtering low-quality or corrupted texts

## Data Organization

Our data pipeline ensures:

- **Reproducibility** - All processing steps are documented and version-controlled
- **Scalability** - Methods that work for both small samples and large corpora
- **Flexibility** - Data structures that support multiple types of analysis
- **Interoperability** - Formats compatible with various computational tools

The notebooks in this section provide step-by-step documentation of how we built and processed these datasets, making the entire process transparent and reproducible.
