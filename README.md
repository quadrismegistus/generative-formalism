# generative-formalism

Code and data for "Generative Aesthetics: On the formal stuckness of AI verse" (_Journal of Cultural Analytics_, vol. 10, no. 3, Sept. 2025)


## Installation

### System utils

```bash
# Install prereqs on Mac
brew install espeak # for Prosodic

# Optional - for latex:
brew install basictex poppler imagemagick ghostscript 
```


### Python

```bash
# Optional
pyenv shell 3.12

# Environment
python -m venv venv
. venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Then run the notebooks in [notebooks/](notebooks).


## Notebooks

The analysis is organized into a series of Jupyter notebooks that can be run sequentially or individually:

### Pipeline

- [`0-Run-All.ipynb`](notebooks/0-Run-All.ipynb) - Master notebook that executes all other notebooks in sequence

### Data

- [`1-Data-A-HistoricalPoems.ipynb`](notebooks/1-Data-A-HistoricalPoems.ipynb) - Loads and samples historical poetry from the Chadwyck-Healey corpus, creating datasets stratified by time period and rhyme annotation
- [`1-Data-B-GenerativePoemsByPrompt.ipynb`](notebooks/1-Data-B-GenerativePoemsByPrompt.ipynb) - Generates AI poetry by prompting language models to write rhymed or unrhymed poems
- [`1-Data-C-GenerativePoemsByCompletion.ipynb`](notebooks/1-Data-C-GenerativePoemsByCompletion.ipynb) - Generates AI poetry using completion-style prompts with historical poem fragments

### Methods

- [`2-Methods-A-MeasuringRhyme.ipynb`](notebooks/2-Methods-A-MeasuringRhyme.ipynb) - Implements and demonstrates computational rhyme detection methods, including validation against human annotations
- [`2-Methods-B-MeasuringRhythm.ipynb`](notebooks/2-Methods-B-MeasuringRhythm.ipynb) - Implements metrical analysis using Prosodic for rhythm and stress pattern detection
- [`2-Methods-C-DetectingMemorization.ipynb`](notebooks/2-Methods-C-DetectingMemorization.ipynb) - Detects memorized poems in AI outputs using both open-source training data and model completion behavior

### Results

- [`3-Results-A-Rhyme.ipynb`](notebooks/3-Results-A-Rhyme.ipynb) - Analyzes and visualizes rhyme patterns across historical periods and AI-generated poetry, including statistical comparisons
- [`3-Results-B-Rhythm.ipynb`](notebooks/3-Results-B-Rhythm.ipynb) - Analyzes stress patterns and metrical regularities in sonnets from Shakespeare, AI models, and historical corpora

### Paper

- [`4-Paper-A-Latex.ipynb`](notebooks/4-Paper-A-Latex.ipynb) - Generates LaTeX tables and formatted content for the research paper
- [`4-Paper-B-Numbers.ipynb`](notebooks/4-Paper-B-Numbers.ipynb) - Computes numerical results, statistics, and key figures referenced in the paper



## Data

The research data is organized in the `data/` directory following a clear pipeline from raw sources to final paper outputs:

### Raw data

- **[`raw/`](data/raw/)** - Original source data and external datasets
  - `corpus/` - Chadwyck-Healey metadata and Shakespeare sonnets
  - `memorization/` - Memorization detection data from Walsh et al. and Dolma corpus matches
  - `rhyme_promptings/` - Raw AI-generated poetry from prompting experiments
  - `rhyme_completions/` - Raw AI-generated poetry from completion experiments

### Processed data

- **[`data_as_replicated/`](data/data_as_replicated/)** - Reproducible processed datasets mirroring structure in `data_as_in_paper`
  - Data regenerated on your own machine
  - Produces new corpus samples stratified by period, subcorpus, rhyme, and sonnet type
  - New analyses for rhyme and rhythm patterns for these new samples

- **[`data_as_in_paper/`](data/data_as_in_paper/)** - Final datasets and analysis used in the paper

  * **Historical poetry corpus samples**
    - `corpus_sample_by_period.csv.gz` - 8K poems stratified by 50-year periods (1600-2000)
    - `corpus_sample_by_period_subcorpus.csv.gz` - 16K poems stratified by period and subcorpus
    - `corpus_sample_by_rhyme.csv.gz` - 2K poems balanced by rhyme annotation (1K rhymed, 1K unrhymed)
    - `corpus_sample_by_sonnet_period.csv.gz` - Sonnet samples stratified by historical period

  * **AI-generated poetry datasets**
    - `genai_rhyme_promptings.csv.gz` - AI poems from rhyme/no-rhyme prompts across models
    - `genai_rhyme_completions.csv.gz` - AI completions of historical poem fragments
    - `genai_poem_memorizations.csv.gz` - Memorization detection results for AI-generated poems

  * **Rhyme analysis results:**
    - `corpus_sample_by_*.rhyme_data.csv` - Computational rhyme measurements for each corpus sample
    - `genai_rhyme_promptings.rhyme_data.csv` - Rhyme analysis for AI-prompted poems
    - `genai_rhyme_completions.rhyme_data.csv` - Rhyme analysis for AI-completed poems
    - `genai_rhyme_completions_real.rhyme_data.csv` - Real historical fragments used for completion
    - `genai_rhyme_completions_gen.rhyme_data.csv` - AI-generated completions of fragments
    - `genai_rhyme_completions_text_vs_instruct.rhyme_data.csv` - Comparison of text vs instruction models
    - `all_memorization_data.rhyme_data.csv` - Combined memorization detection with rhyme analysis

  * **Rhythm analysis results:**
    - `rhythm_data_for_corpus_sample_by_sonnet_period.csv.gz` - Prosodic analysis of historical sonnets
    - `rhythm_data_for_genai_sonnets.csv.gz` - Prosodic analysis of AI-generated sonnets
    - `sonnet_rhythm_data_by_sonnet_period.csv.gz` - Rhythm analysis stratified by period

  * **Statistical analysis:**
    - `*.stats.rhyme_pred_perc_by_period.csv` - Rhyme percentage statistics by historical period
    - `human_genai_rhyme_data.stats.rhyme_pred_perc_by_xcol.csv` - Comparative statistics between human and AI poetry
    - `all_memorization_data.rhyme_data.csv.stats.rhyme_pred_perc_by_found_source_corpus.csv` - Memorization statistics by source corpus

  * **Output directories:**
    - `figures/` - All plots, charts, and visualizations used in the paper (PNG format with CSV data)
    - `tex/` - LaTeX table sources and formatted content for paper publication


## Code

The core analysis code is organized in the `generative_formalism/` module:

- **[`constants.py`](generative_formalism/constants.py)** - Configuration, file paths, model definitions, prompts, and global constants
- **[`utils/`](generative_formalism/utils/)** - General utilities and helper functions
  - `utils.py` - Text processing, hashing, file I/O, and general utilities  
  - `stats.py` - Statistical analysis functions for significance testing
  - `llms.py` - Language model API interactions and text generation

- **[`corpus/`](generative_formalism/corpus/)** - Historical poetry corpus management
  - `corpus.py` - Loading and preprocessing Chadwyck-Healey poetry collections
  - `sample.py` - Stratified sampling by period, subcorpus, and rhyme annotation
  - `tex.py` - LaTeX table generation for corpus statistics

- **[`rhyme/`](generative_formalism/rhyme/)** - Rhyme analysis and AI poetry generation
  - `rhyme_measurements.py` - Computational rhyme detection using prosodic analysis
  - `rhyme_promptings.py` - AI poetry generation via prompting (rhymed/unrhymed)
  - `rhyme_completions.py` - AI poetry generation via completion of historical fragments
  - `rhyme_memorizations.py` - Detection of memorized poems in AI outputs
  - `rhyme_plots.py` - Visualization and statistical plotting for rhyme analysis

- **[`rhythm/`](generative_formalism/rhythm/)** - Metrical analysis and stress patterns
  - `rhythm_measurements.py` - Prosodic parsing for meter, stress, and syllable analysis
  - `rhythm_analysis.py` - Aggregate analysis of stress patterns in sonnets and other forms
