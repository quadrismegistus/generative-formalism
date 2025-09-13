# Main Module: `generative_formalism`

The main module serves as the entry point for the `generative_formalism` package, providing unified access to all functionality and establishing global configuration.

## Module Structure

```python
from generative_formalism import *
```

This import provides access to all core functionality:

- **Corpus management** - Loading and processing poetic corpora
- **Language model interfaces** - Generating poetry with various AI models
- **Prosodic analysis** - Measuring rhyme, meter, and formal features
- **Statistical tools** - Comparing and analyzing results
- **Utilities** - Helper functions and data processing tools

## Global Configuration

The module establishes several global settings:

### Display Options
```python
pd.options.display.max_rows = 100  # Pandas display settings
p9.options.figure_size = (10, 5)   # Plot dimensions
p9.options.dpi = 300               # High-resolution plots
```

### Library Configuration
```python
prosodic.USE_CACHE = False         # Disable prosodic caching
prosodic.LOG_LEVEL = 'CRITICAL'    # Minimize logging output
tqdm.pandas()                      # Enable progress bars for pandas
filterwarnings('ignore')           # Suppress warnings
```

### Environment Setup
The module automatically:
- Loads environment variables from `.env` files
- Configures API access for language models
- Sets up data paths and stash locations
- Initializes caching mechanisms

## Key Features

### Unified Import System
All submodules are imported and made available:
```python
from .constants import *     # Paths and configuration
from .utils import *         # Utility functions  
from .llms import *          # Language model interfaces
from .prosody import *       # Prosodic analysis tools
from .corpus import *        # Corpus management
from .stats import *         # Statistical analysis
from .rhyme_promptings import *     # Rhyme prompting methods
from .rhyme_completions import *    # Rhyme completion analysis
```

### Shared Dependencies
Common libraries are imported once and made available throughout:
- **pandas** and **numpy** for data manipulation
- **plotnine** for visualization
- **prosodic** for phonetic analysis
- **tqdm** for progress tracking
- **nest_asyncio** for async operations

### Caching Infrastructure
Global caching support using:
- **functools.lru_cache** for function-level caching
- **hashstash** for persistent result caching
- **JSONLHashStash** for JSONL-based storage

## Usage Examples

### Basic Analysis Workflow
```python
# Import everything
from generative_formalism import *

# Load historical corpus
historical_poems = get_chadwyck_corpus_sampled_by_period()

# Generate AI poetry
ai_poems = generate_poems_with_model("gpt-4", prompts)

# Analyze rhyme patterns
historical_rhyme = get_rhyme_for_sample(historical_poems)
ai_rhyme = get_rhyme_for_sample(ai_poems)

# Compare results
comparison = compare_rhyme_distributions(historical_rhyme, ai_rhyme)
```

### Custom Configuration
```python
# Override default settings
prosodic.USE_CACHE = True          # Enable caching
p9.options.figure_size = (12, 8)   # Larger plots

# Custom data paths
PATH_DATA = "/custom/data/path"
```

## Error Handling

The module includes robust error handling:
- **Optional dependencies** - Graceful degradation when optional packages aren't available
- **Environment validation** - Checks for required environment variables
- **Path verification** - Ensures data directories exist and are accessible

## Performance Considerations

### Memory Management
- **Lazy loading** - Modules are only imported when needed
- **Efficient data structures** - Optimized pandas and numpy usage
- **Cache management** - Configurable caching to balance speed and memory

### Parallel Processing  
- **Async support** - Built-in support for asynchronous operations
- **Progress tracking** - Visual feedback for long-running operations
- **Batch processing** - Efficient handling of large datasets

This module design provides a clean, powerful interface for computational poetics research while maintaining flexibility and performance.
