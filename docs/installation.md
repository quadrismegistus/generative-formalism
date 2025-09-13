# Installation Guide

This guide provides complete instructions for setting up the computational environment needed to reproduce the research and run the code examples in this book.

## System Requirements

### Operating System
- **macOS** - Tested on macOS 10.15+
- **Linux** - Tested on Ubuntu 18.04+ and CentOS 7+
- **Windows** - Limited testing, some features may require WSL

### Hardware
- **Memory** - 8GB RAM minimum, 16GB+ recommended for large corpus analysis
- **Storage** - 10GB free space for data and environments
- **CPU** - Multi-core processor recommended for parallel processing

## System Dependencies

### macOS
Install system-level dependencies using Homebrew:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system utilities
brew install poppler imagemagick ghostscript espeak
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils imagemagick ghostscript espeak espeak-data
```

### Linux (CentOS/RHEL)
```bash
sudo yum install -y poppler-utils ImageMagick ghostscript espeak
```

## Python Environment Setup

### Python Version
This project requires **Python 3.11** or later. We recommend using `pyenv` for Python version management.

#### Installing pyenv (optional but recommended)
```bash
# macOS
brew install pyenv

# Linux
curl https://pyenv.run | bash
```

#### Setting Python version
```bash
pyenv install 3.11.5
pyenv global 3.11.5  # or pyenv local 3.11.5 for project-specific
```

### Virtual Environment
Create and activate a virtual environment:

```bash
# Navigate to project directory
cd generative-formalism

# Create virtual environment
python -m venv venv

# Activate environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### Installing Dependencies

#### Core Dependencies
```bash
# Upgrade pip and install build tools
pip install -U pip wheel setuptools

# Install the research package and dependencies
pip install -e .

# OR install from requirements.txt
pip install -r requirements.txt
```

#### Jupyter Book Dependencies
```bash
# Install Jupyter Book and related tools
pip install jupyter-book>=0.15.0
pip install jupyterlab>=4.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

#### Development Dependencies (optional)
```bash
# For development and testing
pip install pytest>=7.0.0
pip install black>=23.0.0
pip install flake8>=6.0.0
pip install mypy>=1.0.0
```

## Environment Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your specific settings
nano .env
```

Required environment variables:
```bash
# API Keys (for AI model access)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Data paths (optional, defaults provided)
CHADWYCK_METADATA_URL=https://path/to/metadata.csv
CHADWYCK_CORPUS_URL=https://path/to/corpus.zip

# Computation settings
PYTHONPATH=/path/to/generative-formalism
```

### Data Setup

#### Historical Corpus Data
If you have access to the Chadwyck-Healey poetry corpus:

```bash
# Create data directory structure
mkdir -p data/chadwyck_poetry/txt
mkdir -p data/raw
mkdir -p data/stash

# Download and extract corpus (if you have access)
# Follow specific instructions for your corpus access
```

#### Public Data
For users without access to commercial corpora, we provide sample datasets:

```bash
# Download sample data
python -c "from generative_formalism.corpus import download_sample_data; download_sample_data()"
```

## Verification

### Test Installation
```bash
# Test basic imports
python -c "from generative_formalism import *; print('Import successful')"

# Test core functionality
python -c "from generative_formalism import check_paths; check_paths()"

# Run basic tests
pytest tests/ -v
```

### Test Jupyter Book
```bash
# Test book build
jupyter-book build .

# Serve locally (optional)
jupyter-book build . --path-output ./docs
python -m http.server 8000 -d docs/_build/html
```

## Common Issues and Solutions

### Import Errors
If you encounter import errors:

```bash
# Ensure the package is installed in development mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH

# Reinstall problematic packages
pip uninstall prosodic
pip install prosodic @ git+https://github.com/quadrismegistus/prosodic
```

### Memory Issues
For large corpus analysis:

```bash
# Increase available memory for Python
export PYTHONMAXMEMORY=8G

# Use chunked processing
python -c "from generative_formalism.constants import *; print(f'Cache dir: {PATH_STASH}')"
```

### Audio/Phonetic Issues
If `espeak` or phonetic analysis fails:

```bash
# Test espeak installation
espeak "hello world"

# Reinstall prosodic with dependencies
pip uninstall prosodic
pip install prosodic[full] @ git+https://github.com/quadrismegistus/prosodic
```

### API Access Issues
For language model access:

```bash
# Verify API keys are set
python -c "import os; print('OpenAI key set:', bool(os.getenv('OPENAI_API_KEY')))"

# Test API connectivity
python -c "from generative_formalism.llms import test_api_access; test_api_access()"
```

## Performance Optimization

### Parallel Processing
Enable parallel processing for large-scale analysis:

```bash
# Set number of worker processes
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
```

### Caching
Enable aggressive caching for development:

```bash
# Enable prosodic caching
export PROSODIC_USE_CACHE=true

# Set cache directory
export PROSODIC_CACHE_DIR=./cache/prosodic
```

## Development Setup

For contributors and developers:

```bash
# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Set up pre-commit hooks
pre-commit install

# Run full test suite
pytest tests/ --cov=generative_formalism

# Build documentation
jupyter-book build . --all
```

## Docker Alternative (Experimental)

For a completely isolated environment:

```bash
# Build Docker image
docker build -t generative-formalism .

# Run with mounted data
docker run -v $(pwd)/data:/app/data -p 8888:8888 generative-formalism
```

## Getting Help

If you encounter issues not covered here:

1. **Check the GitHub issues** - Someone may have encountered the same problem
2. **Review the error logs** - Most issues have clear error messages
3. **Verify your environment** - Ensure all dependencies are correctly installed
4. **Create a minimal example** - Isolate the problem to specific functionality
5. **File an issue** - Include your system info, error messages, and reproduction steps
