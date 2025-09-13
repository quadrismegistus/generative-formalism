# Jupyter Book Setup for Generative Formalism

This directory has been configured as a Jupyter Book to provide an interactive, web-based presentation of the research on "Generative formalism: On the formal stuckness of AI verse."

## Quick Start

### Building the Book

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the book:**
   ```bash
   jupyter-book build .
   ```

3. **Open the book:**
   ```bash
   open _build/html/index.html
   ```

### Development Workflow

For active development and writing:

```bash
# Clean previous builds
jupyter-book clean .

# Build with full execution (slower but ensures all outputs are current)
jupyter-book build . --all

# Serve locally for live viewing
python -m http.server 8000 -d _build/html
```

## Book Structure

The book is organized into four main parts:

### Part I: Data Collection and Preparation
- **Historical Corpora** - Processing Chadwyck-Healey poetry collections
- **Prompting for Rhyme** - Various approaches to AI poetry generation
- **Data Collection** - Systematic gathering of AI-generated poetry

### Part II: Methods and Analysis  
- **Rhyme Measurement** - Computational prosodic analysis tools
- **Comparative Analysis** - Statistical methods for corpus comparison

### Part III: Results and Analysis
- **Memorization Studies** - Understanding AI poetry memorization patterns
- **Model Comparisons** - Systematic comparison of different language models
- **Formal Analysis** - Deep dive into specific poetic forms like sonnets

### Part IV: Code Documentation
- **Module Documentation** - Complete API reference for the `generative_formalism` package
- **Usage Examples** - Practical demonstrations of the tools

## Configuration

### Book Configuration (`_config.yml`)
Key settings include:
- **Execution mode**: Force re-execution of notebooks for reproducibility
- **Repository integration**: Links to GitHub for collaboration
- **Interactive features**: Binder and Colab integration for live execution

### Table of Contents (`_toc.yml`)
Defines the book structure and navigation. Modify this file to:
- Add new chapters or sections
- Reorder content
- Create custom groupings

## Interactive Features

### Executable Notebooks
All notebooks in the book can be executed interactively:
- **Download** - Click the download button to get notebook files
- **Binder** - Launch interactive sessions in the cloud
- **Colab** - Open notebooks in Google Colab

### Code Integration
The book includes full documentation of the `generative_formalism` Python package:
- **API Reference** - Complete function and class documentation
- **Usage Examples** - Practical code examples throughout
- **Reproducible Analysis** - All figures and results can be regenerated

## Customization

### Adding New Content

1. **Notebooks**: Add `.ipynb` files to the `notebooks/` directory
2. **Markdown pages**: Create `.md` files for non-executable content
3. **Update TOC**: Modify `_toc.yml` to include new content in navigation

### Styling
Customize the appearance by:
- Modifying `_config.yml` for global settings
- Adding custom CSS in `_static/` directory
- Configuring the Sphinx theme options

### Advanced Features
Enable additional functionality:
- **Bibliography**: Add citations using `references.bib`
- **Cross-references**: Link between sections and figures
- **Executable code**: Include live code examples

## Publishing

### GitHub Pages
Automatic deployment to GitHub Pages:
```bash
# Build the book
jupyter-book build .

# Deploy to gh-pages branch
ghp-import -n -p -f _build/html
```

### Local Serving
For local development:
```bash
# Simple HTTP server
python -m http.server 8000 -d _build/html

# Or using Node.js
npx http-server _build/html -p 8000
```

### PDF Generation
Generate PDF version:
```bash
# Install LaTeX dependencies first
jupyter-book build . --builder pdflatex
```

## Troubleshooting

### Common Issues

**Build failures**: 
- Check that all dependencies are installed
- Ensure notebooks can execute without errors
- Verify file paths in `_toc.yml`

**Missing outputs**:
- Use `jupyter-book build . --all` to force re-execution
- Check that required data files are available
- Verify environment variables are set

**Slow builds**:
- Use cached execution: `jupyter-book config sphinx.execute_notebooks cache`
- Consider pre-executing notebooks separately

### Performance Tips

- **Caching**: Enable execution caching for faster rebuilds
- **Selective execution**: Use `execute_notebooks: 'off'` for static content
- **Parallel processing**: Leverage multiple cores for large corpora

## Contributing

### Content Guidelines
- **Reproducibility**: All analyses should be fully reproducible
- **Documentation**: Include clear explanations and context
- **Code quality**: Follow Python style guidelines and include docstrings

### Review Process
1. Test build locally before committing
2. Ensure all notebooks execute successfully
3. Verify links and cross-references work
4. Check that figures and tables display correctly

## Support

For issues with the book setup:
- Check the [Jupyter Book documentation](https://jupyterbook.org/)
- Review the project's GitHub issues
- Contact the maintainers for research-specific questions

## License

This Jupyter Book setup and documentation are released under the same license as the main research project. See the main README for details.
