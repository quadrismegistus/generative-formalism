# Documentation Build Guide

## Unified Build Script: `build_docs.py`

All documentation building is now handled by a single script with multiple options:

### Quick Start

```bash
source venv/bin/activate
python build_docs.py
```

This opens an interactive menu with all options.

## Command Line Usage

### Check Build Status
```bash
python build_docs.py --status
```
Shows which notebooks are built and which need building.

### Build Missing Notebooks Only
```bash
python build_docs.py --all
```
Builds only notebooks that don't have HTML files yet (resume builds).

### Build Single Notebook
```bash
python build_docs.py notebooks/NotebookName.ipynb
```
Builds one specific notebook with proper CSS styling.

### Build Complete Book
```bash
python build_docs.py --book
```
Builds the entire Jupyter Book with full navigation.

## Key Features

✅ **Live Output** - See real-time execution progress with timestamps
✅ **Proper CSS** - Standalone HTML files with correct styling  
✅ **Resume Builds** - Skip notebooks that are already built
✅ **Status Checking** - See what's built vs missing
✅ **Error Handling** - Continue building even if one notebook fails
✅ **Clean Output** - Files saved to `docs_build/` directory

## Output Structure

- **Individual notebooks**: `docs_build/NotebookName.html`
- **Complete book**: `docs/_build/html/index.html`
- **CSS assets**: `docs_build/_static/` (for proper styling)

## Interactive Menu

Run without arguments for a menu:

```
📚 Jupyter Book Documentation Builder
Choose an option:
1. Show build status
2. Build missing notebooks only
3. Build single notebook  
4. Build complete book
5. Exit
```

## Tips

- **Resumable**: Interrupted builds can be resumed with `--all`
- **Fast previews**: Individual notebooks build much faster than full book
- **Proper styling**: All HTML files include proper CSS and work standalone
- **Live feedback**: See execution progress and timing in real-time
- **Error recovery**: Failed notebooks won't stop the entire build

## Troubleshooting

- **CSS issues**: The script automatically copies CSS assets and fixes paths
- **Missing files**: Use `--status` to see what needs building
- **Long execution**: Individual notebook mode shows live progress
- **Memory issues**: Build individual notebooks instead of full book
