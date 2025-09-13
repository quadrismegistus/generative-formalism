#!/usr/bin/env python3
"""
Unified Jupyter Book Documentation Builder

Features:
- Build individual notebooks or entire book
- Live output streaming during execution  
- Proper CSS and styling for standalone files
- Skip existing files (resume builds)
- Show status of all notebooks
- Multiple output modes

Usage:
  python build_docs.py                    # Interactive menu
  python build_docs.py --all              # Build all missing notebooks  
  python build_docs.py --book             # Build entire book
  python build_docs.py --status           # Show build status
  python build_docs.py notebook.ipynb     # Build single notebook
"""

import os
import sys
import time
import shutil
from pathlib import Path
import subprocess

# Configuration
NOTEBOOKS = [
    "notebooks/1-Data-A-HistoricalCorpora.ipynb",
    "notebooks/1-Data-B1-PromptingForRhyme.ipynb", 
    "notebooks/1-Data-B2-PromptingForRhymeAtScale.ipynb",
    "notebooks/1-Data-B3-CollectingRhymePromptingData.ipynb",
    "notebooks/1-Data-C-PromptingForCompletions.ipynb",n
    "notebooks/2-Methods-A-TestingRhymeMeasurement.ipynb",
    "notebooks/4-MeasuringRhymeInPromptedPoems.ipynb",
    "notebooks/6-MeasuringRhymeInCompletedPoems.ipynb",
    "notebooks/7-TestingPoemMemorization.ipynb",
    "notebooks/8-PlottingPoemMemorization.ipynb",
    "notebooks/9-TextVsInstructionModels.ipynb",
    "notebooks/10-Sonnets.ipynb",
]

OUTPUT_DIR = "docs/book"
BOOK_OUTPUT = "docs/book"

def get_html_info(html_path):
    """Get info about existing HTML file."""
    if os.path.exists(html_path):
        stat = os.stat(html_path)
        size_kb = stat.st_size / 1024
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
        return f"{size_kb:.1f}KB, {mtime}"
    return "Not found"

def check_build_status():
    """Check which notebooks are built and which need building."""
    print(f"🔍 Checking build status...")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"\n📊 STATUS CHECK:")
    print(f"{'='*80}")
    
    existing = []
    missing = []
    
    for notebook in NOTEBOOKS:
        if not os.path.exists(notebook):
            print(f"📂 MISSING SOURCE: {notebook}")
            continue
            
        notebook_name = Path(notebook).stem
        html_file = f"{OUTPUT_DIR}/{notebook_name}.html"
        
        if os.path.exists(html_file):
            html_info = get_html_info(html_file)
            print(f"✅ EXISTS: {notebook_name:<35} ({html_info})")
            existing.append(notebook)
        else:
            print(f"❌ MISSING: {notebook_name}")
            missing.append(notebook)
    
    print(f"{'='*80}")
    print(f"📈 SUMMARY: {len(existing)} built, {len(missing)} missing")
    
    return existing, missing

def copy_css_assets():
    """Copy CSS and static assets for proper styling."""
    full_build_dir = f"{BOOK_OUTPUT}/_build/html"
    
    if not os.path.exists(full_build_dir):
        print("🔧 Creating reference build for CSS assets...")
        try:
            cmd = ["jupyter-book", "build", ".", "--path-output", f"./{BOOK_OUTPUT}"]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            print("✅ Reference build created")
        except:
            print("⚠️  Using fallback CSS")
            return False
    
    # Copy assets to output directory
    target_static = Path(OUTPUT_DIR) / "_static"
    target_static.mkdir(parents=True, exist_ok=True)
    
    source_static = Path(full_build_dir) / "_static"
    if source_static.exists():
        # Copy CSS, JS, and design assets
        for pattern in ["*.css", "*.js"]:
            for file in source_static.glob(pattern):
                shutil.copy2(file, target_static)
        
        # Copy sphinx design assets
        sphinx_design = source_static / "_sphinx_design_static"
        if sphinx_design.exists():
            target_sphinx = target_static / "_sphinx_design_static"
            if target_sphinx.exists():
                shutil.rmtree(target_sphinx)
            shutil.copytree(sphinx_design, target_sphinx)
        
        print("🎨 CSS assets copied")
        return True
    
    return False

def fix_css_in_html(html_file):
    """Fix CSS paths in HTML for standalone viewing."""
    if not os.path.exists(html_file):
        return False
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix CSS paths and add fallback styling
    fixes = [
        ('href="_sphinx_design_static/', 'href="_static/_sphinx_design_static/'),
        ('src="_sphinx_design_static/', 'src="_static/_sphinx_design_static/'),
    ]
    
    modified = False
    for old_path, new_path in fixes:
        if old_path in content:
            content = content.replace(old_path, new_path)
            modified = True
    
    # Add fallback CSS if needed
    if '_static/' not in content:
        fallback_css = '''
<style>
/* Jupyter Book Fallback Styling */
body { 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fff;
}
.cell { margin: 1em 0; padding: 10px; border-left: 3px solid #ddd; }
.cell_input { background: #f8f9fa; border-left-color: #007acc; }
.cell_output { background: #fff; border-left-color: #28a745; }
pre, code { background: #f8f9fa; padding: 8px; border-radius: 4px; overflow-x: auto; }
h1, h2, h3, h4, h5, h6 { color: #2c3e50; margin-top: 1.5em; }
.highlight { background: #f8f9fa; }
</style>
'''
        content = content.replace('</head>', f'{fallback_css}</head>')
        modified = True
    
    if modified:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return modified

def build_notebook(notebook_path, show_output=True):
    """Build a single notebook with live output and proper CSS."""
    notebook_name = Path(notebook_path).stem
    
    if show_output:
        print(f"\n{'='*60}")
        print(f"🔨 Building: {notebook_name}")
        print(f"{'='*60}")
    
    start_time = time.time()
    temp_output = f"{OUTPUT_DIR}/temp_{notebook_name}"
    
    try:
        cmd = [
            "jupyter-book", "build", 
            notebook_path,
            "--path-output", temp_output
        ]
        
        if show_output:
            print(f"📋 Command: {' '.join(cmd)}")
            
            # Stream live output
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    timestamp = time.strftime('%H:%M:%S')
                    clean_line = output.rstrip()
                    if clean_line and 'reading sources' not in clean_line:
                        print(f"[{timestamp}] {clean_line}")
                        sys.stdout.flush()
            
            return_code = process.poll()
        else:
            # Silent build
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return_code = result.returncode
        
        elapsed = time.time() - start_time
        
        if return_code != 0:
            if show_output:
                print(f"❌ Build failed (exit code {return_code})")
            return False
        
        # Find generated HTML and copy to final location
        html_pattern = f"{temp_output}/_build/_page/{notebook_name}/html/{notebook_name}.html"
        if not os.path.exists(html_pattern):
            if show_output:
                print(f"❌ HTML not found: {html_pattern}")
            return False
        
        # Create output directory and copy HTML
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_html = f"{OUTPUT_DIR}/{notebook_name}.html"
        shutil.copy2(html_pattern, final_html)
        
        # Fix CSS paths
        fix_css_in_html(final_html)
        
        # Clean up temp directory
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
        
        if show_output:
            print(f"✅ SUCCESS: {notebook_name} ({elapsed:.1f}s)")
            print(f"📄 HTML: {final_html}")
            print(f"🌐 Open: file://{os.path.abspath(final_html)}")
        
        return True
        
    except Exception as e:
        if show_output:
            print(f"💥 Error: {e}")
        return False

def build_missing_notebooks(auto=False):
    """Build only notebooks that don't have HTML files yet."""
    existing, missing = check_build_status()
    
    if not missing:
        print("\n🎉 All notebooks already built!")
        return
    
    print(f"\n❓ Found {len(missing)} notebooks to build:")
    for nb in missing:
        print(f"   • {Path(nb).stem}")
    
    if not auto:
        response = input(f"\nBuild these {len(missing)} notebooks? (y/n): ").lower()
        if response != 'y':
            print("👋 Cancelled")
            return
    
    print(f"\n🚀 Building {len(missing)} notebooks...")
    
    # Ensure CSS assets are available
    copy_css_assets()
    
    successful = 0
    failed = 0
    
    for i, notebook in enumerate(missing, 1):
        print(f"\n{'🔸'*60}")
        print(f"📓 BUILDING {i}/{len(missing)}: {Path(notebook).stem}")
        print(f"{'🔸'*60}")
        
        if build_notebook(notebook):
            successful += 1
        else:
            failed += 1
            if not auto and i < len(missing):
                cont = input(f"\n❓ Continue with remaining notebooks? (y/n): ")
                if cont.lower() != 'y':
                    break
    
    print(f"\n{'='*60}")
    print(f"🏁 RESULTS: ✅ {successful} built, ❌ {failed} failed")
    print(f"📁 All HTML files: {OUTPUT_DIR}/")

def build_full_book():
    """Build the complete Jupyter Book."""
    print("📚 Building complete Jupyter Book...")
    
    try:
        cmd = ["jupyter-book", "build", ".", "--path-output", f"./{BOOK_OUTPUT}"]
        
        print(f"📋 Command: {' '.join(cmd)}")
        start_time = time.time()
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                timestamp = time.strftime('%H:%M:%S')
                clean_line = output.rstrip()
                if clean_line:
                    print(f"[{timestamp}] {clean_line}")
                    sys.stdout.flush()
        
        return_code = process.poll()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            print(f"\n✅ Book built successfully in {elapsed:.1f}s")
            print(f"🌐 Open: file://{os.path.abspath(f'{BOOK_OUTPUT}/_build/html/index.html')}")
            return True
        else:
            print(f"\n❌ Book build failed (exit code {return_code})")
            return False
            
    except Exception as e:
        print(f"💥 Error building book: {e}")
        return False

def main():
    """Main entry point with command line options."""
    if len(sys.argv) == 1:
        # Interactive menu
        print("📚 Jupyter Book Documentation Builder")
        print("Choose an option:")
        print("1. Show build status")
        print("2. Build missing notebooks only") 
        print("3. Build single notebook")
        print("4. Build complete book")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            check_build_status()
        elif choice == "2":
            build_missing_notebooks()
        elif choice == "3":
            notebook = input("Enter notebook path: ").strip()
            if os.path.exists(notebook):
                copy_css_assets()
                build_notebook(notebook)
            else:
                print(f"❌ Not found: {notebook}")
        elif choice == "4":
            build_full_book()
        elif choice == "5":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
    
    else:
        # Command line mode
        arg = sys.argv[1]
        
        if arg == "--status":
            check_build_status()
        elif arg == "--all":
            build_missing_notebooks(auto=True)
        elif arg == "--book":
            build_full_book()
        elif arg == "--help":
            print(__doc__)
        elif os.path.exists(arg):
            copy_css_assets()
            build_notebook(arg)
        else:
            print(f"❌ Unknown option or file not found: {arg}")
            print("Use --help for usage information")

if __name__ == "__main__":
    main()
