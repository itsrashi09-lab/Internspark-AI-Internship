"""
Utility script to convert all Python scripts to Jupyter notebooks.
Run this after installing jupytext: pip install jupytext

Usage:
    python convert_to_notebooks.py
"""

import subprocess
import os
import sys

def convert_scripts():
    """Convert all .py scripts in scripts/ to .ipynb notebooks."""
    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    notebooks_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    
    # Create notebooks directory
    os.makedirs(notebooks_dir, exist_ok=True)
    
    # Find all .py scripts
    scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
    
    if not scripts:
        print("No Python scripts found in scripts/ directory.")
        return
    
    print(f"Found {len(scripts)} scripts to convert:\n")
    
    for script in sorted(scripts):
        script_path = os.path.join(scripts_dir, script)
        notebook_name = script.replace('.py', '.ipynb')
        notebook_path = os.path.join(notebooks_dir, notebook_name)
        
        print(f"  Converting: {script} -> notebooks/{notebook_name}")
        
        try:
            subprocess.run(
                ["jupytext", "--to", "notebook", "--output", notebook_path, script_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✅ Success!")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error: {e.stderr}")
        except FileNotFoundError:
            print("  ❌ jupytext not found. Install it with: pip install jupytext")
            sys.exit(1)
        
        print()
    
    print(f"\nAll notebooks saved to: {notebooks_dir}/")
    print("\nTo execute notebooks and capture output:")
    print("  jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb")

if __name__ == "__main__":
    convert_scripts()
