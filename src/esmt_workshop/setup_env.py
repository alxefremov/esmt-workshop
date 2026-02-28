import subprocess
import sys
import os
from pathlib import Path


def _is_colab():
    return "google.colab" in sys.modules or os.path.exists("/content")

def _find_project_root():
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return None

def setup():
    repo_dir = "esmt-workshop"

    project_root = _find_project_root()
    is_colab = _is_colab()

    if project_root is not None and not is_colab:
        # --- Running locally inside the repo ---
        print(f"Local environment detected (project root: {project_root})")

        # Add src/ to sys.path so imports work
        src_dir = str(project_root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            print(f"Added {src_dir} to sys.path")

        print("Setup complete!")
        
    else:
        # Install dependencies
        # On Colab, the root might be found but we still need to install reqs.
        if is_colab:
             print("Google Colab environment detected.")
             
        requirements_path = os.path.join(repo_dir, "requirements.txt")
        
        # If running from inside the repo in Colab, the path might be just requirements.txt
        if project_root is not None and (project_root / "requirements.txt").exists():
             requirements_path = str(project_root / "requirements.txt")
             
        if os.path.exists(requirements_path):
            print(f"Installing dependencies from {requirements_path}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path])
            print("Setup complete!")
        else:
            print(f"Warning: Requirements file not found at {requirements_path}")
