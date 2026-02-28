import subprocess
import sys
import os
from pathlib import Path


def _find_project_root():
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return None


def setup():
    repo_url = "https://github.com/alxefremov/esmt-workshop.git"
    repo_dir = "esmt-workshop"

    project_root = _find_project_root()

    if project_root is not None:
        # --- Running locally inside the repo ---
        print(f"Local environment detected (project root: {project_root})")

        # Add src/ to sys.path so imports work
        src_dir = str(project_root / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            print(f"Added {src_dir} to sys.path")

        print("Setup complete!")
    else:
        # --- Running on Colab or outside the repo ---
        if not os.path.exists(repo_dir):
            print(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url], check=True)
        else:
            print(f"Directory '{repo_dir}' already exists. Skipping clone.")

        # Install dependencies
        requirements_path = os.path.join(repo_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print(f"Installing dependencies from {requirements_path}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path])
            print("Setup complete!")
        else:
            print(f"Warning: Requirements file not found at {requirements_path}")
