import os
import sys
from pathlib import Path

def get_roots():
    # Resolve repository root both for local runs and Google Colab.
    PROJECT_ROOT: Path | None = None

    for candidate in [Path.cwd(), *Path.cwd().parents, Path('/content/esmt-workshop')]:
        if (candidate / 'src' / 'esmt_workshop').exists():
            PROJECT_ROOT = candidate
            break

    if PROJECT_ROOT is None:
        raise FileNotFoundError(
            'Project root not found. Run this notebook from the ESMT_Workshop repository.'
        )

    assert PROJECT_ROOT is not None  # help the type checker after the guard above

    # Make local package importable inside notebook execution context.
    SRC_DIR = PROJECT_ROOT / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    print('PROJECT_ROOT =', PROJECT_ROOT)
    
    return {
        'PROJECT_ROOT': PROJECT_ROOT,
        'SRC_DIR': SRC_DIR,
    }
