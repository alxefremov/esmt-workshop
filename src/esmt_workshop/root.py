import sys
from pathlib import Path

def get_root():
    root = [
        it for it in [Path("src"), Path('esmt-workshop/src')]
        if (it / 'esmt_workshop').exists()
    ][0].parent
    sys.path.insert(0, str(root / 'src'))
    print('PROJECT_ROOT =', root)
    return root
