import os
import sys
from pathlib import Path

# Make `import utils` work when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
