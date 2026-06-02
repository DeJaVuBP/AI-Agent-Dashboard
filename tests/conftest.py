import sys
from pathlib import Path

# Ensure repository root is on sys.path so `ai_agent` package is importable during tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
