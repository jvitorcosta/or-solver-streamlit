import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import _run_or_solver_application

if __name__ == "__main__":
    _run_or_solver_application()
