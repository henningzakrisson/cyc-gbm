"""Allow running the numerical illustration as a package.

Usage:
    python -m numerical_illustration
    python numerical_illustration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numerical_illustration.numerical_illustration import main

main()
