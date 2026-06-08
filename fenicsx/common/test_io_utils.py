import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from common.io_utils import (
    print_banner
)

print_banner(
    "TEST IO"
)