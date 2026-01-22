import sys
import os

# Add VEXAIRL directory to path for local imports (vex_core, pushback, etc.)
_vexairl_dir = os.path.dirname(os.path.abspath(__file__))
if _vexairl_dir not in sys.path:
    sys.path.insert(0, _vexairl_dir)