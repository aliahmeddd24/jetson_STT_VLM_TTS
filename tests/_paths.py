import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "services")):      # inside the app image
    ROOT = "/app"
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
for p in ("services/orchestrator", "services/action_server", "services/vlm", "scripts/bench", "tests"):
    full = os.path.join(ROOT, p)
    if full not in sys.path:
        sys.path.insert(0, full)
