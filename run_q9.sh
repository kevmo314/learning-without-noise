#!/bin/bash
# q9 after q4 frees its GPU memory (the 2B runs leave little headroom).
cd "$(dirname "$0")"
while pgrep -f "experiments.q4_mitigations" >/dev/null; do sleep 30; done
PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=. uv run python -u -m experiments.q9_deterministic > results/q9_deterministic.log 2>&1; echo "q9 exit $?"
