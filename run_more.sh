#!/bin/bash
# The reviewer-requested tests, in three more lanes alongside the first (the model is tiny).
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
lane(){ for q in "$@"; do PYTHONPATH=. uv run python -u -m experiments.$q > results/$q.log 2>&1; echo "$q exit $?"; done; }
lane q10_rotate_noise q11_optimizer & lane q9_deterministic & wait; echo DONE
