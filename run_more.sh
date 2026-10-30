#!/bin/bash
# The reviewer-requested tests, in three more lanes alongside the first (the model is tiny).
cd "$(dirname "$0")"
lane(){ for q in "$@"; do PYTHONPATH=. uv run python -u -m experiments.$q > results/$q.log 2>&1; echo "$q exit $?"; done; }
lane q9_deterministic & lane q10_rotate_noise & lane q11_optimizer & wait; echo DONE
