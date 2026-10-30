#!/bin/bash
# Run every question once the base model exists: three lanes in parallel (the model is tiny; runs are cached and shared).
cd "$(dirname "$0")"
until [ -f results/base.pt ]; do sleep 30; done
lane(){ for q in "$@"; do PYTHONPATH=. uv run python -u -m experiments.$q > results/$q.log 2>&1; echo "$q exit $?"; done; }
lane q1_reproduce q5_measure_noise &
lane q2_noise_share q3_which_data q4_mitigations &
lane q6_subspace q7_spikes q8_data_quality &
wait; echo DONE
