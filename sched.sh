#!/bin/bash
# Run figure scripts one after another on one GPU, optionally after another figure script finishes.
# usage: ./sched.sh GPU_PCI_INDEX WAIT_FOR|- f1_curves f2_injection ...
cd "$(dirname "$0")"
gpu=$1; wait_for=$2; shift 2
[ "$wait_for" != - ] && while pgrep -f "figures_src[.]$wait_for" >/dev/null; do sleep 30; done
for f in "$@"; do
 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ~/.local/bin/uv run python -u -m figures_src.$f > results/$f.log 2>&1; echo "$f exit $?"
done
