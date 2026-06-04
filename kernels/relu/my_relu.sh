#!/usr/bin/env bash
set -e

name=${1:-relu_f16}
dtype=${2:-float16}

ncu --nvtx \
  --nvtx-include "profiling/" \
  --set full \
  --import-source yes \
  -o "$name" \
  -- python3 my_relu.py --profiling "$name" --dtype "$dtype"