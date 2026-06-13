#!/usr/bin/env bash
set -e

name=${1:-all_reduce_sum_f16x8_pack}

ncu --nvtx \
  --nvtx-include "profiling/" \
  -k regex:"$name"_kernel \
  --set full \
  --import-source yes \
  -f \
  -o "$name" \
  -- python3 my_all_reduce.py --profiling "$name"
