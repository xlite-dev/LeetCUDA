file_name=bank_conflict_test

nvcc "$file_name.cu" -arch=sm_90 -o "$file_name"

ncu --nvtx \
    --nvtx-include "profiling/" \
    --set full \
    -f \
    -o "$file_name" \
    -- "$file_name"