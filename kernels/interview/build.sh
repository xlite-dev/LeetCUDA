#!/usr/bin/env bash
# build.sh — Compile leetcuda bench (multi-TU) for one or more SM architectures.
#
# 全部 .cu 位于 bench/ 目录（kernel 与 host 封装仍在根目录 *.cuh，经 -I 解析），
# 按模块拆分为多个翻译单元并行编译（--jobs）：
#   bench/{base,sgemv,sgemm,hgemm,fp8_gemm,fp4_gemm,flash_attn,ffpa_attn,utils}.cu
#   + bench/bench_leetcuda.cu（仅 main 与原型声明）。
# 每个 TU 一个 .o（ccache 缓存粒度 = 单 TU），落 bin/<arch>/，最后链接成单个 bin。
#
# 为什么需要 weaken 步骤：base.cuh 的非模板 __global__ kernel 会经 include 链
# 进入多个 TU（nvcc 为其生成强符号 host stub，多 TU 重复定义 = 链接错误）。
# 约定 base.o 为这些符号的规范定义；链接前对 sgemv/sgemm/hgemm/flash_attn/
# ffpa_attn 五个 .o 执行 objcopy --weaken-symbols（列表由 nm base.o 动态提取，
# objcopy 写临时文件再 mv，避免污染 ccache 硬链接产物）。
#
# Usage:
#   ./build.sh --arch sm_89 --jobs 8     # Ada (RTX 40 series)
#   ./build.sh --arch sm_90a --jobs 8    # Hopper (H100/H200)
#   ./build.sh --arch sm_120a --jobs 8   # Blackwell (RTX 5090 / PRO 5000/6000)
#   ./build.sh --arch sm_120f --jobs 8   # Blackwell family target (CUDA >= 13.2)
#   ./build.sh --arch all --jobs 8       # All five architectures (逐 arch，arch 内并行)
#   ./build.sh --arch sm_XX --jobs 8     # Generic SM arch (无 NOTES_V2_XXX 宏)
#   ./build.sh --clean                   # Remove build artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# All build artifacts (.o and .bin) go to ./bin
OUT_DIR="$SCRIPT_DIR/bin"
mkdir -p "$OUT_DIR"

# ccache detection
USE_CCACHE=0
if command -v ccache &>/dev/null; then
  USE_CCACHE=1
  # ccache settings for reliable nvcc caching (ref: ffpa-attn/tools/build_fast.sh)
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_mtime,time_macros,locale,pch_defines}"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
  echo "[build.sh] ccache detected — compilercheck=content, sloppiness=$CCACHE_SLOPPINESS"
fi

NVCC="/usr/local/cuda/bin/nvcc"
if [[ ! -x "$NVCC" ]]; then
  echo "[ERROR] nvcc not found at $NVCC" >&2
  exit 1
fi

# common flags (shared across all architectures)
COMMON_FLAGS=(
  -std=c++20
  -O3
  --expt-relaxed-constexpr
  --use_fast_math
  -I .
  -I ../../third-party/cutlass/include
  -I ../../third-party/cudnn-frontend/include
)

# architecture configurations
declare -A ARCH_GENCODE
declare -A ARCH_DEFINES
declare -A ARCH_LIB_PATH
declare -A ARCH_LIBS
declare -A ARCH_OUTPUT

# sm_86 — Ampere (RTX 30 series, 3080)
ARCH_GENCODE[sm_86]="-gencode arch=compute_86,code=sm_86"
ARCH_DEFINES[sm_86]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_86]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_86]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_86]="leetcuda_bench_sm86.bin"

# sm_89 — Ada (Ampere RTX 40 series)
ARCH_GENCODE[sm_89]="-gencode arch=compute_89,code=sm_89"
ARCH_DEFINES[sm_89]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_89]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_89]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_89]="leetcuda_bench_sm89.bin"

# sm_90a — Hopper (H100/H200)
ARCH_GENCODE[sm_90a]="-gencode arch=compute_90a,code=sm_90a"
ARCH_DEFINES[sm_90a]="-DNOTES_V2_ENABLE_WGMMA -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_90a]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_90a]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_90a]="leetcuda_bench_sm90a.bin"

# sm_120a — Blackwell (RTX 5090 / PRO 5000/6000)
ARCH_GENCODE[sm_120a]="-gencode arch=compute_120a,code=sm_120a"
ARCH_DEFINES[sm_120a]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN -DNOTES_V2_ENABLE_SM120_FP4"
ARCH_LIB_PATH[sm_120a]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_120a]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_120a]="leetcuda_bench_sm120a.bin"

# sm_120f — Blackwell family-specific target (RTX 5090 / PRO 5000/6000, CUDA >= 13.2).
# setmaxnreg experiment target: defines NOTES_V2_ENABLE_SETMAXNREGS so
# NOTES_V2_REG_{DE}ALLOC actually emit PTX, plus NOTES_V2_FORCE_INLINE_ASYNC_PROXY
# so TMA helpers use raw `asm volatile` with a shared::cta destination (the
# cuda::ptx wrappers issue shared::cluster, which ptxas treats as an extern-call
# boundary and drops setmaxnreg with C7506). With those two macros the rebalancing
# survives on sm_120a AND sm_120f alike (112 USETMAXREG == PTX count on both);
# sm_120f additionally keeps the whole sm_120 family binary-compatible.
ARCH_GENCODE[sm_120f]="-gencode arch=compute_120f,code=sm_120f"
ARCH_DEFINES[sm_120f]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN -DNOTES_V2_ENABLE_SETMAXNREGS -DNOTES_V2_FORCE_INLINE_ASYNC_PROXY -DNOTES_V2_ENABLE_SM120_FP4"
ARCH_LIB_PATH[sm_120f]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_120f]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_120f]="leetcuda_bench_sm120f.bin"

VALID_ARCHS="sm_86 sm_89 sm_90a sm_120a sm_120f"

# translation units（链接用显式列表，防 stale .o 混入；weaken 后 base.o 强符号恒胜）
TUS=(base sgemv sgemm hgemm fp8_gemm fp4_gemm flash_attn ffpa_attn utils bench_leetcuda)
# 经 .cuh include 链传递包含 base.cuh（非模板 kernel = 强符号）的 TU；
# base.o 之外的这些 TU 需要 weaken 去重。
WEAKEN_TUS=(sgemv sgemm hgemm flash_attn ffpa_attn)

# CLI
usage() {
  cat <<EOF
Usage: $0 --arch <name> [--jobs N] [--clean] [-h]

Architectures:
  sm_86     Ampere (RTX 30 series)
  sm_89     Ada Lovelace (RTX 40 series)
  sm_90a    Hopper (H100/H200)
  sm_120a   Blackwell (RTX 5090 / PRO 5000/6000)
  sm_120f   Blackwell family target (keeps setmaxnreg; CUDA >= 13.2)
  all       Build all five architectures
  sm_XX     Generic SM arch (e.g., sm_80 for A100)

Options:
  --jobs N  并行编译的 TU 数（默认: min(nproc --all, 32)）
  --clean   Remove .o and .bin files, then exit
  -h, --help  Show this help

Generic arch notes:
  Generic arches use no NOTES_V2_XXX flags (no CuTe/WGMMA/TMA/CUDNN).
  Output: bin/leetcuda_bench_smXX.bin (e.g., bin/leetcuda_bench_sm80.bin), linked with -lcublas -lcuda only.
EOF
  exit 0
}

resolve_default_jobs() {
  local n
  n="$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  n="${n//[!0-9]/}"
  [[ -z "$n" || "$n" -lt 1 ]] && n=8
  (( n > 32 )) && n=32
  echo "$n"
}

ARCH=""
JOBS="$(resolve_default_jobs)"
CLEAN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      [[ $# -lt 2 ]] && { echo "[ERROR] --arch requires a value" >&2; exit 1; }
      ARCH="$2"; shift 2 ;;
    --jobs)
      [[ $# -lt 2 ]] && { echo "[ERROR] --jobs requires a value" >&2; exit 1; }
      JOBS="$2"; shift 2 ;;
    --clean)
      CLEAN_ONLY=1; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "[ERROR] Unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --jobs must be a positive integer (got: $JOBS)" >&2
  exit 1
fi

# clean
if [[ "$CLEAN_ONLY" == "1" ]]; then
  echo "[clean] Removing build artifacts in bin/ ..."
  rm -rf "$OUT_DIR"
  echo "[clean] Done."
  exit 0
fi

if [[ -z "$ARCH" ]]; then
  echo "[ERROR] --arch is required. Use -h for help." >&2
  exit 1
fi

# compile one TU (background-safe)
compile_tu() {
  local tu="$1" objdir="$2" gencode="$3" defines="$4"
  local cmd
  if [[ "$USE_CCACHE" == "1" ]]; then
    cmd=(ccache "$NVCC")
  else
    cmd=("$NVCC")
  fi
  cmd+=("${COMMON_FLAGS[@]}" $defines $gencode -c "bench/${tu}.cu" -o "${objdir}/${tu}.o")
  echo "  [compile:${tu}] ${cmd[*]}"
  "${cmd[@]}"
}

# build one architecture (compile in parallel, weaken, link)
# build_one <tag> <gencode> <defines> <lib_path> <libs> <output>
build_one() {
  local tag="$1" gencode="$2" defines="$3" lib_path="$4" libs="$5" output="$6"
  local objdir="$OUT_DIR/$tag"
  rm -rf "$objdir"
  mkdir -p "$objdir"

  echo "=== Building $tag -> $output ($JOBS jobs) ==="
  local t0
  t0=$(date +%s)

  # Step 1: 并行编译全部 TU（wait -n 限流，失败即终止并杀掉其余 job 及其 nvcc 子进程）
  local pids=() tu p rc failed=0
  kill_jobs() {
    local p
    for p in "${pids[@]}"; do
      pkill -P "$p" 2>/dev/null || true
      kill "$p" 2>/dev/null || true
    done
  }
  for tu in "${TUS[@]}"; do
    compile_tu "$tu" "$objdir" "$gencode" "$defines" &
    pids+=($!)
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      wait -n || {
        rc=$?
        echo "[ERROR] a compile job failed (rc=$rc); killing siblings" >&2
        kill_jobs
        exit "$rc"
      }
    done
  done
  for p in "${pids[@]}"; do
    wait "$p" || { echo "[ERROR] compile job (pid $p) failed" >&2; failed=1; }
  done
  if (( failed != 0 )); then
    kill_jobs
    exit 1
  fi

  # Step 2: weaken base.cuh 派生的重复强符号（base.o 为规范定义）
  local weaken_list="$objdir/weaken_syms.txt"
  nm "$objdir/base.o" | awk '$2 ~ /^[TDB]$/ { print $3 }' | sort -u > "$weaken_list"
  local nweak
  nweak=$(wc -l < "$weaken_list")
  for tu in "${WEAKEN_TUS[@]}"; do
    objcopy --weaken-symbols="$weaken_list" "$objdir/$tu.o" "$objdir/$tu.o.w" \
      && mv "$objdir/$tu.o.w" "$objdir/$tu.o"
    echo "  [weaken:${tu}] weakened $nweak base.o strong symbols"
  done

  # Step 3: link（显式 TU 列表，防 stale .o 混入；weaken 后 base.o 的强符号恒胜弱符号）
  local link_cmd=("$NVCC")
  for tu in "${TUS[@]}"; do
    link_cmd+=("$objdir/$tu.o")
  done
  link_cmd+=(-o "$OUT_DIR/$output" $lib_path $libs)
  echo "  [link]     ${link_cmd[*]}"
  "${link_cmd[@]}"

  local t1
  t1=$(date +%s)
  echo "  [OK] bin/$output  (elapsed $((t1 - t0))s)"
  echo ""
}

# main
if [[ "$ARCH" == "all" ]]; then
  for a in $VALID_ARCHS; do
    build_one "$a" "${ARCH_GENCODE[$a]}" "${ARCH_DEFINES[$a]}" \
      "${ARCH_LIB_PATH[$a]}" "${ARCH_LIBS[$a]}" "${ARCH_OUTPUT[$a]}"
  done
elif [[ -n "${ARCH_GENCODE[$ARCH]:-}" ]]; then
  # Predefined arch (sm_86/sm_89/sm_90a/sm_120a/sm_120f)
  build_one "$ARCH" "${ARCH_GENCODE[$ARCH]}" "${ARCH_DEFINES[$ARCH]}" \
    "${ARCH_LIB_PATH[$ARCH]}" "${ARCH_LIBS[$ARCH]}" "${ARCH_OUTPUT[$ARCH]}"
elif [[ "$ARCH" == sm_* ]]; then
  # Generic arch (e.g., sm_80): no NOTES_V2_XXX flags, cublas/cuda only
  echo "[build.sh] Generic architecture: $ARCH (no NOTES_V2_XXX flags)"
  local_arch_num="${ARCH#sm_}"
  build_one "$ARCH" \
    "-gencode arch=compute_${local_arch_num},code=sm_${local_arch_num}" \
    "" \
    "-L/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
    "-lcublas -lcuda" \
    "leetcuda_bench_sm${local_arch_num}.bin"
else
  echo "[ERROR] Unknown architecture: $ARCH. Valid: $VALID_ARCHS, all, sm_XX" >&2
  exit 1
fi

echo "=== All builds complete ==="
