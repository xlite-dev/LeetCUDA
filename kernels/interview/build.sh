#!/usr/bin/env bash
# build.sh — Compile notes-v2.cu for one or more SM architectures.
#
# Uses ccache (when available) in a two-step compile+link workflow:
#   1. ccache nvcc ... -c notes-v2.cu -o notes-v2.o   (cached)
#   2. nvcc notes-v2.o -o notes_v2_<arch>.bin ...       (uncached link)
#
# Usage:
#   ./build.sh --arch sm_89       # Ada (RTX 40 series)
#   ./build.sh --arch sm_90a      # Hopper (H100/H200)
#   ./build.sh --arch sm_120a     # Blackwell (RTX 5090 / PRO 5000/6000)
#   ./build.sh --arch all         # All three architectures
#   ./build.sh --clean            # Remove build artifacts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── ccache detection ──────────────────────────────────────────────
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

# ── common flags (shared across all architectures) ────────────────
COMMON_FLAGS=(
  -std=c++20
  -O3
  --expt-relaxed-constexpr
  --use_fast_math
  -I ../../third-party/cutlass/include
  -I ../../third-party/cudnn-frontend/include
)

# ── architecture configurations ───────────────────────────────────
# Each arch is described by: gencode, extra defines, stubs lib path, extra libs, output name.
declare -A ARCH_GENCODE
declare -A ARCH_DEFINES
declare -A ARCH_LIB_PATH
declare -A ARCH_LIBS
declare -A ARCH_OUTPUT

# sm_89 — Ada (Ampere RTX 40 series)
ARCH_GENCODE[sm_89]="-gencode arch=compute_89,code=sm_89"
ARCH_DEFINES[sm_89]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_89]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_89]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_89]="notes_v2_cute_sm89.bin"

# sm_90a — Hopper (H100/H200)
ARCH_GENCODE[sm_90a]="-gencode arch=compute_90a,code=sm_90a"
ARCH_DEFINES[sm_90a]="-DNOTES_V2_ENABLE_WGMMA -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_90a]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_90a]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_90a]="notes_v2_sm90a.bin"

# sm_120a — Blackwell (RTX 5090 / PRO 5000/6000)
ARCH_GENCODE[sm_120a]="-gencode arch=compute_120a,code=sm_120a"
ARCH_DEFINES[sm_120a]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN"
ARCH_LIB_PATH[sm_120a]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"
ARCH_LIBS[sm_120a]="-lcublas -lcudnn -lnvrtc -lcuda"
ARCH_OUTPUT[sm_120a]="notes_v2_sm120a.bin"

VALID_ARCHS="sm_89 sm_90a sm_120a"

# ── CLI ───────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $0 --arch <name>   [--clean] [-h]

Architectures:
  sm_89     Ada Lovelace (RTX 40 series)
  sm_90a    Hopper (H100/H200)
  sm_120a   Blackwell (RTX 5090 / PRO 5000/6000)
  all       Build all three architectures
  sm_XX     Generic SM arch (e.g., sm_86 for Ampere RTX 30 series)

Options:
  --clean   Remove .o and .bin files, then exit
  -h, --help  Show this help

Generic arch notes:
  Generic arches use no NOTES_V2_XXX flags (no CuTe/WGMMA/TMA/CUDNN).
  Output: notes_v2_smXX.bin (e.g., notes_v2_sm86.bin), linked with -lcublas -lcuda only.
EOF
  exit 0
}

ARCH=""
CLEAN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"; shift 2 ;;
    --clean)
      CLEAN_ONLY=1; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2; usage ;;
  esac
done

# ── clean ─────────────────────────────────────────────────────────
if [[ "$CLEAN_ONLY" == "1" ]]; then
  echo "[clean] Removing build artifacts..."
  rm -f notes-v2.o
  rm -f notes_v2_*.bin
  echo "[clean] Done."
  exit 0
fi

if [[ -z "$ARCH" ]]; then
  echo "[ERROR] --arch is required. Use -h for help." >&2
  exit 1
fi

# ── build one architecture ────────────────────────────────────────
build_one() {
  local arch="$1"
  local gencode="${ARCH_GENCODE[$arch]}"
  local defines="${ARCH_DEFINES[$arch]}"
  local lib_path="${ARCH_LIB_PATH[$arch]}"
  local libs="${ARCH_LIBS[$arch]}"
  local output="${ARCH_OUTPUT[$arch]}"

  echo "=== Building $arch -> $output ==="
  local t0
  t0=$(date +%s)

  # Step 1: compile (ccache-cached when available)
  local compile_cmd
  if [[ "$USE_CCACHE" == "1" ]]; then
    compile_cmd=(ccache "$NVCC")
  else
    compile_cmd=("$NVCC")
  fi
  compile_cmd+=(
    "${COMMON_FLAGS[@]}"
    $defines
    $gencode
    -c notes-v2.cu -o notes-v2.o
  )
  echo "  [compile] ${compile_cmd[*]}"
  "${compile_cmd[@]}"

  # Step 2: link
  local link_cmd=("$NVCC" notes-v2.o -o "$output" $lib_path $libs)
  echo "  [link]    ${link_cmd[*]}"
  "${link_cmd[@]}"

  local t1
  t1=$(date +%s)
  echo "  [OK] $output  (${t1}-${t0}s, elapsed $((t1 - t0))s)"
  echo ""
}

# ── main ──────────────────────────────────────────────────────────
if [[ "$ARCH" == "all" ]]; then
  for a in $VALID_ARCHS; do
    build_one "$a"
  done
else
  if [[ -n "${ARCH_GENCODE[$ARCH]:-}" ]]; then
    # Predefined arch (sm_89/sm_90a/sm_120a)
    build_one "$ARCH"
  else
    # Generic arch (e.g., sm_86, sm_80)
    echo "[build.sh] Generic architecture: $ARCH (no NOTES_V2_XXX flags)"
    arch_num="${ARCH#sm_}"
    local_gencode="-gencode arch=compute_${arch_num},code=sm_${arch_num}"
    local_output="notes_v2_sm${arch_num}.bin"
    
    echo "=== Building $ARCH -> $local_output ==="
    t0=$(date +%s)

    # Step 1: compile
    if [[ "$USE_CCACHE" == "1" ]]; then
      compile_cmd=(ccache "$NVCC")
    else
      compile_cmd=("$NVCC")
    fi
    compile_cmd+=(
      "${COMMON_FLAGS[@]}"
      $local_gencode
      -c notes-v2.cu -o notes-v2.o
    )
    echo "  [compile] ${compile_cmd[*]}"
    "${compile_cmd[@]}"

    # Step 2: link
    link_cmd=("$NVCC" notes-v2.o -o "$local_output" -L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcublas -lcuda)
    echo "  [link]    ${link_cmd[*]}"
    "${link_cmd[@]}"

    t1=$(date +%s)
    echo "  [OK] $local_output  (${t1}-${t0}s, elapsed $((t1 - t0))s)"
  fi
fi

echo "=== All builds complete ==="
