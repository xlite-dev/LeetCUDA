#!/usr/bin/env bash
# book/tests/build_tests.sh — 编译并运行最小测试（RFC-0.6，规范 BOOK_PLAN §6）
# 用法：./build_tests.sh [--arch sm_120a] [--ch ch04] [--all] [--no-run]
#   --arch     目标架构（默认 sm_120a，本机 RTX PRO 5000）
#   --ch       只构建指定章（如 ch04；省略时构建已有测试中 ch01-ch02 基础集）
#   --all      构建全部已存在的 ch*_*.cu
#   --no-run   只编译不运行
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../../../.. && pwd)"        # LeetCUDA 仓库根（tests/ -> book/ -> interview/ -> kernels/ -> LeetCUDA/）
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=sm_120a
RUN=1
SELECT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch) ARCH="$2"; shift 2 ;;
    --ch) SELECT="$2"; shift 2 ;;
    --all) SELECT="all"; shift ;;
    --no-run) RUN=0; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

INCLUDES="-I${ROOT}/third-party/cutlass/include -I../.."

if [[ "$SELECT" == "all" ]]; then
  SOURCES=$(ls ch*_*.cu 2>/dev/null || true)
elif [[ -n "$SELECT" ]]; then
  SOURCES=$(ls "${SELECT}"_*.cu 2>/dev/null || true)
else
  # 默认：已实现测试的章节（随 RFC-C..F 推进逐步加入）
  SOURCES=""
  for ch in ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 \
            ch15 ch16 ch17 ch18 ch19 ch20 ch21 ch22 ch23 ch24 ch25 ch26 ch26b ch26c; do
    f=$(ls ${ch}_*.cu 2>/dev/null || true)
    SOURCES="$SOURCES $f"
  done
fi

[[ -z "${SOURCES// /}" ]] && { echo "no test sources found (SELECT=${SELECT:-default})"; exit 0; }

# ch13 (WGMMA) 仅 sm_90a 可编译：其它 arch 下显式跳过（运行验收=sm_90a 编译+设备 SKIP）
if [[ "$ARCH" != "sm_90a" ]]; then
  N_BEFORE=$(echo $SOURCES | wc -w)
  SOURCES=$(echo $SOURCES | tr ' ' '\n' | grep -v '^ch13_' | tr '\n' ' ')
  [[ $(echo $SOURCES | wc -w) -lt $N_BEFORE ]] && echo "(ch13 skipped: requires -arch sm_90a; see ch13 §13.10 for explicit build)"
fi

mkdir -p .build
FAIL=0
for src in $SOURCES; do
  exe=".build/${src%.cu}"
  echo "== $src ($ARCH)"
  if ! $NVCC -arch "$ARCH" -std=c++20 -O2 $INCLUDES -o "$exe" "$src" 2> ".build/${src%.cu}.buildlog"; then
    echo "BUILD FAIL: $src"; cat ".build/${src%.cu}.buildlog"; FAIL=1; continue
  fi
  if [[ $RUN -eq 1 ]]; then
    if ! "$exe"; then echo "RUN FAIL: $src"; FAIL=1; fi
  fi
done
[[ $FAIL -eq 0 ]] && echo "ALL OK" || echo "FAILURES PRESENT"
exit $FAIL
