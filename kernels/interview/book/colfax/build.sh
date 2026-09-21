#!/usr/bin/env bash
# book/colfax/build.sh — xelatex×3 + biber 构建 + 验收检查（仿 ../build.sh）
# 用法：cd kernels/interview/book/colfax && ./build.sh [--draft]
#   --draft：跳过 undefined references 断言（分批集成期间前后片引用未齐时用）
set -euo pipefail
cd "$(dirname "$0")"
export TEXMFCNF=../../tex:  # 尾部冒号=追加系统默认配置

STRICT=1
if [ "${1:-}" = "--draft" ]; then STRICT=0; fi

MAIN=colfax-cute-zh

echo "== xelatex pass 1/3"
xelatex -interaction=nonstopmode $MAIN.tex >/dev/null 2>&1 || { echo "BUILD FAILED, last 40 lines of $MAIN.log:"; tail -40 $MAIN.log; exit 1; }
echo "== biber"
biber $MAIN >/dev/null 2>&1 || { echo "BIBER FAILED, last 30 lines of $MAIN.blg:"; tail -30 $MAIN.blg; exit 1; }
echo "== xelatex pass 2/3"
xelatex -interaction=nonstopmode $MAIN.tex >/dev/null 2>&1 || { echo "BUILD FAILED, last 40 lines of $MAIN.log:"; tail -40 $MAIN.log; exit 1; }
echo "== xelatex pass 3/3"
xelatex -interaction=nonstopmode $MAIN.tex >/dev/null 2>&1 || { echo "BUILD FAILED, last 40 lines of $MAIN.log:"; tail -40 $MAIN.log; exit 1; }

if grep -q '^! ' $MAIN.log; then
  echo "LaTeX Error found in $MAIN.log:"; grep -n '^! ' $MAIN.log; exit 1
fi
if [ "$STRICT" -eq 1 ]; then
  if grep -qE '(There were undefined references|undefined citation|Reference .`.*. on page .* undefined)' $MAIN.log; then
    echo "Undefined references/citations found in $MAIN.log:"; grep -cE 'Reference .*undefined' $MAIN.log || true; exit 1
  fi
else
  UNDEF=$(grep -cE 'Reference .`.*. on page .* undefined' $MAIN.log || true)
  echo "[draft] undefined refs: ${UNDEF}（分批集成期允许）"
fi
OVER=$(grep -oE 'Overfull \\hbox \([0-9]+\.[0-9]+pt' $MAIN.log | grep -oE '[0-9]+\.[0-9]+' | awk '$1>=1.0' | wc -l || true)
echo "Overfull hbox >=1pt: ${OVER}"
MISSING=$(grep -c 'Missing character' $MAIN.log || true)
if [ "${MISSING}" -ne 0 ]; then
  echo "Missing characters (tofu) found in $MAIN.log:"; grep 'Missing character' $MAIN.log | sort -u | head -10; exit 1
fi
echo "Missing characters: 0"

PAGES=$(pdfinfo $MAIN.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
echo "OK: $MAIN.pdf (${PAGES:-?} pages)"

rm -f $MAIN.aux $MAIN.log $MAIN.out $MAIN.toc $MAIN.bbl $MAIN.bcf $MAIN.run.xml $MAIN.blg
echo "cleaned intermediates"
