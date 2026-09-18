#!/usr/bin/env bash
# book/build.sh — 两遍 xelatex 构建 + 验收检查 + 中间文件清理（RFC-0.5）
# 用法：cd kernels/interview/book && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
export TEXMFCNF=../tex:  # 尾部冒号=追加系统默认配置；纯 ../tex/ 会找不到 xelatex.fmt

echo "== xelatex pass 1/2"
xelatex -interaction=nonstopmode book.tex >/dev/null 2>&1 || { echo "BUILD FAILED, last 40 lines of book.log:"; tail -40 book.log; exit 1; }
echo "== xelatex pass 2/2"
xelatex -interaction=nonstopmode book.tex >/dev/null 2>&1 || { echo "BUILD FAILED, last 40 lines of book.log:"; tail -40 book.log; exit 1; }

if grep -q '^! ' book.log; then
  echo "LaTeX Error found in book.log:"; grep -n '^! ' book.log; exit 1
fi
if grep -qE '(There were undefined references|undefined citation|Reference .`.*. on page .* undefined)' book.log; then
  echo "Undefined references/citations found in book.log:"; grep -nE 'undefined (reference|citation)|Reference .*undefined' book.log | head -20; exit 1
fi
OVER=$(grep -oE 'Overfull \\hbox \([0-9]+\.[0-9]+pt' book.log | grep -oE '[0-9]+\.[0-9]+' | awk '$1>=1.0' | wc -l || true)
echo "Overfull hbox >=1pt: ${OVER}"
MISSING=$(grep -c 'Missing character' book.log || true)
if [ "${MISSING}" -ne 0 ]; then
  echo "Missing characters (tofu) found in book.log:"; grep 'Missing character' book.log | sort -u | head -10; exit 1
fi
echo "Missing characters: 0"

PAGES=$(pdfinfo book.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
echo "OK: book.pdf (${PAGES:-?} pages)"

# 中间文件清理（.tex/.pdf 由 .gitignore 规则管理）
rm -f book.aux book.log book.out book.toc
echo "cleaned intermediates"
