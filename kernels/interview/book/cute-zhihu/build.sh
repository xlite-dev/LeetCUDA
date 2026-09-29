#!/usr/bin/env bash
# build.sh — CUDA 知乎优秀文章合集构建脚本
# 用法：cd book/cute-zhihu/ && bash build.sh
set -e
cd "$(dirname "$0")"
rm -f cute-zhihu.aux cute-zhihu.toc
# 封面 standalone 先编译（cover-tikz.pdf 被上层 gitignore 忽略，不入库）
xelatex -interaction=nonstopmode cover-tikz.tex > .tmp_cover.log 2>&1
TEXMFCNF=../../tex: xelatex -interaction=nonstopmode cute-zhihu.tex > .tmp_build1.log 2>&1
TEXMFCNF=../../tex: xelatex -interaction=nonstopmode cute-zhihu.tex > .tmp_build2.log 2>&1
echo "errors: $(grep -c '^!' .tmp_build2.log || true)"
echo "missingchar: $(grep -c 'Missing character' .tmp_build2.log || true)"
echo "overfull: $(grep -c 'Overfull' .tmp_build2.log || true)"
grep -o 'Output written on cute-zhihu.pdf ([0-9]* pages' .tmp_build2.log || true
