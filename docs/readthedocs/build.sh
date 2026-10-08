#!/usr/bin/env bash
# 本地一键预览：检查依赖 → 书稿 tex 转 MyST + TikZ → sphinx-build → 打印预览路径。
#
# 用法：
#   conda activate cdit
#   ./build.sh                 # 全量转换并构建
#   ./build.sh --only ch00-profiling   # 只转一章（调试）
#   ./build.sh --skip-tikz     # 复用已有 SVG，只重跑文本转换
set -euo pipefail

cd "$(dirname "$0")"
BUILD_DIR=build
SRC_DIR="$BUILD_DIR/src"
HTML_DIR="$BUILD_DIR/html"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "缺少依赖：$1${2:+（$2）}" >&2
    exit 1
  }
}

require python "conda activate cdit"
require xelatex "texlive-xetex"
require dvisvgm "dvisvgm 包"
PANDOC_BIN="$(command -v pandoc || true)"
if [ -z "$PANDOC_BIN" ]; then
  python - <<'PY' || { echo "缺少 pandoc：pip install pypandoc-binary 或 apt install pandoc" >&2; exit 1; }
import pypandoc, sys
sys.exit(0 if pypandoc.get_pandoc_path() else 1)
PY
fi

echo "==> 转换书稿"
python -m convert --out "$SRC_DIR" --work "$BUILD_DIR/tmp" "$@"

echo "==> 构建 HTML"
sphinx-build -T -b html -d "$BUILD_DIR/doctrees" -c . "$SRC_DIR" "$HTML_DIR"

echo "==> 内容核对"
python -m convert.verify --src "$SRC_DIR" --html "$HTML_DIR"

echo
echo "预览：file://$(pwd)/$HTML_DIR/index.html"
