#!/usr/bin/env bash
# book/build.sh — 构建 + 验收 + 中间文件清理
# 三本产物：book.pdf（全一册）、book-vol1.pdf（上册）、book-vol2.pdf（下册）
# 用法：
#   ./build.sh                 # 全量：三份封面 + 上下册 + 全一册（约 10 分钟）
#   ./build.sh --only vols     # 只编上下两册（book | v1 | v2 | vols）
#   ./build.sh --only book     # 只编全一册
#   ./build.sh --cover-only    # 只重编三份封面
set -euo pipefail
cd "$(dirname "$0")"
export TEXMFCNF=../tex:  # 尾部冒号=追加系统默认配置；纯 ../tex/ 会找不到 xelatex.fmt

#: 每个 PDF 的 xelatex 遍数。两遍不够：首遍 aux 不存在时本册 \ref 全出 ??，分页比稳定态短，
#: 第二遍写出的 .toc 页码因此过期（实测全一册 24 处、上册 8 处目录页码差 1）。第三遍拿稳定
#: aux 收目录，页码才与正文一致。
PASSES=3

ONLY=all
COVER_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only)
      [ $# -ge 2 ] || { echo "--only 需要参数（book | v1 | v2 | vols）"; exit 2; }
      ONLY="$2"; shift 2 ;;
    --cover-only) COVER_ONLY=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "未知参数：$1（见 ./build.sh --help）"; exit 2 ;;
  esac
done

# 封面含编译当月的动态日期：每次构建都重编，保证 PDF 首页日期与当月一致
# （write-leetcuda-book 硬规则 11；figures/misc/cover.pdf 入库，RTD 站点也读它）
build_cover() {  # $1=jobname $2=期望卷标（主封面 0）
  (cd figures/misc && xelatex -interaction=nonstopmode "$1.tex" >/dev/null 2>&1) || {
    echo "COVER BUILD FAILED: $1，last 40 lines of figures/misc/$1.log:"
    tail -40 "figures/misc/$1.log"; return 1; }
  # 卷标参数没生效就会静默出成主书封面，必须校验
  grep -q "^LEETCOVER volume=$2\$" "figures/misc/$1.log" || {
    echo "COVER TAG MISSING: $1 期望 LEETCOVER volume=$2"; return 1; }
  # 卷标是中文文本，字体缺字会静默出方框
  missing=$(grep -c 'Missing character' "figures/misc/$1.log" || true)
  [ "${missing}" -eq 0 ] || {
    echo "COVER MISSING CHAR: $1 有 ${missing} 处缺字"; grep 'Missing character' "figures/misc/$1.log" | sort -u | head -5; return 1; }
  # standalone 的页面尺寸由图形包围盒决定：卷标溢出会把画布撑大，进而被封面 hook 压扁
  pdfinfo "figures/misc/$1.pdf" | awk '/^Page size:/{exit !(($3-595.28<0.5 && 595.28-$3<0.5) && ($5-841.89<0.5 && 841.89-$5<0.5))}' || {
    echo "COVER SIZE WRONG: $1 = $(pdfinfo "figures/misc/$1.pdf" | awk '/^Page size:/{print $3" x "$5}')"; return 1; }
}

echo "== covers (成书日期 $(date '+%Y 年 %-m 月'))"
build_cover cover-tikz 0 && cp figures/misc/cover-tikz.pdf figures/misc/cover.pdf
build_cover cover-vol1 1
build_cover cover-vol2 2
if [ "$COVER_ONLY" -eq 1 ]; then
  rm -f figures/misc/cover-tikz.aux figures/misc/cover-tikz.log \
        figures/misc/cover-vol1.aux figures/misc/cover-vol1.log \
        figures/misc/cover-vol2.aux figures/misc/cover-vol2.log
  echo "OK: cover.pdf cover-vol1.pdf cover-vol2.pdf"
  exit 0
fi

# 单遍 xelatex；失败时把该册 log 尾部打出来
build_pass() {  # $1=jobname $2=pass 序号 $3=总遍数
  echo "== $1 xelatex pass $2/$3"
  xelatex -interaction=nonstopmode "$1.tex" >/dev/null 2>&1 || {
    echo "BUILD FAILED: $1，last 40 lines of $1.log:"; tail -40 "$1.log"; return 1; }
}

build_passes() {  # $1=jobname
  local index
  for index in $(seq 1 "$PASSES"); do
    build_pass "$1" "$index" "$PASSES"
  done
}

# 日志门禁：错误 / 未定义引用 / 缺字；overfull 只报数（排版既有状态，不作门禁）
check_log() {  # $1=jobname
  if grep -q '^! ' "$1.log"; then
    echo "LaTeX Error found in $1.log:"; grep -n '^! ' "$1.log"; return 1
  fi
  if grep -qE '(There were undefined references|undefined citation|Reference .`.*. on page .* undefined)' "$1.log"; then
    echo "Undefined references/citations found in $1.log:"
    grep -nE 'undefined (reference|citation)|Reference .*undefined' "$1.log" | head -20; return 1
  fi
  over=$(grep -oE 'Overfull \\hbox \([0-9]+\.[0-9]+pt' "$1.log" | grep -oE '[0-9]+\.[0-9]+' | awk '$1>=1.0' | wc -l || true)
  missing=$(grep -c 'Missing character' "$1.log" || true)
  echo "  overfull hbox >=1pt: ${over}   missing characters: ${missing}"
  if [ "${missing}" -ne 0 ]; then
    echo "Missing characters (tofu) found in $1.log:"; grep 'Missing character' "$1.log" | sort -u | head -10
    return 1
  fi
}

# 单册 aux 缺失时提示先编对侧：跨册引用靠 xr 读对侧 .aux，缺了会静默出 ??
require_aux() {  # $1=对侧 jobname $2=对侧册名
  if [ ! -f "$1.aux" ]; then
    echo "缺 $1.aux：跨册引用需要在 $1 编译之后才能解析。"
    echo "先跑 ./build.sh（全量）或 ./build.sh --only ${2}。"
    exit 1
  fi
  newest=$(find chapters appendices figures book-vol1.tex book-vol2.tex preamble.tex \
             -newer "$1.aux" \( -name '*.tex' -o -name '*.pdf' \) -print -quit 2>/dev/null || true)
  if [ -n "$newest" ]; then
    echo "WARN: $1.aux 比 $newest 旧，跨册引用与章号可能过期（全量构建可消除）"
  fi
}

build_target() {  # $1=jobname
  check_log "$1" || exit 1
  pages=$(pdfinfo "$1.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  echo "OK: $1.pdf (${pages:-?} pages)"
}

# 上下册：下册先出一遍 aux，上册整轮到稳定（每遍都读得到下册 aux），下册再收两遍。
# 顺序不能颠倒：上册若拿不到下册 aux，ch26b 的跨册引用会先出 ?? 再变章号，行宽变化
# 会让该章之后的分页挪动，而目录里留着的仍是上一遍的页码。
build_volumes() {
  echo "== check_volume_split（章表 / setcounter 接续）"
  python3 scripts/check_volume_split.py || { echo "SPLIT CHECK FAILED（修完再构建）"; exit 1; }
  build_pass book-vol2 1 "$PASSES"   # 先落下下册 aux
  build_passes book-vol1             # 上册整轮：aux → 稳定分页 → 目录
  build_pass book-vol2 2 "$PASSES"   # 下册：本册引用 + 分页
  build_pass book-vol2 3 "$PASSES"   # 下册：目录
  build_target book-vol1
  build_target book-vol2
}

case "$ONLY" in
  all)
    build_volumes
    build_passes book
    build_target book
    ;;
  vols) build_volumes ;;
  v1)
    require_aux book-vol2 v2
    python3 scripts/check_volume_split.py || { echo "SPLIT CHECK FAILED（修完再构建）"; exit 1; }
    build_passes book-vol1
    build_target book-vol1
    ;;
  v2)
    require_aux book-vol1 v1
    python3 scripts/check_volume_split.py || { echo "SPLIT CHECK FAILED（修完再构建）"; exit 1; }
    build_passes book-vol2
    build_target book-vol2
    ;;
  book)
    build_passes book
    build_target book
    # 只编全一册时上下册不一定重编，若本次动过章/part 集合会在这里暴露（不中止）
    python3 scripts/check_volume_split.py || {
      echo "WARN: 分册清单未同步（见上）；提交前须同步 book-vol1/2.tex 与 \\setcounter 并跑全量 ./build.sh"
    }
    ;;
  *) echo "--only 只接受 book | v1 | v2 | vols"; exit 2 ;;
esac

echo "== 跨文件 lint（kaogu 引文 / 跨章串表 / 图-码索引）"
python3 scripts/lint_book.py || { echo "LINT FAILED（见上，修完再提交）"; exit 1; }

# 分册一致性终检：部分号序列 + 882 条 label 编号。必须在清理 aux 之前跑。
if [ -f book.aux ] && [ -f book-vol1.aux ] && [ -f book-vol2.aux ]; then
  echo "== check_volume_split --compare-aux（部分号 / 章图公式编号）"
  python3 scripts/check_volume_split.py --compare-aux || {
    echo "AUX CHECK FAILED（若刚用 --only 编单册，book.aux 可能过期，跑一次 ./build.sh 全量）"
    exit 1; }
else
  echo "== 跳过 aux 终检（三本未齐编；全量 ./build.sh 会跑）"
fi

# 中间文件清理：保留 .aux —— 单册迭代（--only v1/v2）要靠对侧 aux 解析跨册引用；
# aux/log/toc/out 均在 kernels/.gitignore 的 book 段内，不留 git 噪音。
rm -f ./*.log ./*.out ./*.toc \
      figures/misc/cover-tikz.aux figures/misc/cover-tikz.log \
      figures/misc/cover-vol1.aux figures/misc/cover-vol1.log \
      figures/misc/cover-vol2.aux figures/misc/cover-vol2.log
echo "cleaned intermediates (aux 保留供单册迭代)"
