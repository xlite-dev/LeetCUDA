"""TikZ 片段 → SVG（standalone 文档，多进程并行）。

每个片段单独起一个 ``standalone`` 文档：xelatex 出的 PDF 页宽高就等于图形外接框，
天然紧致裁剪，dvisvgm 再读这个 PDF 出 SVG。两条被实测否定的近路记在这里，避免以后
再走一遍：

1. **不要直接让 dvisvgm 读 xelatex 的 XDV**：字形推进量解析不可靠，CJK 文本会互相
   重叠（ch00/ch01 的图全乱），而同一份内容转成 PDF 再读就完全正确。
2. **不要用一个多页文档批量编译再按页导出**：PDF 输入的 ``--bbox`` 选项无效，每张
   SVG 都会是整页（737pt）大小，图形缩在左上角。

颜色与 tikz 库从 ``book/preamble.tex`` 程序化抽取，单一样本不漂移。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

#: 从 preamble 抽取的、与绘图相关的行。
_PREAMBLE_KEEP = re.compile(r"^\s*\\(?:definecolor|usetikzlibrary|tikzset|setmonofont|setCJKmonofont)")


#: xelatex 编译超时（秒）。
_COMPILE_TIMEOUT = 600

#: 字号处理方式：字形转矢量路径（默认，任何查看器一致）。
FONT_PATHS = "paths"

#: 字号处理方式：内嵌 woff2 字体子集（体积小、文字可选，但依赖查看器支持 @font-face）。
FONT_WOFF2 = "woff2"

#: 可选的字号模式。
FONT_MODES = (FONT_PATHS, FONT_WOFF2)

#: 缓存版本：编译链路变化时递增，强制全部重编。
_CACHE_VERSION = "standalone-pdf-1"

#: 默认并行进程数（受 CPU 与内存限制，见 README）。
DEFAULT_JOBS = 8


@dataclass
class Snippet:
  """一个待编译的图片段。

  :param image_id: token 载荷标识。
  :param source: 片段 tex（``work/tikz/<id>.tex``）。
  :param dest: 目标 SVG。
  :param digest: 片段内容哈希，用于增量跳过。
  """
  image_id: str
  source: Path
  dest: Path
  digest: str


@dataclass
class Report:
  """编译结果。

  :param ok: 成功的 image_id。
  :param failed: 失败的 image_id → 错误摘要。
  :param skipped: 命中缓存、未重编的 image_id。
  :param mode: 编译模式（当前只有 ``parallel``）。
  :param seconds: 编译耗时。
  :param font_mode: 字号处理方式（``paths`` / ``woff2``）。
  """
  ok: list[str] = field(default_factory=list)
  failed: dict[str, str] = field(default_factory=dict)
  skipped: list[str] = field(default_factory=list)
  mode: str = "parallel"
  seconds: float = 0.0
  font_mode: str = FONT_PATHS

  def to_json(self) -> str:
    """序列化为 JSON 文本。

    :returns: JSON 字符串。
    """
    return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def collect(manifests: list, work_dir: Path, src_dir: Path) -> list[Snippet]:
  """从各章清单收集 tikz 片段。

  :param manifests: ``preprocess.Manifest`` 列表。
  :param work_dir: 临时目录。
  :param src_dir: 站点源树。
  :returns: 片段列表（按章、按序）。
  """
  snippets: list[Snippet] = []
  for manifest in manifests:
    for image in manifest.images.values():
      if not image.origin.endswith(".tex"):
        continue
      source = work_dir / image.origin
      if not source.is_file():
        continue
      digest = hashlib.sha1(source.read_bytes()).hexdigest()
      snippets.append(Snippet(image.image_id, source, src_dir / image.dest, digest))
  return snippets


def load_report(work_dir: Path) -> Report:
  """读回编译报告；不存在时返回空报告。

  :param work_dir: 临时目录。
  :returns: 编译报告。
  """
  path = work_dir / "tikz-report.json"
  if not path.is_file():
    return Report()
  raw = json.loads(path.read_text(encoding="utf-8"))
  return Report(**raw)


def font_args(font_mode: str) -> list[str]:
  """给出 dvisvgm 的字号处理参数。

  ``paths``（默认）把字形转成矢量路径：任何查看器都渲染一致，代价是文件更大
  （约 2.5 倍）且文字不可选；``woff2`` 内嵌字体子集，文件小、文字可选中，但依赖
  查看器支持 ``@font-face``（VS Code 预览等不支持，会显示成重叠乱码）。

  :param font_mode: ``paths`` 或 ``woff2``。
  :returns: dvisvgm 参数列表。
  :raises ValueError: 未知模式。
  """
  if font_mode == FONT_PATHS:
    return ["--no-fonts", "-d", "2"]
  if font_mode == FONT_WOFF2:
    return ["--font-format=woff2"]
  raise ValueError(f"未知的字号模式：{font_mode}")


def build(
  snippets: list[Snippet],
  book_dir: Path,
  work_dir: Path,
  force: bool = False,
  font_mode: str = FONT_PATHS,
  jobs: int = DEFAULT_JOBS,
) -> Report:
  """编译全部片段。

  :param snippets: 片段列表。
  :param book_dir: 书根目录（提供 ``preamble.tex``）。
  :param work_dir: 临时目录。
  :param force: 忽略缓存全部重编。
  :param font_mode: 字号处理方式，见 :func:`font_args`。
  :param jobs: 并行编译进程数。
  :returns: 编译报告。
  """
  started = time.monotonic()
  report = Report(font_mode=font_mode)
  staged = []
  for snippet in snippets:
    if not force and snippet.dest.is_file() and _cached(work_dir, snippet, font_mode):
      report.skipped.append(snippet.image_id)
      continue
    staged.append(snippet)

  if staged:
    head = _preamble_head(book_dir)
    build_dir = work_dir / "tikz-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    report.mode = "parallel"
    tasks = [(snippet, head, build_dir, font_mode) for snippet in staged]
    workers = max(1, min(jobs, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
      for image_id, ok, message in pool.map(_compile_worker, tasks):
        if ok:
          report.ok.append(image_id)
        else:
          report.failed[image_id] = message

  report.seconds = round(time.monotonic() - started, 1)
  _save_cache(work_dir, snippets, font_mode, report.failed)
  (work_dir / "tikz-report.json").write_text(report.to_json(), encoding="utf-8")
  return report


def _cached(work_dir: Path, snippet: Snippet, font_mode: str) -> bool:
  """判断片段是否命中缓存（缓存键含字号模式）。

  :param work_dir: 临时目录。
  :param snippet: 片段。
  :param font_mode: 字号处理方式。
  :returns: 是否命中。
  """
  cache = _read_cache(work_dir)
  return cache.get(snippet.image_id) == f"{snippet.digest}@{font_mode}@{_CACHE_VERSION}"


def _read_cache(work_dir: Path) -> dict[str, str]:
  """读取片段哈希缓存。

  :param work_dir: 临时目录。
  :returns: image_id → 哈希。
  """
  path = work_dir / "tikz-cache.json"
  if not path.is_file():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return {}


def _save_cache(
  work_dir: Path,
  snippets: list[Snippet],
  font_mode: str,
  failed: dict[str, str],
) -> None:
  """写回片段哈希缓存。

  **只记编译成功的片段**：失败片段若被记成「已完成」，下次运行会跳过它，磁盘上那份
  陈旧或残缺的 SVG 就永远留在页面上（ch00 的 occupancy 图丢文字就是这么来的）。

  :param work_dir: 临时目录。
  :param snippets: 全部片段。
  :param font_mode: 字号处理方式（随哈希一起记录，切换模式会触发重编）。
  :param failed: 失败的 image_id → 原因。
  """
  cache = _read_cache(work_dir)
  for snippet in snippets:
    if snippet.image_id in failed or not snippet.dest.is_file():
      cache.pop(snippet.image_id, None)
      continue
    cache[snippet.image_id] = f"{snippet.digest}@{font_mode}@{_CACHE_VERSION}"
  (work_dir / "tikz-cache.json").write_text(
    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _preamble_head(book_dir: Path) -> str:
  """从 ``preamble.tex`` 抽取绘图所需的定义。

  :param book_dir: 书根目录。
  :returns: 可直接拼进模板的定义行。
  """
  preamble = (book_dir / "preamble.tex").read_text(encoding="utf-8")
  lines = [line.rstrip() for line in preamble.splitlines() if _PREAMBLE_KEEP.match(line)]
  return "\n".join(lines)


def _template(head: str) -> str:
  """生成编译模板头部（standalone 文档）。

  :param head: 从 preamble 抽取的定义。
  :returns: 模板头（不含正文）。
  """
  document_class = "\\documentclass[border=6pt]{standalone}"
  fonts = (
    "\\IfFontExistsTF{Noto Sans CJK SC}{%\n"
    "  \\setCJKmainfont{Noto Sans CJK SC}%\n"
    "  \\setCJKsansfont{Noto Sans CJK SC}%\n"
    "  \\setCJKmonofont{Noto Sans Mono CJK SC}%\n"
    "}{%\n"
    "  \\setCJKmainfont{FandolSong-Regular.otf}%\n"
    "  \\setCJKsansfont{FandolHei-Regular.otf}%\n"
    "  \\setCJKmonofont{FandolHei-Regular.otf}%\n"
    "}\n"
    "\\IfFontExistsTF{DejaVu Sans Mono}{\\setmonofont{DejaVu Sans Mono}}{}"
  )
  return (
    f"{document_class}\n"
    "\\usepackage{xcolor}\n"
    "\\usepackage{tikz}\n"
    "\\usepackage{amsmath,amssymb}\n"
    "\\usepackage{fontspec}\n"
    "\\usepackage[UTF8,fontset=none]{ctex}\n"
    f"{head}\n"
    f"{fonts}\n"
    "\\pagestyle{empty}\n"
    "\\begin{document}\n"
  )


def _compile_worker(args: tuple[Snippet, str, Path, str]) -> tuple[str, bool, str]:
  """多进程工作函数（必须模块级，才能被 pickle）。

  :param args: ``(片段, 模板定义, 工作目录, 字号模式)``。
  :returns: ``(image_id, 是否成功, 错误摘要)``。
  """
  snippet, head, build_dir, mode = args
  ok, message = _compile_one(snippet, head, build_dir, mode)
  if not ok:
    # 失败必须清掉旧产物：否则页面继续用上一轮的图，看起来「编译全绿」却内容不对。
    snippet.dest.unlink(missing_ok=True)
  return snippet.image_id, ok, message


def _safe_name(image_id: str) -> str:
  """把 image_id 变成安全的文件名。

  :param image_id: 片段标识。
  :returns: 文件名（不含扩展名）。
  """
  return re.sub(r"[^A-Za-z0-9_.-]", "-", image_id)


def _compile_one(
  snippet: Snippet,
  head: str,
  build_dir: Path,
  mode: str = FONT_PATHS,
) -> tuple[bool, str]:
  """编译单个片段：standalone → PDF → SVG。

  :param snippet: 片段。
  :param head: 模板定义。
  :param build_dir: 编译工作目录。
  :param mode: 字号处理方式。
  :returns: ``(是否成功, 错误摘要)``。
  """
  name = _safe_name(snippet.image_id)
  tex = build_dir / f"{name}.tex"
  tex.write_text(
    _template(head) + snippet.source.read_text(encoding="utf-8") + "\n\\end{document}\n",
    encoding="utf-8")
  code, log = _run(
    ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], build_dir)
  if code != 0:
    return False, f"xelatex 失败：{_tail(log)}"
  pdf = build_dir / f"{name}.pdf"
  if not pdf.is_file():
    return False, f"未生成 PDF：{_tail(log)}"
  snippet.dest.parent.mkdir(parents=True, exist_ok=True)
  code, log = _run(
    ["dvisvgm", "--pdf", *font_args(mode), "--no-styles", "-o", str(snippet.dest), pdf.name],
    build_dir)
  if code != 0 or not snippet.dest.is_file():
    return False, f"dvisvgm 失败：{_tail(log)}"
  # dvisvgm 偶尔「成功」却没写出任何字形（PDF 里有文字、SVG 里没有对应输出），
  # 页面上的表现就是图里文字全丢。这里做一次交叉核对，把这种情况当失败处理。
  if _pdf_has_text(pdf) and not _svg_has_glyphs(snippet.dest):
    return False, "SVG 缺少字形输出（PDF 有文字但 SVG 里没有 use/text）"
  return True, ""


def _pdf_has_text(pdf: Path) -> bool:
  """判断 PDF 里是否绘制了文字。

  :param pdf: 编译产物。
  :returns: 含文字绘制算子为 True。
  """
  try:
    raw = pdf.read_bytes()
  except OSError:
    return False
  return b"Tj" in raw or b"TJ" in raw


def _svg_has_glyphs(svg: Path) -> bool:
  """判断 SVG 里是否输出了字形。

  ``--no-fonts`` 模式下文字会变成 ``<use>`` 引用 + ``<defs>`` 里的路径，
  因此「有 ``<use>`` 或 ``<text>``」即视为有字形。

  :param svg: SVG 产物。
  :returns: 有字形输出为 True。
  """
  try:
    text = svg.read_text(encoding="utf-8", errors="replace")
  except OSError:
    return False
  return "<use" in text or "<text" in text


def _run(command: list[str], cwd: Path) -> tuple[int, str]:
  """执行外部命令。

  :param command: 命令与参数。
  :param cwd: 工作目录。
  :returns: ``(退出码, 标准输出与标准错误合并文本)``。
  """
  try:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT, check=False)
  except (OSError, subprocess.TimeoutExpired) as exc:
    return 1, f"{type(exc).__name__}: {exc}"
  return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _tail(log: str) -> str:
  """取日志尾部作为错误摘要。

  :param log: 日志文本。
  :returns: 末尾若干行的摘要。
  """
  lines = [line for line in log.splitlines() if line.strip()]
  interesting = [line for line in lines if line.startswith("!") or "Error" in line]
  picked = interesting[-3:] if interesting else lines[-3:]
  return " / ".join(picked)[:400]
