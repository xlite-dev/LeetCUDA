"""TikZ 片段 → SVG（standalone 文档，多进程并行 + 批量摊薄导言区）。

xelatex 出 PDF、dvisvgm 再读这个 PDF 出 SVG：standalone 的页宽高就等于图形外接框，
天然紧致裁剪。两条被实测否定的近路记在这里，避免以后再走一遍：

1. **不要直接让 dvisvgm 读 xelatex 的 XDV**：字形推进量解析不可靠，CJK 文本会互相
   重叠（ch00/ch01 的图全乱），而同一份内容转成 PDF 再读就完全正确。
2. **不要直接让 dvisvgm 读多页 PDF**：PDF 输入的 ``--bbox`` 选项无效，每张 SVG 都会
   是整页（737pt）大小，图形缩在左上角。

批量编译走的是「多页 PDF → 拆单页 → 逐页 dvisvgm」：单张编译有 99% 的时间花在导言区
（fontspec/ctex/Noto 字体加载），一个 ``multi=tikzpicture`` 文档顺序拼 N 个片段只付一次
导言区成本；``pdfseparate`` 拆出的单页 PDF 页尺寸仍是各自图形的外接框，逐页转 SVG 与
单片编译逐字节相同（全量 md5 对比验证）。整批失败（xelatex 报错、页数不符、进程池崩溃）
时回退成逐片编译，错误仍能定位到具体片段。

颜色与 tikz 库从 ``book/preamble.tex`` 程序化抽取，单一样本不漂移。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

from . import figtext

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

#: 缓存版本：编译链路变化时递增，强制全部重编。**批大小不影响产物，不要写进键**。
_CACHE_VERSION = "standalone-batch-2"

#: 默认并行进程数（受 CPU 与内存限制，见 README）。
DEFAULT_JOBS = 8

#: 单个 xelatex 文档里最多拼多少个片段（``--tikz-batch`` 默认值；1 = 逐片编译）。
DEFAULT_BATCH = 32

#: 批编译的超时基数（秒）：单张挂住不能吃掉整站构建预算。
_BATCH_TIMEOUT = 120

#: 批路径里 ``pdfinfo`` / ``dvisvgm`` 的单步超时（秒）。逐片路径仍是 600s，
#: 这两个只在批路径收紧，避免「某一页挂住」把整批拖到平台超时。
_PAGE_TIMEOUT = 30
_SVG_TIMEOUT = 60

#: 整批失败后逐片回退的时间预算（秒）。预算耗尽就把剩余片段直接标失败——宁可
#: 快速红在内容核对里，也不要拖到平台 900s 被杀、连一句错误都留不下。
_FALLBACK_BUDGET = 240


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
  :param mode: 实际走的编译路径：``parallel``（逐片）、``batch29``（这里写的是**有效批
    大小**，不是 ``--tikz-batch`` 的上限）、``batch29+fallback3``（含逐片回退）。
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


def prune_stale(
  snippets: list[Snippet],
  work_dir: Path,
  chapters: list[str],
) -> list[str]:
  """删掉本次已不存在的陈旧片段。

  章里的图被合并或删掉后，旧片段会留在 ``work/tikz``：``verify`` 的「TikZ 图保留文字」
  按目录枚举，会把它当成丢字的图误报（ch20 的 ``-t011`` 就是这么来的）。这里按
  ``<chap_id>-t<序号>`` 归属清理，只动本次转换到的章，不波及其它书或其它章。

  :param snippets: 本次实际产出的片段。
  :param work_dir: 临时目录（片段在 ``tikz/`` 下）。
  :param chapters: 本次转换的章标识。
  :returns: 被删掉的片段名。
  """
  keep = {snippet.source.stem for snippet in snippets}
  removed: list[str] = []
  for path in sorted((work_dir / "tikz").glob("*.tex")):
    if path.stem in keep:
      continue
    if not any(belongs(path.stem, chapter) for chapter in chapters):
      continue
    path.unlink()
    removed.append(path.stem)
  return removed


def belongs(stem: str, chap_id: str) -> bool:
  """判断片段名是否属于某一章。

  :param stem: 片段名（``<chap_id>-t003``）。
  :param chap_id: 章标识。
  :returns: 是否属于该章。
  """
  prefix = f"{chap_id}-t"
  return stem.startswith(prefix) and stem[len(prefix):].isdigit()


def prune_atlas(dest_dir: Path, tasks: list[Snippet], chapters: list[str]) -> list[str]:
  """删掉英文图集里本次已不存在的成品。

  图被删掉后旧的英文 SVG 会留在 ``_static/figures-en``，核对会数出「英文图 196/195」
  这种多于预期张数的结果。这里按章归属清理，只动本次转换到的章。

  :param dest_dir: 图集目录（``_static/figures-en``）。
  :param tasks: 本次产出的英文片段。
  :param chapters: 本次转换的章标识。
  :returns: 被删掉的文件名。
  """
  keep = {task.dest.name for task in tasks}
  removed: list[str] = []
  if not dest_dir.is_dir():
    return removed
  for path in sorted(dest_dir.glob("*.svg")):
    if path.name in keep:
      continue
    if not any(belongs(path.stem, chapter) for chapter in chapters):
      continue
    path.unlink()
    removed.append(path.name)
  return removed


def collect_translated(
  snippets: list[Snippet],
  work_dir: Path,
  dest_dir: Path,
  dictionary: dict[str, str],
) -> tuple[list[Snippet], int, int]:
  """按词典生成英文片段，并给出英文图集的编译任务。

  只输出**全部文字都命中词典**的图：半中半英的图比原图更难读，未命中的图由前端回退
  到中文原图（见 ``_static/translate.js`` 的换图逻辑）。产物写到站点 ``_static``
  下的子目录——Sphinx 会整目录复制，浏览器直接可达，不必走 docutils 图片节点。

  digest 取英文片段内容的哈希，所以词典一改、对应图自动重编。

  :param snippets: 中文图集的任务列表。
  :param work_dir: 临时目录（英文片段与编译缓存放它的 ``en`` 子目录，与中文分开）。
  :param dest_dir: 英文图集输出目录。
  :param dictionary: 掩码中文单元 → 掩码英文单元。
  :returns: ``(英文片段列表, 命中单元数, 未命中单元数)``。
  """
  source_dir = work_dir / "snippets"
  source_dir.mkdir(parents=True, exist_ok=True)
  tasks: list[Snippet] = []
  hit_total = miss_total = 0
  for snippet in snippets:
    text = snippet.source.read_text(encoding="utf-8")
    english, hit, miss = figtext.translate_snippet(text, dictionary)
    hit_total += hit
    miss_total += miss
    if hit == 0 or miss:
      continue
    target = source_dir / snippet.source.name
    target.write_text(english, encoding="utf-8")
    digest = hashlib.sha1(english.encode("utf-8")).hexdigest()
    tasks.append(Snippet(snippet.image_id, target,
                         dest_dir / f"{snippet.image_id}.svg", digest))
  return tasks, hit_total, miss_total


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
  batch: int = DEFAULT_BATCH,
) -> Report:
  """编译全部片段。

  :param snippets: 片段列表。
  :param book_dir: 书根目录（提供 ``preamble.tex``）。
  :param work_dir: 临时目录。
  :param force: 忽略缓存全部重编。
  :param font_mode: 字号处理方式，见 :func:`font_args`。
  :param jobs: 并行编译进程数。
  :param batch: 单个 xelatex 文档拼多少个片段；``1`` 表示逐片编译（回退开关）。
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
    workers = max(1, min(jobs, len(staged)))
    results, report.mode = _compile_all(staged, head, build_dir, font_mode, workers, batch)
    for image_id, ok, message in results:
      if ok:
        report.ok.append(image_id)
      else:
        report.failed[image_id] = message

  report.seconds = round(time.monotonic() - started, 1)
  _save_cache(work_dir, snippets, font_mode, report.failed)
  (work_dir / "tikz-report.json").write_text(report.to_json(), encoding="utf-8")
  return report


def _compile_all(
  snippets: list[Snippet],
  head: str,
  build_dir: Path,
  font_mode: str,
  workers: int,
  batch: int,
) -> tuple[list[tuple[str, bool, str]], str]:
  """编译待编片段：批编译为主，整批失败时在主进程逐片重编。

  回退放在主进程而不是 worker 内：worker 里串行重编 N 张会让它独占而其余 worker 空转，
  且进程池崩溃时回退代码本身也没了。

  :param snippets: 待编译片段（保持原顺序）。
  :param head: 模板定义。
  :param build_dir: 编译工作目录。
  :param font_mode: 字号处理方式。
  :param workers: 并行进程数。
  :param batch: 单文档片段上限；``1`` 走逐片编译。
  :returns: ``(逐片段结果, 模式标签)``，结果顺序与 ``snippets`` 一致。
  """
  if batch <= 1 or len(snippets) == 1:
    return _compile_parallel([(snippet, head, build_dir, font_mode) for snippet in snippets],
                             workers), "parallel"
  missing = [name for name in ("pdfseparate", "pdfinfo") if not shutil.which(name)]
  if missing:
    print(f"[TikZ] 缺少 {'、'.join(missing)}（poppler-utils），退回逐片编译")
    return _compile_parallel([(snippet, head, build_dir, font_mode) for snippet in snippets],
                             workers), "parallel"

  # 批量文档要求每个片段恰好一个 tikzpicture：多一个就多一页，而页数校验只拦得住
  # 「总数不符」，拦不住「A 出 2 页 + B 出 0 页」这种整批静默错位。这类片段先挑出来。
  odd_ids = {snippet.image_id for snippet in snippets if _picture_count(snippet) != 1}
  batched = [snippet for snippet in snippets if snippet.image_id not in odd_ids]
  odd = [snippet for snippet in snippets if snippet.image_id in odd_ids]
  if odd:
    print(f"[TikZ] {len(odd)} 个片段不是单个 tikzpicture，改逐片编译："
          f"{', '.join(snippet.image_id for snippet in odd[:3])}")

  chunks = _plan_batches(batched, batch, workers) if batched else []
  finished: dict[int, tuple] = {}
  broken = ""
  if chunks:
    args = [(index, chunk, head, build_dir, font_mode) for index, chunk in enumerate(chunks)]
    finished, broken = _run_batches(args, min(workers, len(args)))
  if broken:
    print(f"[TikZ] 进程池异常（{broken}），未回报的批次改逐片重编")

  entries: list[tuple[str, bool, str]] = []
  retry: list[list[Snippet]] = []
  for index, chunk in enumerate(chunks):
    result = finished.get(index)
    if result is None or result[2]:
      retry.append(chunk)
      continue
    entries.extend(result[1])
  if retry:
    print(f"[TikZ] 整批回退逐片重编：{len(retry)} 批、"
          f"{sum(len(chunk) for chunk in retry)} 个片段")
  serial = [snippet for chunk in retry for snippet in chunk] + odd
  if serial:
    deadline = time.monotonic() + _FALLBACK_BUDGET
    expired = 0
    for snippet in serial:
      if time.monotonic() > deadline:
        # 预算耗尽：清旧产物并标失败，让内容核对直接红，而不是拖到平台整体超时。
        snippet.dest.unlink(missing_ok=True)
        entries.append((snippet.image_id, False, "回退预算耗尽：批编译系统性失败"))
        expired += 1
        continue
      entries.append(_compile_worker((snippet, head, build_dir, font_mode)))
    if expired:
      print(f"[TikZ] 回退预算 {_FALLBACK_BUDGET}s 耗尽，{expired} 个片段标记失败")
  mode = f"batch{len(chunks[0])}" if chunks else "parallel"
  if broken or retry:
    mode += f"+fallback{len(retry)}"
  return _in_order(snippets, entries), mode


def _compile_parallel(
  tasks: list[tuple[Snippet, str, Path, str]],
  workers: int,
) -> list[tuple[str, bool, str]]:
  """逐片并行编译；进程池崩溃时全部标失败，不把异常抛给调用方。

  :param tasks: 逐片段任务参数。
  :param workers: 并行进程数。
  :returns: ``(image_id, 是否成功, 错误摘要)`` 列表。
  """
  try:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      return list(pool.map(_compile_worker, tasks))
  except BrokenProcessPool as exc:
    print(f"[TikZ] 进程池崩溃（{type(exc).__name__}: {exc}），{len(tasks)} 个片段标失败")
    return [(snippet.image_id, False, f"进程池崩溃：{exc}") for snippet, _, _, _ in tasks]


def _picture_count(snippet: Snippet) -> int:
  """片段里 ``tikzpicture`` 环境的个数（批量编译要求恰好一个）。

  :param snippet: 片段。
  :returns: 个数。
  """
  return snippet.source.read_text(encoding="utf-8").count("\\begin{tikzpicture}")


def _plan_batches(snippets: list[Snippet], batch: int, workers: int) -> list[list[Snippet]]:
  """按原顺序切块。

  批数取「``batch`` 上限所需的批数」与 worker 数的较大者：只按上限切会让「40 个片段 +
  8 个 worker」塌成 2 个 worker，本地增量构建反而更慢。

  :param snippets: 片段列表。
  :param batch: 单批片段上限。
  :param workers: 并行进程数。
  :returns: 批次列表（顺序拼接后等于原列表）。
  """
  needed = max(1, math.ceil(len(snippets) / batch))
  count = min(len(snippets), max(needed, workers))
  size = math.ceil(len(snippets) / count)
  return [snippets[index:index + size] for index in range(0, len(snippets), size)]


def _in_order(
  snippets: list[Snippet],
  entries: list[tuple[str, bool, str]],
) -> list[tuple[str, bool, str]]:
  """把（可能乱序、含回退追加的）结果按片段原顺序重排。

  :param snippets: 片段列表。
  :param entries: 逐片段结果。
  :returns: 与 ``snippets`` 同序的结果列表。
  """
  by_id = {entry[0]: entry for entry in entries}
  return [by_id.get(snippet.image_id, (snippet.image_id, False, "编译结果缺失"))
          for snippet in snippets]


def _run_batches(
  tasks: list[tuple],
  workers: int,
) -> tuple[dict[int, tuple], str]:
  """并行执行批次任务，进程池崩溃时不抛出、交由调用方回退。

  :param tasks: 批次任务参数。
  :param workers: 并行进程数。
  :returns: ``({批序号: 批结果}, 崩溃信息)``；未回报的批序号不在字典里。
  """
  finished: dict[int, tuple] = {}
  broken = ""
  try:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(_compile_batch, task): task[0] for task in tasks}
      for future in as_completed(futures):
        index = futures[future]
        try:
          finished[index] = future.result()
        except Exception as exc:
          broken = f"{type(exc).__name__}: {exc}"
  except BrokenProcessPool as exc:
    broken = f"{type(exc).__name__}: {exc}"
  return finished, broken


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


def _template(head: str, multi: bool = False) -> str:
  """生成编译模板头部（standalone 文档）。

  :param head: 从 preamble 抽取的定义。
  :param multi: 是否每个 ``tikzpicture`` 独立成页（批量编译用）。
  :returns: 模板头（不含正文）。
  """
  options = "border=6pt,multi=tikzpicture" if multi else "border=6pt"
  document_class = f"\\documentclass[{options}]{{standalone}}"
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


def _compile_batch(args: tuple) -> tuple[int, list[tuple[str, bool, str]], bool]:
  """多进程工作函数：一个多页文档编译一批片段（必须模块级，才能被 pickle）。

  :param args: ``(批序号, 片段列表, 模板定义, 编译工作目录, 字号模式)``。
  :returns: ``(批序号, 逐片段结果, 是否整批失败需回退)``。
  """
  index, chunk, head, build_dir, mode = args
  work = build_dir / f"batch-{index:03d}"
  try:
    # 每批独立子目录：``tikz-build`` 是持久目录，同名 page-*.pdf 会串到上一轮。
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    body = "\n".join(snippet.source.read_text(encoding="utf-8").rstrip() for snippet in chunk)
    (work / "batch.tex").write_text(
      _template(head, multi=True) + body + "\n\\end{document}\n", encoding="utf-8")
    timeout = _BATCH_TIMEOUT + 3 * len(chunk)
    code, log = _run(
      ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "batch.tex"], work, timeout)
    if code != 0:
      print(f"[TikZ] 批 {index}（{_span(chunk)}）xelatex 失败，回退逐片：{_tail(log)}")
      return index, [], True
    pdf = work / "batch.pdf"
    # 页数校验是必要非充分条件：能挡住「总数与片段数不符」，挡不住 A 出 2 页 B 出 0 页。
    # 靠 ``_picture_count`` 前置挑出非单图片段来堵这个口子，产物正确性最终由视觉验收兜底。
    pages = _pdf_pages(pdf, work)
    if pages != len(chunk):
      print(f"[TikZ] 批 {index}（{_span(chunk)}）页数 {pages} != 片段数 {len(chunk)}，回退逐片")
      return index, [], True
    code, log = _run(["pdfseparate", "batch.pdf", "page-%03d.pdf"], work, timeout)
    if code != 0:
      print(f"[TikZ] 批 {index}（{_span(chunk)}）拆页失败，回退逐片：{_tail(log)}")
      return index, [], True
    entries: list[tuple[str, bool, str]] = []
    for number, snippet in enumerate(chunk, start=1):
      page = work / f"page-{number:03d}.pdf"
      if not page.is_file():
        snippet.dest.unlink(missing_ok=True)
        entries.append((snippet.image_id, False, f"拆页缺页：{page.name}"))
        continue
      ok, message = _pdf_to_svg(page, snippet.dest, work, mode, _SVG_TIMEOUT)
      if not ok:
        snippet.dest.unlink(missing_ok=True)
      entries.append((snippet.image_id, ok, message))
    return index, entries, False
  except KeyboardInterrupt:
    raise
  except Exception as exc:
    print(f"[TikZ] 批 {index}（{_span(chunk)}）异常，回退逐片：{type(exc).__name__}: {exc}")
    return index, [], True


def _span(chunk: list[Snippet]) -> str:
  """批的片段范围（日志用，省得去数 batch.tex 的行号）。

  :param chunk: 片段列表。
  :returns: ``首片段 … 末片段`` 形式的描述。
  """
  return f"{chunk[0].image_id} … {chunk[-1].image_id}（{len(chunk)} 个）"


def _pdf_pages(pdf: Path, cwd: Path, timeout: int = _PAGE_TIMEOUT) -> int:
  """读 PDF 页数（``pdfinfo``，来自 poppler-utils）。

  :param pdf: PDF 文件。
  :param cwd: 命令工作目录。
  :param timeout: 超时秒数。
  :returns: 页数；读不到返回 0。
  """
  code, log = _run(["pdfinfo", str(pdf)], cwd, timeout)
  if code != 0:
    return 0
  for line in log.splitlines():
    if line.startswith("Pages:"):
      try:
        return int(line.split()[1])
      except (IndexError, ValueError):
        return 0
  return 0


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
  return _pdf_to_svg(pdf, snippet.dest, build_dir, mode)


def _pdf_to_svg(
  pdf: Path,
  dest: Path,
  work: Path,
  mode: str,
  timeout: int = _COMPILE_TIMEOUT,
) -> tuple[bool, str]:
  """单页 PDF → SVG，并交叉核对是否真的输出了字形。

  dvisvgm 偶尔「成功」却没写出任何字形（PDF 里有文字、SVG 里没有对应输出），页面上
  的表现就是图里文字全丢。这里做一次交叉核对；偶发失败允许重试一次（并行编译时曾
  出现读到未写完 SVG 的情况）。**``pdf`` 必须是单页**：多页 PDF 里只要任意一页有字，
  ``_pdf_has_text`` 就恒为真，这道防线会失效。

  :param pdf: 单页 PDF。
  :param dest: 目标 SVG。
  :param work: 编译工作目录（同时也是 dvisvgm 的相对路径基准）。
  :param mode: 字号处理方式。
  :param timeout: 单次 dvisvgm 的超时秒数（批路径传更短的值）。
  :returns: ``(是否成功, 错误摘要)``。
  """
  dest.parent.mkdir(parents=True, exist_ok=True)
  command = ["dvisvgm", "--pdf", *font_args(mode), "--no-styles", "-o", str(dest), pdf.name]
  code, log = _run(command, work, timeout)
  if code != 0 or not dest.is_file():
    return False, f"dvisvgm 失败：{_tail(log)}"
  if _pdf_has_text(pdf) and not _svg_has_glyphs(dest):
    dest.unlink(missing_ok=True)
    code, log = _run(command, work, timeout)
    if code != 0 or not dest.is_file():
      return False, f"dvisvgm 重试失败：{_tail(log)}"
    if not _svg_has_glyphs(dest):
      return False, "SVG 缺少字形输出（PDF 有文字但 SVG 里没有 use/text）"
  return True, ""


def _pdf_has_text(pdf: Path) -> bool:
  """判断 PDF 里是否绘制了文字。

  PDF 的文本算子一般在**被压缩的内容流**里，直接搜字节会漏判——漏判就会把「dvisvgm
  只写出 2 KB 空壳」的失败当成成功（ch00/ch09 两张英文图曾如此，缓存还会把它锁死）。
  这里把 ``stream…endstream`` 解压后再搜。

  :param pdf: 编译产物。
  :returns: 含文字绘制算子为 True。
  """
  try:
    raw = pdf.read_bytes()
  except OSError:
    return False
  if b"Tj" in raw or b"TJ" in raw:
    return True
  for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
    try:
      data = zlib.decompress(match.group(1).rstrip(b"\r\n"))
    except zlib.error:
      continue
    if b"Tj" in data or b"TJ" in data:
      return True
  return False


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


def _run(command: list[str], cwd: Path, timeout: int = _COMPILE_TIMEOUT) -> tuple[int, str]:
  """执行外部命令。

  :param command: 命令与参数。
  :param cwd: 工作目录。
  :param timeout: 超时秒数（批任务用更短的独立超时，避免单张挂住吃掉整站预算）。
  :returns: ``(退出码, 标准输出与标准错误合并文本)``。
  """
  try:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
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
