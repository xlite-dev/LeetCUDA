#!/usr/bin/env python3
"""lint_book.py — 书稿的三类跨文件检查（R22 三个 P0 都落在这里）。

单章通读抓不到“两个文件说法不一致”，这三类检查各对应一次真实缺陷：

1. ``citations``：kaogu 框引用的英文原文必须逐字出现在被引文件里
   （ch14 曾把 ``tensorRank must be 3, 4, or 5`` 当成 cuda.h 的原话）。
2. ``tables``：两张表的数值矩阵完全相同却分属不同章 → 疑似串表
   （ch12 曾整表复制 ch24 的 CuTe 数据）。
3. ``figures``：图里的偏移索引（``K[k+2]``）必须等于同章代码表达式
   （``k + kStagesK - 1``）在章内参数下的取值（ch17 图里写 k+2、代码是 k+1）。

用法：
  python3 scripts/lint_book.py                    # 全部检查
  python3 scripts/lint_book.py --only tables      # 单类（逗号分隔）
  python3 scripts/lint_book.py --root <book_dir>  # 换书根（回归测试用）
退出码：0 = 无 FAIL（WARN 不计入），1 = 有 FAIL。

能力边界（刻意不做假承诺）：
- citations 只保证“引文存在且落在标注区间内”，不判断推论对错。
- tables 只报“数值矩阵雷同”，同表复用是否合理由人判（故为 WARN）。
- figures 只覆盖 ``X[k±c]`` 形式的偏移索引与 ``x[k±term]`` 形式的代码表达式；
  图与代码同错、或索引以文字描述（“P1 只装 K[0]”）时抓不到。
"""
import argparse
import re
import sys
from pathlib import Path

try:
  import yaml
except ImportError:  # 只影响 anchors 与白名单的读法，检查本身仍可跑
  yaml = None

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent  # kernels/interview/book
SRC_ROOT = BOOK.parent  # kernels/interview
REPO_ROOT = SRC_ROOT.parent.parent  # 仓库根

#: 英文引文的最短词数；再短（如 ``k + 1``）视为标识符，不参与核对。
_MIN_QUOTE_WORDS = 3

#: 引文最短字符数，以及必须含一个长词——用来滤掉 ``idx < N`` 这类代码片段。
_MIN_QUOTE_CHARS = 22
_MIN_QUOTE_LONG_WORD = 6


def _tex_files(book):
  """书稿里的全部 tex（章 + 附录）。

  :param book: 书根目录。
  :returns: 排好序的路径列表。
  """
  files = sorted((book / "chapters").glob("*.tex")) + sorted((book / "appendices").glob("*.tex"))
  return [path for path in files if path.is_file()]


def _blocks(text, env):
  """取出某个环境的所有块。

  :param text: 源文本。
  :param env: 环境名。
  :returns: ``(起始行号, 块文本)`` 列表。
  """
  pattern = re.compile(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", re.S)
  return [(text.count("\n", 0, match.start()) + 1, match.group(0)) for match in pattern.finditer(text)]


def _line_of(text, needle):
  """片段在源文本里的行号。

  :param text: 源文本。
  :param needle: 子串。
  :returns: 行号（找不到返回 0）。
  """
  index = text.find(needle)
  return text.count("\n", 0, index) + 1 if index >= 0 else 0


def _norm(text):
  """把 TeX/排版噪声抹平，便于逐字匹配。

  :param text: 原始文本。
  :returns: 归一化后的文本（小写、单空格、无宏）。
  """
  text = re.sub(r"\\_", "_", text)
  text = re.sub(r"\\[%&#$]", lambda m: m.group(0)[1], text)
  text = re.sub(r"\\[a-zA-Z]+\*?\s*", "", text)
  text = text.replace("{", "").replace("}", "")
  text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
  text = re.sub(r"[\u2010-\u2015]", "-", text)
  text = re.sub(r"\s+", " ", text)
  return text.strip().casefold()


def _squash(text):
  """抹掉一切非字母数字，只留词序。

  引文和源文件的行首注释符（``*``、``//``、``- ``）与标点差异都在这层被吸收，
  否则跨行的 ``* - \\p xxx`` 这类注释必然匹配失败。

  :param text: 归一化文本。
  :returns: 只剩小写字母数字与单空格的文本。
  """
  return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _ascii_words(text):
  """引文里的英文词。

  :param text: 归一化文本。
  :returns: 词列表。
  """
  return re.findall(r"[A-Za-z][A-Za-z'\-]*", text)


def _allow_list(book):
  """读 ``scripts/lint_allow.yaml`` 的引文白名单。

  白名单只用于“本来就不在源码里”的引文（编译器诊断、外部规范原文），
  每一条都要在 yaml 里写明理由。

  :param book: 书根目录。
  :returns: 归一化后的引文集合。
  """
  path = book / "scripts" / "lint_allow.yaml"
  if not path.is_file():
    return set()
  text = path.read_text(encoding="utf-8")
  if yaml is not None:
    items = (yaml.safe_load(text) or {}).get("quotes") or []
  else:
    items, inside = [], False
    for line in text.splitlines():
      if re.match(r"^quotes\s*:", line):
        inside = True
      elif inside and line.startswith("  - "):
        items.append(line[4:].strip())
  return {_squash(_norm(item)) for item in items}


def _anchor_paths(book):
  """anchors.yaml 登记的源文件名 → 实际路径。

  :param book: 书根目录。
  :returns: 名字到路径的字典（找不到的不放进去）。
  """
  paths = {}
  anchor = book / "scripts" / "anchors.yaml"
  if anchor.is_file():
    text = anchor.read_text(encoding="utf-8")
    if yaml is not None:
      names = list((yaml.safe_load(text) or {}).get("files") or {})
    else:
      names, inside = [], False
      for line in text.splitlines():
        if re.match(r"^files\s*:", line):
          inside = True
        elif inside and re.match(r"^\S", line):
          inside = False
        elif inside and re.match(r"^  ([^\s:]+):\s*$", line):
          names.append(re.match(r"^  ([^\s:]+):", line).group(1))
    for name in names:
      found = _locate(name)
      if found:
        paths[name] = found
  cuda_h = Path("/usr/local/cuda/include/cuda.h")
  if cuda_h.is_file():
    paths["cuda.h"] = cuda_h
  return paths


def _locate(name):
  """按文件名在仓库里找源文件。

  先找 ``kernels/interview``（anchors 登记区），再退回整仓（``bench/`` 下的
  测试文件不在登记表里，但书里会引它们）。

  :param name: 文件名（如 ``hgemm.cuh``）。
  :returns: 路径或 None。
  """
  for root in (SRC_ROOT, BOOK, REPO_ROOT):
    found = sorted((path for path in root.rglob(name) if path.is_file()), key=lambda p: len(p.parts))
    if found:
      return found[0]
  return None


def _ptx_doc(section):
  """按 PTX 节号找本地镜像文档。

  镜像按大节切文件（``9.7.14.5-...md``），小节号（``9.7.14.5.15``）在文件内部，
  所以先按“文件里出现过该节号”找，再退回按前缀匹配。

  :param section: 节号（如 ``9.7.15.5.1.2.2``）。
  :returns: 路径或 None。
  """
  roots = (
    REPO_ROOT / ".github" / "skills",
    REPO_ROOT / ".codebuddy" / "skills",
    REPO_ROOT.parent / ".codebuddy" / "skills",
    Path.home() / ".codebuddy" / "skills",
  )
  docs = []
  for skills in roots:
    if skills.is_dir():
      docs += list(skills.glob("*/references/ptx-docs/**/*.md"))
  if not docs:
    return None
  parts = section.split(".")
  for cut in range(len(parts), 1, -1):
    prefix = ".".join(parts[:cut])
    for path in docs:
      if path.name.startswith(prefix + "-") or path.name.startswith(prefix + "."):
        return path
  for path in docs:
    if re.search(r"^#+\s*" + re.escape(section) + r"\b", path.read_text(encoding="utf-8",
                                                                      errors="replace"), re.M):
      return path
  return None


_CITE_RE = re.compile(r"([A-Za-z0-9_\\\-.]+\.(?:cuh|cu|hpp|h|py|json|ya?ml|md|tex))"
                      r"(?:\s*(?:L|:)\s*(\d+)(?:\s*(?:--|~|-)\s*L?(\d+))?)?")
_PTX_RE = re.compile(r"§\s*(\d+(?:\.\d+)+)")
_QUOTE_RE = re.compile(r"``(.+?)''", re.S)
_TT_RE = re.compile(r"\\texttt\{([^{}]{12,})\}")
_BOLD_RE = re.compile(r"\\textbf\{([^{}]{12,})\}")


def _clean_name(name):
  """还原被 TeX 转义的文件名。

  :param name: 匹配到的名字（可能含 ``\\_``、``\\allowbreak{}``）。
  :returns: 原始文件名。
  """
  return name.replace("\\allowbreak{}", "").replace("\\_", "_").replace("\\", "")


def _quotes(block):
  """从 kaogu 块里抽英文引文。

  ``\\texttt`` 与 ``\\textbf`` 都要收：ch14 曾把编造的原文写在框标题的
  ``\\textbf`` 里，只收 ``\\texttt`` 会漏掉这一类。

  :param block: 块文本。
  :returns: 引文列表。
  """
  candidates = [match.group(1) for match in _QUOTE_RE.finditer(block)]
  candidates += [match.group(1) for match in _TT_RE.finditer(block)]
  candidates += [match.group(1) for match in _BOLD_RE.finditer(block)]
  quotes = []
  for raw in candidates:
    if re.search(r"[\u4e00-\u9fff]", raw) or "$" in raw:
      continue
    text = _norm(raw)
    words = _ascii_words(text)
    if len(words) < _MIN_QUOTE_WORDS or len(text) < _MIN_QUOTE_CHARS:
      continue
    if not any(len(word) >= _MIN_QUOTE_LONG_WORD for word in words):
      continue
    if text.count(" ") < 3:
      continue
    quotes.append((raw, text))
  return quotes


def _find_line(text, norm):
  """按引文前几个词定位源文件里的行号。

  :param text: 源文件原文。
  :param norm: 归一化引文。
  :returns: 行号（定位不到返回 0）。
  """
  words = _ascii_words(norm)[:6]
  if len(words) < 3:
    return 0
  pattern = r"\s+".join(re.escape(word) for word in words)
  match = re.search(pattern, text, re.I)
  return text.count("\n", 0, match.start()) + 1 if match else 0


def check_citations(book):
  """kaogu 引文逐字回引。

  :param book: 书根目录。
  :returns: ``(fails, warns)`` 两条消息列表。
  """
  fails, warns = [], []
  sources = _anchor_paths(book)
  allowed = _allow_list(book)
  cache = {}
  skipped = 0
  for path in _tex_files(book):
    text = path.read_text(encoding="utf-8")
    for start, block in _blocks(text, "kaogu"):
      block = block.replace("\\allowbreak{}", "")
      refs = []
      unresolved = []
      for match in _CITE_RE.finditer(block):
        name = _clean_name(match.group(1))
        source_path = sources.get(name) or _locate(name)
        if source_path:
          sources[name] = source_path
          refs.append((name, source_path, int(match.group(2) or 0), int(match.group(3) or 0)))
        elif name.lower().endswith((".cuh", ".cu", ".h", ".hpp")):
          unresolved.append(name)
      for match in _PTX_RE.finditer(block):
        doc = _ptx_doc(match.group(1))
        if doc:
          refs.append((f"PTX §{match.group(1)}", doc, 0, 0))
        else:
          unresolved.append(f"PTX §{match.group(1)}")
      if not refs and not unresolved:
        continue
      for raw, norm in _quotes(block):
        needle = _squash(norm)
        if needle in allowed:
          skipped += 1
          continue
        hit = None
        for name, source, begin, end in refs:
          key = str(source)
          if key not in cache:
            original = source.read_text(encoding="utf-8", errors="replace")
            cache[key] = (_squash(_norm(original)), original)
          squashed, original = cache[key]
          if needle in squashed:
            hit = (name, begin, end, original)
            break
        where = f"{path.name}:{start}"
        if hit is None:
          if unresolved:
            warns.append(f"{where} 引文在已解析来源中未命中，且来源未解析：{unresolved}；"
                         f"引文：{raw.strip()[:56]}…")
          else:
            fails.append(f"{where} 引文未在来源中找到：{raw.strip()[:72]}…")
          continue
        name, begin, end, original = hit
        if begin and end:
          line = _find_line(original, norm)
          if line and not (begin - 3 <= line <= end + 3):
            warns.append(f"{where} 引文命中 {name} 第 {line} 行，但标注区间是 L{begin}-{end}")
  if skipped:
    warns.append(f"白名单跳过 {skipped} 条引文（见 scripts/lint_allow.yaml）")
  return fails, warns


_TABULAR_RE = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.S)
_TAB_LABEL_RE = re.compile(r"\\label\{(tab:[^}]+)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _tables(book):
  """抽取所有表格的数值矩阵。

  :param book: 书根目录。
  :returns: ``(label, 章名, 行号, 数值元组)`` 列表。
  """
  tables = []
  for path in _tex_files(book):
    text = path.read_text(encoding="utf-8")
    for match in _TABULAR_RE.finditer(text):
      before = text[max(0, match.start() - 800):match.start()]
      labels = _TAB_LABEL_RE.findall(before)
      if not labels:
        continue
      body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", match.group(1))
      numbers = tuple(_NUM_RE.findall(body))
      if len(numbers) < 4:
        continue
      refs = set(re.findall(r"\\ref\{(ch:[0-9a-z]+)\}", before))
      tables.append((labels[-1], path.name, _line_of(text, match.group(0)[:40]), numbers, refs))
  return tables


def check_tables(book):
  """跨章表格数值矩阵雷同。

  :param book: 书根目录。
  :returns: ``(fails, warns)``。
  """
  tables = _tables(book)
  warns = []
  seen = {}
  for label, chapter, line, numbers, refs in tables:
    chapter_id = chapter.split("-")[0]
    if numbers in seen:
      other_label, other_chapter, other_line, other_refs = seen[numbers]
      other_id = other_chapter.split("-")[0]
      declared = f"ch:{other_id[2:]}" in refs or f"ch:{chapter_id[2:]}" in other_refs
      if other_chapter != chapter and not declared:
        warns.append(f"{chapter}:{line} {label} 的数值矩阵与 {other_chapter}:{other_line} "
                     f"{other_label} 完全相同 → 疑似串表（若属跨章引用请在表前 ref 该章）")
    else:
      seen[numbers] = (label, chapter, line, refs)
  return [], warns


_PARAM_RE = re.compile(r"(kStages[KV]|kValTileK|Sk|Sv|S_k|S_v)\s*[=＝]\s*(\d+)")
_CODE_IDX_RE = re.compile(r"([A-Za-z])\[\s*k((?:\s*[+-]\s*[A-Za-z0-9_]+)*)\s*\]")
_TERM_RE = re.compile(r"([+-])\s*([A-Za-z0-9_]+)")
_FIG_IDX_RE = re.compile(r"([A-Za-z])\s*\[\s*k\s*([+-])\s*(\d+)\s*\]")
_MATH_STRIP_RE = re.compile(r"\\(?:mathrm|text|mathtt)\{([^{}]*)\}")


def _plain(text):
  """去掉数学包装与花括号，便于解析索引表达式。

  :param text: 源文本。
  :returns: 纯文本副本。
  """
  text = _MATH_STRIP_RE.sub(r"\1", text)
  return text.replace("{", "").replace("}", "")


def _without(text, env):
  """删掉某环境的块。

  图内文字必须排除在“代码表达式”来源之外，否则 ``K[k+2]`` 会用自己的图给自己作证。

  :param text: 源文本。
  :param env: 环境名。
  :returns: 删掉该环境后的文本。
  """
  pattern = re.compile(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", re.S)
  return pattern.sub(" ", text)


def _params(text):
  """章内声明的循环/流水参数。

  :param text: 章文本（已去数学包装）。
  :returns: 参数名 → 整数。
  """
  params = {}
  for name, value in _PARAM_RE.findall(text):
    params.setdefault(name.replace("_", ""), int(value))
  return params


def _code_offsets(body, params):
  """章内 ``x[k±…]`` 表达式按章内参数实例化后的偏移。

  :param body: 去掉图块后的章文本。
  :param params: 参数名 → 整数。
  :returns: ``(缓冲区字母 → 偏移集合, 未解析符号集合)``。
  """
  offsets, unresolved = {}, set()
  for letter, tail in _CODE_IDX_RE.findall(body):
    total = 0
    for sign, term in _TERM_RE.findall(tail):
      if term.isdigit():
        value = int(term)
      elif term.replace("_", "") in params:
        value = params[term.replace("_", "")]
      else:
        unresolved.add(term)
        total = None
        break
      total += value if sign == "+" else -value
    if total is not None:
      offsets.setdefault(letter, set()).add(total)
  return offsets, unresolved


def check_figures(book):
  """图内偏移索引与代码表达式同解。

  :param book: 书根目录。
  :returns: ``(fails, warns)``。
  """
  fails, warns = [], []
  for path in _tex_files(book):
    raw = path.read_text(encoding="utf-8")
    figures = _blocks(raw, "tikzpicture")
    if not figures:
      continue
    body = _plain(_without(raw, "tikzpicture"))
    params = _params(body)
    offsets, unresolved = _code_offsets(body, params)
    for start, block in figures:
      for letter, sign, value in _FIG_IDX_RE.findall(block):
        wanted = int(value) * (1 if sign == "+" else -1)
        known = offsets.get(letter, set())
        where = f"{path.name}:{start}"
        if not known:
          warns.append(f"{where} 图内 {letter}[k{sign}{value}] 在正文/代码里找不到同族表达式（无法核对）")
        elif wanted in known:
          continue
        elif unresolved:
          warns.append(f"{where} 图内 {letter}[k{sign}{value}] 与代码偏移 {sorted(known)} 不一致，"
                       f"但章内存在未声明参数 {sorted(unresolved)}（无法判定）")
        else:
          fails.append(f"{where} 图内 {letter}[k{sign}{value}] 与代码偏移 {sorted(known)} 不一致"
                       f"（章内参数 {params}）")
  return fails, warns


CHECKS = {
  "citations": check_citations,
  "tables": check_tables,
  "figures": check_figures,
}


def main():
  """命令行入口。

  :returns: 退出码（0 无 FAIL，1 有 FAIL）。
  """
  parser = argparse.ArgumentParser(description="书稿跨文件 lint（引文/串表/图-码索引）")
  parser.add_argument("--root", default=str(BOOK), help="书根目录（默认本脚本所在书）")
  parser.add_argument("--only", default="", help="只跑某几类（逗号分隔：citations,tables,figures）")
  parser.add_argument("--quiet-warn", action="store_true", help="不打印 WARN")
  args = parser.parse_args()
  book = Path(args.root).resolve()
  names = [name.strip() for name in args.only.split(",") if name.strip()] or list(CHECKS)
  unknown = [name for name in names if name not in CHECKS]
  if unknown:
    raise SystemExit(f"未知检查项：{unknown}")

  total_fail = 0
  for name in names:
    fails, warns = CHECKS[name](book)
    status = "FAIL" if fails else "OK"
    print(f"[{status}] {name}: {len(fails)} fail / {len(warns)} warn")
    for message in fails:
      print(f"  FAIL {message}")
    if not args.quiet_warn:
      for message in warns:
        print(f"  WARN {message}")
    total_fail += len(fails)
  return 1 if total_fail else 0


if __name__ == "__main__":
  sys.exit(main())
