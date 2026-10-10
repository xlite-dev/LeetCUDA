"""代码块 / 公式英译词典的抽取与合并（人工产线，不参与站点构建）。

代码块与公式被 ``notranslate`` 保护、机器翻译碰不到，英文版靠词典 +
``translate.js`` 按文本节点替换（``swapCodeComments`` / ``swapMathText``）。本脚本
承担产线的两端，中间的翻译由人工/模型逐条完成：

1. ``extract``：从 ``build/html`` 抽出 ``--what code``（``<pre>``）或 ``--what math``
   （``.math`` 容器）内含中文的文本节点（唯一化、按文件聚合并均衡切成若干组），
   产出 ``groups/groupN.json`` 供翻译；
2. ``merge``：合并各组译文（``parts/groupN.json``）并全量校验（key 覆盖、value 无
   汉字、代码块的非中文行 / 公式的 LaTeX 骨架逐字保留），写入 ``_static/code-en.json``
   或 ``_static/math-en.js``。

节点抽取与 ``convert.verify`` 共用 ``convert.cjk_nodes``；key 是文本节点 trim 后的
原文，JS 端替换时保留节点前后空白（缩进不动）。

用法::

    python -m tools.code_dict extract --what math --groups 8 --out .tmp/math-i18n
    python -m tools.code_dict merge --what math --parts .tmp/math-i18n/parts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert.cjk_nodes import CJK, CJK_BAD, CJK_RUN, MathTextCollector, PreTextCollector

#: 站点构建产物（抽取源）。
HTML_DIR = Path(__file__).resolve().parent.parent / "build" / "html"

#: 词典产物（Sphinx 整目录拷贝，浏览器直接可达）。
DICT_PATH = Path(__file__).resolve().parent.parent / "_static" / "code-en.json"

#: 公式词典产物（JS 赋值格式，conf.py 在 translate.js 前同步加载）。
MATH_PATH = Path(__file__).resolve().parent.parent / "_static" / "math-en.js"

#: 词典种类 → 页面容器选择器（产物路径见 DICT_PATH / MATH_PATH）。
TARGETS = {"code": {"selector": "pre"}, "math": {"selector": ".math"}}


def collect_nodes(what: str) -> dict[str, list[str]]:
  """按文件抽取含中文的代码块/公式文本节点（唯一化，保持原书出现顺序）。

  :param what: ``code``（``pre``）或 ``math``（``.math`` 容器）。
  :returns: 文件名（不含扩展名）→ 该文件首次出现的唯一节点列表。
  """
  seen: set[str] = set()
  per_file: dict[str, list[str]] = {}
  for path in sorted(HTML_DIR.glob("*.html")):
    if what == "code":
      collector = PreTextCollector()
    else:
      collector = MathTextCollector()
    collector.feed(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in collector.nodes:
      trimmed = node.strip()
      if trimmed and trimmed not in seen:
        seen.add(trimmed)
        hits.append(trimmed)
    if hits:
      per_file[path.stem] = hits
  return per_file


def cmd_extract(what: str, groups: int, out: Path) -> None:
  """抽取节点并按文件装箱成 ``groups/groupN.json``（同文件同组，上下文连贯）。

  :param what: ``code`` 或 ``math``。
  :param groups: 组数。
  :param out: 输出目录（``groups/`` 建在其下）。
  """
  per_file = collect_nodes(what)
  # first-fit decreasing：文件按条数降序放进当前最空的组。
  loads = [0] * groups
  assignment: dict[str, int] = {}
  for name in sorted(per_file, key=lambda name: -len(per_file[name])):
    group = min(range(groups), key=lambda index: loads[index])
    assignment[name] = group
    loads[group] += len(per_file[name])

  group_nodes: list[list[str]] = [[] for _ in range(groups)]
  group_files: list[list[str]] = [[] for _ in range(groups)]
  for name in per_file:  # 已按文件名排序，组内保持原书顺序
    group = assignment[name]
    group_files[group].append(name)
    group_nodes[group].extend(per_file[name])

  (out / "groups").mkdir(parents=True, exist_ok=True)
  for group in range(groups):
    payload = {"files": group_files[group], "nodes": group_nodes[group]}
    (out / "groups" / f"group{group}.json").write_text(
      json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
  total = sum(loads)
  print(f"[{what}] extracted {total} unique nodes -> {out / 'groups'}")
  for group in range(groups):
    print(f"  group{group}: {loads[group]:4d} nodes  {','.join(group_files[group])}")


def check_translation(key: str, value: str, what: str) -> str | None:
  """校验单条译文，返回问题描述（合格返回 ``None``）。

  :param key: 原文（节点 trim 后）。
  :param value: 译文。
  :param what: ``code`` 或 ``math``。
  """
  if not value.strip():
    return "译文为空"
  if CJK_BAD.search(value):
    return f"汉字/中文标点残留: {value[:60]!r}"
  if CJK.search(key) and value == key:
    return "含中文却原样照抄"
  if what == "math":
    # 骨架校验：译文必须等于「原文把每段连续中文原位替换成任意非中文文本」，
    # LaTeX 结构（命令、花括号、数字、空白、换行）逐字未动。实现：取原文的
    # 非中文片段序列，要求其在译文中按原顺序逐字出现；片段之间（即原中文所在
    # 位置）允许任意译文，译文的非中文性已由上方 CJK_BAD 检查保证。
    skeleton = ".*?".join(re.escape(seg) for seg in CJK_RUN.split(key))
    if not re.fullmatch(skeleton, value, re.DOTALL):
      return "非中文骨架被改动（LaTeX 结构必须逐字保留）"
    if key.count("{") != value.count("{") or key.count("}") != value.count("}"):
      return "花括号数量不一致"
  elif "\n" in key:
    # 多行代码节点：不含中文/全角标点的行（命令、代码）必须逐字保留。
    for line in key.split("\n"):
      if not CJK_BAD.search(line) and line.strip() and line not in value:
        return f"纯代码行被改动: {line[:60]!r}"
  return None


def cmd_merge(what: str, parts: Path) -> None:
  """合并各组译文并校验，写入词典产物。

  :param what: ``code``（``_static/code-en.json``）或 ``math``（``_static/math-en.js``，
      JS 赋值格式，供 ``conf.py`` 在 ``translate.js`` 前同步加载）。
  :param parts: 各组译文所在目录（``parts/groupN.json``）。
  """
  nodes = [node for hits in collect_nodes(what).values() for node in hits]
  merged: dict[str, str] = {}
  problems: list[str] = []
  for part in sorted(parts.glob("group*.json")):
    for key, value in json.loads(part.read_text(encoding="utf-8")).items():
      if key in merged:
        problems.append(f"跨组重复 key: {key[:60]!r}")
      merged[key] = value

  missing = [node for node in nodes if node not in merged]
  problems.extend(f"missing: {item[:60]!r}" for item in missing[:10])
  for key in nodes:
    if key not in merged or len(problems) >= 15:
      continue
    problem = check_translation(key, merged[key], what)
    if problem:
      problems.append(f"{problem} (key: {key[:50]!r})")
  if problems:
    for problem in problems[:15]:
      print("  " + problem)
    raise SystemExit(f"merge failed: {len(problems)} problem(s)")

  entries = {key: merged[key] for key in nodes}
  target = DICT_PATH if what == "code" else MATH_PATH
  if what == "math":
    body = json.dumps(entries, ensure_ascii=False, indent=1)
    target.write_text(
      "// 由 `python -m tools.code_dict merge --what math` 生成：公式英译词典。\n"
      "// key = 公式 tex 源原文（trim 后），value = 英文版 tex（骨架校验保证只翻中文）。\n"
      "// conf.py 把本文件排在 translate.js 之前同步加载；translate.js 于 DOM 解析完时\n"
      "// 同步替换 .math 里的 tex 源，赶在 MathJax 渲染之前（见 swapMathText）。\n"
      f"window.LEETCUDA_MATH_EN = {body};\n",
      encoding="utf-8")
  else:
    target.write_text(
      json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
  print(f"[{what}] written: {target} ({len(nodes)} entries)")


def main() -> None:
  """命令行入口。"""
  parser = argparse.ArgumentParser(description="代码块/公式中文英译词典产线")
  sub = parser.add_subparsers(dest="command", required=True)
  extract = sub.add_parser("extract", help="抽取含中文的代码块/公式文本节点并分组")
  extract.add_argument("--what", choices=sorted(TARGETS), default="code")
  extract.add_argument("--groups", type=int, default=8, help="分组数（默认 8）")
  extract.add_argument("--out", type=Path, required=True, help="输出目录")
  merge = sub.add_parser("merge", help="合并各组译文并写入词典")
  merge.add_argument("--what", choices=sorted(TARGETS), default="code")
  merge.add_argument("--parts", type=Path, required=True, help="parts/groupN.json 所在目录")
  args = parser.parse_args()
  if args.command == "extract":
    cmd_extract(args.what, args.groups, args.out)
  else:
    cmd_merge(args.what, args.parts)


if __name__ == "__main__":
  main()
