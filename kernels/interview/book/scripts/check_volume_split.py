#!/usr/bin/env python3
"""check_volume_split.py — 上下两册与全一册版的一致性检查。

分册用两处字面量把下册接在上册后面：``\\setcounter{chapter}{23}``（上册的编号章数）
与 ``\\setcounter{part}{4}``（上册包含的 part 数）。一旦有人往上册加章、调整 part 划分，
或在两册之间搬动章节而忘了同步，两册各自都自洽，却会出现「上册三十章、下册又从
第三十一章起」「附录从第九部分掉到第八部分」这类静默错位。本脚本把这件事变成可执行断言：

1. ``inputs``：上册 + 下册的 ``\\input`` 序列与 ``book.tex`` 完全一致（顺序也算）。
2. ``counters``：下册两处 ``\\setcounter`` 的取值等于上册实际的编号章数 / 部分数，
   并核对 ``\\chapter*``（不编号章，目前只有 ch19b）没有算进章数。
3. ``parts``（``--compare-aux``）：从三本 aux 的 ``\\contentsline {part}`` 抽部分号序列，
   断言「全一册 = 上册 + 下册」——``\\part`` 没有 label，第 4 项比对覆盖不到它。
4. ``aux``（``--compare-aux``）：``book.aux`` 与两册 aux 的 ``\\newlabel`` 编号逐条相等，
   即两册的章/图/表/公式编号与全一册版同源。

用法：
  python3 scripts/check_volume_split.py                 # 1 + 2（构建前即可跑）
  python3 scripts/check_volume_split.py --compare-aux   # 追加 3 + 4（需先编完三本且未清理 aux）
退出码：0 = 全部 PASS（SKIP 不计入），1 = 有 FAIL。
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent  # kernels/interview/book

_INPUT = re.compile(r"\\input\s*\{([^}]+)\}")
_SET_CHAPTER = re.compile(r"\\setcounter\{chapter\}\{(\d+)\}")
_SET_PART = re.compile(r"\\setcounter\{part\}\{(\d+)\}")
_PART = re.compile(r"\\part\s*\{")
_CHAPTER = re.compile(r"\\chapter(?![a-zA-Z])\s*(\*)?\s*(\[[^\]]*\])?\s*\{")
_NEWLABEL = re.compile(r"\\newlabel\{([^}]*)\}\{")
_CONTENTSLINE_PART = re.compile(r"\\contentsline \{part\}\{[^\n]*?\}\{\d+\}\{part\.(\d+)\}")

AUX_FILES = {
  "全一册": "book.aux",
  "上册": "book-vol1.aux",
  "下册": "book-vol2.aux",
}


def _body(path):
  """主文件/章节文件的正文（去掉整行 ``%`` 注释，避免注释掉的 input 混进来）。

  :param path: tex 文件路径。
  :returns: 去注释后的文本。
  """
  if not path.is_file():
    raise SystemExit(f"找不到文件：{path}")
  text = path.read_text(encoding="utf-8", errors="replace")
  return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))


def _inputs(path):
  """主文件里的 ``\\input`` 序列（去掉共享的 preamble.tex）。

  :param path: 主文件路径。
  :returns: 按出现顺序排列的 input 名。
  """
  return [name for name in _INPUT.findall(_body(path)) if name != "preamble.tex"]


def _chapter_counts(book, names):
  """统计一组章节文件里的编号章与不编号章数量。

  :param book: 书根目录。
  :param names: ``\\input`` 序列。
  :returns: ``(编号章数, \\chapter* 数)``。
  """
  numbered = 0
  starred = 0
  for name in names:
    text = _body(book / (name if name.endswith(".tex") else name + ".tex"))
    for match in _CHAPTER.finditer(text):
      if match.group(1):
        starred += 1
      else:
        numbered += 1
  return numbered, starred


def _match_brace(text, start):
  """从 ``text[start] == '{'`` 起做花括号配对，跳过 ``\\{`` / ``\\}`` 转义。

  :param text: 源文本。
  :param start: 左花括号下标。
  :returns: ``(花括号内容, 右花括号下标)``；不配对时内容为 None。
  """
  if start >= len(text) or text[start] != "{":
    return None, start
  depth = 0
  for index in range(start, len(text)):
    if text[index] not in "{}" or (index and text[index - 1] == "\\"):
      continue
    if text[index] == "{":
      depth += 1
    else:
      depth -= 1
      if depth == 0:
        return text[start + 1:index], index
  return None, len(text)


def _labels(path):
  """解析 aux 里的 ``\\newlabel``，取出 label → number 字段。

  :param path: aux 文件路径。
  :returns: ``{label: number}``。
  """
  text = path.read_text(encoding="utf-8", errors="replace")
  out = {}
  for match in _NEWLABEL.finditer(text):
    group, _ = _match_brace(text, match.end() - 1)
    if group is None:
      continue
    number, _ = _match_brace(group, group.find("{"))
    out[match.group(1)] = number if number is not None else ""
  return out


def _part_numbers(path):
  """从 aux 的 ``\\contentsline {part}`` 抽部分号序列。

  :param path: aux 文件路径。
  :returns: 部分号（int）列表，按目录出现顺序。
  """
  text = path.read_text(encoding="utf-8", errors="replace")
  return [int(number) for number in _CONTENTSLINE_PART.findall(text)]


def _missing_aux(book):
  """列出缺失的 aux 文件名。

  :param book: 书根目录。
  :returns: 缺失的 ``[(册名, 路径)]``。
  """
  return [(name, book / file) for name, file in AUX_FILES.items() if not (book / file).is_file()]


def check_inputs(book):
  """检查 1：上册 + 下册的 input 序列等于全一册版。

  :param book: 书根目录。
  :returns: 是否通过。
  """
  full = _inputs(book / "book.tex")
  vol1 = _inputs(book / "book-vol1.tex")
  vol2 = _inputs(book / "book-vol2.tex")
  split = vol1 + vol2
  if split == full:
    print(f"[OK] inputs: 上册 {len(vol1)} + 下册 {len(vol2)} = 全一册 {len(full)}")
    return True
  print(f"[FAIL] inputs: 上册 {len(vol1)} + 下册 {len(vol2)} = {len(split)}，全一册 {len(full)}")
  only_full = [name for name in full if name not in set(split)]
  only_split = [name for name in split if name not in set(full)]
  if only_full:
    print("  只在全一册版：", only_full[:10])
  if only_split:
    print("  只在分册版：", only_split[:10])
  if not only_full and not only_split:
    for index, (left, right) in enumerate(zip(full, split)):
      if left != right:
        print(f"  集合相同但顺序不同：第 {index + 1} 项 全一册 {left} / 分册 {right}")
        break
  return False


def check_counters(book):
  """检查 2：下册两处 ``\\setcounter`` 等于上册实际计数，且两册部分数相加等于全一册。

  :param book: 书根目录。
  :returns: 是否通过。
  """
  vol1_names = _inputs(book / "book-vol1.tex")
  numbered, starred = _chapter_counts(book, vol1_names)
  vol1_parts = len(_PART.findall(_body(book / "book-vol1.tex")))
  vol2_parts = len(_PART.findall(_body(book / "book-vol2.tex")))
  full_parts = len(_PART.findall(_body(book / "book.tex")))
  vol2 = _body(book / "book-vol2.tex")
  match_ch = _SET_CHAPTER.search(vol2)
  match_part = _SET_PART.search(vol2)
  if match_ch is None or match_part is None:
    print("[FAIL] counters: 下册缺 \\setcounter{chapter}/{part}")
    return False
  set_ch = int(match_ch.group(1))
  set_part = int(match_part.group(1))
  ok = True
  if set_ch == numbered:
    print(f"[OK] counters: chapter 接续 {set_ch}（上册编号章 {numbered}）")
  else:
    print(f"[FAIL] counters: \\setcounter{{chapter}}{{{set_ch}}} != 上册编号章 {numbered}")
    ok = False
  if set_part == vol1_parts:
    print(f"[OK] counters: part 接续 {set_part}（上册部分数 {vol1_parts}）")
  else:
    print(f"[FAIL] counters: \\setcounter{{part}}{{{set_part}}} != 上册部分数 {vol1_parts}")
    ok = False
  if vol1_parts + vol2_parts == full_parts:
    print(f"[OK] counters: 部分总数 上册 {vol1_parts} + 下册 {vol2_parts} = 全一册 {full_parts}")
  else:
    print(f"[FAIL] counters: 部分总数 上册 {vol1_parts} + 下册 {vol2_parts} != 全一册 {full_parts}")
    ok = False
  print(f"  上册 \\chapter* {starred} 个（不编号，不计入接续值）")
  return ok


def check_parts(book):
  """检查 3：三本的 ``\\contentsline {part}`` 部分号序列满足「全一册 = 上册 + 下册」。

  :param book: 书根目录。
  :returns: 是否通过；aux 缺失时返回 None（记为 SKIP）。
  """
  missing = _missing_aux(book)
  if missing:
    print(f"[SKIP] parts: 缺 {[name for name, _ in missing]}，先跑 ./build.sh 再比对")
    return None
  paths = {name: book / file for name, file in AUX_FILES.items()}
  full = _part_numbers(paths["全一册"])
  vol1 = _part_numbers(paths["上册"])
  vol2 = _part_numbers(paths["下册"])
  expect = list(range(1, len(full) + 1))
  if full == vol1 + vol2 == expect and full:
    print(f"[OK] parts: 部分号 上册 {vol1} + 下册 {vol2} = 全一册 {full}")
    return True
  print(f"[FAIL] parts: 部分号 上册 {vol1} + 下册 {vol2} != 全一册 {full}（或非 1..N 连续）")
  return False


def check_aux(book):
  """检查 4：全一册版与两册的 ``\\newlabel`` 编号逐条相等。

  :param book: 书根目录。
  :returns: 是否通过；aux 缺失时返回 None（记为 SKIP）。
  """
  missing = _missing_aux(book)
  if missing:
    print(f"[SKIP] aux: 缺 {[name for name, _ in missing]}，先跑 ./build.sh 再比对")
    return None
  paths = {name: book / file for name, file in AUX_FILES.items()}
  full, vol1, vol2 = (_labels(paths[name]) for name in ("全一册", "上册", "下册"))
  split = {**vol1, **vol2}
  ok = True
  dup = sorted(set(vol1) & set(vol2))
  if dup:
    print(f"[FAIL] aux: 两册重复 label {dup[:10]}")
    ok = False
  if set(full) != set(split):
    only_full = sorted(set(full) - set(split))
    only_split = sorted(set(split) - set(full))
    print(f"[FAIL] aux: label 集合不等（只在全一册 {len(only_full)}，只在分册 {len(only_split)}）")
    if only_full:
      print("  只在全一册：", only_full[:10])
    if only_split:
      print("  只在分册：", only_split[:10])
    ok = False
  diff = [
    (key, full.get(key), split.get(key))
    for key in sorted(set(full) & set(split))
    if full.get(key) != split.get(key)
  ]
  if diff:
    print(f"[FAIL] aux: {len(diff)} 个 label 的编号不一致")
    for key, left, right in diff[:10]:
      print(f"  {key}: 全一册 {left!r} / 分册 {right!r}")
    ok = False
  if ok:
    print(f"[OK] aux: {len(full)} 个 label 编号逐条一致（上册 {len(vol1)} + 下册 {len(vol2)}）")
  return ok


def main():
  """入口。

  :returns: 进程退出码。
  """
  parser = argparse.ArgumentParser(description="上下册与全一册版的一致性检查")
  parser.add_argument("--book", default=str(BOOK), help="书根目录（默认 kernels/interview/book）")
  parser.add_argument("--compare-aux", action="store_true", help="追加部分号与 aux 编号比对")
  args = parser.parse_args()
  book = Path(args.book).resolve()

  results = [check_inputs(book), check_counters(book)]
  if args.compare_aux:
    results += [check_parts(book), check_aux(book)]

  fails = sum(1 for item in results if item is False)
  skips = sum(1 for item in results if item is None)
  passes = len(results) - fails - skips
  print(f"== {passes} pass / {fails} fail / {skips} skip")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
