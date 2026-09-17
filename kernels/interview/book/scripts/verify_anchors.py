#!/usr/bin/env python3
"""verify_anchors.py — 校验源码冻结状态（RFC-A.4）。

校验三层：
1. 文件 SHA256 与 anchors.yaml 登记一致（源码冻结，BOOK_PLAN §5.3）
2. 每章引用区间首/末行文本与锚点一致（防行号漂移）
3. 区间内 #if/#ifdef/#ifndef 与 #endif 成对（linerange 裁剪完整性）

用法：
  python3 verify_anchors.py              # 全部章节
  python3 verify_anchors.py --ch ch04    # 单章（可逗号分隔多个）
退出码：0=全绿，1=有失败项。
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE / ".." / ".."  # kernels/interview/

IF_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
ENDIF_RE = re.compile(r"^\s*#\s*endif\b")


def check_files(anchor):
    fails = []
    for name, meta in anchor["files"].items():
        p = SRC / name
        if not p.exists():
            fails.append(f"MISSING file {name}")
            continue
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != meta["sha256"]:
            fails.append(f"{name}: SHA256 mismatch (frozen={meta['sha256'][:12]}, now={sha[:12]})")
        nlines = len(data.decode("utf-8", errors="replace").splitlines())
        if nlines != meta["lines"]:
            fails.append(f"{name}: line count {nlines} != frozen {meta['lines']}")
    return fails


def check_ref(lines, ref, tag):
    fails = []
    s, e = ref["start"], ref["end"]
    if not (1 <= s <= e <= len(lines)):
        return [f"{tag}: range {s}-{e} out of bounds (file has {len(lines)} lines)"]
    first = lines[s - 1].strip()
    last = lines[e - 1].strip()
    if first != ref["first_anchor"]:
        fails.append(f"{tag}: first line mismatch\n    expect: {ref['first_anchor'][:80]}\n    actual: {first[:80]}")
    if last != ref["last_anchor"]:
        fails.append(f"{tag}: last line mismatch\n    expect: {ref['last_anchor'][:80]}\n    actual: {last[:80]}")
    n_if = sum(1 for ln in lines[s - 1 : e] if IF_RE.match(ln))
    n_end = sum(1 for ln in lines[s - 1 : e] if ENDIF_RE.match(ln))
    if n_if != n_end:
        fails.append(f"{tag}: unbalanced directives in range: {n_if} #if* vs {n_end} #endif")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", default="", help="comma-separated chapter ids, e.g. ch04,ch12 (default: all)")
    args = ap.parse_args()

    anchor = yaml.safe_load((HERE / "anchors.yaml").read_text(encoding="utf-8"))
    want = [c.strip() for c in args.ch.split(",") if c.strip()] or sorted(anchor["chapters"])

    total_fail = 0
    file_fails = check_files(anchor)
    if file_fails:
        total_fail += len(file_fails)
        for m in file_fails:
            print(f"FAIL [files] {m}")

    lines_cache = {}
    for ch in want:
        refs = anchor["chapters"].get(ch)
        if refs is None:
            print(f"FAIL {ch}: not registered in anchors.yaml")
            total_fail += 1
            continue
        fails = []
        for i, ref in enumerate(refs["refs"]):
            f = ref["file"]
            if f not in lines_cache:
                lines_cache[f] = (SRC / f).read_text(encoding="utf-8").splitlines()
            fails += check_ref(lines_cache[f], ref, f"{ch}/{f}[{ref['start']}-{ref['end']}]#{i + 1}")
        if fails:
            total_fail += len(fails)
            for m in fails:
                print(f"FAIL [{ch}] {m}")
        else:
            print(f"OK   {ch} ({len(refs['refs'])} refs)")

    print(f"\n{'ALL GREEN' if total_fail == 0 else f'{total_fail} FAILURES'} ({len(want)} chapters checked)")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
