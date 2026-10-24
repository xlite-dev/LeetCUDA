#!/usr/bin/env python3
"""Scan LeetCUDA book chapters for high-risk correctness claims.

This is a candidate finder, not a judge. Every hit must be entered into a
claim ledger and checked against a definition, header, PTX section, or source
call site. A clean scan does not prove the prose is correct.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# (category, short reason, regex)
PATTERNS = [
    (
        "L0-logic",
        "derivation/implication",
        re.compile(
            r"由此可得|可推出|两边(?:同)?(?:乘|除|加|减)|代入|约去|消去|"
            r"当且仅当|充分必要|充要条件|"
            r"因此[^，。；\n]{0,24}(?:等于|就是|成立)"
        ),
    ),
    (
        "L1-global",
        "global/iff claim",
        re.compile(
            r"size\s*==\s*cosize|cosize\s*==\s*size|双射|当且仅当|"
            r"任何\s*D|在所有\s*D|必然|恰好|所有[^，。；\n]{0,24}都"
        ),
    ),
    (
        "L2-wait",
        "cp.async wait semantics",
        re.compile(
            r"wait\s*<\s*\d|wait_group|不存在的组|等很久|最多[^，。；\n]{0,12}未完成"
        ),
    ),
    (
        "L2-tma",
        "TMA/descriptor/hardware contract",
        re.compile(
            r"OOB|zero-fill|zero fill|cluster|_1\s*\{|make_tma_copy|"
            r"1024\s*(?:\\,)?\s*B|SW64|SW128|alignment|对齐"
        ),
    ),
    (
        "L2-api",
        "CuTe API semantics",
        re.compile(
            r"partition_fragment_B|convert_type|convert_layout_acc_Aregs|"
            r"Aregs|cute::gemm|gemm_ss|gemm_rs|partition_S"
        ),
    ),
    (
        "L3-register",
        "register/resource accounting",
        re.compile(
            r"可用寄存器的一半|256\s*(?:个)?\s*(?:f32|寄存器)|寄存器上限|"
            r"spill|每线程[^，。；\n]{0,12}寄存器|CTA\s*总|私有副本|共享寄存器|"
            r"逻辑值|逻辑累加值"
        ),
    ),
    (
        "L3-type",
        "conversion vs reinterpretation",
        re.compile(
            r"零拷贝|原地(?:覆盖|重排|复用)|不新增|新\s*f16|新 fragment|"
            r"convert_type|同一批[^，。；\n]{0,12}寄存器"
        ),
    ),
    (
        "L3-batch",
        "batch/divisibility capacity",
        re.compile(r"VCPB|VChunksPerBatch|只够一个|只接受|整批|整除"),
    ),
    (
        "L4-figure",
        "figure/caption/QA residue",
        re.compile(
            r"就地覆盖为\s*P|S\s*/\s*P|Phase 2 就地|rows?\s*0--1|"
            r"K1[^，。；\n]{0,10}复写|Q\s*跨\s*q-tile|"
            r"消除[^，。；\n]{0,8}尾损|任何[^，。；\n]{0,8}最优"
        ),
    ),
]

SKIP_DIRS = {".git", ".tmp", "build", "dist", "__pycache__"}
DEFAULT_SUFFIXES = {".tex"}


def iter_files(paths: list[Path], suffixes: set[str]):
    for path in paths:
        if path.is_file():
            if path.suffix in suffixes:
                yield path
            continue
        if not path.is_dir():
            print(f"warning: skip missing path: {path}", file=sys.stderr)
            continue
        for child in sorted(path.rglob("*")):
            if any(part in SKIP_DIRS for part in child.parts):
                continue
            if child.is_file() and child.suffix in suffixes:
                yield child


def scan_file(path: Path):
    hits = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, 1):
        seen = set()
        for category, reason, regex in PATTERNS:
            match = regex.search(line)
            if not match or category in seen:
                continue
            seen.add(category)
            text = line.strip()
            if len(text) > 220:
                text = text[:217] + "..."
            hits.append(
                {
                    "file": str(path),
                    "line": lineno,
                    "category": category,
                    "reason": reason,
                    "match": match.group(0),
                    "text": text,
                }
            )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan book .tex files for high-risk correctness claims "
        "(candidate list only; verify every hit)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="tex files or directories (e.g. book/chapters)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of text"
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        help="file suffix to scan; default .tex, repeatable",
    )
    args = parser.parse_args()

    suffixes = set(args.suffix or DEFAULT_SUFFIXES)
    files = list(iter_files(args.paths, suffixes))
    if not files:
        print("no files matched", file=sys.stderr)
        return 1

    all_hits = []
    for path in files:
        all_hits.extend(scan_file(path))

    counts = Counter(hit["category"] for hit in all_hits)
    if args.json:
        print(
            json.dumps(
                {
                    "files": [str(p) for p in files],
                    "hit_count": len(all_hits),
                    "by_category": dict(sorted(counts.items())),
                    "hits": all_hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"scanned {len(files)} file(s), {len(all_hits)} candidate line(s)")
    for hit in all_hits:
        print(
            f"{hit['file']}:{hit['line']}: "
            f"[{hit['category']}] {hit['text']}"
        )
    print("\nby category:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    print(
        "\nNOTE: candidates only. Enter each hit into the claim ledger and "
        "verify against definition/header/PTX/source; a clean scan is not "
        "proof of correctness."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
