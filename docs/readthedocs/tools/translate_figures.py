"""图内文字的英译词典生成工具（英文图集用，构建时一次性跑）。

英文图集需要 ``i18n/figures-en.json``（掩码后的中文单元 → 掩码后的英文单元）。本工具
把片段里的节点文字掩码后分批送 Google 翻译（与站点英文模式同一个引擎，风格一致），
再逐条校验占位符完整、可还原。结果写回词典、按批保存，可中断续跑。

    python -m tools.translate_figures --report                 # 只看覆盖情况
    python -m tools.translate_figures --proxy http://127.0.0.1:7890
    python -m tools.translate_figures --only '时间' --only '有效'

之所以不手工翻：图内文字有 1700 多条（多数是整句），人工翻不现实；也不用整篇 MT
替换页面文字——那只覆盖 HTML，图片是做不到的，这正是本工具存在的理由。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert import figtext  # noqa: E402

#: 请求之间的间隔（秒）：别把免费接口打爆。
_DELAY = 0.3

#: 批量分隔符：实测机器翻译会原样保留这对括号。
_SEPARATOR = " ⟪SEP⟫ "

#: 单条翻译失败时的重试次数。
_RETRIES = 3

_ENDPOINT = "https://translate.googleapis.com/translate_a/single"


def translate_batch(texts: list[str], proxy: str = "") -> list[str]:
  """把一批文本合并成一次请求交给 Google 翻译。

  :param texts: 待翻译文本（已掩码）。
  :param proxy: 代理地址，空串表示直连。
  :returns: 与输入等长的译文列表。
  :raises ValueError: 返回的分段数与输入不一致。
  """
  return _split(_request(_SEPARATOR.join(texts), proxy), len(texts))


def translate_one(text: str, proxy: str = "") -> str:
  """翻译单条文本（批量失败时的回退路径）。

  :param text: 待翻译文本（已掩码）。
  :param proxy: 代理地址。
  :returns: 译文。
  """
  return _request(text, proxy)


def _request(text: str, proxy: str) -> str:
  """发一次翻译请求。

  :param text: 待翻译文本。
  :param proxy: 代理地址。
  :returns: 译文原文。
  :raises urllib.error.URLError: 网络失败。
  """
  params = [("client", "gtx"), ("sl", "zh-CN"), ("tl", "en"), ("dt", "t"), ("q", text)]
  url = _ENDPOINT + "?" + urllib.parse.urlencode(params)
  handlers = []
  if proxy:
    handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
  opener = urllib.request.build_opener(*handlers)
  request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
  with opener.open(request, timeout=60) as response:
    payload = json.loads(response.read().decode("utf-8"))
  return "".join(item[0] for item in payload[0] if item and item[0])


def _split(text: str, expected: int) -> list[str]:
  """按分隔符切回译文。

  :param text: 译文原文。
  :param expected: 期望段数。
  :returns: 译文列表。
  :raises ValueError: 段数不符。
  """
  parts = [part.strip() for part in text.split("⟪SEP⟫")]
  if len(parts) != expected:
    raise ValueError(f"分段数不符：期望 {expected}，得到 {len(parts)}")
  return parts


def valid(source: str, target: str) -> bool:
  """校验译文可还原且占位符完整。

  :param source: 中文原文（掩码后）。
  :param target: 英文译文（掩码后）。
  :returns: 可用为 True。
  """
  want = collections.Counter(figtext._PLACEHOLDER_RE.findall(source))
  got = collections.Counter(figtext._PLACEHOLDER_RE.findall(target))
  if want != got:
    return False
  try:
    highest = max((int(key) for key in want), default=-1)
    figtext.unmask(target, [""] * (highest + 1))
  except (IndexError, ValueError):
    return False
  return True


def main() -> int:
  """命令行入口。

  :returns: 退出码。
  """
  parser = argparse.ArgumentParser(description="生成图内文字的英译词典")
  parser.add_argument("--work", default="build/tmp/tikz", help="TikZ 片段目录")
  parser.add_argument("--dictionary", default="i18n/figures-en.json", help="词典路径")
  parser.add_argument("--proxy", default="", help="代理地址，例如 http://127.0.0.1:7890")
  parser.add_argument("--batch", type=int, default=40, help="每次请求合并多少条")
  parser.add_argument("--only", action="append", default=[], help="只翻译这些单元（可重复）")
  parser.add_argument("--limit", type=int, default=0, help="最多翻译多少条（0 = 不限）")
  parser.add_argument("--report", action="store_true", help="只报告覆盖情况，不发请求")
  args = parser.parse_args()

  snippets = sorted(Path(args.work).glob("*.tex"))
  if not snippets:
    print(f"没有片段：{args.work}（先跑 python -m convert）")
    return 1
  dictionary_path = Path(args.dictionary)
  dictionary = figtext.load_dictionary(dictionary_path)
  covered, total, missing = figtext.coverage(snippets, dictionary)
  print(f"词典 {dictionary_path}：{len(dictionary)} 条")
  print(f"单元覆盖：{covered}/{total}（{covered * 100 // max(total, 1)}%），未覆盖 {len(missing)}")

  if args.report:
    for unit in missing[:15]:
      print("  未覆盖:", unit[:70])
    return 0

  pending = args.only or [unit for unit in missing if figtext._CJK_RE.search(unit)]
  if args.limit:
    pending = pending[:args.limit]
  if not pending:
    print("没有待翻译的单元。")
    return 0
  print(f"待翻译 {len(pending)} 条，按 {args.batch} 条一批发送…")

  done = failed = 0
  for start in range(0, len(pending), args.batch):
    batch = pending[start:start + args.batch]
    try:
      results = translate_batch(batch, args.proxy)
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
      print(f"  第 {start // args.batch + 1} 批失败（{type(exc).__name__}: {exc}），逐条重试")
      results = []
      for unit in batch:
        results.append(_retry(unit, args.proxy))
        time.sleep(_DELAY)
    for unit, english in zip(batch, results):
      if english and valid(unit, english):
        dictionary[unit] = english
        done += 1
      else:
        failed += 1
        print(f"  跳过（占位符不符）：{unit[:50]!r} → {english[:50]!r}")
    figtext.save_dictionary(dictionary_path, dictionary)
    time.sleep(_DELAY)
    print(f"  进度 {done + failed}/{len(pending)}（成功 {done}）")

  figtext.save_dictionary(dictionary_path, dictionary)
  covered, total, _missing = figtext.coverage(snippets, dictionary)
  print(f"完成：词典 {len(dictionary)} 条，单元覆盖 {covered}/{total}，失败 {failed}")
  return 0


def _retry(unit: str, proxy: str) -> str:
  """单条翻译并重试。

  :param unit: 待翻译单元。
  :param proxy: 代理地址。
  :returns: 译文（失败时为空串）。
  """
  for attempt in range(_RETRIES):
    try:
      return translate_one(unit, proxy)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
      if attempt == _RETRIES - 1:
        print(f"  单条失败：{unit[:40]!r}（{type(exc).__name__}）")
      time.sleep(_DELAY * (attempt + 2))
  return ""


if __name__ == "__main__":
  raise SystemExit(main())
