"""从仓库根 README.md 取书稿 PDF 的下载链接。

站点侧栏「下载 PDF」按钮与构建后的内容核对都以 ``README.md`` 里的链接引用定义
``[leetcuda-pdf]: <url>`` 为准：链接随版本发布变化，构建期现读现用，不在站点里硬编码。
解析失败要让构建红灯，不要静默降级成「没有按钮」。
"""
from __future__ import annotations

import re
from pathlib import Path

#: 链接引用定义行；正文里的用法（``[x][leetcuda-pdf]`` / ``[x](leetcuda-pdf)``）不算。
_DEFINITION = re.compile(r"^\[leetcuda-pdf\]:\s*(\S+)\s*$", re.MULTILINE)

#: 报错里给出的期望写法。
_EXPECTED = "[leetcuda-pdf]: https://…/leetcuda-YYYYMMDD.pdf"

#: 仓库根 README.md（相对本文件：docs/readthedocs/convert/ → 仓库根）。
REPO_README = Path(__file__).resolve().parents[3] / "README.md"


def pdf_link(readme: str | Path = REPO_README) -> str:
  """解析 README 里 ``[leetcuda-pdf]`` 链接定义的 URL。

  定义必须**唯一**：发新版时是改这一行，不是再添一行——两行都留着会让站点与核对
  一起自洽地指向旧 PDF，没人会发现。

  :param readme: 仓库根 README.md 路径。
  :returns: PDF 下载地址（已去掉可选的尖括号包裹）。
  :raises FileNotFoundError: 找不到 README。
  :raises ValueError: 定义不是恰好一处，或地址不是 http(s)。
  """
  path = Path(readme)
  if not path.is_file():
    raise FileNotFoundError(f"找不到 README：{path}（书稿 PDF 的下载链接由它提供）")
  text = path.read_text(encoding="utf-8", errors="replace")
  found = _DEFINITION.findall(text)
  if not found:
    raise ValueError(f"{path} 里没有 [leetcuda-pdf] 链接定义；期望写法：{_EXPECTED}")
  if len(found) > 1:
    raise ValueError(
      f"{path} 里有 {len(found)} 处 [leetcuda-pdf] 链接定义，只能有一处"
      f"（发新版改这一行，不要再添一行）：{found}")
  url = found[0]
  if url.startswith("<") and url.endswith(">"):
    url = url[1:-1]
  if not url.startswith(("http://", "https://")):
    raise ValueError(f"{path} 的 [leetcuda-pdf] 不是 http(s) 地址：{url!r}；期望写法：{_EXPECTED}")
  return url
