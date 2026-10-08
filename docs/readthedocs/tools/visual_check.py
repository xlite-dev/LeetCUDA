"""逐页视觉验收：真浏览器渲染 + 截图 + 公式错误抓取。

`convert.verify` 查的是文件层面的痕迹（锚点、表格数、字面残迹），查不出「页面上
到底长什么样」。本脚本用无头 Chromium 打开每个页面，等 MathJax 渲染完，然后把三
类证据落盘：

1. **公式错误**：Count ``mjx-merror`` 元素并取出它的 TeX 报错原文——这是「公式渲染
   崩溃」最直接的信号，比看截图更早暴露问题；
2. **页面字面残迹**：排除 ``pre``/``code`` 后统计 ``{=html}``、孤立 ``$$``、
   ``\\begin{`` 之类的可见文本；
3. **整页截图**：供人工肉眼核对排版（可用 :func:`contact_sheet` 合并成接触表）。

依赖（可选，不进站点构建）：``pip install playwright && python -m playwright install
chromium``。用法：

    python -m tools.visual_check --html build/html --out build/shots/chapters
    python -m tools.visual_check --html build/html --sheet build/shots/sheet.png
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

#: 页面里不应出现的可见字面文本（排除代码块、公式容器与隐藏元素）。
ARTIFACT_PATTERNS = {
  "pandoc 原始 HTML 残迹": "{=html}",
  "pandoc 标题属性": "{#",
  "冒号围栏": ":::",
  "字面 \\begin{": "\\begin{",
  "字面 \\ref{": "\\ref{",
  "字面 $$": "$$",
  "未渲染的行内公式": "\\(",
  "未渲染的显示公式": "\\[",
  "未渲染的强调标记": "**",
}

#: 检查可见字面文本时排除的内容：代码块、脚本、MathJax 产物，以及 Sphinx 用来承载
#: 公式原文的 ``.math`` 容器（那里的 ``\begin{…}`` 是 MathJax 的输入，不是可见文本）。
VISIBLE_TEXT_JS = """
() => {
  const skipTags = new Set(['PRE', 'CODE', 'SCRIPT', 'STYLE', 'MJX-CONTAINER', 'MJX-MATH',
                            'MJX-MERROR', 'MJX-ASSISTIVE-MML']);
  const parts = [];
  const hidden = (element) => {
    if (!element.offsetParent && element.tagName !== 'BODY') return true;
    const style = window.getComputedStyle(element);
    return style.display === 'none' || style.visibility === 'hidden';
  };
  const walk = (node) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        parts.push(child.nodeValue);
        continue;
      }
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      const tag = child.tagName.toUpperCase();
      if (skipTags.has(tag)) continue;
      if (child.classList && (child.classList.contains('math') || child.classList.contains('MathJax'))) continue;
      if (hidden(child)) continue;
      walk(child);
    }
  };
  walk(document.body);
  return parts.join('');
}
"""

MATH_STATE_JS = """
() => {
  const errors = [...document.querySelectorAll('mjx-merror')].map(
    (node) => (node.getAttribute('data-mjx-error') || node.textContent || '').trim());
  return {
    containers: document.querySelectorAll('mjx-container').length,
    display: document.querySelectorAll('mjx-container[display="true"]').length,
    errors: errors,
  };
}
"""

#: 版式几何：图片与表格是否居中、有没有元素撑出正文栏（见 README「版式」一节）。
LAYOUT_JS = """
() => {
  const content = document.querySelector('.rst-content .section')
    || document.querySelector('.rst-content');
  if (!content) return null;
  const column = content.getBoundingClientRect();
  const offCenter = (rect) => Math.abs((rect.left - column.left) - (column.right - rect.right)) > 3;
  const out = {tables: 0, tables_off_center: 0, figures: 0, figures_off_center: 0,
               wide_math: 0, math_escaped: 0, page_overflow: 0};
  for (const table of content.querySelectorAll('table')) {
    const rect = table.getBoundingClientRect();
    out.tables += 1;
    if (rect.width <= column.width + 1 && offCenter(rect)) out.tables_off_center += 1;
  }
  for (const img of content.querySelectorAll('p > img:only-child')) {
    out.figures += 1;
    if (offCenter(img.getBoundingClientRect())) out.figures_off_center += 1;
  }
  for (const node of content.querySelectorAll('mjx-container[display="true"]')) {
    if (node.scrollWidth > node.clientWidth + 2) out.wide_math += 1;
    if (node.getBoundingClientRect().right > column.right + 2) out.math_escaped += 1;
  }
  if (document.documentElement.scrollWidth > window.innerWidth + 2) out.page_overflow = 1;
  return out;
}
"""


@dataclass
class PageReport:
  """单页验收结果。

  :param name: 页面名（不含扩展名）。
  :param math_containers: 渲染出的公式个数。
  :param math_display: 其中显示公式个数。
  :param math_errors: MathJax 报错原文。
  :param artifacts: 命中的字面残迹 → 出现次数。
  :param console_errors: 控制台 error 文本。
  :param failed_requests: 加载失败的资源。
  :param mathjax_loaded: 页面是否加载到 MathJax。
  :param screenshot: 截图路径。
  :param seconds: 该页耗时。
  """
  name: str
  math_containers: int = 0
  math_display: int = 0
  math_errors: list[str] = field(default_factory=list)
  artifacts: dict[str, int] = field(default_factory=dict)
  console_errors: list[str] = field(default_factory=list)
  failed_requests: list[str] = field(default_factory=list)
  mathjax_loaded: bool = True
  layout: dict[str, int] = field(default_factory=dict)
  screenshot: str = ""
  seconds: float = 0.0

  @property
  def ok(self) -> bool:
    """是否通过。

    判据是「页面上没有可见的未渲染数学、没有公式报错、没有字面残迹、版式没有跑偏」
    ——而不是「MathJax 是否加载」：侧栏含数学时，没有自身公式的页面（``search`` 等）
    也不会加载 MathJax，那是 Sphinx 的正常行为。

    :returns: 全部通过为 True。
    """
    if self.math_errors or self.artifacts or self.console_errors:
      return False
    return not (self.layout.get("tables_off_center") or self.layout.get("figures_off_center")
                or self.layout.get("math_escaped") or self.layout.get("page_overflow"))


def serve(directory: Path) -> tuple[str, socketserver.TCPServer]:
  """在后台线程里起一个静态文件服务（MathJax 走 CDN，需要 http 协议）。

  :param directory: 站点目录。
  :returns: ``(基地址, 服务对象)``。
  """
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
  httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
  httpd.allow_reuse_address = True
  thread = threading.Thread(target=httpd.serve_forever, daemon=True)
  thread.start()
  host, port = httpd.server_address[:2]
  return f"http://{host}:{port}", httpd


def check_pages(
  html_dir: Path,
  out_dir: Path,
  width: int = 1400,
  only: list[str] | None = None,
  timeout_ms: int = 30000,
  proxy: str = "",
) -> list[PageReport]:
  """逐页渲染、截图并抓取证据。

  :param html_dir: Sphinx 输出目录。
  :param out_dir: 截图与报告输出目录。
  :param width: 视口宽度。
  :param only: 只检查这些页面（不含扩展名），None 表示全部。
  :param timeout_ms: 单页渲染超时。
  :param proxy: 代理地址（MathJax 走 CDN，需要能出网；本地服务自动绕过）。
  :returns: 每页结果。
  """
  from playwright.sync_api import sync_playwright

  out_dir.mkdir(parents=True, exist_ok=True)
  pages = sorted(path for path in html_dir.glob("*.html"))
  if only:
    pages = [path for path in pages if path.stem in set(only)]

  launch_args = ["--no-sandbox"]
  if proxy:
    launch_args.extend([f"--proxy-server={proxy}", "--proxy-bypass-list=127.0.0.1;localhost"])

  base, httpd = serve(html_dir)
  reports: list[PageReport] = []
  try:
    with sync_playwright() as playwright:
      browser = playwright.chromium.launch(args=launch_args)
      context = browser.new_context(viewport={"width": width, "height": 1000})
      for path in pages:
        page = context.new_page()
        console_errors: list[str] = []
        failed_requests: list[str] = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))
        page.on("requestfailed",
                lambda request: failed_requests.append(f"{request.url} {request.failure}"))
        started = time.monotonic()
        page.goto(f"{base}/{path.name}", wait_until="load", timeout=timeout_ms)
        # 页面本身没有公式时 Sphinx 不会加载 MathJax，这不算问题。
        needs_math = page.evaluate("() => !!document.querySelector('.math')")
        mathjax_loaded = page.evaluate("() => !!window.MathJax") or not needs_math
        if page.evaluate("() => !!window.MathJax"):
          try:
            page.evaluate("() => MathJax.startup.promise")
          except Exception:  # MathJax 报错时 promise 不会 resolve，这里不该中断验收
            pass
        page.wait_for_timeout(200)
        state = page.evaluate(MATH_STATE_JS)
        text = page.evaluate(VISIBLE_TEXT_JS)
        layout = page.evaluate(LAYOUT_JS) or {}
        shot = out_dir / f"{path.stem}.png"
        page.screenshot(path=str(shot), full_page=True)
        report = PageReport(
          name=path.stem,
          math_containers=state["containers"],
          math_display=state["display"],
          math_errors=[error for error in state["errors"] if error],
          artifacts={name: text.count(token) for name, token in ARTIFACT_PATTERNS.items()
                     if text.count(token)},
          console_errors=[error for error in console_errors if "favicon" not in error][:5],
          failed_requests=[item for item in failed_requests if "favicon" not in item][:5],
          mathjax_loaded=mathjax_loaded,
          layout=layout,
          screenshot=str(shot),
          seconds=round(time.monotonic() - started, 1),
        )
        reports.append(report)
        mark = "OK  " if report.ok else "FAIL"
        print(f"[{mark}] {report.name}: MathJax={report.mathjax_loaded} "
              f"公式 {report.math_containers}（显示 {report.math_display}），"
              f"公式错误 {len(report.math_errors)}，残迹 {report.artifacts}，"
              f"资源失败 {len(report.failed_requests)}，"
              f"表 {report.layout.get('tables', 0)} 张/偏离 {report.layout.get('tables_off_center', 0)}，"
              f"图 {report.layout.get('figures', 0)} 张/偏离 {report.layout.get('figures_off_center', 0)}，"
              f"越界公式 {report.layout.get('math_escaped', 0)}，{report.seconds}s")
        page.close()
      browser.close()
  finally:
    httpd.shutdown()

  (out_dir / "visual-report.json").write_text(
    json.dumps([report.__dict__ for report in reports], ensure_ascii=False, indent=2),
    encoding="utf-8")
  return reports


def contact_sheet(screenshots: list[Path], target: Path, columns: int = 3, scale: float = 0.32) -> Path:
  """把多张整页截图拼成一张接触表（便于逐页肉眼核对）。

  :param screenshots: 截图路径。

  :param target: 输出图片。
  :param columns: 每行几张。
  :param scale: 缩放比例。
  :returns: 输出路径。
  """
  from PIL import Image, ImageDraw

  thumbs = []
  for path in screenshots:
    image = Image.open(path).convert("RGB")
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    thumbs.append((path.stem, image.resize(size)))
  if not thumbs:
    raise ValueError("没有截图可拼")
  cell_w = max(thumb.width for _, thumb in thumbs)
  cell_h = max(thumb.height for _, thumb in thumbs) + 18
  rows = (len(thumbs) + columns - 1) // columns
  sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
  draw = ImageDraw.Draw(sheet)
  for index, (name, thumb) in enumerate(thumbs):
    x = (index % columns) * cell_w
    y = (index // columns) * cell_h
    sheet.paste(thumb, (x, y + 18))
    draw.text((x + 4, y + 3), name, fill="black")
  sheet.save(target)
  return target


def main() -> int:
  """命令行入口。

  :returns: 进程退出码（0 表示全部通过）。
  """
  root = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description="逐页视觉验收")
  parser.add_argument("--html", default=str(root / "build" / "html"), help="Sphinx 输出目录")
  parser.add_argument("--out", default=str(root / "build" / "shots" / "chapters"), help="截图输出目录")
  parser.add_argument("--only", default="", help="只检查指定页（逗号分隔）")
  parser.add_argument("--width", type=int, default=1400, help="视口宽度")
  parser.add_argument("--sheet", default="", help="把截图拼成接触表并保存到该路径")
  parser.add_argument("--sheet-columns", type=int, default=3, help="接触表列数")
  parser.add_argument("--proxy", default="", help="代理地址，例如 http://127.0.0.1:7890")
  args = parser.parse_args()

  only = [item.strip() for item in args.only.split(",") if item.strip()]
  reports = check_pages(Path(args.html).resolve(), Path(args.out).resolve(), args.width, only,
                        proxy=args.proxy)
  failed = [report for report in reports if not report.ok]
  print(f"\n合计 {len(reports)} 页，需处理 {len(failed)} 页：{[report.name for report in failed]}")

  if args.sheet:
    shots = [Path(report.screenshot) for report in reports]
    print(f"接触表：{contact_sheet(shots, Path(args.sheet).resolve(), args.sheet_columns)}")
  return 0 if not failed else 1


if __name__ == "__main__":
  raise SystemExit(main())
