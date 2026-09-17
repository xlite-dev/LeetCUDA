# FIG-20-1 audit（colex 线性化表）

- 生成器：`.tmp/drawio/gen_fig20_1.py`（87+2 cells，可复现）
- 迭代：5 轮。C1 数带贴表+箭头穿格+下标方框 → C2 数带右移 RX=470+waypoint(x=405) → C3 `&#10;` 换行破坏 HTML 渲染（title 拆 cell）→ C4 `<sub>` 标签在 CJK 混排下仍方框 → C5 根因=**bold+CJK 混排触发字体 fallback 缺字形**，CJK 去 bold + 公式/中文分 cell 后清零。
- 渲染铁律（后续图复用）：① value 禁用 `&#10;`（拆 cell）② CJK 文本禁 bold/italic ③ 下标用纯文本 c0/c1 ④ unicode 下标 U+2080 不可用 ⑤ 高亮色蓝 #2171b5 / 绿 #2e8540，粗边框同色。
- 导出：drawio-headless -s 3（300dpi），2640×1350px。
