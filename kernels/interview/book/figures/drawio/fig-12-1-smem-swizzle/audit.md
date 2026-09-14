# FIG-12-1 smem XOR swizzle — audit

- 生成器: `gen.py`（本目录）；导出 `/usr/local/bin/drawio-headless -x -s 3`
- 迭代: 2 轮（v1 右侧结论卡文字溢出框高 → v2 拆为 4 个独立框修复）
- 验收: XML 特征（diagram id=fig12-1 + 标题）+ PNG 尺寸 3242x2254 + OCR 全文复核
  （两张 8x3 表数据、quad 序列带 x2、4 个结论框无截断、XOR 箭头标签）
- 执行模式: coordinator 自审（本会话 view_image 显示层缓存故障，改用 XML+OCR+PIL 三件套）
- 规格: ch12 ASCII 图 + caption；蓝本 common.cuh L185-235 布局表
- 渲染铁律遵从: 格宽 78px（两位数）、CJK 无 bold、value 无 &#10;、箭头显式 mxPoint
