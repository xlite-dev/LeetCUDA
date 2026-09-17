# Example: drawio 重建 smoke test（m1 存储分层架构图）

2026-09-11 drawio 管线 smoke test 的固化样例，演示「知乎图 → 可编辑 drawio」的标准流程。
正式重建任务（RFC-H）以此为例。

## 文件

| 文件 | 说明 |
|---|---|
| `m1.drawio` | 重建件：81 cells **全原生元素**（0 image/svg cell），中文存储分层架构图 |
| `m1_preview.png` | headless CLI 导出预览（238K）——**CJK 渲染已验证无豆腐块** |
| `m1.audit.md` | inventory + 审查记录 + smoke test 结论（工具链/执行模式） |

参考原图：`/workspace/dev/vipshop/.github/skills/drawio-reconstruction/examples/m1.png`（不入库，用路径引用）。

## 复现命令

```bash
# 1. 校验 XML（81 cells / 0 image/svg）
python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/check_drawio.py m1.drawio

# 2. headless 导出（wrapper = xvfb-run + --no-sandbox，dbus 报错为噪音）
DRAWIO_PATH=/usr/local/bin/drawio-headless \
  python3 /workspace/dev/vipshop/.github/skills/drawio-reconstruction/scripts/export_drawio.py m1.drawio out.png

# 3. 逐字核对用的放大 crop（bbox 清单见 m1.audit.md；正式重建必做）
python3 - <<'EOF'
from PIL import Image
im = Image.open('<参考原图路径>')
im.crop((x, y, w, h)).resize((w*3, h*3), Image.LANCZOS).save('crop.png')
EOF
```

## 关键结论（详见 m1.audit.md）

1. 工具链全通：deb 安装（Ubuntu noble 无 apt 包）→ `drawio-headless` wrapper → `export_drawio.py`/`check_drawio.py`。
2. **执行模式（用户拍板）**：主 agent 直接做 inventory+重建+导出+自审，**不派 task agent**（多 agent 闭环实测卡死）；自审需在 audit 标注。
3. 本样例保真度为「结构级」（布局/色彩/图标/箭头/主要文本正确，小字为合理占位）；正式重建须对照放大 crop **逐字核对**后替换占位文本。
4. 水印规则：知乎图的水印/作者角标/平台 logo 一律标记为非内容元素，不得进入重建产物（本样例为 skill 示例图，无水印场景）。
