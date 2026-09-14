#!/usr/bin/env python3
"""几何碰撞检测：text 与 box 重叠对，并按 z-order 区分"被覆盖"与普通交叠。

- COVER: text 先于实心 box 定义（后画 box 盖住文字）-> 必须修
- HIT:   其他相交（文字画在框上/框内）-> 需人工判断是否合法（卡片内文字属正常）
"""
import re, sys

def scan(path, tol=2):
    s = open(path).read()
    els = []
    for oi, m in enumerate(re.finditer(
        r'<mxCell (?:id="([^"]*)" )?value="([^"]*)" style="([^"]*)" vertex="1"[^>]*>'
        r'<mxGeometry x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"', s)):
        i, v, st, x, y, w, h = m.groups()
        is_text = st.startswith('text')
        solid = 'fillColor=none' not in st and 'fill=none' not in st
        els.append((i or '-', float(x), float(y), float(w), float(h), is_text, v, oi, solid))
    covers, hits = [], []
    for t in els:
        if not t[5]:
            continue
        tx1, ty1, tx2, ty2 = t[1], t[2], t[1] + t[3], t[2] + t[4]
        for b in els:
            if b[5] or b[7] == t[7]:
                continue
            bx1, by1, bx2, by2 = b[1], b[2], b[1] + b[3], b[2] + b[4]
            ix = min(tx2, bx2) - max(tx1, bx1)
            iy = min(ty2, by2) - max(ty1, by1)
            if ix > tol and iy > tol:
                tarea = max((tx2 - tx1) * (ty2 - ty1), 1)
                frac = ix * iy / tarea
                row = (f"text[{t[6][:40]}] x={tx1:.0f} y {ty1:.0f}-{ty2:.0f}  X  "
                       f"box[{b[6][:14]}] x={bx1:.0f} y {by1:.0f}-{by2:.0f} (ox={ix:.0f}, oy={iy:.0f}, {frac:.0%})")
                if b[7] > t[7] and b[8] and frac > 0.25:
                    covers.append(row)
                else:
                    hits.append(row)
    print(f'== {path.name}: {len(els)} els | COVER={len(covers)} HIT={len(hits)}')
    for r in covers:
        print('  COVER', r)
    for r in hits:
        print('  HIT  ', r)
    return len(covers)

if __name__ == '__main__':
    import pathlib
    tot = sum(scan(pathlib.Path(p)) for p in sys.argv[1:])
    print('total COVER:', tot)
