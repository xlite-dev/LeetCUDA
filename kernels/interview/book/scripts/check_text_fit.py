#!/usr/bin/env python3
"""drawio 渲染级体检：估算 text 渲染宽度，检测 (a) 越出页宽 (b) 侵入右侧元素。
CJK/全角按 fontSize*1.02、ASCII 按 fontSize*0.58 保守估算；html 实体先解码。"""
import re, sys, glob

def decode(v):
    v = v.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').replace('&quot;', '"')
    v = re.sub(r'&#\d+;', 'x', v)
    return re.sub(r'<[^>]+>', '', v)

def width_est(t, fs):
    w = 0.0
    for ch in t:
        o = ord(ch)
        cjk = (0x2E80 <= o <= 0x9FFF) or (0x3000 <= o <= 0x303F) or (0xFF00 <= o <= 0xFFEF) or (0x2018 <= o <= 0x201D) or o in (0x2192, 0x2190, 0x2264, 0x2265, 0x2248, 0x00D7, 0x00B7)
        w += fs * (1.02 if cjk else 0.58)
    return w

def parse(path):
    s = open(path).read()
    pw = re.search(r'pageWidth="(\d+)"', s)
    page_w = int(pw.group(1)) if pw else 0
    texts, boxes = [], []
    for m in re.finditer(r'<mxCell (?:id="([^"]*)" )?value="([^"]*)" style="([^"]*)" vertex="1"[^>]*><mxGeometry x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"', s):
        i, v, st, x, y, w, h = m.groups()
        x, y, w, h = float(x), float(y), float(w), float(h)
        fs_m = re.search(r'fontSize=([0-9.]+)', st)
        fs = float(fs_m.group(1)) if fs_m else 13.0
        if st.startswith('text'):
            t = decode(v)
            if t.strip():
                texts.append((i or '-', x, y, w, h, fs, t, width_est(t, fs)))
        else:
            boxes.append((x, y, w, h, decode(v)[:12]))
    return page_w, texts, boxes

def scan(path, tol=3):
    page_w, texts, boxes = parse(path)
    probs = []
    for i, x, y, w, h, fs, t, est in texts:
        r = x + est
        if page_w and r > page_w - 5:
            probs.append(('PAGE', i, y, t, est, f'right={r:.0f} > pageW-5={page_w-5}'))
        for bx, by, bw, bh, bv in boxes:
            if bx <= x + 2:      # 框在文字起点左侧（宿主卡/背景）→ 跳过
                continue
            iy = min(y + h, by + bh) - max(y, by)
            if iy > tol and bx < r - tol:
                probs.append(('COLLIDE', i, y, t, est, f'侵入右侧框 x={bx:.0f}-{bx+bw:.0f} [{bv}]（渲染到 {r:.0f}）'))
                break
    return probs

if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('figures/drawio/*/*.drawio'))
    total = 0
    for f in files:
        probs = scan(f)
        if probs:
            print(f'== {f}')
            for kind, i, y, t, est, info in probs:
                print(f'  [{kind}] y={y:4.0f} est={est:4.0f} {i:10s} {info}')
                print(f'           | {t[:78]}')
            total += len(probs)
    print(f'TOTAL render-level issues: {total} in {len(files)} files')
