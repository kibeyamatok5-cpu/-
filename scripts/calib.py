"""通り芯から縮尺・傾きを校正する。
画像上の通り芯（丸い記号から伸びる細線）の位置を測り、図面に書かれた寸法値と
一致するように「画像の画素 → 実寸mm」の変換（回転・拡大縮小・縦横比）を求める。
"""
import json, sys, cv2, numpy as np
from scipy.optimize import least_squares

def detect_bubbles(ink):
    blur = cv2.GaussianBlur(255 - ink, (5, 5), 1.5)
    cs = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=100,
                          param2=35, minRadius=44, maxRadius=66)
    out = []
    t = np.linspace(0, 2*np.pi, 180, endpoint=False)
    H, W = ink.shape
    for x, y, r in (cs[0] if cs is not None else []):
        ring = np.zeros(len(t), bool)
        for d in range(-3, 4):
            qx = np.clip((x+(r+d)*np.cos(t)).astype(int), 0, W-1)
            qy = np.clip((y+(r+d)*np.sin(t)).astype(int), 0, H-1)
            ring |= ink[qy, qx] > 0
        if ring.mean() > 0.85:
            out.append((float(x), float(y), float(r)))
    return out

def refine_center(ink, x, y, r):
    """円周のインク画素に円を当てはめて中心を精密化"""
    H, W = ink.shape
    x0, y0 = int(x-r-12), int(y-r-12)
    sub = ink[max(0,y0):y0+int(2*r+24), max(0,x0):x0+int(2*r+24)]
    ys, xs = np.nonzero(sub)
    xs = xs + max(0,x0); ys = ys + max(0,y0)
    d = np.hypot(xs-x, ys-y)
    m = np.abs(d - r) < 6
    xs, ys = xs[m].astype(float), ys[m].astype(float)
    A = np.c_[2*xs, 2*ys, np.ones_like(xs)]
    b = xs**2 + ys**2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, np.sqrt(c + cx**2 + cy**2)

def trace_axis(ink, cx, cy, r, ang0, span=25, length=(1.15, 7.0)):
    """記号から伸びる直線を探し、その中心線上の点列を返す"""
    H, W = ink.shape
    best = None
    for da in np.arange(-span, span+0.01, 0.25):
        a = np.deg2rad(ang0 + da)
        s = np.arange(r*length[0], r*length[1], 1.0)
        px = (cx + s*np.cos(a)).astype(int); py = (cy + s*np.sin(a)).astype(int)
        ok = (px >= 0) & (py >= 0) & (px < W) & (py < H)
        v = ink[py[ok], px[ok]].mean() if ok.any() else 0
        if best is None or v > best[0]:
            best = (v, a)
    v, a = best
    d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
    pts = []
    for s in np.arange(r*length[0], r*length[1], 4.0):
        c = np.array([cx, cy]) + s*d
        offs = np.arange(-7, 7.01, 0.5)
        q = c[None, :] + offs[:, None]*n[None, :]
        qi = np.round(q).astype(int)
        ok = (qi[:, 0] >= 0) & (qi[:, 1] >= 0) & (qi[:, 0] < W) & (qi[:, 1] < H)
        if not ok.all(): continue
        w = ink[qi[:, 1], qi[:, 0]].astype(float)
        if w.sum() < 2 or w.sum() > 20*255: continue   # 交差部・空白はとばす
        o = (w*offs).sum()/w.sum()
        pts.append(c + o*n)
    pts = np.array(pts)
    if len(pts) < 10:
        return v, a, None
    # 外れ値除去して直線当てはめ
    for _ in range(3):
        m = pts.mean(0); u, s_, vt = np.linalg.svd(pts-m); dd = vt[0]; nn = np.array([-dd[1], dd[0]])
        res = (pts-m)@nn
        keep = np.abs(res) < max(1.5, 3*np.std(res))
        pts = pts[keep]
    return v, a, pts

def trace_families(page_cfg, ink, key="families"):
    bub = [refine_center(ink, *b) for b in detect_bubbles(ink)]
    # 隣の丸と接していて自動検出できなかった丸（位置の目安を設定で与え、精密化する）
    for b in page_cfg.get("extra_bubbles", []):
        bub.append(refine_center(ink, *b))
    fams = {}
    for f in page_cfg.get(key, []):
        x0, y0, x1, y1 = f["region"]
        cand = [b for b in bub if x0 <= b[0] <= x1 and y0 <= b[1] <= y1]
        if "band" in f:   # 2点を結ぶ直線の近く（斜めの並び）だけを採用
            ax, ay, bx, by = f["band"]
            d = np.array([bx-ax, by-ay], float); d /= np.linalg.norm(d)
            cand = [b for b in cand if abs((b[0]-ax)*d[1] - (b[1]-ay)*d[0]) < 60]
        key = f["sort"]
        cand.sort(key=lambda b: b[0]*key[0] + b[1]*key[1])
        if len(cand) != len(f["labels"]):
            raise SystemExit(f"{f['name']}: bubbles {len(cand)} != labels {len(f['labels'])}: {cand}")
        lines = []
        for lab, b in zip(f["labels"], cand):
            v, a, pts = trace_axis(ink, b[0], b[1], b[2], f["ang"], length=f.get("len", (1.15, 7.0)))
            lines.append(dict(label=lab, bubble=b, ang=float(np.rad2deg(a)), cover=float(v/255), pts=pts))
        fams[f["name"]] = (f, lines)
    return fams


def fit(page_cfg, ink):
    fams = trace_families(page_cfg, ink)
    # 未知数: 画素→mm のアフィン変換 (a,b,c,d,tx,ty) と、斜め通り芯群ごとの向き・原点
    oblique = [n for n, (f, _) in fams.items() if f["kind"] == "oblique"]
    def unpack(p):
        M = np.array([[p[0], p[1]], [p[2], p[3]]]); t = p[4:6]
        ob = {n: (p[6+2*i], p[7+2*i]) for i, n in enumerate(oblique)}
        return M, t, ob
    def resid(p, mode):
        M, t, ob = unpack(p)
        r = []
        for n, (f, lines) in fams.items():
            for ln, nom in zip(lines, f["pos"]):
                if ln["pts"] is None: continue
                q = ln["pts"] @ M.T + t
                if f["kind"] == "x": r.append(q[:, 0] - nom)
                elif f["kind"] == "y": r.append(q[:, 1] - nom)
                else:
                    th, c = ob[n]
                    nrm = np.array([np.cos(th), np.sin(th)])
                    r.append(q @ nrm - c - nom)
        r = np.concatenate(r)
        if mode == "similarity":   # 回転＋一様な拡大縮小（縦横同率）
            r = np.r_[r, 1e5*(p[0]+p[3]), 1e5*(p[1]-p[2])]
        elif mode == "scale_xy":   # 回転＋画像の縦横それぞれの拡大縮小（スキャナの送り方向の伸び）
            # M = R(θ)·diag(sx,-sy) なら 列ベクトルが直交する
            r = np.r_[r, 1e8*(p[0]*p[1] + p[2]*p[3])/(p[0]**2+p[2]**2)]
        return r
    s0 = page_cfg["scale_guess"]
    p0 = [s0, 0, 0, -s0, 0, 0]
    for n in oblique:
        p0 += [np.deg2rad(fams[n][0]["normal_guess"]), 0.0]
    # 初期の平行移動: 第1通り芯
    results = {}
    for mode in ("similarity", "scale_xy", "affine"):
        sol = least_squares(lambda p: resid(p, mode), p0, x_scale="jac")
        sol = least_squares(lambda p: resid(p, mode), sol.x, x_scale="jac")
        results[mode] = sol
    return fams, results, unpack

def report(fams, M, t, ob):
    rows = []
    for n, (f, lines) in fams.items():
        coords = []
        for ln in lines:
            if ln["pts"] is None: coords.append(None); continue
            q = ln["pts"] @ M.T + t
            if f["kind"] == "x": coords.append(q[:, 0].mean())
            elif f["kind"] == "y": coords.append(q[:, 1].mean())
            else:
                th, c = ob[n]; nrm = np.array([np.cos(th), np.sin(th)])
                coords.append((q @ nrm).mean() - c)
        for i in range(len(lines)-1):
            if coords[i] is None or coords[i+1] is None: continue
            meas = coords[i+1]-coords[i]; nom = f["pos"][i+1]-f["pos"][i]
            rows.append((n, lines[i]["label"], lines[i+1]["label"], nom, meas, meas-nom))
        ok = [i for i, c in enumerate(coords) if c is not None]
        i0, i1 = ok[0], ok[-1]
        meas = coords[i1]-coords[i0]; nom = f["pos"][i1]-f["pos"][i0]
        rows.append((n, lines[i0]["label"], lines[i1]["label"]+" (全長)", nom, meas, meas-nom))
    return rows

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    page = sys.argv[2]
    pc = cfg[page]
    im = cv2.imread(pc["image"], 0)
    ink = ((im < 128)*255).astype(np.uint8)
    fams, results, unpack = fit(pc, ink)
    out = {}
    for mode, sol in results.items():
        M, t, ob = unpack(sol.x)
        r = sol.fun[:{"similarity": -2, "scale_xy": -1, "affine": None}[mode]]
        sx = np.hypot(M[0,0], M[1,0]); sy = np.hypot(M[0,1], M[1,1])
        rot = np.rad2deg(np.arctan2(M[1,0], M[0,0]))
        shear = np.rad2deg(np.arccos(abs(M[:,0]@M[:,1])/(sx*sy)))
        print(f"== {mode}: 横 {1/sx:.2f}px→1m? 画素寸法 x={sx:.4f}mm/px y={sy:.4f}mm/px 比={sx/sy:.5f} "
              f"回転={rot+0:.4f}° 直交角={shear:.4f}° RMS={np.sqrt(np.mean(r**2)):.1f}mm")
        for n in ob: print(f"   {n}: 法線角 {np.rad2deg(ob[n][0]) % 360:.3f}°")
        rows = report(fams, M, t, ob)
        for row in rows: print("   %-3s %-4s→%-10s 図面 %7.0f  測定 %8.1f  誤差 %+6.1f mm" % row)
        out[mode] = dict(M=M.tolist(), t=t.tolist(), oblique={k: list(v) for k, v in ob.items()},
                         rms=float(np.sqrt(np.mean(r**2))), sx=sx, sy=sy, rot=rot, ortho=shear,
                         rows=[list(map(lambda v: v if isinstance(v, str) else float(v), row)) for row in rows])
    # 同じ用紙の別の図（例：508 左上の小さな平面図）：間隔だけを確認する
    if pc.get("secondary"):
        M, t, ob = unpack(results["affine"].x)
        sec = trace_families(pc, ink, "secondary")
        srows = []
        for n, (f, lines) in sec.items():
            co = []
            for ln in lines:
                q = ln["pts"] @ M.T + t
                co.append(q[:, 0].mean() if f["kind"] == "x" else q[:, 1].mean())
            for i in range(len(lines) - 1):
                srows.append([n, lines[i]["label"], lines[i + 1]["label"], f["pos"][i + 1] - f["pos"][i],
                              float(co[i + 1] - co[i]), float(co[i + 1] - co[i] - (f["pos"][i + 1] - f["pos"][i]))])
            srows.append([n, lines[0]["label"], lines[-1]["label"] + " (全長)", f["pos"][-1] - f["pos"][0],
                          float(co[-1] - co[0]), float(co[-1] - co[0] - (f["pos"][-1] - f["pos"][0]))])
            # この図の通り芯の位置（本体の座標系での値）。本体の図とのずれ
            srows.append([n, "offset", lines[0]["label"], f["pos"][0], float(co[0]), float(co[0] - f["pos"][0])])
        for r_ in srows:
            print("   副図 %-6s %-6s→%-10s 図面 %7.0f  測定 %9.1f  差 %+8.1f" % tuple(r_))
        out["secondary"] = srows
        out["axes_secondary"] = {n: [dict(label=l["label"], bubble=l["bubble"], ang=l["ang"], cover=l["cover"],
                                     p0=(l["pts"][0].tolist() if l["pts"] is not None else None),
                                     p1=(l["pts"][-1].tolist() if l["pts"] is not None else None)) for l in lines]
                                 for n, (f, lines) in sec.items()}
    out["axes"] = {n: [dict(label=l["label"], bubble=l["bubble"], ang=l["ang"], cover=l["cover"],
                       p0=(l["pts"][0].tolist() if l["pts"] is not None else None),
                       p1=(l["pts"][-1].tolist() if l["pts"] is not None else None)) for l in lines]
                   for n, (f, lines) in fams.items()}
    json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
