"""画像の一範囲から図形を取り出す（画素座標のまま）。

1. 小さな汚れ（スキャンのゴミ）を除く
2. 小さな塊を「点」「短い線（破線の1片）」「文字・記号のかけら」に分ける
3. 破線・一点鎖線・二点鎖線：同一直線上に並ぶ短い線と点をつないで1本にする
4. 文字：文字のかけらの集まりを文字のまとまりにする
5. 残りの線：細線化してなぞり、線分・円弧・円にする
6. 交点で切れた同一直線上の線分を1本に戻し、かすれで途切れた線もつなぐ
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
import tracer as T

NOISE_AREA = 5          # これ未満の面積の塊はスキャンの汚れとみなす
SMALL_MAX = 120         # 小さな塊とみなす外接矩形の最大辺(px)


def comp_info(lab, stats, i):
    x, y, w, h, a = stats[i]
    sub = lab[y:y + h, x:x + w] == i
    ys, xs = np.nonzero(sub)
    pts = np.c_[xs + x, ys + y].astype(float)
    return pts, (x, y, w, h, a)


def classify_small(pts, bbox):
    """小さな塊を 点 / 直線片 / 円弧片 / 文字・記号のかけら に分ける"""
    x, y, w, h, a = bbox
    if max(w, h) <= 13 and a >= NOISE_AREA and a / (w * h) > 0.35:
        return "dot", None
    m, d, p0, p1, dev = T.fit_line(pts)
    L = np.hypot(*(p1 - p0)) + 1
    th = a / L
    if L >= max(9, 2.5 * th) and dev <= th / 2 + 1.6 and th <= 14:
        return "dash", dict(m=m, d=d, p0=p0, p1=p1, L=L, th=th)
    fc = T.fit_circle(pts)
    if fc is not None:
        c, r, cdev = fc
        a0, a1, sw = T.arc_angles(c, pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))])
        # 円弧に沿った長さ
        ang = np.unwrap(np.sort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        k = int(np.argmax(gaps))   # 円周上で点がない最大の区間 = 円弧の外側
        start = ang[(k + 1) % len(ang)]
        sweep = 2 * np.pi - gaps[k]
        arcL = sweep * r
        th2 = a / max(1, arcL)
        if 12 < r < 3000 and cdev <= th2 / 2 + 1.6 and arcL >= max(9, 2.5 * th2) and sweep < 1.7 * np.pi:
            return "arcdash", dict(c=c, r=r, a0=start, sweep=sweep, L=arcL, th=th2,
                                   p0=c + r * np.array([np.cos(start), np.sin(start)]),
                                   p1=c + r * np.array([np.cos(start + sweep), np.sin(start + sweep)]))
    return "glyph", None


# ---------------------------------------------------------------- 破線類の連結
def link_dashes(dashes, dots, max_gap=75, ang_tol=5.0, lat_tol=3.0):
    """短い線分(dashes)と点(dots)を直線上で連結する。

    dashes: list of dict(p0,p1,d,L,th)  dots: (M,2)
    戻り値: chains: list of list of ('dash', i) / ('dot', j) を並び順に
    """
    n = len(dashes)
    if n == 0:
        return []
    ends = np.array([[*dashes[i]["p0"]] for i in range(n)] + [[*dashes[i]["p1"]] for i in range(n)])
    tree = cKDTree(ends)
    dot_tree = cKDTree(dots) if len(dots) else None
    best = {}  # (i, side) -> (gap, j, side_j)
    for i in range(n):
        di = dashes[i]
        for side in (0, 1):
            e = di["p1"] if side else di["p0"]
            out = (di["p1"] - di["p0"]) / max(1e-6, np.hypot(*(di["p1"] - di["p0"])))
            if side == 0:
                out = -out
            for k in tree.query_ball_point(e, max_gap):
                j, sj = k % n, k // n
                if j == i:
                    continue
                dj = dashes[j]
                ang = np.degrees(np.arccos(min(1, abs(np.dot(di["d"], dj["d"])))))
                if ang > ang_tol:
                    continue
                ej = dj["p1"] if sj else dj["p0"]
                v = ej - e
                along = np.dot(v, out)
                if along < -1:
                    continue
                # 相手の両端が自分の延長線上にあること
                nrm = np.array([-di["d"][1], di["d"][0]])
                lat = max(abs(np.dot(dj["p0"] - di["m"], nrm)), abs(np.dot(dj["p1"] - di["m"], nrm)))
                if lat > lat_tol + 0.02 * along:
                    continue
                # 相手のもう一方の端は更に遠くにあること
                ej2 = dj["p0"] if sj else dj["p1"]
                if np.dot(ej2 - e, out) < along:
                    continue
                g = along
                if (i, side) not in best or g < best[(i, side)][0]:
                    best[(i, side)] = (g, j, sj)
    # 相互に最良の組だけを連結
    nxt = {}
    for (i, s), (g, j, sj) in best.items():
        if best.get((j, sj), (None, None, None))[1:] == (i, s):
            nxt[(i, s)] = (j, sj, g)
    seen = set()
    chains = []
    for i in range(n):
        if i in seen:
            continue
        # 端までさかのぼる
        cur, side = i, 0
        guard = 0
        while (cur, side) in nxt and guard < 10000:
            j, sj, g = nxt[(cur, side)]
            cur, side = j, 1 - sj
            guard += 1
            if cur == i:
                break
        # cur の side 側が始端。反対側へ進む
        chain = []
        start = cur
        s_out = 1 - side
        visited = set()
        while cur not in visited:
            visited.add(cur)
            chain.append(cur)
            if (cur, s_out) not in nxt:
                break
            j, sj, g = nxt[(cur, s_out)]
            cur, s_out = j, 1 - sj
        seen.update(chain)
        chains.append(chain)
    return chains


def chain_pattern(chain, dashes, dots, dot_tree):
    """連結した短線の並びから、間に挟まる点を数えて線種を判定する"""
    segs = [dashes[i] for i in chain]
    # 並びの向き
    P = np.vstack([np.vstack([s["p0"], s["p1"]]) for s in segs])
    m, d, a, b, dev = T.fit_line(P)
    t = lambda p: float(np.dot(p - a, d))
    iv = sorted([(min(t(s["p0"]), t(s["p1"])), max(t(s["p0"]), t(s["p1"]))) for s in segs])
    gaps, dots_in = [], []
    used_dots = []
    nrm = np.array([-d[1], d[0]])
    for (s0, e0), (s1, e1) in zip(iv[:-1], iv[1:]):
        g = s1 - e0
        gaps.append(g)
        cnt = 0
        if dot_tree is not None and g > 4:
            mid = a + d * (e0 + s1) / 2
            for k in dot_tree.query_ball_point(mid, g / 2 + 2):
                q = dots[k]
                if abs(np.dot(q - a, nrm) - np.dot(m - a, nrm)) < 3.5 and e0 - 1 < t(q) < s1 + 1:
                    cnt += 1
                    used_dots.append(k)
        dots_in.append(cnt)
    lens = [e - s for s, e in iv]
    return dict(a=a + d * iv[0][0], b=a + d * iv[-1][1], d=d, lens=lens, gaps=gaps,
                dots=dots_in, used_dots=used_dots, dev=dev)


def decide_linetype(pat):
    """破線類の判定。かすれによる微小な切れ目しかない場合は実線とする。
    一点鎖線・二点鎖線は、すき間の6割以上に点が1つ・2つ入っている場合だけ認める。"""
    gaps = np.array(pat["gaps"]) if pat["gaps"] else np.array([0.0])
    lens = np.array(pat["lens"])
    dots = np.array(pat["dots"]) if pat["dots"] else np.array([0])
    if np.median(gaps) < 4.5:
        return "CONTINUOUS"
    if len(lens) < 3:
        return None
    big = gaps > 4.5
    d = dots[big]
    if len(d) == 0:
        return None
    if len(lens) >= 4 and (d == 2).mean() >= 0.6:
        return "PHANTOM"
    if len(lens) >= 4 and (d == 1).mean() >= 0.6:
        return "CENTER"
    if (d > 0).mean() <= 0.2:
        return "DASHED"
    return None


# ---------------------------------------------------------------- 線の連結（同一直線上）
def join_lines(lines, max_gap=8.0, ang_tol=3.0, lat_tol=2.0, wratio=1.7):
    """端どうしが近く同一直線上にある線分をつなぐ（交点で切れたもの・かすれたもの）"""
    changed = True
    lines = [dict(l) for l in lines]
    while changed:
        changed = False
        alive = [l for l in lines if not l.get("dead")]
        if len(alive) < 2:
            break
        E = np.array([l["p0"] for l in alive] + [l["p1"] for l in alive])
        n = len(alive)
        tree = cKDTree(E)
        pairs = tree.query_pairs(max_gap + 0.01)
        cand = []
        for k1, k2 in pairs:
            i, j = k1 % n, k2 % n
            if i == j:
                continue
            A, B = alive[i], alive[j]
            da = A["p1"] - A["p0"]; db = B["p1"] - B["p0"]
            la, lb = np.hypot(*da), np.hypot(*db)
            if la < 1 or lb < 1:
                continue
            ang = np.degrees(np.arccos(min(1, abs(np.dot(da, db)) / (la * lb))))
            if ang > ang_tol + (6 if min(la, lb) < 12 else 0):
                continue
            wa, wb = A.get("w", np.nan), B.get("w", np.nan)
            if not (np.isnan(wa) or np.isnan(wb)) and max(wa, wb) / max(0.5, min(wa, wb)) > wratio:
                continue
            P = np.vstack([A["p0"], A["p1"], B["p0"], B["p1"]])
            m, d, a, b, dev = T.fit_line(P)
            if dev > lat_tol:
                continue
            # 重なりではなく端どうしの連結であること
            ta = sorted([np.dot(A["p0"] - m, d), np.dot(A["p1"] - m, d)])
            tb = sorted([np.dot(B["p0"] - m, d), np.dot(B["p1"] - m, d)])
            gap = max(ta[0], tb[0]) - min(ta[1], tb[1])
            if gap > max_gap or gap < -3:
                continue
            cand.append((gap, i, j))
        cand.sort()
        used = set()
        for gap, i, j in cand:
            if i in used or j in used:
                continue
            A, B = alive[i], alive[j]
            P = np.vstack([A["p0"], A["p1"], B["p0"], B["p1"]])
            m, d, a, b, dev = T.fit_line(P)
            LA, LB = np.hypot(*(A["p1"] - A["p0"])), np.hypot(*(B["p1"] - B["p0"]))
            ws = [(A.get("w", np.nan), LA), (B.get("w", np.nan), LB)]
            ws = [(x, l) for x, l in ws if not np.isnan(x)]
            w = sum(x * l for x, l in ws) / sum(l for x, l in ws) if ws else np.nan
            new = dict(A)
            new.update(p0=a, p1=b, w=w, nseg=A.get("nseg", 1) + B.get("nseg", 1))
            A["dead"] = B["dead"] = True
            lines.append(new)
            used.update((i, j))
            changed = True
        lines = [l for l in lines if not l.get("dead")]
    return lines


# ---------------------------------------------------------------- 曲がった破線の連結
def piece_ends(p):
    """片の両端の位置と外向きの接線"""
    if p["type"] == "dash":
        d = (p["p1"] - p["p0"]) / max(1e-6, np.hypot(*(p["p1"] - p["p0"])))
        return [(p["p0"], -d), (p["p1"], d)]
    a0, sw = p["a0"], p["sweep"]
    t0 = np.array([np.sin(a0), -np.cos(a0)])            # 角度が減る向き
    a1 = a0 + sw
    t1 = np.array([-np.sin(a1), np.cos(a1)])            # 角度が増える向き
    return [(p["p0"], t0), (p["p1"], t1)]


def piece_points(p, n=6):
    if p["type"] == "dash":
        return np.linspace(p["p0"], p["p1"], n)
    t = p["a0"] + np.linspace(0, p["sweep"], n)
    return p["c"] + p["r"] * np.c_[np.cos(t), np.sin(t)]


def link_curved(pieces, max_gap=60, turn_tol=50.0, conn_tol=30.0):
    n = len(pieces)
    if n < 2:
        return [[i] for i in range(n)]
    E, U = [], []
    for p in pieces:
        for e, u in piece_ends(p):
            E.append(e); U.append(u)
    E, U = np.array(E), np.array(U)
    tree = cKDTree(E)
    best = {}
    cosT = np.cos(np.deg2rad(turn_tol))
    cosC = np.cos(np.deg2rad(conn_tol))
    for k in range(2 * n):
        i = k // 2
        for k2 in tree.query_ball_point(E[k], max_gap):
            j = k2 // 2
            if j == i:
                continue
            v = E[k2] - E[k]
            g = np.hypot(*v)
            if g > 2 and (np.dot(v / g, U[k]) < cosC or np.dot(v / g, -U[k2]) < cosC):
                continue
            if np.dot(-U[k2], U[k]) < cosT:
                continue
            thi, thj = pieces[i]["th"], pieces[j]["th"]
            if max(thi, thj) / max(0.5, min(thi, thj)) > 1.8:
                continue
            if k not in best or g < best[k][0]:
                best[k] = (g, k2)
    nxt = {}
    for k, (g, k2) in best.items():
        if best.get(k2, (None, None))[1] == k:
            nxt[k] = k2
    seen, chains = set(), []
    for i in range(n):
        if i in seen:
            continue
        # 始端へさかのぼる
        k = 2 * i
        guard = 0
        while k in nxt and guard < 5000:
            k2 = nxt[k]
            k = k2 ^ 1
            guard += 1
            if k // 2 == i:
                break
        start_k = k
        chain, ks = [], []
        k = start_k
        visited = set()
        while k // 2 not in visited:
            visited.add(k // 2)
            chain.append(k // 2); ks.append(k)
            out = k ^ 1
            if out not in nxt:
                break
            k = nxt[out]
        seen.update(chain)
        chains.append((chain, ks))
    return chains


def curved_chain_record(chain, ks, pieces, dots, dot_tree):
    """曲線状の破線列を1つの図形（線分・円弧・折れ線）にまとめる"""
    seq = []
    for i, k in zip(chain, ks):
        pts = piece_points(pieces[i], 8)
        if k % 2 == 1:      # 入口が p1 側 → 逆順
            pts = pts[::-1]
        seq.append(pts)
    lens = [pieces[i]["L"] for i in chain]
    gaps, dots_in, used = [], [], []
    for a, b in zip(seq[:-1], seq[1:]):
        e, f = a[-1], b[0]
        g = np.hypot(*(f - e))
        gaps.append(g)
        cnt = 0
        if dot_tree is not None and g > 4:
            for kd in dot_tree.query_ball_point((e + f) / 2, g / 2 + 2):
                q = dots[kd]
                v = f - e
                t = np.clip(np.dot(q - e, v) / max(1e-6, np.dot(v, v)), 0, 1)
                if np.hypot(*(q - (e + t * v))) < 3.5:
                    cnt += 1; used.append(kd)
        dots_in.append(cnt)
    P = np.vstack(seq)
    pat = dict(lens=lens, gaps=gaps, dots=dots_in, used_dots=used)
    m, d, a, b, dev = T.fit_line(P)
    if dev <= 2.0:
        return dict(kind="line", p0=P[0] if np.dot(P[0]-m, d) <= np.dot(P[-1]-m, d) else P[-1],
                    p1=P[-1] if np.dot(P[0]-m, d) <= np.dot(P[-1]-m, d) else P[0], pat=pat, pts=P)
    fc = T.fit_circle(P)
    if fc is not None and fc[2] <= 2.5:
        c, r, cdev = fc
        a0, a1, sw = T.arc_angles(c, P)
        if sw > 2 * np.pi - 0.25 or np.hypot(*(P[0] - P[-1])) < 1.2 * max(gaps + [5]):
            return dict(kind="circle", c=c, r=r, pat=pat, pts=P)
        return dict(kind="arc", c=c, r=r, a0=a0, a1=a1, pat=pat, pts=P)
    return dict(kind="poly", pts=P, pat=pat)


def regular(pat):
    g = np.array(pat["gaps"], float); l = np.array(pat["lens"], float)
    if len(g) < 2:
        return False
    big = g[g > 4.5]
    if len(big) < 2 or len(big) < 0.7 * len(g):
        return False
    return (np.std(big) / np.mean(big) < 0.45) and (np.std(l) / np.mean(l) < 0.6)


# ---------------------------------------------------------------- 範囲の処理
GRAPHIC_LEN = 50      # これ以上の長さの直線は図形（文字ではない）とみなす


def find_filled_dots(ink, min_d=11, max_side=40):
    """塗りつぶした点（寸法の点・記号の黒丸など）を探す"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_d, min_d))
    op = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(op, connectivity=8)
    out = []
    H, W = ink.shape
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) <= max_side and min(w, h) >= min_d - 1 and a / (w * h) > 0.6 and max(w, h) / min(w, h) < 1.6:
            c = cen[i]; r = np.sqrt(a / np.pi)
            # 周囲が細い線だけなら独立した黒丸（太線の角や重なりではない）
            hits = 0; tot = 0
            for rr in (r + 4, r + 5.5, r + 7):
                qx = np.clip((c[0] + rr * np.cos(t)).astype(int), 0, W - 1)
                qy = np.clip((c[1] + rr * np.sin(t)).astype(int), 0, H - 1)
                hits += ink[qy, qx].sum(); tot += len(t)
            if hits / tot < 0.22:
                out.append(dict(c=c.astype(float), r=float(r)))
    return out


def vectorize_mask(m, origin):
    """小さな画素のまとまりを細線化して線分・円弧にする（文字の形・記号用）"""
    m = m.astype(np.uint8)
    if m.sum() == 0:
        return []
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    paths, junctions = T.trace_skeleton(np.pad(m, 1), np.pad(dt, 1))
    out = []
    for p in paths:
        q = p["pts"] - 1
        if len(q) < 3:
            continue
        for s in T.segment_path(T.smooth_path(q, 1), tol=1.2, closed=p["closed"]):
            for k in ("p0", "p1", "c"):
                if k in s:
                    s[k] = s[k] + np.array(origin, float)
            s["pts"] = s["pts"] + np.array(origin, float)
            s["w"] = float(np.median(2 * dt[np.clip(np.round(q[:, 1]).astype(int), 0, m.shape[0] - 1),
                                            np.clip(np.round(q[:, 0]).astype(int), 0, m.shape[1] - 1)]))
            out.append(s)
    return out


def process(ink, origin=(0, 0)):
    """ink: 0/1 の画像（範囲を切り出したもの）。結果の座標はページ全体の画素座標。"""
    ox, oy = origin
    off = np.array([ox, oy], float)
    H, W = ink.shape
    n, lab, stats, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    kinds = np.zeros(n, object)
    pieces, dots, dot_comp = [], [], []
    noise_px = 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < NOISE_AREA:
            kinds[i] = "noise"; noise_px += a
            continue
        if max(w, h) > SMALL_MAX or a > 2500:
            kinds[i] = "large"
            continue
        pts, bb = comp_info(lab, stats, i)
        k, info = classify_small(pts, bb)
        kinds[i] = k
        if k in ("dash", "arcdash"):
            info["type"] = k; info["comp"] = i
            pieces.append(info)
        elif k == "dot":
            dots.append(pts.mean(0)); dot_comp.append(i)
    dots = np.array(dots) if dots else np.zeros((0, 2))
    dot_tree = cKDTree(dots) if len(dots) else None
    used_piece = np.zeros(len(pieces), bool)
    used_dot = np.zeros(len(dots), bool)
    dashed, solid_from_frag = [], []

    # --- B1 直線の破線類
    sidx = [k for k, p in enumerate(pieces) if p["type"] == "dash"]
    for ch in link_dashes([pieces[k] for k in sidx], dots):
        if len(ch) < 2:
            continue
        ch_p = [sidx[c] for c in ch]
        pat = chain_pattern(ch, [pieces[k] for k in sidx], dots, dot_tree)
        lt = decide_linetype(pat)
        if lt is None or (lt != "CONTINUOUS" and not regular(pat)):
            continue
        w = float(np.median([pieces[k]["th"] for k in ch_p]))
        comps = [pieces[k]["comp"] for k in ch_p] + [dot_comp[k] for k in pat["used_dots"]]
        rec = dict(kind="line", p0=pat["a"], p1=pat["b"], w=w, linetype=lt, pat=pat, comps=comps)
        (solid_from_frag if lt == "CONTINUOUS" else dashed).append(rec)
        used_piece[ch_p] = True
        used_dot[pat["used_dots"]] = True

    # --- B2 曲がった破線類
    ridx = [k for k in range(len(pieces)) if not used_piece[k]]
    for ch, ks in link_curved([pieces[k] for k in ridx]):
        if len(ch) < 3:
            continue
        ch_p = [ridx[c] for c in ch]
        rec = curved_chain_record(ch, ks, [pieces[k] for k in ridx], dots, dot_tree)
        lt = decide_linetype(rec["pat"])
        if lt is None or lt == "CONTINUOUS" or not regular(rec["pat"]):
            continue
        rec["w"] = float(np.median([pieces[k]["th"] for k in ch_p]))
        rec["linetype"] = lt
        rec["comps"] = [pieces[k]["comp"] for k in ch_p] + [dot_comp[k] for k in rec["pat"]["used_dots"]]
        dashed.append(rec)
        used_piece[ch_p] = True
        used_dot[rec["pat"]["used_dots"]] = True

    # --- C 大きな塊：細線化してなぞる
    big = np.isin(lab, np.nonzero(kinds == "large")[0]).astype(np.uint8)
    dt = cv2.distanceTransform(big, cv2.DIST_L2, 5)
    paths, junctions = T.trace_skeleton(big, dt)
    prims = []
    for p in paths:
        pts = p["pts"]
        L = len(pts)
        if (p["ends"][0] > 0) != (p["ends"][1] > 0):
            jc = junctions.get(max(p["ends"]))
            wloc = dt[int(jc[1]), int(jc[0])] if jc is not None else 2
            if L < 2.2 * wloc + 3:
                continue
        q = T.trim_at_junctions(pts, p["ends"], junctions, dt)
        if len(q) < 3:
            continue
        q = T.smooth_path(q, 1)
        segs = T.segment_path(q, tol=1.4, closed=p["closed"])
        for k, s in enumerate(segs):
            s["src"] = "skel"
            # 分岐点で切った線の端を分岐点まで延ばす（交点・角のすき間をなくす）
            if s["kind"] == "line":
                for side, jid in ((0, p["ends"][0] if k == 0 else -1), (1, p["ends"][1] if k == len(segs) - 1 else -1)):
                    jc = junctions.get(jid) if jid > 0 else None
                    if jc is None:
                        continue
                    a_, b_ = s["p0"], s["p1"]
                    v = b_ - a_; L = np.hypot(*v)
                    if L < 1:
                        continue
                    u = v / L
                    t = np.dot(jc - a_, u)
                    newp = a_ + t * u
                    end = a_ if side == 0 else b_
                    if np.hypot(*(newp - end)) < 2 * dt[int(jc[1]), int(jc[0])] + 8 and \
                            abs(np.dot(jc - a_, np.array([-u[1], u[0]]))) < dt[int(jc[1]), int(jc[0])] + 3:
                        if side == 0:
                            s["p0"] = newp
                        else:
                            s["p1"] = newp
            prims.append(s)
    for k, p in enumerate(pieces):
        if used_piece[k]:
            continue
        if p["type"] == "dash":
            prims.append(dict(kind="line", p0=p["p0"], p1=p["p1"], pts=np.vstack([p["p0"], p["p1"]]),
                              src="piece", comp=p["comp"]))
        else:
            t = p["a0"] + np.linspace(0, p["sweep"], 12)
            prims.append(dict(kind="arc", c=p["c"], r=p["r"], a0=p["a0"], a1=p["a0"] + p["sweep"],
                              pts=p["c"] + p["r"] * np.c_[np.cos(t), np.sin(t)], src="piece", comp=p["comp"]))
    for s in prims:
        if s["kind"] == "line":
            s["w"] = T.measure_width(ink, s["p0"], s["p1"])
        else:
            s["w"] = T.measure_width_curve(ink, s["pts"])

    # --- D 同一直線上の線をつなぐ → 太さを全長で測り直す
    lines = [s for s in prims if s["kind"] == "line"] + solid_from_frag
    curves = [s for s in prims if s["kind"] != "line"]
    lines = join_lines(lines)
    for s in lines:
        if s.get("nseg", 1) > 1 or s.get("linetype") == "CONTINUOUS":
            w = T.measure_width(ink, s["p0"], s["p1"])
            if not np.isnan(w):
                s["w"] = w

    # --- E 図形として確実なものを塗り、残りを文字の候補にする
    gmask = np.zeros((H, W), np.uint8)
    def length(s):
        return np.hypot(*(s["p1"] - s["p0"])) if s["kind"] == "line" else \
            (2 * np.pi * s["r"] if s["kind"] == "circle" else abs(s["a1"] - s["a0"]) * s["r"])
    def is_graphic(s):
        if s["kind"] == "line":
            return length(s) >= GRAPHIC_LEN
        return s["r"] >= 26 and length(s) >= 45
    def paint(m, s, extra=1.5):
        w = s.get("w", 3.0)
        w = 3.0 if (w is None or np.isnan(w)) else w
        th = max(1, int(round(w + 2 * extra)))
        if s["kind"] == "line":
            cv2.line(m, tuple(np.round(s["p0"]).astype(int)), tuple(np.round(s["p1"]).astype(int)), 1, th)
        else:
            cv2.polylines(m, [np.round(s["pts"]).astype(np.int32)], s["kind"] == "circle", 1, th)
    for s in lines + curves:
        if is_graphic(s):
            paint(gmask, s)
    for s in dashed:
        gmask[np.isin(lab, s["comps"])] = 1
    fdots = find_filled_dots(ink)
    for d in fdots:
        cv2.circle(gmask, tuple(np.round(d["c"]).astype(int)), int(d["r"] + 2), 1, -1)
    resid = (ink > 0) & (gmask == 0)
    rn, rlab, rst, rcen = cv2.connectedComponentsWithStats(resid.astype(np.uint8), connectivity=8)
    cand = np.zeros((H, W), np.uint8)
    straight = []
    resid_geo = []
    for i in range(1, rn):
        x, y, w, h, a = rst[i]
        if a >= NOISE_AREA and max(w, h) > 110:
            # 文字にしては大きい残り：図形としてなぞる（取りこぼし防止）
            m_ = (rlab[y:y + h, x:x + w] == i)
            for q in vectorize_mask(m_, (x, y)):
                q["src"] = "resid"
                resid_geo.append(q)
            continue
        if a >= NOISE_AREA and max(w, h) <= 110:
            m_ = (rlab[y:y + h, x:x + w] == i)
            ys_, xs_ = np.nonzero(m_)
            P_ = np.c_[xs_, ys_].astype(float)
            _, _, q0, q1, dev = T.fit_line(P_)
            L_ = np.hypot(*(q1 - q0)) + 1
            if (L_ >= 15 and dev <= a / L_ / 2 + 1.6) or a < 20:
                straight.append((i, (x, y, w, h)))      # 直線状・ごく小さいかけら：文字の種にはしない
                continue
            cand[y:y + h, x:x + w] |= m_.astype(np.uint8)
    seed = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    for i, (x, y, w, h) in straight:
        m_ = (rlab[y:y + h, x:x + w] == i)
        if seed[y:y + h, x:x + w][m_].mean() > 0.9:      # 文字の中に収まる直線（「ー」「1」など）
            cand[y:y + h, x:x + w] |= m_.astype(np.uint8)
    # 文字のまとまり（近い文字の候補をつなげる）
    blk = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    bn, blab, bst, bcen = cv2.connectedComponentsWithStats(blk, connectivity=8)
    blocks = []
    keep_blk = np.zeros(bn, bool)
    for i in range(1, bn):
        x, y, w, h, a = bst[i]
        inkc = int(cand[y:y + h, x:x + w][blab[y:y + h, x:x + w] == i].sum())
        if inkc < 60 or max(w, h) < 22:     # 小さすぎるもの（線の切れ端など）は文字にしない
            continue
        keep_blk[i] = True
        blocks.append(dict(id=i, bbox=(x + ox, y + oy, w, h)))
    # 文字のまとまりの画素
    tmask = ((cand > 0) & keep_blk[blab]).astype(np.uint8)
    for q in resid_geo:
        w0 = q.get("w", np.nan)
        q["w"] = T.measure_width(ink, q["p0"], q["p1"]) if q["kind"] == "line" else T.measure_width_curve(ink, q["pts"])
        if np.isnan(q["w"]):
            q["w"] = w0            # 短すぎて測れない場合は細線化のときの太さ
        (lines if q["kind"] == "line" else curves).append(q)
    # 文字のまとまりの中にある短い図形は文字の一部（いちばん多く重なるまとまりに入れる）
    def text_block_of(s):
        pts = s["pts"] if "pts" in s else np.vstack([s["p0"], s["p1"]])
        q = np.round(pts).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1); q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        ids = blab[q[:, 1], q[:, 0]]
        ok = keep_blk[ids] & (seed[q[:, 1], q[:, 0]] > 0)
        if ok.mean() <= 0.85:
            return None
        v, c = np.unique(ids[ok], return_counts=True)
        return int(v[np.argmax(c)])
    tp = {}
    keep_l, keep_c = [], []
    for lst, keep in ((lines, keep_l), (curves, keep_c)):
        for s in lst:
            bid = None if is_graphic(s) else text_block_of(s)
            if bid is None:
                keep.append(s)
            else:
                tp.setdefault(bid, []).append(s)
    lines, curves = keep_l, keep_c
    # 文字のまとまりの形を線にする。直線数本だけでできたまとまりは図形（記号の角・線の切れ端）とみなす
    for b in blocks:
        x, y, w, h = b["bbox"]; x -= ox; y -= oy
        m_ = (tmask[y:y + h, x:x + w] > 0) & (blab[y:y + h, x:x + w] == b["id"])
        b["prims"] = vectorize_mask(m_, (x + ox, y + oy))
        for s in tp.get(b["id"], []):
            s2 = dict(s)
            for k in ("p0", "p1", "c"):
                if k in s2:
                    s2[k] = s2[k] + off
            if "pts" in s2:
                s2["pts"] = s2["pts"] + off
            b["prims"].append(s2)
        kinds_ = [q["kind"] for q in b["prims"]]
        Ls = [np.hypot(*(q["p1"] - q["p0"])) for q in b["prims"] if q["kind"] == "line"]
        only_lines = len(kinds_) > 0 and all(k == "line" for k in kinds_)
        b["auto"] = "G" if (only_lines and (len(kinds_) <= 4 or (len(kinds_) <= 10 and np.median(Ls) >= 25))) else "T"
        # 文字のまとまりの画素（文字の形の検証・文字の位置決め用）
    tm2 = np.zeros((H, W), np.uint8)
    for bid, lst in tp.items():
        for s in lst:
            paint(tm2, s, extra=1.0)
    tmask = ((tmask > 0) | ((tm2 > 0) & (ink > 0) & (gmask == 0))).astype(np.uint8)

    def mv(s):
        s = dict(s)
        for k in ("p0", "p1", "c"):
            if k in s:
                s[k] = s[k] + off
        if "pts" in s:
            s["pts"] = s["pts"] + off
        return s
    return dict(lines=[mv(s) for s in lines], curves=[mv(s) for s in curves], dashed=[mv(s) for s in dashed],
                fdots=[mv(d) for d in fdots], text_mask=tmask, text_blocks=blocks, blk_lab=blab, keep_blk=keep_blk,
                origin=origin, noise_px=noise_px)
