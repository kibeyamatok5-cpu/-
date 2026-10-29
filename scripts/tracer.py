"""画像の線をなぞってベクター図形（線分・円弧・円・折れ線）にする部品集。

座標はすべて画像の画素座標 (x右向き, y下向き)。実寸mmへの変換は最後に別途行う。
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skan import Skeleton


# ---------------------------------------------------------------- 基本の当てはめ
def fit_line(pts):
    """全最小二乗で直線を当てはめ、(中心, 方向, 端点0, 端点1, 最大ずれ) を返す"""
    m = pts.mean(0)
    if len(pts) < 2:
        return m, np.array([1.0, 0.0]), m, m, 0.0
    u, s, vt = np.linalg.svd(pts - m, full_matrices=False)
    d = vt[0]
    t = (pts - m) @ d
    n = np.array([-d[1], d[0]])
    dev = np.abs((pts - m) @ n)
    return m, d, m + t.min() * d, m + t.max() * d, float(dev.max())


def fit_circle(pts):
    """代数的円当てはめ (Kasa法)"""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    try:
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    dev = np.abs(np.hypot(x - cx, y - cy) - r)
    return np.array([cx, cy]), float(r), float(dev.max())


def arc_angles(c, pts):
    """点列が円周上で描く範囲（始角, 終角, 掃引角）。角度は画像座標で反時計回り（y下向き）"""
    a = np.unwrap(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return float(a[0]), float(a[-1]), float(abs(a[-1] - a[0]))


# ---------------------------------------------------------------- 点列の分割当てはめ
def segment_path(pts, tol=1.3, min_arc_pts=14, closed=False):
    """画素の点列を、線分・円弧の並びに分割する（再帰分割）。"""
    out = []

    def rec(p):
        if len(p) < 2:
            return
        m, d, a, b, dev = fit_line(p)
        if dev <= tol:
            fwd = np.dot(p[0] - m, d) < np.dot(p[-1] - m, d)
            out.append(dict(kind="line", p0=a if fwd else b, p1=b if fwd else a, pts=p))
            return
        if len(p) >= min_arc_pts:
            fc = fit_circle(p)
            if fc is not None:
                c, r, cdev = fc
                a0, a1, sweep = arc_angles(c, p)
                if cdev <= tol and r < 4000 and sweep > np.deg2rad(18) and r > 3:
                    out.append(dict(kind="arc", c=c, r=r, a0=a0, a1=a1, pts=p))
                    return
        # 弦から最も離れた点で分割
        ch = p[-1] - p[0]
        L = np.hypot(*ch)
        if L < 1e-6:
            k = int(np.argmax(np.hypot(*(p - p[0]).T)))
        else:
            n = np.array([-ch[1], ch[0]]) / L
            k = int(np.argmax(np.abs((p - p[0]) @ n)))
        k = min(max(k, 2), len(p) - 3)
        if k <= 1 or k >= len(p) - 2:
            out.append(dict(kind="line", p0=p[0], p1=p[-1], pts=p))
            return
        rec(p[:k + 1])
        rec(p[k:])

    if closed:
        fc = fit_circle(pts) if len(pts) >= min_arc_pts else None
        if fc is not None:
            c, r, cdev = fc
            if cdev <= max(tol, 0.06 * r) and r > 3:
                return [dict(kind="circle", c=c, r=r, pts=pts)]
    rec(pts)
    return merge_adjacent(out, tol)


def merge_adjacent(prims, tol):
    """同じ点列から出た隣り合う線分が一直線なら結合する"""
    if not prims:
        return prims
    res = [prims[0]]
    for q in prims[1:]:
        p = res[-1]
        if p["kind"] == "line" and q["kind"] == "line":
            P = np.vstack([p["pts"], q["pts"]])
            m, d, a, b, dev = fit_line(P)
            if dev <= tol:
                s = np.dot(P[0] - m, d) < np.dot(P[-1] - m, d)
                res[-1] = dict(kind="line", p0=a if s else b, p1=b if s else a, pts=P)
                continue
        if p["kind"] == "arc" and q["kind"] == "arc":
            P = np.vstack([p["pts"], q["pts"]])
            fc = fit_circle(P)
            if fc and fc[2] <= tol:
                a0, a1, sw = arc_angles(fc[0], P)
                res[-1] = dict(kind="arc", c=fc[0], r=fc[1], a0=a0, a1=a1, pts=P)
                continue
        res.append(q)
    return res


# ---------------------------------------------------------------- 線の太さ
def measure_width(ink, p0, p1, step=3.0, margin=None):
    """線分に直交する方向にインクの連続長を測り、太さ(px)を返す"""
    H, W = ink.shape
    v = p1 - p0
    L = np.hypot(*v)
    if L < 1:
        return np.nan
    d = v / L
    n = np.array([-d[1], d[0]])
    if margin is None:
        margin = min(6.0, L / 4)
    ts = np.arange(margin, L - margin + 1e-6, step)
    if len(ts) == 0:
        ts = np.array([L / 2])
    offs = np.arange(-12, 12.01, 0.25)
    runs = []
    for t in ts:
        c = p0 + t * d
        q = np.round(c[None, :] + offs[:, None] * n[None, :]).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < W) & (q[:, 1] < H)
        v_ = np.zeros(len(offs), bool)
        v_[ok] = ink[q[ok, 1], q[ok, 0]] > 0
        i0 = len(offs) // 2
        # 中心付近のインクを探す
        cand = np.nonzero(v_[i0 - 8:i0 + 9])[0]
        if len(cand) == 0:
            continue
        j = i0 - 8 + cand[np.argmin(np.abs(cand - 8))]
        lo = j
        while lo > 0 and v_[lo - 1]:
            lo -= 1
        hi = j
        while hi < len(offs) - 1 and v_[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(offs) - 1:
            continue
        runs.append((hi - lo) * 0.25 + 1.0)
    if not runs:
        return np.nan
    runs = np.sort(runs)
    return float(runs[int(0.4 * (len(runs) - 1))])


def measure_width_curve(ink, pts):
    ws = []
    for i in range(0, len(pts) - 6, 6):
        w = measure_width(ink, pts[i], pts[i + 6], step=3, margin=1)
        if not np.isnan(w):
            ws.append(w)
    return float(np.median(ws)) if ws else np.nan


# ---------------------------------------------------------------- 骨格線の追跡
def trace_skeleton(mask, dt):
    """mask（線の画素）を細線化し、分岐点間の点列を返す。

    戻り値: paths: list of dict(pts=(N,2) xy, ends=(junction_id or -1, ...), closed=bool)
            junctions: dict id -> 中心(xy)
    """
    sk = skeletonize(mask > 0)
    if sk.sum() == 0:
        return [], {}
    S = Skeleton(sk)
    deg = S.degrees  # 画素ごとの近傍数（平坦化された配列）
    coords = S.coordinates  # (row, col)
    # 分岐画素の塊ごとに番号をつける
    jmask = np.zeros(mask.shape, np.uint8)
    jidx = np.nonzero(deg >= 3)[0]
    jr, jc = coords[jidx, 0].astype(int), coords[jidx, 1].astype(int)
    jmask[jr, jc] = 1
    jmask = cv2.dilate(jmask, np.ones((3, 3), np.uint8))
    nj, jlab = cv2.connectedComponents(jmask, connectivity=8)
    # 分岐画素の塊ごとの中心（塊の数が多くても速いように、まとめて集計する）
    ys, xs = np.nonzero((jlab > 0) & sk)
    ids = jlab[ys, xs]
    cnt = np.bincount(ids, minlength=nj)
    sx = np.bincount(ids, weights=xs, minlength=nj); sy = np.bincount(ids, weights=ys, minlength=nj)
    junctions = {k: np.array([sx[k] / cnt[k], sy[k] / cnt[k]]) for k in range(1, nj) if cnt[k] > 0}
    paths = []
    for i in range(S.n_paths):
        rc = S.path_coordinates(i)
        pts = np.c_[rc[:, 1], rc[:, 0]].astype(float)
        ids = S.path(i)
        e0 = int(jlab[int(rc[0, 0]), int(rc[0, 1])]) if deg[ids[0]] >= 3 else -1
        e1 = int(jlab[int(rc[-1, 0]), int(rc[-1, 1])]) if deg[ids[-1]] >= 3 else -1
        closed = (ids[0] == ids[-1]) and len(ids) > 3 and e0 == -1
        if e0 == -1 and e1 == -1 and len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < 1.5:
            closed = True
        paths.append(dict(pts=pts, ends=(e0, e1), closed=closed))
    return paths, junctions


def trim_at_junctions(pts, ends, junctions, dt):
    """分岐点付近の骨格のゆがみを除くため、端から局所の線幅分だけ点を落とす"""
    a, b = 0, len(pts)
    if ends[0] > 0:
        c = junctions.get(ends[0], pts[0])
        r = dt[int(round(c[1])), int(round(c[0]))] + 2.0 if 0 <= int(c[1]) < dt.shape[0] else 3
        while a < b - 2 and np.hypot(*(pts[a] - c)) < r:
            a += 1
    if ends[1] > 0:
        c = junctions.get(ends[1], pts[-1])
        r = dt[int(round(c[1])), int(round(c[0]))] + 2.0 if 0 <= int(c[1]) < dt.shape[0] else 3
        while b > a + 2 and np.hypot(*(pts[b - 1] - c)) < r:
            b -= 1
    return pts[a:b]


def smooth_path(pts, k=2):
    """画素の階段状のギザギザを弱めるため移動平均（端点は保持）"""
    if len(pts) < 2 * k + 3:
        return pts
    out = pts.copy()
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    for j in range(2):
        v = np.convolve(np.r_[np.full(k, pts[0, j]), pts[:, j], np.full(k, pts[-1, j])], ker, mode="valid")
        out[:, j] = v
    out[0], out[-1] = pts[0], pts[-1]
    return out
