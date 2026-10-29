"""取り出した図形をレイヤーに振り分ける。

判定の根拠（推測で意味を決めつけないため、根拠が弱いものは「要確認」や記号名のままのレイヤーにする）
- 図枠：用紙の外枠より外側、表題欄の中
- 通り芯：校正で見つけた通り芯の線と同じ直線上にある線、通り芯記号の丸
- 寸法：通り芯記号と寸法線が並ぶ範囲（通り芯の外側）の線・点・数字
- 配管：太線。線の途中に挟まる文字（目視で確認した「R」など）で系統を決め、つながっている太線に広げる
- ハッチング：等間隔に並ぶ短い平行線の群
- それ以外の細線・中線：建築（背景）
"""
import re
import numpy as np
import cv2
from scipy.spatial import cKDTree

W_THIN = 4.75     # これ未満を細線 (px, 600dpi)
W_THICK = 6.75    # これ以上を太線
W_XTHICK = 8.4    # これ以上を極太線（主な配管）
# 太さの区分ごとの線の太さ（lineweight, 1/100mm）：区分ごとの実測の中央値に近い値
LW_CLASS = {"thin": 15, "med": 20, "thick": 30, "xthick": 40}


def wclass4(w):
    c = wclass(w)
    if c == "thick" and w >= W_XTHICK:
        return "xthick"
    return c

LAYERS = [
    # name, 日本語の意味, linetype, ACI color
    ("FRAME", "図枠・表題欄の線", "CONTINUOUS", 8),
    ("FRAME_TEXT", "表題欄の文字（線のまま）", "CONTINUOUS", 8),
    ("AXIS", "通り芯（記号から伸びる線）", "CONTINUOUS", 1),
    ("AXIS_SYM", "通り芯記号の丸", "CONTINUOUS", 1),
    ("AXIS_TEXT", "通り芯符号（X1・A1 など、文字データ）", "CONTINUOUS", 1),
    ("DIM", "寸法線・寸法の点", "CONTINUOUS", 3),
    ("DIM_TEXT", "寸法値（文字データ）", "CONTINUOUS", 3),
    ("A_THIN", "建築 細線（壁・建具・家具など）", "CONTINUOUS", 8),
    ("A_MED", "中線（建築の外形・引出線などが混在、用途は要確認）", "CONTINUOUS", 9),
    ("A_DASHED", "建築 破線（上部・隠れ線など）", "DASHED", 8),
    ("A_HATCH", "ハッチング・階段の段など（等間隔の平行線群）", "CONTINUOUS", 252),
    ("L_CENTER", "一点鎖線（用途は要確認）", "CENTER", 30),
    ("L_PHANTOM", "二点鎖線（用途は要確認）", "PHANTOM", 34),
    ("P_R", "配管「R」系統（線の途中に R。雨水排水管と思われるが要確認）", "CONTINUOUS", 5),
    ("P_R_MARK", "配管の線の途中の記号「R」（線のまま）", "CONTINUOUS", 5),
    ("P_NOMARK", "太線の配管（線の途中に記号なし。系統は要確認）", "CONTINUOUS", 6),
    ("P_DASHED", "破線の太線・中線（配管と思われる。系統は要確認）", "DASHED", 200),
    ("P_LEADER", "配管径などの引出線・下線", "CONTINUOUS", 140),
    ("P_SIZE", "配管径などの数値（文字データ）", "CONTINUOUS", 140),
    ("SYM", "器具・ドレン・桝などの記号（丸・四角・黒丸など）", "CONTINUOUS", 4),
    ("TEXT", "文字（文字データ）", "CONTINUOUS", 7),
    ("TEXT_SHAPE", "文字（判読に自信がないため線の形のまま）", "CONTINUOUS", 7),
    ("CHECK", "要確認（分類できなかったもの）", "CONTINUOUS", 2),
    ("REF_IMAGE", "参照用の原図画像（初期は非表示）", "CONTINUOUS", 8),
]
LAYER_INFO = {n: dict(jp=j, lt=l, color=c) for n, j, l, c in LAYERS}
AXIS_TEXT_RATIO = 0.80    # 通り芯記号の文字の高さ／丸の半径（506のX・Y通り22個で実測した中央値 0.80）


def wclass(w):
    if w is None or np.isnan(w):
        return "nan"
    return "thin" if w < W_THIN else ("med" if w < W_THICK else "thick")


def length(s):
    if s["kind"] == "line":
        return float(np.hypot(*(s["p1"] - s["p0"])))
    if s["kind"] == "circle":
        return 2 * np.pi * s["r"]
    if s["kind"] == "arc":
        return abs(s["a1"] - s["a0"]) * s["r"]
    P = s["pts"]
    return float(np.sum(np.hypot(*np.diff(P, axis=0).T)))


def ends(s):
    if s["kind"] == "line":
        return [s["p0"], s["p1"]]
    return [s["pts"][0], s["pts"][-1]]


def mid(s):
    if s["kind"] == "line":
        return (s["p0"] + s["p1"]) / 2
    if s["kind"] == "circle":
        return s["c"]
    return s["pts"][len(s["pts"]) // 2]


# ---------------------------------------------------------------- 図枠
def detect_frame(ink):
    """表題欄の上端・下端と、スキャナの読み取り端（右端の灰色の帯）を求める。
    この図面には外枠の縦線がなく、用紙の縁に目盛り（短い線）があるだけ。"""
    H, W = ink.shape
    row = ink[:, 500:W - 500].mean(1)
    long_rows = np.nonzero(row > 0.5)[0]
    long_rows = long_rows[long_rows > 0.85 * H]
    title_top = int(long_rows.min()) if len(long_rows) else H
    title_bot = int(long_rows.max()) if len(long_rows) else H
    d = np.convolve(ink[1000:min(H, 6000)].mean(0), np.ones(20) / 20, "same")
    x1 = W
    for x in range(W - 300, W - 60):
        if d[x:x + 60].mean() > 0.018:
            x1 = x
            break
    return dict(title_top=title_top, title_bot=title_bot, scan_x1=int(x1) - 5, W=W, H=H, edge=220)


# ---------------------------------------------------------------- 本体
def classify(r, calib, review, frame, page_cfg=None, ink=None):
    ents = []

    def add(layer, s, **kw):
        e = dict(s); e["layer"] = layer; e.update(kw); ents.append(e)

    # ---- 文字のまとまりと目視判定の対応
    rv = review["blocks"]
    rc = np.array([b["c"] for b in rv]) if rv else np.zeros((0, 2))
    rtree = cKDTree(rc) if len(rc) else None
    blocks = []
    for b in r["text_blocks"]:
        x, y, w, h = b["bbox"]
        c = np.array([x + w / 2, y + h / 2])
        lab = None
        if rtree is not None:
            d, k = rtree.query(c)
            if d < 12:
                lab = rv[k]["label"]
        if lab is None:
            lab = "G" if b["auto"] == "G" else "?"
        b = dict(b); b["label"] = lab; b["c"] = c
        blocks.append(b)

    # ---- 通り芯
    axes = []
    all_axes = dict(calib["axes"]); all_axes.update(calib.get("axes_secondary", {}))
    for fam, lst in all_axes.items():
        for a in lst:
            if a["p0"] is None:
                continue
            bc = np.array(a["bubble"][:2]); br = a["bubble"][2]
            p1 = np.array(a["p1"])
            d = (p1 - bc) / np.linalg.norm(p1 - bc)
            rot = 0.0
            if page_cfg:
                for f in page_cfg["families"] + page_cfg.get("secondary", []):
                    if f["name"] == fam:
                        rot = f.get("label_rot", 0.0)
            axes.append(dict(fam=fam, label=a["label"], c=bc, r=br, d=d, far=float(np.dot(p1 - bc, d)), rot=rot))
    zones = {}
    for fam in all_axes:
        P = []
        for a in axes:
            if a["fam"] == fam:
                # 寸法線は記号から半径の約4倍以内にある。それより先（建物側）は寸法の範囲に含めない
                P += [a["c"], a["c"] + a["d"] * min(a["far"] + 30, 4.5 * a["r"])]
        if len(P) >= 3:
            zones[fam] = cv2.convexHull(np.array(P, np.float32))

    def in_zone(p, margin=45):
        for fam, hull in zones.items():
            if cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) >= -margin:
                return fam
        return None

    fam_dir = {}
    for a in axes:
        fam_dir.setdefault(a["fam"], a["d"])

    def along_family(s, fam):
        """寸法線・補助線は通り芯に平行か直角"""
        if s["kind"] != "line":
            return s["kind"] in ("circle", "arc") and s["r"] < 15
        v = s["p1"] - s["p0"]; L = np.hypot(*v)
        if L < 1:
            return False
        c = abs(np.dot(v / L, fam_dir[fam]))
        return c > np.cos(np.deg2rad(3)) or c < np.sin(np.deg2rad(3))

    def axis_of(s):
        if s["kind"] != "line":
            return None
        v = s["p1"] - s["p0"]; L = np.hypot(*v)
        if L < 5:
            return None
        u = v / L
        for a in axes:
            if abs(np.dot(u, a["d"])) < np.cos(np.deg2rad(2.0)):
                continue
            n = np.array([-a["d"][1], a["d"][0]])
            if max(abs(np.dot(s["p0"] - a["c"], n)), abs(np.dot(s["p1"] - a["c"], n))) > 5:
                continue
            t0, t1 = np.dot(s["p0"] - a["c"], a["d"]), np.dot(s["p1"] - a["c"], a["d"])
            if min(t0, t1) > a["r"] - 10 and max(t0, t1) < a["far"] + 60:
                return a
        return None

    def bubble_of(p, r_=None):
        for a in axes:
            if np.hypot(*(p - a["c"])) < (a["r"] * 0.9 if r_ is None else 10) and \
                    (r_ is None or abs(r_ - a["r"]) < 0.2 * a["r"]):
                return a
        return None

    def scan_edge(p):
        return p[0] >= frame["scan_x1"]

    def in_frame(p):
        return p[1] < frame["title_top"] - 4 and not scan_edge(p)

    def edge_tick(s):
        """用紙の縁の目盛り（縁に直交する短い線）"""
        if s["kind"] != "line" or length(s) > 150:
            return False
        m_ = mid(s); v = np.abs(s["p1"] - s["p0"]); e = frame["edge"]
        if m_[1] < e and v[1] > 3 * v[0]:
            return True
        if (m_[0] < e or m_[0] > frame["scan_x1"] - 250) and v[0] > 3 * v[1]:
            return True
        return False

    # ---- 文字のまとまりの処理
    inline = []      # 線の途中の記号
    text_polys = []  # 文字データにした文字の範囲
    leaders_from = []
    for b in blocks:
        lab = b["label"]
        prims = b.get("prims", [])
        if lab == "N":
            continue
        if scan_edge(b["c"]):
            continue
        if not in_frame(b["c"]):
            for s in prims:
                add("FRAME_TEXT", s)
            continue
        if lab == "AX":
            a = bubble_of(b["c"])
            if a is None:
                for s in prims:
                    add("TEXT_SHAPE", s)
                continue
            a.setdefault("blocks", []).append(b)
            continue
        if lab.startswith("L:"):
            sym = lab[2:]
            inline.append(dict(c=b["c"], sym=sym, prims=prims))
            continue
        if lab == "S":
            for s in prims:
                add("SYM", s)
            continue
        if lab == "G":
            for s in prims:
                ents.append(dict(s, layer=None, src="block"))
            continue
        if lab.startswith("T:"):
            txt = lab[2:]
            if txt.endswith("@180"):       # 目視で、自動推定の向きと上下が逆と判定したもの
                txt = txt[:-4]
                b["angle"] = (b.get("angle") or 0.0) + 180.0
            # 文字の中の長い直線（下線・寸法線）と、文字の帯の外にある線は線として残す
            hgt = text_height(b)
            keep = [s for s in prims if s["kind"] == "line" and length(s) > max(1.3 * hgt, 30)]
            import write_dxf as WD
            band = WD.text_band_px(dict(block=b, exclude=keep), r["text_mask"], r["origin"])
            if band is not None:
                poly = np.array([band["bl"], band["br"], band["tr"], band["tl"]], np.float32)
                for s in prims:
                    if any(s is k for k in keep):
                        continue
                    if cv2.pointPolygonTest(poly, tuple(map(float, mid(s))), True) < -3:
                        keep.append(s)
                hgt = float(np.hypot(*(band["tl"] - band["bl"])))
                # 文字の帯を前後に少し広げた範囲（「1」など直線の字が線側に取られた分を文字に戻すため）
                u_ = band["br"] - band["bl"]; u_ = u_ / max(1e-6, np.hypot(*u_))
                v_ = band["tl"] - band["bl"]; v_ = v_ / max(1e-6, np.hypot(*v_))
                ex = 0.7 * hgt; ey = 0.15 * hgt
                text_polys.append((np.array([band["bl"] - ex * u_ - ey * v_, band["br"] + ex * u_ - ey * v_,
                                             band["tr"] + ex * u_ + ey * v_, band["tl"] - ex * u_ + ey * v_],
                                            np.float32), hgt))
            zone = in_zone(b["c"], margin=20)
            numeric = re.fullmatch(r"[0-9.,x×/φ]+(pe|PE|VP|vp|A)?", txt) is not None
            layer = "DIM_TEXT" if zone else ("P_SIZE" if numeric else "TEXT")
            ents.append(dict(kind="text", text=txt, layer=layer, block=b, height=hgt, exclude=keep))
            for s in keep:
                ents.append(dict(s, layer=("DIM" if zone else "P_LEADER"), src="underline"))
            if not zone and numeric:
                leaders_from.append(b)
            continue
        for s in prims:                      # "?"
            add("TEXT_SHAPE", s)

    # 通り芯符号：記号の順番で決まる符号を文字データにする
    for a in axes:
        if not a.get("blocks"):
            continue      # この範囲に符号の文字がない
        # 記号の中の文字の大きさを実測
        ents.append(dict(kind="text", text=a["label"], layer="AXIS_TEXT", center=a["c"],
                         height=a["r"] * AXIS_TEXT_RATIO, rot_mm=a["rot"]))

    # ---- 線・曲線
    geo = []
    for s in r["lines"]:
        geo.append(dict(s, src="line"))
    for s in r["curves"]:
        geo.append(dict(s, src="curve"))
    for s in r["dashed"]:
        geo.append(dict(s, src="dashed"))
    geo += [e for e in ents if e.get("layer", "x") is None]
    ents = [e for e in ents if e.get("layer", "x") is not None]

    rest = []
    dropped = 0
    absorbed = 0
    for s in geo:
        m_ = mid(s)
        if s["kind"] == "line" and wclass(s.get("w")) in ("med", "thick") and text_polys:
            hit = False
            for poly, hg in text_polys:
                if length(s) < 1.5 * hg and all(cv2.pointPolygonTest(poly, tuple(map(float, q)), False) >= 0
                                                for q in (s["p0"], s["p1"])):
                    hit = True; break
            if hit:
                absorbed += 1; continue      # 文字データの一部（「1」の縦棒など）
        if scan_edge(m_):
            dropped += 1; continue
        if not in_frame(m_) or edge_tick(s):
            add("FRAME", s); continue
        if s["kind"] in ("circle", "arc") and bubble_of(s["c"], s["r"]) is not None:
            add("AXIS_SYM", s); continue
        if axis_of(s) is not None:
            add("AXIS", s); continue
        z = in_zone(m_)
        if z and all(in_zone(e) for e in ends(s)) and s.get("linetype", "CONTINUOUS") == "CONTINUOUS" \
                and along_family(s, z):
            add("DIM", s); continue
        rest.append(s)

    # ---- 配管（太線）
    # 配管の候補：太線で、長いもの（60px=約2.5mm以上）か、長い太線に端がつながるもの
    def thick_solid(s):
        return wclass(s.get("w")) == "thick" and s.get("linetype", "CONTINUOUS") == "CONTINUOUS" and length(s) >= 20
    tcand = [s for s in rest if thick_solid(s)]
    longs = [s for s in tcand if length(s) >= 60]
    lsamp = []
    for e in longs:
        if e["kind"] == "line":
            for t in np.linspace(0, 1, max(2, int(length(e) / 4))):
                lsamp.append(e["p0"] + t * (e["p1"] - e["p0"]))
        else:
            lsamp += list(e["pts"])
    ltree = cKDTree(np.array(lsamp)) if lsamp else None
    def is_pipe_cand(s):
        if not thick_solid(s):
            return False
        if length(s) >= 60:
            return True
        return ltree is not None and min(ltree.query(np.array(ends(s)))[0]) < 10
    thick = [s for s in rest if is_pipe_cand(s)]
    others = [s for s in rest if not is_pipe_cand(s)]
    n = len(thick)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)
    E = []
    for i, s in enumerate(thick):
        for e in ends(s):
            E.append((e, i))
    if E:
        tree = cKDTree(np.array([e for e, i in E]))
        for k1, k2 in tree.query_pairs(16):
            union(E[k1][1], E[k2][1])
        # 同一直線上で、交差部などで少し離れた太線どうし
        for i, si in enumerate(thick):
            if si["kind"] != "line":
                continue
            for j in range(i + 1, n):
                sj = thick[j]
                if sj["kind"] != "line" or find(i) == find(j):
                    continue
                vi = si["p1"] - si["p0"]; vj = sj["p1"] - sj["p0"]
                li, lj = np.hypot(*vi), np.hypot(*vj)
                if li < 5 or lj < 5 or abs(np.dot(vi, vj)) / (li * lj) < np.cos(np.deg2rad(3)):
                    continue
                gap = min(np.hypot(*(a - b)) for a in (si["p0"], si["p1"]) for b in (sj["p0"], sj["p1"]))
                if gap > 80:
                    continue
                ui = vi / li; nrm = np.array([-ui[1], ui[0]])
                if max(abs(np.dot(sj["p0"] - si["p0"], nrm)), abs(np.dot(sj["p1"] - si["p0"], nrm))) < 5:
                    union(i, j)
        # T字：端点が他の太線の途中に接する
        for e, i in E:
            for j, s in enumerate(thick):
                if j == i or s["kind"] != "line":
                    continue
                if seg_dist(e, s["p0"], s["p1"]) < 9:
                    union(i, j)
    # 太線のすき間に挟まった記号を探す（目視判定で「L:」にしたもの以外も拾う）
    # 自動で読むのは、目視でこの図面に出てくると確認した記号だけ（OCRの誤読で系統を作らない）
    known_syms = set(m["sym"] for m in inline)
    gap_unknown = []
    templates = build_templates(ink, inline, thick) if ink is not None else []
    for i, si in enumerate(thick):
        if si["kind"] != "line":
            continue
        for j in range(i + 1, n):
            sj = thick[j]
            if sj["kind"] != "line":
                continue
            vi = si["p1"] - si["p0"]; vj = sj["p1"] - sj["p0"]
            li, lj = np.hypot(*vi), np.hypot(*vj)
            if li < 25 or lj < 25 or abs(np.dot(vi, vj)) / (li * lj) < np.cos(np.deg2rad(4)):
                continue
            pairs_ = [(a, b) for a in (si["p0"], si["p1"]) for b in (sj["p0"], sj["p1"])]
            a, b = min(pairs_, key=lambda ab: np.hypot(*(ab[0] - ab[1])))
            gap = np.hypot(*(a - b))
            if not (22 <= gap <= 110):
                continue
            ui = vi / li; nrm = np.array([-ui[1], ui[0]])
            if max(abs(np.dot(sj["p0"] - si["p0"], nrm)), abs(np.dot(sj["p1"] - si["p0"], nrm))) > 6:
                continue
            # すき間が2本の外側に向いていること（重なりではない）
            far_i = si["p0"] if np.allclose(a, si["p1"]) else si["p1"]
            if np.dot(b - a, a - far_i) <= 0:
                continue
            c = (a + b) / 2
            if any(np.hypot(*(m["c"] - c)) < 45 for m in inline):
                continue
            sym = match_gap_symbol(ink, a, b, templates) if ink is not None else None
            # すき間の中の図形（記号の線）を集める
            u_ = (b - a) / gap
            def inside(q):
                t = np.dot(q - a, u_); o = abs(np.dot(q - a, np.array([-u_[1], u_[0]])))
                return -3 < t < gap + 3 and o < 30
            members = [x for x in others if not x.get("_used") and x["kind"] != "dot" and inside(mid(x))
                       and length(x) < 60]
            if not members:
                continue
            if sym in known_syms:
                inline.append(dict(c=c, sym=sym, prims=[], auto=True, members=members))
            else:
                gap_unknown.append(c)
    # 線の途中の記号：記号の両側にある太線に系統を付け、両側どうしもつなぐ
    sysmark = {}
    for m in inline:
        near = []
        for i, s in enumerate(thick):
            if s["kind"] != "line":
                continue
            v = s["p1"] - s["p0"]; L = np.hypot(*v)
            if L < 1:
                continue
            u = v / L; nrm = np.array([-u[1], u[0]])
            off = abs(np.dot(m["c"] - s["p0"], nrm))
            de = min(np.hypot(*(m["c"] - s["p0"])), np.hypot(*(m["c"] - s["p1"])))
            if off < 14 and de < 75:
                near.append(i)
        for i in near:
            sysmark.setdefault(i, set()).add(m["sym"])
        for i in near[1:]:
            union(near[0], i)
        m["near"] = near
    comp_sym = {}
    for i in range(n):
        if i in sysmark:
            comp_sym.setdefault(find(i), set()).update(sysmark[i])
    checks = []
    for i, s in enumerate(thick):
        syms = comp_sym.get(find(i), set())
        if len(syms) == 1:
            add("P_" + next(iter(syms)), s)
        elif len(syms) > 1:
            add("CHECK", s, why="太線に異なる記号 " + ",".join(sorted(syms)))
            checks.append(("太線に異なる記号", mid(s)))
        else:
            add("P_NOMARK", s)
    for m in inline:
        lay = "P_%s_MARK" % m["sym"]
        if lay not in LAYER_INFO:
            LAYER_INFO[lay] = dict(jp="配管の線の途中の記号「%s」（線のまま）" % m["sym"], lt="CONTINUOUS", color=5)
        for s in m["prims"]:
            add(lay, s)
        for s in m.get("members", []):
            s["_used"] = lay
    for c in gap_unknown:
        checks.append(("太線のすき間に記号らしきものがあるが読めない", c))
        if not m["near"]:
            checks.append(("記号「%s」の両側に太線が見つからない" % m["sym"], m["c"]))

    # ---- 引出線（配管径の数字の下線につながる線）
    pipe_pts = [e for e in ents if e.get("layer", "").startswith("P_") and e["kind"] == "line"]
    und = [e for e in ents if e.get("src") == "underline" and e["layer"] == "P_LEADER"]
    lead_ends = [p for u in und for p in (u["p0"], u["p1"])]
    for b in leaders_from:
        x, y, w, h = b["bbox"]
        for s in others:
            if s["kind"] != "line" or s.get("layer"):
                continue
            # まとまりのすぐ下にある水平に近い線
            if s.get("_used"):
                continue
            c = mid(s)
            if x - 10 < c[0] < x + w + 10 and y + 0.5 * h < c[1] < y + h + 18 and length(s) > 0.5 * w:
                s["_used"] = "P_LEADER"; lead_ends += [s["p0"], s["p1"]]
    for hop in range(2):
        if not lead_ends:
            break
        lt = cKDTree(np.array(lead_ends))
        new = []
        for s in others:
            if s["kind"] != "line" or s.get("_used") or wclass(s.get("w")) == "thick" or length(s) > 900:
                continue
            for e in ends(s):
                if lt.query(e)[0] < 10:
                    s["_used"] = "P_LEADER"; new += [s["p0"], s["p1"]]; break
        lead_ends += new

    # ---- ハッチング（等間隔の平行線群）
    hatch_idx = find_hatch([s for s in others if not s.get("_used")])

    # ---- 記号（配管の端につく丸・黒丸）
    pipe_end_pts = [p for e in ents if e.get("layer", "").startswith("P_") and e["kind"] in ("line",)
                    for p in (e["p0"], e["p1"])]
    pe_tree = cKDTree(np.array(pipe_end_pts)) if pipe_end_pts else None

    # 配管の線の近くか・文字のまとまりの中か を調べる道具
    pipe_lines = [e for e in ents if e.get("layer", "").startswith("P_") and not e["layer"].endswith("_MARK")
                  and e["kind"] == "line"]
    samp, slay = [], []
    for e in pipe_lines:
        L = max(2, int(length(e) / 4))
        for t in np.linspace(0, 1, L):
            samp.append(e["p0"] + t * (e["p1"] - e["p0"])); slay.append(e["layer"])
    st_ = cKDTree(np.array(samp)) if samp else None

    def pipe_near(s):
        if st_ is None:
            return None
        d, k = st_.query(np.array(ends(s) + [mid(s)]))
        k = k[d < 8]
        return slay[k[0]] if len(k) else None
    boxes = np.array([b["bbox"] for b in blocks]) if blocks else np.zeros((0, 4))

    def in_block_box(p):
        if not len(boxes):
            return False
        return bool(((boxes[:, 0] <= p[0]) & (p[0] <= boxes[:, 0] + boxes[:, 2]) &
                     (boxes[:, 1] <= p[1]) & (p[1] <= boxes[:, 1] + boxes[:, 3])).any())

    # 短い太線・中線が集まっている所は手描きの太い文字とみなす（3本以上が40px以内）
    shorts = [x for x in others if wclass(x.get("w")) in ("thick", "med") and length(x) < 60]
    sh_tree = cKDTree(np.array([mid(x) for x in shorts])) if shorts else None

    def short_cluster(s):
        if sh_tree is None:
            return False
        return len(sh_tree.query_ball_point(mid(s), 40)) >= 4     # 自分を含めて4本

    for s in others:
        if s.get("_used"):
            add(s["_used"], s); continue
        if id(s) in hatch_idx:
            add("A_HATCH", s); continue
        lt = s.get("linetype", "CONTINUOUS")
        wc = wclass(s.get("w"))
        if lt == "DASHED":
            add("A_DASHED" if wc == "thin" else "P_DASHED", s); continue
        if lt == "CENTER":
            add("L_CENTER", s); continue
        if lt == "PHANTOM":
            add("L_PHANTOM", s); continue
        if s["kind"] in ("circle", "arc") and s["r"] < 45 and pe_tree is not None:
            if pe_tree.query(s["c"])[0] < s["r"] + 14:
                add("SYM", s); continue
        if wc == "thick":
            if length(s) < 60:
                lay = pipe_near(s)
                if lay:
                    add(lay, s); continue
                if in_block_box(mid(s)) or short_cluster(s):
                    add("TEXT_SHAPE", s); continue
                add("CHECK", s, why="短い太線（文字の一部か配管の一部か不明）"); continue
            add("P_NOMARK", s); continue
        if wc == "nan":
            add("CHECK", s, why="線の太さを測れない"); continue
        add("A_THIN" if wc == "thin" else "A_MED", s)

    for d in r["fdots"]:
        s = dict(kind="dot", c=d["c"], r=d["r"])
        if scan_edge(d["c"]):
            continue
        if not in_frame(d["c"]):
            add("FRAME", s)
        elif in_zone(d["c"]):
            add("DIM", s)
        elif in_block_box(d["c"]):
            add("TEXT_SHAPE", s)       # 太い文字（「室」など）の線の重なりが黒丸に見えたもの
        else:
            add("SYM", s)
    # かすれで切れた実線をつないでから（切れ目にインクが残るもの）、白い切れ目が等間隔に並ぶものを破線にする
    if ink is not None:
        for lay, gmax in (("DIM", 45), ("AXIS", 45), ("A_THIN", 30), ("A_MED", 30)):
            ents = merge_broken(ents, lay, gmax, ink)
    ents = merge_dash_chains(ents)
    return ents, checks


def merge_dash_chains(ents):
    """取り出し後の短い直線のうち、同一直線上に等間隔で3本以上並ぶもの（壁などに接していて
    最初の段階で破線と判定できなかったもの）を、1本の破線にまとめる。"""
    import extract as E
    src_layers = {"A_THIN", "A_MED", "CHECK", "P_NOMARK"}
    cand = [e for e in ents if e["kind"] == "line" and e["layer"] in src_layers and 8 <= length(e) <= 90]
    if len(cand) < 3:
        return ents
    pieces = []
    for e in cand:
        d = (e["p1"] - e["p0"]) / length(e)
        w = e.get("w")
        pieces.append(dict(p0=e["p0"], p1=e["p1"], d=d, m=(e["p0"] + e["p1"]) / 2, L=length(e),
                           th=w if (w is not None and not np.isnan(w)) else 4.0))
    used = set(); new = []
    for ch in E.link_dashes(pieces, np.zeros((0, 2)), max_gap=60, ang_tol=4.0, lat_tol=3.0):
        if len(ch) < 3:
            continue
        ws = np.array([pieces[i]["th"] for i in ch])
        if ws.max() > 1.8 * ws.min():
            continue
        pat = E.chain_pattern(ch, pieces, np.zeros((0, 2)), None)
        if E.decide_linetype(pat) != "DASHED" or not E.regular(pat):
            continue
        w = float(np.median(ws))
        lay = "A_DASHED" if wclass(w) == "thin" else "P_DASHED"
        new.append(dict(kind="line", p0=pat["a"], p1=pat["b"], w=w, linetype="DASHED", pat=pat, layer=lay,
                        src="dash2"))
        used.update(id(cand[i]) for i in ch)
    return [e for e in ents if id(e) not in used] + new


def merge_broken(ents, layer, max_gap, ink):
    """スキャンのかすれで切れた線をつなぐ。同じレイヤーで同一直線上にあり、
    切れ目にインクのかけらが残っている（＝もとは1本の線）場合だけつなぐ。
    切れ目が真っ白な場合（扉の開口など、もともと切れている線）はつながない。"""
    H, W = ink.shape
    def ok(e):
        return e["layer"] == layer and e["kind"] == "line" and e.get("linetype", "CONTINUOUS") == "CONTINUOUS"
    L = [e for e in ents if ok(e)]
    other = [e for e in ents if not ok(e)]
    changed = True
    while changed and len(L) > 1:
        changed = False
        E_ = np.array([e["p0"] for e in L] + [e["p1"] for e in L]); n = len(L)
        tree = cKDTree(E_)
        dead = set(); new = []
        for k1, k2 in sorted(tree.query_pairs(max_gap), key=lambda kk: np.hypot(*(E_[kk[0]] - E_[kk[1]]))):
            i, j = k1 % n, k2 % n
            if i == j or i in dead or j in dead:
                continue
            A, B = L[i], L[j]
            va, vb = A["p1"] - A["p0"], B["p1"] - B["p0"]
            la, lb = np.hypot(*va), np.hypot(*vb)
            if la < 3 or lb < 3 or abs(np.dot(va, vb)) / (la * lb) < np.cos(np.deg2rad(2.5)):
                continue
            P = np.vstack([A["p0"], A["p1"], B["p0"], B["p1"]])
            m = P.mean(0); u_, s_, vt = np.linalg.svd(P - m, full_matrices=False); d = vt[0]
            nrm = np.array([-d[1], d[0]])
            if np.abs((P - m) @ nrm).max() > 2.0:
                continue
            ta = sorted([np.dot(A["p0"] - m, d), np.dot(A["p1"] - m, d)])
            tb = sorted([np.dot(B["p0"] - m, d), np.dot(B["p1"] - m, d)])
            g0, g1 = (ta[1], tb[0]) if ta[0] < tb[0] else (tb[1], ta[0])
            if g1 - g0 < -2 or g1 - g0 > max_gap:
                continue
            if g1 - g0 > 3:
                ts = np.arange(g0 + 1, g1 - 1, 1.0)
                q = np.round(m + ts[:, None] * d).astype(int)
                q[:, 0] = np.clip(q[:, 0], 1, W - 2); q[:, 1] = np.clip(q[:, 1], 1, H - 2)
                hit = np.zeros(len(q), bool)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        hit |= ink[q[:, 1] + dy, q[:, 0] + dx] > 0
                if hit.mean() < 0.2:
                    continue
            t = (P - m) @ d
            wa, wb = A.get("w", np.nan), B.get("w", np.nan)
            wa = np.nan if wa is None else wa; wb = np.nan if wb is None else wb
            w = np.nansum([wa * la, wb * lb]) / max(1e-9, (la if not np.isnan(wa) else 0) + (lb if not np.isnan(wb) else 0)) \
                if not (np.isnan(wa) and np.isnan(wb)) else np.nan
            e = dict(A); e.update(p0=m + t.min() * d, p1=m + t.max() * d, w=w)
            dead.update((i, j)); new.append(e); changed = True
        L = [e for k, e in enumerate(L) if k not in dead] + new
    return other + L


def gap_glyph(ink, a, b, pad=4):
    """太線の端 a と b の間にある記号の画像を、線の向きを水平にして切り出し 32x32 に正規化する"""
    import ocr
    c = (a + b) / 2
    gap = np.hypot(*(b - a))
    u = (b - a) / max(gap, 1e-6)
    R_ = int(gap / 2 + 40)
    x0, y0 = int(c[0]) - R_, int(c[1]) - R_
    if x0 < 0 or y0 < 0:
        return None
    sub = ink[y0:y0 + 2 * R_, x0:x0 + 2 * R_].astype(np.uint8)
    ang = np.degrees(np.arctan2(u[1], u[0]))
    rimg, M = ocr.rotate(sub, ang)
    hh, ww = rimg.shape
    cy, cx = hh // 2, ww // 2
    win = rimg[cy - 30:cy + 30, max(0, cx - int(gap / 2) + pad):cx + int(gap / 2) - pad]
    if win.size == 0 or win.sum() < 30:
        return None
    ys, xs = np.nonzero(win)
    g = win[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if min(g.shape) < 8:
        return None
    return cv2.resize(g.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)


def build_templates(ink, inline, thick):
    """目視で確認した線の途中の記号から、照合用のお手本を作る"""
    out = []
    for m in inline:
        # 記号の両側の太線の端を探す
        best = None
        for s in thick:
            if s["kind"] != "line":
                continue
            for e in (s["p0"], s["p1"]):
                d = np.hypot(*(e - m["c"]))
                if d < 75:
                    best = (best or []) + [e]
        if not best or len(best) < 2:
            continue
        P = np.array(best)
        k = np.argsort(np.hypot(*(P - m["c"]).T))[:2]
        a, b = P[k[0]], P[k[1]]
        if np.dot(a - m["c"], b - m["c"]) > 0:
            continue
        for aa, bb in ((a, b), (b, a)):
            g = gap_glyph(ink, aa, bb)
            if g is not None:
                out.append((m["sym"], g))
    return out


def match_gap_symbol(ink, a, b, templates, thr=0.9):
    if not templates:
        return None
    g = gap_glyph(ink, a, b)
    if g is None:
        return None
    best = (None, 0.0)
    for sym, t in templates:
        # 少しぼかして重なり具合を比べる
        ga = cv2.GaussianBlur(g, (5, 5), 1.5); ta = cv2.GaussianBlur(t, (5, 5), 1.5)
        sc = float((ga * ta).sum() / np.sqrt((ga * ga).sum() * (ta * ta).sum() + 1e-9))
        if sc > best[1]:
            best = (sym, sc)
    GAP_SCORES.append(((a + b) / 2, best[0], best[1]))
    return best[0] if best[1] >= thr else None


GAP_SCORES = []


def read_gap_symbol(ink, a, b, u):
    """太線のすき間の文字を読む（線の向きに合わせて回して OCR）。確信がなければ None"""
    import ocr
    c = (a + b) / 2
    half = int(np.hypot(*(b - a)) / 2) + 6
    x0, y0 = int(c[0] - half - 30), int(c[1] - half - 30)
    sub = ink[max(0, y0):int(c[1] + half + 30), max(0, x0):int(c[0] + half + 30)].astype(np.uint8)
    if sub.size == 0:
        return None
    ang = np.degrees(np.arctan2(u[1], u[0]))
    best = None
    for extra in (0, 180, 90, -90):
        rimg, M = ocr.rotate(sub, ang + extra)
        hh, ww = rimg.shape
        # 線の太線部分を除いて、中央の文字だけ残す
        m = np.zeros_like(rimg); cy, cx = hh // 2, ww // 2
        m[cy - 28:cy + 28, cx - half + 4:cx + half - 4] = 1
        crop = rimg * m
        t, conf = ocr.run_tess(crop, "eng", 10, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if t and (best is None or conf > best[1]):
            best = (t.strip(), conf)
    if best and len(best[0]) == 1 and best[1] >= 80:
        return best[0]
    return None


def text_height(b):
    """文字の高さ(px)：向きをそろえたときの短辺"""
    x, y, w, h = b["bbox"]
    if b.get("angle"):
        a = np.deg2rad(b["angle"])
        # 回転した外接矩形の短辺の近似
        return float(min(w, h) * 0.8)
    return float(min(w, h) * 0.8)


def seg_dist(p, a, b):
    v = b - a; L2 = np.dot(v, v)
    if L2 < 1e-9:
        return float(np.hypot(*(p - a)))
    t = np.clip(np.dot(p - a, v) / L2, 0, 1)
    return float(np.hypot(*(p - (a + t * v))))


def find_hatch(cands, min_n=5):
    """短い平行線が等間隔に並ぶ群を探す。戻り値は該当する図形の id の集合"""
    L = [s for s in cands if s["kind"] == "line" and 8 <= length(s) <= 400 and wclass(s.get("w")) != "thick"]
    if len(L) < min_n:
        return set()
    C = np.array([mid(s) for s in L])
    ang = np.array([np.arctan2(*(s["p1"] - s["p0"])[::-1]) % np.pi for s in L])
    tree = cKDTree(C)
    parent = list(range(len(L)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in tree.query_pairs(45):
        da = abs(ang[i] - ang[j]); da = min(da, np.pi - da)
        if da > np.deg2rad(4):
            continue
        u = np.array([np.cos(ang[i]), np.sin(ang[i])]); n = np.array([-u[1], u[0]])
        dperp = abs(np.dot(C[j] - C[i], n))
        if not (5 <= dperp <= 40):
            continue
        # 長さ方向に重なっていること
        ti = sorted([np.dot(L[i]["p0"] - C[i], u), np.dot(L[i]["p1"] - C[i], u)])
        tj = sorted([np.dot(L[j]["p0"] - C[i], u), np.dot(L[j]["p1"] - C[i], u)])
        ov = min(ti[1], tj[1]) - max(ti[0], tj[0])
        if ov < 0.5 * min(ti[1] - ti[0], tj[1] - tj[0]):
            continue
        if max(length(L[i]), length(L[j])) > 1.6 * min(length(L[i]), length(L[j])):
            continue
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(L)):
        groups.setdefault(find(i), []).append(i)
    out = set()
    for g in groups.values():
        if len(g) < min_n:
            continue
        u = np.array([np.cos(ang[g[0]]), np.sin(ang[g[0]])]); n = np.array([-u[1], u[0]])
        offs = np.sort(C[g] @ n)
        dd = np.diff(offs)
        dd = dd[dd > 2]
        if len(dd) >= min_n - 1 and np.std(dd) / np.mean(dd) < 0.35:
            out.update(id(L[i]) for i in g)
    return out
