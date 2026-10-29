"""振り分けた図形を、実寸mm座標のDXFに書き出す。"""
import numpy as np
import cv2
import ezdxf
from ezdxf.enums import TextEntityAlignment
import classify as C
import ocr

PX_MM_PAPER = 25.4 / 600.0          # 1画素の紙面上の長さ(mm)
LW_TABLE = [5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211]


class Xform:
    """画素 → 実寸mm（校正で求めた変換）"""
    def __init__(self, M, t):
        self.M = np.array(M, float); self.t = np.array(t, float)
        self.Mi = np.linalg.inv(self.M)
        self.s = float(np.sqrt(abs(np.linalg.det(self.M))))
        self.flip = np.linalg.det(self.M) < 0

    def __call__(self, p):
        p = np.asarray(p, float)
        return p @ self.M.T + self.t

    def inv(self, q):
        return (np.asarray(q, float) - self.t) @ self.Mi.T

    def angle(self, a_img_deg):
        """画像上の向き（度, y下向き座標の atan2）→ mm 座標上の向き"""
        a = np.deg2rad(a_img_deg)
        v = self.M @ np.array([np.cos(a), np.sin(a)])
        return float(np.degrees(np.arctan2(v[1], v[0])))


def nearest_lw(mm):
    v = mm * 100
    return min(LW_TABLE, key=lambda x: abs(x - v))


def text_geometry(e, tmask, origin, X):
    """文字のまとまりの、文字の高さ・左下の位置・向きを求める（mm）"""
    g = text_band_px(e, tmask, origin)
    if g is None:
        return None
    bl, br, tl = g["bl"], g["br"], g["tl"]
    P0, P1, PT = X(bl), X(br), X(tl)
    rot = float(np.degrees(np.arctan2(*(P1 - P0)[::-1])))
    return dict(insert=P0, end=P1, height=float(np.hypot(*(PT - P0))), rot=rot, w_mm=float(np.hypot(*(P1 - P0))))


def text_band_px(e, tmask, origin):
    """文字の帯（左下・右下・左上の画素座標）"""
    b = e["block"]
    crop, o = ocr.block_crop(tmask, b["bbox"], origin, pad=4)
    # 下線・寸法線など、線として残す部分は文字の位置決めから除く
    for s in e.get("exclude", []):
        w = s.get("w", 4.0); w = 4.0 if np.isnan(w) else w
        cv2.line(crop, tuple(np.round(s["p0"] - o).astype(int)), tuple(np.round(s["p1"] - o).astype(int)), 0,
                 int(round(w + 4)))
    ang = b.get("angle") or 0.0
    if ang:
        rimg, M = ocr.rotate(crop, ang)
    else:
        rimg, M = crop, np.array([[1, 0, 0], [0, 1, 0]], float)
    ys, xs = np.nonzero(rimg)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # 下線などの長い線が含まれると高さを誤るため、行ごとのインク量で文字の帯を求める
    rows = rimg[:, x0:x1].sum(1)
    band = np.nonzero(rows > 0.06 * (x1 - x0))[0]
    if len(band):
        y0, y1 = band.min(), band.max() + 1
        cols = np.nonzero(rimg[y0:y1].sum(0))[0]
        if len(cols):
            x0, x1 = cols.min(), cols.max() + 1
    Mi = cv2.invertAffineTransform(M)
    def back(px, py):
        q = Mi @ np.array([px, py, 1.0])
        return q + np.array(o, float)
    bl = back(x0, y1); br = back(x1, y1); tl = back(x0, y0); tr = back(x1, y0)
    return dict(bl=bl, br=br, tl=tl, tr=tr)


_FONT = None


def width_factor(text, h, w_target):
    """原図の文字列の長さに合わせる幅係数（等幅ゴシックの字幅で計算）"""
    global _FONT
    from PIL import ImageFont
    if _FONT is None:
        _FONT = ImageFont.truetype("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", 100)
    nat = _FONT.getlength(text) / 100.0 * h
    if nat <= 0:
        return 1.0
    return float(np.clip(w_target / nat, 0.5, 2.0))


def arc_mm(c, r, a0, a1, X):
    """画像上の円弧（中心・半径・始角・終角, ラジアン）を mm 座標の DXF 円弧(反時計回り)にする"""
    am = (a0 + a1) / 2
    P = lambda a: X(c + r * np.array([np.cos(a), np.sin(a)]))
    cm = X(c)
    s, m, e = P(a0), P(am), P(a1)
    ang = lambda q: np.degrees(np.arctan2(q[1] - cm[1], q[0] - cm[0])) % 360
    sa, ma, ea = ang(s), ang(m), ang(e)
    # 始点→終点 反時計回りで中点を通るか
    def ccw_between(a, b, x):
        return (x - a) % 360 < (b - a) % 360
    if not ccw_between(sa, ea, ma):
        sa, ea = ea, sa
    return cm, r * X.s, sa, ea


def build_linetypes(ents, X):
    """破線類のピッチを図面から測り、線種を定義する（mm, 線種尺度1で原図と同じ見え方）"""
    out = {}
    for lt in ("DASHED", "CENTER", "PHANTOM"):
        dash, gap = [], []
        for e in ents:
            if e.get("linetype") == lt and "pat" in e:
                p = e["pat"]
                dash += list(p["lens"]); gap += [g for g in p["gaps"] if g > 4.5]
        if not dash:
            continue
        D = float(np.median(dash)) * X.s; G = float(np.median(gap)) * X.s if gap else D * 0.3
        if lt == "DASHED":
            pat = [D, -G]
            desc = f"破線 実線{D:.0f} 空き{G:.0f}"
        elif lt == "CENTER":
            g2 = G / 2
            pat = [D, -g2 * 0.8, 0.0, -g2 * 0.8]
            desc = f"一点鎖線 実線{D:.0f} 空き{G:.0f}"
        else:
            g3 = G / 3
            pat = [D, -g3 * 0.7, 0.0, -g3 * 0.7, 0.0, -g3 * 0.7]
            desc = f"二点鎖線 実線{D:.0f} 空き{G:.0f}"
        out[lt] = dict(pattern=pat, desc=desc, dash_px=float(np.median(dash)),
                       gap_px=float(np.median(gap)) if gap else None)
    return out


def write(ents, X, path, tmask, origin, ref_image=None, title=""):
    doc = ezdxf.new("R2013", setup=False)
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LUNITS"] = 2
    doc.header["$LTSCALE"] = 1.0
    doc.header["$CELTSCALE"] = 1.0
    doc.header["$PSLTSCALE"] = 0
    doc.header["$LWDISPLAY"] = 1
    lts = build_linetypes(ents, X)
    for name, d in lts.items():
        doc.linetypes.add(name, pattern=[sum(abs(x) for x in d["pattern"])] + d["pattern"], description=d["desc"])
    if "JP" not in doc.styles:
        doc.styles.add("JP", font="msgothic.ttc")
    # レイヤーごとの線の太さ（実測の中央値）
    wl = {}
    for e in ents:
        if e.get("w") is not None and not np.isnan(e.get("w", np.nan)):
            wl.setdefault(e["layer"], []).append((e["w"], C.length(e) if e["kind"] in ("line", "arc", "circle", "poly") else 1))
    used = sorted(set(e["layer"] for e in ents) | ({"REF_IMAGE"} if ref_image else set()))
    layer_rows = []
    for ln in used:
        info = C.LAYER_INFO.get(ln, dict(jp="", lt="CONTINUOUS", color=7))
        lt = info["lt"] if info["lt"] in lts or info["lt"] == "CONTINUOUS" else "CONTINUOUS"
        if ln in wl:
            ws = np.array(wl[ln]); order = np.argsort(ws[:, 0]); cw = np.cumsum(ws[order, 1])
            wmed = float(ws[order, 0][np.searchsorted(cw, cw[-1] / 2)])
            lw = nearest_lw(wmed * PX_MM_PAPER)
        else:
            wmed, lw = None, 13
        lay = doc.layers.add(ln, color=info["color"], linetype=lt, lineweight=lw)
        lay.description = info["jp"]
        if ln == "REF_IMAGE":
            lay.off()
        layer_rows.append(dict(name=ln, jp=info["jp"], linetype=lt, color=info["color"], lw=lw, w_px=wmed))
    msp = doc.modelspace()
    counts = {}
    for e in ents:
        a = dict(layer=e["layer"])
        k = e["kind"]
        wc = C.wclass4(e.get("w", np.nan)) if e.get("w") is not None else "nan"
        if k in ("line", "arc", "circle", "poly") and wc in C.LW_CLASS:
            a["lineweight"] = C.LW_CLASS[wc]      # 実測した太さの区分を図形ごとに反映
        if k == "line":
            msp.add_line(tuple(X(e["p0"])), tuple(X(e["p1"])), dxfattribs=a)
        elif k == "circle":
            msp.add_circle(tuple(X(e["c"])), e["r"] * X.s, dxfattribs=a)
        elif k == "arc":
            cm, rm, sa, ea = arc_mm(e["c"], e["r"], e["a0"], e["a1"], X)
            msp.add_arc(tuple(cm), rm, sa, ea, dxfattribs=a)
        elif k == "poly":
            pl = msp.add_lwpolyline([tuple(q) for q in X(e["pts"])], dxfattribs=a)
            if e.get("linetype") in lts:
                pl.dxf.linetype = e["linetype"]
                pl.dxf.flags = pl.dxf.flags | 128     # 線種を頂点ごとに切らずに連続させる
        elif k == "dot":
            cm = X(e["c"]); rm = e["r"] * X.s
            msp.add_circle(tuple(cm), rm, dxfattribs=a)
            h = msp.add_hatch(dxfattribs=a)
            h.set_solid_fill()
            h.paths.add_edge_path().add_arc(tuple(cm), rm, 0, 360)
        elif k == "text":
            if "block" in e:
                g = text_geometry(e, tmask, origin, X)
                if g is None:
                    continue
                wf = width_factor(e["text"], g["height"], g["w_mm"])
                t = msp.add_text(e["text"], height=g["height"], rotation=g["rot"],
                                 dxfattribs=dict(a, style="JP", width=wf))
                t.set_placement(tuple(g["insert"]), align=TextEntityAlignment.LEFT)
                e["_geom"] = g
            else:
                cm = X(e["center"]); h = e["height"] * X.s
                rot = e.get("rot_mm", 0.0)
                t = msp.add_text(e["text"], height=h, rotation=rot, dxfattribs=dict(a, style="JP"))
                t.set_placement(tuple(cm), align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            continue
        if e.get("linetype") in lts and k in ("line", "arc", "circle"):
            pass   # レイヤーの線種(BYLAYER)を使う。破線は破線用レイヤーに入っている
        counts[e["layer"]] = counts.get(e["layer"], 0) + 1
    if ref_image:
        img_def = doc.add_image_def(filename=ref_image["file"], size_in_pixel=ref_image["size_px"])
        msp.add_image(img_def, insert=ref_image["insert"], size_in_units=ref_image["size_mm"], rotation=0,
                      dxfattribs=dict(layer="REF_IMAGE"))
        counts["REF_IMAGE"] = 1
    doc.saveas(path)
    for row in layer_rows:
        row["count"] = counts.get(row["name"], 0)
    return layer_rows, lts


def make_reference_image(gray, X, x0, y0, x1, y1, path):
    """原図（の一部）を、DXF の mm 座標の向きにそろえて書き出す（参照画像用）。
    戻り値: 画像の左下の mm 座標、mm での大きさ、画素数"""
    # 範囲の四隅を mm に
    corners = X(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float))
    mx0, my0 = corners.min(0); mx1, my1 = corners.max(0)
    s = X.s
    W = int(np.ceil((mx1 - mx0) / s)); H = int(np.ceil((my1 - my0) / s))
    # 出力画素 (u,v) → mm (mx0 + u s, my1 - v s) → 元画像の画素
    A = np.array([[s, 0, mx0], [0, -s, my1]], float)
    Mi = X.Mi; t = X.t
    # 元画素 = Mi @ (A @ [u v 1] - t)
    B = Mi @ A
    B[:, 2] = Mi @ (A[:, 2] - t)
    out = cv2.warpAffine(gray, B, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    cv2.imwrite(path, out)
    return dict(insert=(float(mx0), float(my0)), size_mm=(W * s, H * s), size_px=(W, H))
