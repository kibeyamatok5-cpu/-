"""DXFを読み直して検証する。

- ezdxf の監査機能でエラーがないか確認
- DXF の図形を画像に描き戻し、原図（薄い灰色）と重ねた確認画像を作る（線種・線の太さも反映）
- 原図のインクのうち、どの図形にも覆われなかった割合（取りこぼし率）と、
  原図にインクがない所に描かれた図形の割合（余計な線）を数える
- 短い線のまま残った破線・一点鎖線の片がないかを数える
"""
import numpy as np
import cv2
import ezdxf
from ezdxf import colors as ezcolors
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
PX_MM_PAPER = 25.4 / 600.0


def aci_rgb(aci):
    if aci in (7, 255):
        return (0, 0, 0)
    r, g, b = ezcolors.aci2rgb(aci)
    return (int(r), int(g), int(b))


def dash_polyline(P, pattern, s_px):
    """点列 P(px) を線種のパターン(mm)で区切り、描く区間の列を返す"""
    if not pattern:
        return [P]
    pat = [abs(x) / s_px for x in pattern]
    sign = [x >= 0 for x in pattern]
    out, cur = [], []
    k, rem = 0, pat[0]
    for a, b in zip(P[:-1], P[1:]):
        seg = np.hypot(*(b - a)); pos = 0.0
        while seg - pos > 1e-9:
            step = min(rem, seg - pos)
            q0 = a + (b - a) * pos / seg; q1 = a + (b - a) * (pos + step) / seg
            if sign[k]:
                if pat[k] == 0:
                    out.append(np.array([q0, q0 + 0.01]))
                else:
                    cur += [q0, q1]
            pos += step; rem -= step
            if rem <= 1e-9:
                if cur:
                    out.append(np.array(cur)); cur = []
                k = (k + 1) % len(pat); rem = pat[k] if pat[k] > 0 else 0.0
                if pat[k] == 0 and sign[k]:
                    out.append(np.array([q1, q1 + 0.01]))
                    k = (k + 1) % len(pat); rem = pat[k]
    if cur:
        out.append(np.array(cur))
    return out


def entity_paths(e, Xinv):
    """DXF 図形 → 画素座標の点列（複数）"""
    t = e.dxftype()
    if t == "LINE":
        return [Xinv(np.array([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))]
    if t in ("CIRCLE", "ARC"):
        c = np.array([e.dxf.center.x, e.dxf.center.y]); r = e.dxf.radius
        if t == "CIRCLE":
            a = np.linspace(0, 2 * np.pi, 90)
        else:
            sa, ea = np.deg2rad(e.dxf.start_angle), np.deg2rad(e.dxf.end_angle)
            if ea < sa:
                ea += 2 * np.pi
            a = np.linspace(sa, ea, max(8, int((ea - sa) / 0.05)))
        return [Xinv(c + r * np.c_[np.cos(a), np.sin(a)])]
    if t == "LWPOLYLINE":
        return [Xinv(np.array([p[:2] for p in e.get_points()]))]
    return []


def render(doc, X, shape, origin, base_ink=None, scale=1.0):
    """DXF を画像に描く。戻り値: (確認用カラー画像, 図形の画素マスク)"""
    ox, oy = origin
    H, W = shape
    Hs, Ws = int(H * scale), int(W * scale)
    img = np.full((Hs, Ws, 3), 255, np.uint8)
    if base_ink is not None:
        b = cv2.resize(base_ink.astype(np.uint8) * 255, (Ws, Hs), interpolation=cv2.INTER_AREA)
        img[b > 60] = (215, 215, 215)
    mask = np.zeros((H, W), np.uint8)
    def ltpat(name):
        try:
            lt = doc.linetypes.get(name)
            p = list(lt.simplified_line_pattern())
            return [x if i % 2 == 0 else -x for i, x in enumerate(p)]
        except Exception:
            return []
    layers = {l.dxf.name: l for l in doc.layers}
    msp = doc.modelspace()
    Xinv = lambda q: X.inv(q) - np.array([ox, oy])
    texts = []
    for e in msp:
        lay = layers.get(e.dxf.layer)
        if lay is None or not lay.is_on():
            continue
        col = aci_rgb(lay.color)[::-1]
        lw = lay.dxf.lineweight if lay.dxf.lineweight > 0 else 13
        wpx = max(1, int(round(lw / 100 / PX_MM_PAPER)))
        ltn = e.dxf.linetype if e.dxf.hasattr("linetype") and e.dxf.linetype.upper() != "BYLAYER" else lay.dxf.linetype
        pat = ltpat(ltn) if ltn.upper() != "CONTINUOUS" else []
        t = e.dxftype()
        if t == "TEXT":
            texts.append((e, col)); continue
        if t == "HATCH":
            for p in e.paths:
                for ed in getattr(p, "edges", []):
                    if ed.EDGE_TYPE == "ArcEdge":
                        c = Xinv(np.array([ed.center[0], ed.center[1]])); r = ed.radius / X.s
                        cv2.circle(mask, tuple(np.round(c).astype(int)), int(round(r)), 1, -1)
                        cv2.circle(img, tuple(np.round(c * scale).astype(int)), max(1, int(round(r * scale))), col, -1)
            continue
        for P in entity_paths(e, Xinv):
            for seg in dash_polyline(P, pat, X.s) if pat else [P]:
                q = np.round(seg).astype(np.int32)
                cv2.polylines(mask, [q], False, 1, wpx)
                cv2.polylines(img, [np.round(seg * scale).astype(np.int32)], False, col, max(1, int(round(wpx * scale))),
                              cv2.LINE_AA)
    # 文字
    if texts:
        pil = Image.fromarray(img[:, :, ::-1]); dr = ImageDraw.Draw(pil)
        mpil = Image.fromarray(mask * 255); mdr = ImageDraw.Draw(mpil)
        for e, col in texts:
            h_px = e.dxf.height / X.s
            font = ImageFont.truetype(FONT, max(6, int(round(h_px / 0.75))))   # 文字の高さ＝大文字・数字の高さとして描く
            q_ = e.dxf.insert if e.dxf.halign == 0 else e.dxf.align_point
            ins = np.array([q_.x, q_.y])
            p = Xinv(ins)
            rot = e.dxf.rotation
            # mm 座標の回転 → 画像上の回転
            v = X.Mi @ np.array([np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))])
            ang_img = np.degrees(np.arctan2(v[1], v[0]))
            txt = e.dxf.text
            tw = int(font.getlength(txt)) + 4; th = int(h_px * 1.6) + 4
            for target, d, c, sc in ((pil, None, col[::-1], scale), (mpil, None, 255, 1.0)):
                tile = Image.new("L", (int(tw), int(th)), 0)
                ImageDraw.Draw(tile).text((2, th - 2), txt, font=font, fill=255, anchor="ls")
                wf = e.dxf.width if e.dxf.hasattr("width") else 1.0
                if abs(wf - 1) > 0.01:
                    tile = tile.resize((max(1, int(tw * wf)), int(th)))
                    tw = tile.size[0]
                tile = tile.rotate(-ang_img, expand=True, resample=Image.BILINEAR)
                if e.dxf.halign == 0:
                    # 左下基準：回転後の左下点の位置
                    a = np.deg2rad(ang_img)
                    # 元タイルの (2, th-2) が挿入点
                    cx, cy = tw / 2, th / 2
                    dx, dy = 2 - cx, (th - 2) - cy
                    rx = dx * np.cos(a) - dy * np.sin(a); ry = dx * np.sin(a) + dy * np.cos(a)
                    px0 = p[0] - (tile.size[0] / 2 + rx); py0 = p[1] - (tile.size[1] / 2 + ry)
                else:
                    px0 = p[0] - tile.size[0] / 2; py0 = p[1] - tile.size[1] / 2
                if sc != 1.0:
                    tile = tile.resize((max(1, int(tile.size[0] * sc)), max(1, int(tile.size[1] * sc))))
                    px0 *= sc; py0 *= sc
                if target is pil:
                    colimg = Image.new("RGB", tile.size, col[::-1])
                    target.paste(colimg, (int(px0), int(py0)), tile)
                else:
                    target.paste(255, (int(px0), int(py0)), tile)
        img = np.array(pil)[:, :, ::-1].copy()
        mask = (np.array(mpil) > 0).astype(np.uint8)
    return img, mask


def coverage(ink, mask, tol=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    md = cv2.dilate(mask, k)
    inkd = cv2.dilate(ink.astype(np.uint8), k)
    ink_n = int(ink.sum())
    missed = int(((ink > 0) & (md == 0)).sum())
    drawn = int(mask.sum())
    extra = int(((mask > 0) & (inkd == 0)).sum())
    return dict(ink_px=ink_n, missed_px=missed, missed_ratio=missed / max(1, ink_n),
                drawn_px=drawn, extra_px=extra, extra_ratio=extra / max(1, drawn),
                missed_map=((ink > 0) & (md == 0)), extra_map=((mask > 0) & (inkd == 0)))


def audit(path):
    doc = ezdxf.readfile(path)
    a = doc.audit()
    return doc, dict(errors=len(a.errors), fixes=len(a.fixes),
                     messages=[str(x.message) for x in a.errors][:20])


def measure_axes_in_dxf(doc, calib, page_cfg, X):
    """DXF の AXIS レイヤーの線から、通り芯の間隔を測り、図面の寸法値と比べる。
    各通り芯記号の中心を通り、その通り芯の向きに平行な AXIS 線を探して使う。"""
    segs = []
    for e in doc.modelspace().query('LINE[layer=="AXIS"]'):
        p0 = np.array([e.dxf.start.x, e.dxf.start.y]); p1 = np.array([e.dxf.end.x, e.dxf.end.y])
        v = p1 - p0; L = np.hypot(*v)
        if L > 1:
            segs.append((p0, p1, v / L, L))
    rows = []
    fams = list(page_cfg["families"]) + list(page_cfg.get("secondary", []))
    axes = dict(calib["axes"]); axes.update(calib.get("axes_secondary", {}))
    for f in fams:
        fam = f["name"]
        if fam not in axes:
            continue
        found = []
        for a, nom in zip(axes[fam], f["pos"]):
            if a["p0"] is None:
                continue
            bc = X(np.array(a["p0"]))      # なぞった通り芯の線上の点（記号の丸は線から少しずれることがある）
            d = X(np.array(a["p1"])) - X(np.array(a["p0"])); d /= np.linalg.norm(d)
            best = None
            for p0, p1, u, L in segs:
                if abs(np.dot(u, d)) < np.cos(np.deg2rad(3)):
                    continue
                n = np.array([-u[1], u[0]])
                off = abs(np.dot(bc - p0, n))
                # 記号の中心はその通り芯の延長上にある：中心からのずれが最も小さい線を使う
                if off < 100 and (best is None or off < best[0]):
                    best = (off, p0, p1, L, u, n)
            if best is not None:
                found.append((a["label"], nom, best))
        if len(found) < 2:
            continue
        def dist(b0, b1):
            q = (b1[1] + b1[2]) / 2
            return abs(np.dot(q - b0[1], b0[5]))
        for (l0, n0, b0), (l1, n1, b1) in zip(found[:-1], found[1:]):
            dd = dist(b0, b1)
            rows.append(dict(fam=fam, a=l0, b=l1, nominal=n1 - n0, measured=float(dd), err=float(dd - (n1 - n0))))
        (l0, n0, b0), (l1, n1, b1) = found[0], found[-1]
        dd = dist(b0, b1)
        rows.append(dict(fam=fam, a=l0, b=l1 + "（全長）", nominal=n1 - n0, measured=float(dd), err=float(dd - (n1 - n0))))
    return rows
