"""文字のまとまりを OCR（文字認識）にかける。

向きは文字の並びから推定し、水平にしてから認識する。結果と確からしさ(0-100)を返す。
"""
import numpy as np, cv2, pytesseract, re

def block_crop(tmask, bbox, origin, pad=6):
    x, y, w, h = bbox
    ox, oy = origin
    x0, y0 = x - ox - pad, y - oy - pad
    H, W = tmask.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w + 2 * pad), min(H, y0 + h + 2 * pad)
    return tmask[ya:yb, xa:xb].copy(), (xa + ox, ya + oy)

def orientation(crop):
    ys, xs = np.nonzero(crop)
    if len(xs) < 10:
        return 0.0
    rect = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    (cx, cy), (rw, rh), ang = rect
    if rw < rh:
        ang += 90
        rw, rh = rh, rw
    if rw < 1.6 * rh:         # 正方形に近い（1〜2文字）→向きは決めない
        return None
    return ((ang + 90) % 180) - 90

def rotate(crop, ang):
    h, w = crop.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 4, int(h * cos + w * sin) + 4
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    r = cv2.warpAffine(crop * 255, M, (nw, nh), flags=cv2.INTER_LINEAR)
    return (r > 100).astype(np.uint8), M

def run_tess(img01, lang, psm, whitelist=None):
    im = 255 - img01 * 255
    im = cv2.copyMakeBorder(im, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # tesseract は文字高さ 30px 前後が得意
    ys = np.nonzero(img01.any(1))[0]
    hgt = (ys.max() - ys.min() + 1) if len(ys) else 30
    f = 34.0 / max(10, hgt) if hgt > 60 or hgt < 20 else 1.0
    if f != 1.0:
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(im, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    txt = "".join(t for t, c in words) if lang.startswith("jpn") else " ".join(t for t, c in words)
    conf = min(c for t, c in words)
    return txt, conf

def recognize(crop):
    """戻り値: dict(text, conf, angle)  angle は画像上の文字の回転角（度、反時計回り正, y下向き画像基準）"""
    ang = orientation(crop)
    cands = []
    angs = [0.0] if ang is None else [ang, ang + 180]
    if ang is not None and abs(ang) > 3:
        pass
    for a in angs:
        r, M = rotate(crop, a) if a else (crop, None)
        for lang, psm, wl in (("eng", 7, "0123456789.,x×/-()ABCDRVPpeHφ"), ("jpn", 7, None)):
            t, c = run_tess(r, lang, psm, wl)
            if t:
                cands.append(dict(text=t, conf=c, angle=a, lang=lang))
    if not cands:
        return dict(text="", conf=0.0, angle=0.0 if ang is None else ang, lang="")
    return max(cands, key=lambda d: d["conf"])
