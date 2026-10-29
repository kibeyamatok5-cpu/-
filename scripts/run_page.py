"""本番：1ページを変換して、DXF・確認画像・検証結果を作る。

使い方: python3 run_page.py <作業フォルダ> <出力フォルダ> <図面番号> <DXFの名前>
  例:    python3 run_page.py work output 506 506_1F
前提: 作業フォルダに p-00N.png（ページ画像）、calib_<番号>.json（校正）、full_<番号>.pkl（線の取り出し結果）
"""
import sys, os, json, pickle
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import classify as C, write_dxf as WD, verify as V, extract as E

work, outdir, page, name = sys.argv[1:5]
pkl = sys.argv[5] if len(sys.argv) > 5 else os.path.join(work, f"full_{page}.pkl")
os.makedirs(outdir, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
pcfg = json.load(open(os.path.join(here, "pages.json")))[page]
r = pickle.load(open(pkl, "rb"))
calib = json.load(open(os.path.join(work, f"calib_{page}.json")))
rvp = os.path.join(here, "reviews", f"{page}_text.json")
review = json.load(open(rvp)) if os.path.exists(rvp) else dict(blocks=[])
im = cv2.imread(os.path.join(work, pcfg["image"]), 0); ink = (im < 128).astype(np.uint8)
frame = C.detect_frame(ink)
ents, checks = C.classify(r, calib, review, frame, pcfg, ink=ink)
X = WD.Xform(calib["affine"]["M"], calib["affine"]["t"])

# 参照用の原図画像（DXFの座標にそろえたもの）
H, W = ink.shape
ref = WD.make_reference_image(im, X, 0, 0, W, H, os.path.join(outdir, f"{name}_原図.png"))
ref["file"] = f"{name}_原図.png"
dxf = os.path.join(outdir, f"{name}.dxf")
rows, lts = WD.write(ents, X, dxf, r["text_mask"], r["origin"], ref_image=ref)
doc, aud = V.audit(dxf)

# 重ね合わせ画像（DXFを読み直して描く）
img, mask = V.render(doc, X, (H, W), (0, 0), base_ink=ink)
cv2.imwrite(os.path.join(outdir, f"{name}_重ね合わせ.png"), cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
cv2.imwrite(os.path.join(work, f"{name}_重ね合わせ_原寸.png"), img)
cov = V.coverage(ink, mask)
mm = np.full((H, W, 3), 255, np.uint8)
mm[ink > 0] = (200, 200, 200); mm[cov["missed_map"]] = (0, 0, 255); mm[cov["extra_map"]] = (255, 0, 0)
cv2.imwrite(os.path.join(outdir, f"{name}_取りこぼし図.png"), cv2.resize(mm, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

# 取りこぼしの内訳：意図して除いたもの（用紙右端の帯・スキャンの細かい汚れ）
miss = cov["missed_map"]
edge = int(miss[:, frame["scan_x1"]:].sum())
n_, l_, s_, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
speck = np.isin(l_, np.nonzero(s_[:, 4] < E.NOISE_AREA)[0][1:])
speck_n = int((miss & speck).sum())
body = miss.copy(); body[:, frame["scan_x1"]:] = False; body &= ~speck
covd = {k: v for k, v in cov.items() if not k.endswith("map")}
covd.update(edge_px=edge, speck_px=speck_n, real_missed_px=int(body.sum()),
            real_missed_ratio=float(body.sum() / max(1, ink.sum())))

# 破線類が短い線のまま残っていないか：破線類以外のレイヤーの短い直線で、等間隔に3本以上並ぶもの
dashed_layers = {"A_DASHED", "P_DASHED", "L_CENTER", "L_PHANTOM"}
shorts = []
for e in ents:
    if e["kind"] == "line" and e["layer"] not in dashed_layers and not e["layer"].startswith("TEXT") \
            and e["layer"] not in ("A_HATCH", "FRAME") and 8 <= C.length(e) <= 60:
        d = (e["p1"] - e["p0"]) / C.length(e)
        shorts.append(dict(p0=e["p0"], p1=e["p1"], d=d, m=(e["p0"] + e["p1"]) / 2, L=C.length(e),
                           th=e.get("w", 4.0), layer=e["layer"]))
leftover = []
for ch in E.link_dashes(shorts, np.zeros((0, 2)), max_gap=40):
    if len(ch) >= 3:
        pat = E.chain_pattern(ch, shorts, np.zeros((0, 2)), None)
        if E.regular(pat):
            leftover.append(dict(n=len(ch), at=X((pat["a"] + pat["b"]) / 2).tolist(),
                                 layers=sorted(set(shorts[i]["layer"] for i in ch))))

# 通り芯の間隔（DXF上で測定）
axrows = V.measure_axes_in_dxf(doc, calib, pcfg, X)

# 要確認の一覧（図面上の場所を通り芯で示す）
def where(p_mm):
    out = []
    for f in pcfg["families"]:
        if f["kind"] == "x":
            v = p_mm[0]
        elif f["kind"] == "y":
            v = p_mm[1]
        else:
            continue
        pos = np.array(f["pos"]); lab = f["labels"]
        k = int(np.clip(np.searchsorted(pos, v), 1, len(pos) - 1))
        if pos[0] - 3000 <= v <= pos[-1] + 3000:
            out.append(f"{lab[k-1]}〜{lab[k]}" if pos[k - 1] <= v <= pos[k] else (f"{lab[0]}の外" if v < pos[0] else f"{lab[-1]}の外"))
    return " / ".join(dict.fromkeys(out))
chk = []
for e in ents:
    if e["layer"] == "CHECK":
        p = C.mid(e) if e["kind"] != "dot" else e["c"]
        chk.append(dict(why=e.get("why", ""), px=[float(p[0]), float(p[1])]))
for why, c in checks:
    chk.append(dict(why=why, px=[float(c[0]), float(c[1])]))
# 近いものはまとめる（120px＝約2m以内）
groups = []
for c in chk:
    for g in groups:
        if g["why"] == c["why"] and np.hypot(*(np.array(g["px"]) - c["px"])) < 120:
            g["n"] += 1; break
    else:
        groups.append(dict(c, n=1))
for g in groups:
    q = X(np.array(g["px"]))
    g["mm"] = [round(float(q[0])), round(float(q[1]))]
    g["where"] = where(q)

res = dict(page=page, name=name, rows=rows, lts=lts, audit=aud, coverage=covd, axes=axrows,
           leftover_dashes=leftover, checks=groups, frame=frame, ref=ref,
           calib=dict(M=calib["affine"]["M"], t=calib["affine"]["t"], rms=calib["affine"]["rms"],
                      sx=calib["affine"]["sx"], sy=calib["affine"]["sy"], rot=calib["affine"]["rot"],
                      ortho=calib["affine"]["ortho"], rows=calib["affine"]["rows"],
                      secondary=calib.get("secondary")),
           text_counts={k: sum(1 for e in ents if e["kind"] == "text" and e["layer"] == k)
                        for k in ("AXIS_TEXT", "DIM_TEXT", "P_SIZE", "TEXT")},
           review_counts={})
from collections import Counter
res["review_counts"] = dict(Counter(b["label"][:2] for b in review["blocks"]))
json.dump(res, open(os.path.join(work, f"result_{page}.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("audit", aud)
print("coverage", covd)
print("leftover dashed chains", len(leftover))
print("checks", len(groups))
for row in axrows:
    print("AX %s %s→%s 図面 %.0f DXF %.0f 誤差 %+.0f" % (row["fam"], row["a"], row["b"], row["nominal"], row["measured"], row["err"]))
for row in rows:
    print(row["name"], row["count"])
