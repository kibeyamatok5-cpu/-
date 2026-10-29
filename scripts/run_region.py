"""範囲を処理して結果を保存し、文字の切り出し一覧を作る"""
import sys, pickle, glob, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import extract as E, ocr, contact
img, x0, y0, x1, y1, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
im = cv2.imread(img, 0); ink = (im < 128).astype(np.uint8)
r = E.process(ink[y0:y1, x0:x1].copy(), (x0, y0)); r.pop("blk_lab")
for i, b in enumerate(r["text_blocks"]):
    crop, o = ocr.block_crop(r["text_mask"], b["bbox"], r["origin"])
    a = ocr.orientation(crop); b["angle"] = a if a is not None else 0.0
    b["idx"] = b["_lab"] = i
pickle.dump(r, open(out + ".pkl", "wb"))
for f in glob.glob(out + "_sheet*.png"): os.remove(f)
T = [b for b in r["text_blocks"] if b["auto"] == "T"]; G = [b for b in r["text_blocks"] if b["auto"] == "G"]
for name, lst in (("T", T), ("G", G)):
    for s in range(0, len(lst), 48):
        contact.sheet(lst[s:s + 48], r["text_mask"], r["origin"], ink, f"{out}_sheet{name}_{s:03d}.png", start=0)
print(len(r["text_blocks"]), len(T), len(G))
