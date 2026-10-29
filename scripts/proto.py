"""試作：1ページ目の一部（多目的ホール周辺）を変換し、確認用の資料を作る"""
import sys, os, json, pickle
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import classify as C, write_dxf as WD, verify as V

work = sys.argv[1]; outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
r = pickle.load(open(os.path.join(work, "proto.pkl"), "rb"))
calib = json.load(open(os.path.join(work, "calib_506.json")))
review = json.load(open(os.path.join(os.path.dirname(__file__), "reviews", "506_text.json")))
im = cv2.imread(os.path.join(work, "p-000.png"), 0); ink = (im < 128).astype(np.uint8)
frame = C.detect_frame(ink)
print("frame", frame)
pc = json.load(open(os.path.join(os.path.dirname(__file__), "pages.json")))["506"]
ents, checks = C.classify(r, calib, review, frame, pc, ink=ink)
cal = calib["affine"]
X = WD.Xform(cal["M"], cal["t"])
dxf = os.path.join(outdir, "506_1F_試作.dxf")
x0, y0 = r["origin"]; H, W = r["text_mask"].shape
ref = WD.make_reference_image(im, X, x0, y0, x0 + W, y0 + H, os.path.join(outdir, "506_1F_試作_原図.png"))
ref["file"] = "506_1F_試作_原図.png"
rows, lts = WD.write(ents, X, dxf, r["text_mask"], r["origin"], ref_image=ref)
json.dump(ref, open(os.path.join(work, "proto_ref.json"), "w"))
doc, aud = V.audit(dxf)
print("audit", aud)
sub = ink[y0:y0 + H, x0:x0 + W]
img, mask = V.render(doc, X, (H, W), (x0, y0), base_ink=sub)
cv2.imwrite(os.path.join(outdir, "506_試作_重ね合わせ.png"), img)
cv2.imwrite(os.path.join(outdir, "506_試作_重ね合わせ_縮小.png"), cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
cov = V.coverage(sub, mask)
mm = np.zeros((H, W, 3), np.uint8) + 255
mm[sub > 0] = (220, 220, 220); mm[cov["missed_map"]] = (0, 0, 255); mm[cov["extra_map"]] = (255, 0, 0)
cv2.imwrite(os.path.join(work, "proto_missed.png"), cv2.resize(mm, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
print("coverage", {k: v for k, v in cov.items() if not k.endswith("map")})
pickle.dump(dict(ents=ents, checks=checks, rows=rows, lts=lts, aud=aud,
                 cov={k: v for k, v in cov.items() if not k.endswith("map")}, frame=frame),
            open(os.path.join(work, "proto_result.pkl"), "wb"))
for row in rows:
    print(row)
print(lts)
pickle.dump(C.GAP_SCORES, open(os.path.join(work, "gap_scores.pkl"), "wb"))
axrows = [a for a in V.measure_axes_in_dxf(doc, calib, pc, X) if a["fam"] == "B"]   # 試作範囲にあるのは B 通り
for row in axrows:
    print("AX", row)
json.dump(axrows, open(os.path.join(work, "proto_axes.json"), "w"), ensure_ascii=False, indent=1)
