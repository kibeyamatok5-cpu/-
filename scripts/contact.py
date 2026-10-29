"""文字のまとまりの切り出し一覧画像（番号付き）を作る。目視での読み取り・照合用。"""
import sys, pickle, numpy as np, cv2
import ocr

def sheet(blocks, tmask, origin, page_ink, path, start=0, per=48, cols=6, cell=(300, 170)):
    cw, ch = cell
    rows = (min(per, len(blocks) - start) + cols - 1) // cols
    img = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for k in range(start, min(start + per, len(blocks))):
        b = blocks[k]
        x, y, w, h = b["bbox"]
        pad = 14
        crop = page_ink[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        tm, o = ocr.block_crop(tmask, b["bbox"], origin, pad=pad)
        vis = np.full(crop.shape + (3,), 255, np.uint8)
        vis[crop > 0] = (190, 190, 190)
        hh, ww = min(tm.shape[0], crop.shape[0]), min(tm.shape[1], crop.shape[1])
        vis[:hh, :ww][tm[:hh, :ww] > 0] = (0, 0, 0)
        a = b.get("angle")
        if a not in (None, 0.0):
            g = (255 - vis[:, :, 0]).astype(np.uint8)
            r, M = ocr.rotate((g > 128).astype(np.uint8), a)
            vis = np.full(r.shape + (3,), 255, np.uint8); vis[r > 0] = (0, 0, 0)
        s = min((cw - 10) / vis.shape[1], (ch - 30) / vis.shape[0], 2.0)
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        i = k - start
        r0, c0 = (i // cols) * ch, (i % cols) * cw
        img[r0 + 25:r0 + 25 + vis.shape[0], c0 + 5:c0 + 5 + vis.shape[1]] = vis
        cv2.rectangle(img, (c0, r0), (c0 + cw - 1, r0 + ch - 1), (200, 200, 255), 1)
        cv2.putText(img, f"#{b.get('_lab', k)} {b.get('ocr', '')}", (c0 + 4, r0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(path, img)
