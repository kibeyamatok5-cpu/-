import sys, pickle, numpy as np, cv2
import extract as E

def draw(res, ink, origin, scale=1.0, path="dbg.png"):
    ox, oy = origin
    H, W = ink.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[ink > 0] = (200, 200, 200)
    def P(p): return (int(round(p[0]-ox)), int(round(p[1]-oy)))
    def col(w):
        if np.isnan(w): return (255, 0, 255)
        if w < 4.5: return (255, 0, 0)      # 細: 青
        if w < 7: return (0, 160, 0)        # 中: 緑
        return (0, 0, 255)                  # 太: 赤
    for s in res["lines"]:
        cv2.line(img, P(s["p0"]), P(s["p1"]), col(s["w"]), 1, cv2.LINE_AA)
    for s in res["curves"]:
        pts = np.array([P(p) for p in s["pts"]], np.int32)
        cv2.polylines(img, [pts], s["kind"] == "circle", (0, 140, 255) if s["kind"] != "line" else (0,0,0), 1, cv2.LINE_AA)
    for s in res["dashed"]:
        c = {"DASHED": (200, 0, 200), "CENTER": (0, 200, 200), "PHANTOM": (120, 60, 0)}[s["linetype"]]
        if s["kind"] == "line":
            cv2.line(img, P(s["p0"]), P(s["p1"]), c, 2, cv2.LINE_AA)
        else:
            cv2.polylines(img, [np.array([P(p) for p in s["pts"]], np.int32)], False, c, 2, cv2.LINE_AA)
    gm = res["text_mask"] > 0
    img[gm] = (0, 120, 255)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)
