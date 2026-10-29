"""通り芯記号（丸）の候補を検出して一覧表示する（調査用）"""
import sys, cv2, numpy as np
im = cv2.imread(sys.argv[1], 0)
ink = (im < 128).astype(np.uint8) * 255
# 円周の検出: Hough
blur = cv2.GaussianBlur(255 - ink, (5, 5), 1.5)
cs = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=100, param2=35, minRadius=44, maxRadius=66)
out = []
if cs is not None:
    for x, y, r in cs[0]:
        # 円周上のインク率で確認
        t = np.linspace(0, 2*np.pi, 180, endpoint=False)
        px = (x + r*np.cos(t)).astype(int); py = (y + r*np.sin(t)).astype(int)
        ok = (px > 0) & (py > 0) & (px < im.shape[1]) & (py < im.shape[0])
        ring = np.zeros(len(t), bool)
        for d in (-3,-2,-1,0,1,2,3):
            qx = np.clip((x+(r+d)*np.cos(t)).astype(int),0,im.shape[1]-1); qy=np.clip((y+(r+d)*np.sin(t)).astype(int),0,im.shape[0]-1)
            ring |= ink[qy, qx] > 0
        cov = ring.mean()
        if cov > 0.85:
            out.append((x, y, r, cov))
for x, y, r, c in sorted(out, key=lambda a: (a[1], a[0])):
    print(f"{x:7.1f} {y:7.1f} {r:5.1f} {c:.2f}")
