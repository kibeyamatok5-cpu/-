"""3ページの検証結果をまとめて、検証レポート・レイヤー対応表・要確認一覧を作る。

使い方: python3 make_report.py <作業フォルダ> <出力フォルダ>
"""
import sys, os, json, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import classify as C

work, outdir = sys.argv[1:3]
PAGES = [("506", "506_1F", "1階"), ("507", "507_2F", "2階"), ("508", "508_3F_RF", "3階・屋上階")]
ACI = {1: "赤", 2: "黄", 3: "緑", 4: "水色", 5: "青", 6: "紫(マゼンタ)", 7: "黒/白", 8: "濃い灰", 9: "薄い灰",
       30: "橙", 34: "茶", 140: "空色", 200: "紫", 252: "灰"}
R = {p: json.load(open(os.path.join(work, f"result_{p}.json"))) for p, _, _ in PAGES}

# ---- レイヤー対応表
names = []
for p, _, _ in PAGES:
    for row in R[p]["rows"]:
        if row["name"] not in names:
            names.append(row["name"])
order = [n for n, *_ in C.LAYERS]
names.sort(key=lambda n: order.index(n) if n in order else len(order))
info = {}
for p, _, _ in PAGES:
    for row in R[p]["rows"]:
        info.setdefault(row["name"], row)
with open(os.path.join(outdir, "レイヤー対応表.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["レイヤー名", "日本語の意味", "線種", "色(ACI番号)", "色", "線の太さ(mm)※レイヤーの既定値",
                "図形の数 506_1F", "図形の数 507_2F", "図形の数 508_3F_RF", "合計"])
    for n in names:
        i = info[n]
        cnt = [next((r["count"] for r in R[p]["rows"] if r["name"] == n), 0) for p, _, _ in PAGES]
        w.writerow([n, i["jp"], i["linetype"], i["color"], ACI.get(i["color"], ""), f"{i['lw'] / 100:.2f}"] + cnt + [sum(cnt)])

# ---- 要確認一覧
with open(os.path.join(outdir, "要確認一覧.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["図面", "内容", "件数（近くのものをまとめた数）", "X座標(mm)", "Y座標(mm)", "場所（通り芯）", "画像上の位置(px)"])
    for p, name, _ in PAGES:
        for g in sorted(R[p]["checks"], key=lambda g: (g["why"], g["mm"][1], g["mm"][0])):
            w.writerow([name, g["why"], g["n"], g["mm"][0], g["mm"][1], g["where"], f"{g['px'][0]:.0f},{g['px'][1]:.0f}"])

# ---- 検証レポート
L = ["# 検証レポート：大樹町生涯学習センター 衛生設備 平面図（506・507・508）", ""]
L += ["## 1. 縮尺と寸法", "",
      "- 各ページで、通り芯記号の丸から伸びる線の位置を画像から測り、図面に書かれた通り芯の寸法値に合うように"
      "「画素→実寸mm」の変換（回転・縦横の拡大縮小・わずかな斜めのずれ、の一様な変換）を求めました。",
      "- 座標の原点は、3ページ共通で X1通り・Y1通りの交点（1階図で決まる位置）です。2・3階も同じ座標になるので、"
      "CADで3枚を重ねると通り芯が一致します（X6通り＝27,500mm、Y2通り＝5,000mm など）。単位は mm（$INSUNITS=4）。", ""]
L += ["| 図面 | 1画素の大きさ(横/縦 mm) | 紙面上の縮尺 | 回転の補正 | 縦横の直交のずれ | 通り芯の当てはまり(RMS) |", "|---|---|---|---|---|---|"]
for p, name, fl in PAGES:
    c = R[p]["calib"]
    s = np.sqrt(abs(np.linalg.det(np.array(c["M"]))))
    L.append(f"| {name}（{fl}） | {c['sx']:.3f} / {c['sy']:.3f} | 約 1/{s / (25.4 / 600):.0f} | {c['rot']:+.3f}° | {90 - c['ortho']:.2f}° | {c['rms']:.0f} mm |")
L += ["", "### 通り芯の間隔：図面の寸法値と、DXF上で測った値", "",
      "DXFの通り芯（AXISレイヤー）の線どうしの距離を測っています。", ""]
for p, name, fl in PAGES:
    L += [f"**{name}（{fl}）**", "", "| 通り芯 | 区間 | 図面の寸法 | DXF上 | 誤差 |", "|---|---|---|---|---|"]
    for a in R[p]["axes"]:
        bold = "**" if "全長" in a["b"] else ""
        L.append(f"| {a['fam']} | {bold}{a['a']}〜{a['b']}{bold} | {a['nominal']:.0f} | {a['measured']:.0f} | {a['err']:+.0f} mm |")
    errs = [abs(a["err"]) for a in R[p]["axes"] if "全長" not in a["b"]]
    if errs:
        L.append(f"| | 一区間の誤差 最大 | | | {max(errs):.0f} mm（紙面で {max(errs) / R[p]['calib']['sx'] * 25.4 / 600:.2f} mm） |")
    sec = R[p]["calib"].get("secondary")
    if sec:
        L += ["", "同じ用紙の左上にある別の小さな平面図（X6〜X14・Y2〜Y6）の通り芯の間隔（縮尺の確認用）：", "",
              "| 区間 | 図面の寸法 | 測定 | 誤差 |", "|---|---|---|---|"]
        for r_ in sec:
            if r_[1] == "offset":
                continue
            L.append(f"| {r_[1]}〜{r_[2]} | {r_[3]:.0f} | {r_[4]:.0f} | {r_[5]:+.0f} mm |")
        L += ["", "※この小さな図は用紙上の別の場所に描かれているため、DXFでも用紙上の位置のまま（本体の図とは重ならない位置）に置いています。"]
    L.append("")
L += ["- 一区間ごとの誤差は、紙面で 0.1〜0.4mm 程度です。手描きの原図自体の誤差と思われ、一様な変換以外の補正（部分的な引き伸ばし）はしていません。", ""]

L += ["## 2. 取りこぼし・余計な線", "",
      "DXFを読み直して画像に描き戻し、原図と重ねて比べました（2画素＝紙面0.08mmまでのずれは一致とみなす）。", "",
      "| 図面 | 原図の線の画素 | 覆われなかった割合 | うち意図して除いたもの※ | 実質の取りこぼし | 原図にない所に描いた割合 |",
      "|---|---|---|---|---|---|"]
for p, name, fl in PAGES:
    c = R[p]["coverage"]
    L.append(f"| {name} | {c['ink_px']:,} | {c['missed_ratio'] * 100:.1f}% | {(c['edge_px'] + c['speck_px']) / c['ink_px'] * 100:.1f}% | "
             f"**{c['real_missed_ratio'] * 100:.1f}%** | {c['extra_ratio'] * 100:.1f}% |")
L += ["", "※ 用紙右端のスキャナの灰色の帯と、5画素未満の細かい汚れ。",
      "- 「原図にない所に描いた」分の大半は、文字データにした文字（ゴシック体）と手描き文字の形の違いによるものです。"
      "原図にない線を新たに作ることはしていません。",
      "- 取りこぼし図（`*_取りこぼし図.png`）：赤＝原図にあるのにDXFにない所、青＝原図にない所にDXFの図形がある所。", ""]

L += ["## 3. 破線・一点鎖線", ""]
for p, name, fl in PAGES:
    lts = R[p]["lts"]
    s = "、".join(f"{k}（{v['desc']}mm）" for k, v in lts.items()) or "なし"
    lo = R[p]["leftover_dashes"]
    L.append(f"- {name}：線種 {s}。破線類以外のレイヤーで、短い線が等間隔に3本以上並んだまま残っている所：{len(lo)}か所"
             + ("（" + "、".join(f"({a['at'][0]:.0f}, {a['at'][1]:.0f}) {'・'.join(a['layers'])}" for a in lo[:10]) + (" ほか" if len(lo) > 10 else "") + "）" if lo else ""))
L += ["- 破線類は、短い線の集まりではなく1本の線にして、実測したピッチの線種を割り当てています（線種尺度1のままで原図と同じ見え方）。", ""]

L += ["## 4. DXFの監査", ""]
for p, name, fl in PAGES:
    a = R[p]["audit"]
    L.append(f"- {name}.dxf：エラー {a['errors']} 件、自動修正 {a['fixes']} 件（ezdxf の audit）")
L += ["- 形式：AutoCAD 2013 形式（AC1027）。Rhino 8・AutoCAD で開けます。文字は UTF-8、フォント指定は MS ゴシック。", ""]

L += ["## 5. 文字", ""]
for p, name, fl in PAGES:
    t = R[p]["text_counts"]; rc = R[p]["review_counts"]
    L.append(f"- {name}：文字データにしたもの 通り芯符号 {t['AXIS_TEXT']}・寸法値 {t['DIM_TEXT']}・配管径など {t['P_SIZE']}・その他 {t['TEXT']}。"
             f"目視で判定した文字のまとまり {sum(rc.values())} 個（うち 判読に自信がなく線のまま {rc.get('?', 0)}、"
             f"配管の線の途中の記号 {rc.get('L:', 0)}、記号 {rc.get('S', 0)}、図形 {rc.get('G', 0)}）。")
L += ["- 文字認識ソフト（tesseract）は手描き文字をほとんど読めなかったため、文字のまとまりを切り出した画像を目で見て判定しました。"
      "数字は1文字ずつ確認し、欠け・重なりがあるものや自信のないものは文字データにせず、線の形のまま TEXT_SHAPE レイヤーに残しています。", ""]

L += ["## 6. 要確認", "", "`要確認一覧.csv` に、場所（mm座標と通り芯）付きで載せています。内容ごとの件数：", ""]
for p, name, fl in PAGES:
    from collections import Counter
    cnt = Counter()
    for g in R[p]["checks"]:
        cnt[g["why"]] += g["n"]
    L.append(f"- {name}：" + "、".join(f"{k} {v}" for k, v in cnt.most_common()))
L.append("")
open(os.path.join(outdir, "検証レポート.md"), "w").write("\n".join(L))
print("\n".join(L))
