"""目視判定の結果を保存する（ブロック番号→判定）。指定のない番号は自動判定（図形らしければG、それ以外は?）。"""
import sys, pickle, json
from collections import Counter

def save(pkl, page, T, AX=(), N=(), S=(), LR=(), Q=(), G=(), default_T="G"):
    r = pickle.load(open(pkl, "rb"))
    lab = {}
    for k, v in T.items(): lab[k] = "T:" + v
    for k in AX: lab[k] = "AX"
    for k in N: lab[k] = "N"
    for k in S: lab[k] = "S"
    for k in LR: lab[k] = "L:R"
    for k in Q: lab[k] = "?"
    for k in G: lab[k] = "G"
    out = []
    for b in r["text_blocks"]:
        x, y, w, h = map(int, b["bbox"])
        l = lab.get(b["idx"], "G" if b["auto"] == "G" else default_T)
        out.append(dict(c=[x + w / 2, y + h / 2], bbox=[x, y, w, h], label=l))
    note = ("目視判定: T:=文字データ化 / L:=配管の線種記号（線のまま配管レイヤーへ）/ S=記号 / G=図形 / "
            "AX=通り芯記号 / ?=判読不確実（線のまま文字形状レイヤーへ）/ N=スキャンの汚れ")
    json.dump(dict(page=page, note=note, blocks=out), open(f"/home/user/-/scripts/reviews/{page}_text.json", "w"),
              ensure_ascii=False, indent=0)
    print(page, len(out), Counter(o["label"][:2] for o in out))
