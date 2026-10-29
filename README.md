# 大樹町生涯学習センター 衛生設備図 スキャンPDF → DXF 変換

## 成果物（`output/`）
| ファイル | 内容 |
|---|---|
| `506_1F.dxf` `507_2F.dxf` `508_3F_RF.dxf` | ページごとのDXF（AutoCAD 2013形式、単位mm、実寸の座標） |
| `*_重ね合わせ.png` | 原図（薄い灰色）にDXFの図形（レイヤーの色）を重ねた確認画像（DXFを読み直して描いたもの） |
| `*_取りこぼし図.png` | 赤＝原図にあるのにDXFにない所、青＝原図にない所にDXFの図形がある所 |
| `*_原図.png` | DXFの座標にそろえた原図画像（参照用。DXFには非表示レイヤー REF_IMAGE で配置済み） |
| `レイヤー対応表.csv` | レイヤー名・日本語の意味・線種・色・太さ・図形の数 |
| `要確認一覧.csv` | 分類に自信のない箇所の一覧（mm座標・通り芯で場所を表示） |
| `検証レポート.md` | 縮尺・寸法の誤差・取りこぼし率・破線の統合・監査結果 |
| `試作_506/` | 試作のときの成果物（参考） |

## 座標
- 3ページ共通で、X1通りとY1通りの交点が (0, 0)。X方向が右、Y方向が上（図面のY1→Y8の向き）。
- 2・3階図も同じ座標なので、3つのDXFを重ねると通り芯がそろいます。

## Rhino 8 で参照画像を使う場合
Rhino は DXF に入っている画像を読み込まないことがあります。そのときは `*_原図.png` を
「Picture」コマンドで、`検証レポート.md` に書いてある左下・右上の座標に置いてください。

## 再実行のしかた
```
pip install opencv-python-headless ezdxf scikit-image scipy skan pytesseract pymupdf
mkdir -p work input && cp 大樹町生涯学習センター図面.pdf input/ && cd work && pdfimages -png ../input/*.pdf p
for p in 506 507 508; do python3 ../scripts/calib.py ../scripts/pages.json $p calib_$p.json; done
python3 ../scripts/run_region.py p-000.png 0 0 9930 7020 full_506
python3 ../scripts/run_region.py p-001.png 0 0 9930 7020 full_507
python3 ../scripts/run_region.py p-002.png 0 0 9930 7020 full_508
python3 ../scripts/run_page.py . ../output 506 506_1F
python3 ../scripts/run_page.py . ../output 507 507_2F
python3 ../scripts/run_page.py . ../output 508 508_3F_RF
python3 ../scripts/make_report.py . ../output
```
文字の目視判定の結果は `scripts/reviews/*.json` に保存してあり、再実行時はそれを使います
（線の取り出しの設定を変えて文字のまとまりが変わった場合は、`run_region.py` が作る切り出し一覧で判定し直す必要があります）。

## スクリプト（`scripts/`）
- `calib.py` / `pages.json`：通り芯から縮尺・傾きを校正（ページごとの通り芯の寸法を `pages.json` に記載）
- `tracer.py` / `extract.py`：画像の線をなぞって線分・円弧・円にする、破線類を1本にまとめる、文字のまとまりを切り分ける
- `run_region.py`：範囲（ページ全体）を処理し、文字のまとまりの切り出し一覧を作る
- `classify.py`：レイヤーへの振り分け（通り芯・寸法・配管系統・ハッチング・記号・文字など）
- `write_dxf.py`：DXFの書き出し（線種・線の太さ・文字・参照画像）
- `verify.py`：監査・重ね合わせ画像・取りこぼし率・通り芯の間隔の測定
- `run_page.py`：1ページ分の変換と検証 / `make_report.py`：3ページの報告のまとめ
- `save_review.py`・`reviews/`：文字の目視判定の記録
