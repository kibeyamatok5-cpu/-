# 大樹町生涯学習センター 衛生設備図 スキャンPDF → DXF 変換

- `scripts/` 変換に使うスクリプト一式（Python）
  - `calib.py` 通り芯から縮尺・傾きを校正 / `pages.json` ページごとの通り芯の寸法
  - `tracer.py` `extract.py` 画像の線をなぞって図形にする
  - `classify.py` レイヤーへの振り分け / `write_dxf.py` DXF書き出し / `verify.py` 検証・重ね合わせ画像
  - `reviews/` 文字のまとまりの目視判定の記録
- `output/` 成果物（いまは試作のみ: `output/試作_506/`）

再実行（試作）:
```
pip install opencv-python-headless ezdxf scikit-image scipy skan pytesseract pymupdf
mkdir -p work input && cp 大樹町生涯学習センター図面.pdf input/ && cd work && pdfimages -png ../input/*.pdf p
python3 ../scripts/calib.py ../scripts/pages.json 506 calib_506.json
python3 ../scripts/run_region.py p-000.png 5250 0 9800 2900 proto
python3 ../scripts/proto.py . out_proto
```
