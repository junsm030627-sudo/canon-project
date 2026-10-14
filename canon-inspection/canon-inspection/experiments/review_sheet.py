"""review_list.csv 의 이상 판정 사진을 스텝별로 한 장에 모아 눈으로 확인한다.
사용: py review_sheet.py --root "<root>" --step 9 --n 12      → review_step9.png
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--step", type=int, required=True); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(); R = Path(a.root)
    rows = [r for r in csv.DictReader(open(R / "review_list.csv", encoding="utf-8-sig")) if int(r["스텝"]) == a.step]
    rows = sorted(rows, key=lambda r: int(float(r["일치율"])))[: a.n]
    tiles = []
    for r in rows:
        p = R / r["기종"] / r["제품"] / f"step_{a.step}.jpg"
        if not p.exists(): continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR); im = cv2.resize(im, (320, 240))
        cv2.putText(im, f"{r['제품']} {r['일치율']}", (4, 18), 0, 0.55, (0, 255, 255), 2); tiles.append(im)
    if not tiles: print("표시할 사진이 없습니다"); return
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 4)
    img = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    o = R / f"review_step{a.step}.png"; cv2.imencode(".png", img)[1].tofile(str(o)); print("저장:", o)
if __name__ == "__main__":
    main()
