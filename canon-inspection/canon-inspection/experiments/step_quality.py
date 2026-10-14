"""촬영 품질 점검(밝기·선명도). 스텝 8처럼 '너무 어두운' 사진, 흐린 사진을 잡는다.
정상 데이터의 분포에서 임계값을 정한다(하위 0.5%).
사용: py step_quality.py --root "<root>" --step 8 --out out_q8
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from step_dots import collect, find_ng

def qual(path):
    g = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (512, 384))
    return float(np.mean(g)), float(np.percentile(g, 95)), float(cv2.Laplacian(g, cv2.CV_64F).var())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", default="out_q"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    items = collect(a.root, a.step, per_model=10**6)[0]
    ng = find_ng(a.root, a.step)
    Q = np.array([qual(i["path"]) for i in items])
    names = ["평균밝기", "밝기95%", "선명도"]
    thr = [np.percentile(Q[:, k], 0.5) * 0.9 for k in range(3)]
    print(f"정상 {len(Q)}장")
    for k in range(3):
        print(f"  {names[k]}: min {Q[:,k].min():.1f}  0.5% {np.percentile(Q[:,k],0.5):.1f}  중앙 {np.median(Q[:,k]):.1f}  → 임계 {thr[k]:.1f}")
    bad = (Q < np.array(thr)).any(1)
    print(f"정상 오탐 {int(bad.sum())}장 ({bad.mean()*100:.2f}%)")
    rows = [(i["model"], i["serial"], "normal", *Q[j], "FAIL" if bad[j] else "OK") for j, i in enumerate(items)]
    for i in ng:
        q = qual(i["path"]); f = any(q[k] < thr[k] for k in range(3))
        rows.append((i["model"], i["serial"], "NG", *q, "FAIL" if f else "OK"))
        why = [names[k] for k in range(3) if q[k] < thr[k]]
        print(f"NG {i['model']}/{i['serial']}: 밝기 {q[0]:.1f}/{q[1]:.1f} 선명도 {q[2]:.1f} → {'FAIL('+','.join(why)+')' if f else 'OK(놓침)'}")
    with open(out / "result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["model", "serial", "kind", "mean", "p95", "sharp", "result"]); w.writerows(rows)
if __name__ == "__main__":
    main()
