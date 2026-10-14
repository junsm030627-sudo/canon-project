"""스텝 7(적재 한도 라벨 부품) '파란 부품 존재' 점검.
정상: 파란 부품이 화면의 큰 면적을 차지. 부품 누락(NG): 파란 면적 ~0.
사용:  py step_blue.py --root "<root>" --step 7 --out out_blue7
"""
import argparse, csv
SMIN = 80
from pathlib import Path
import numpy as np, cv2
from step_dots import collect, find_ng, imread_gray

def blue_frac(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 192))
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(h, (95, SMIN, 50), (135, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(m)
    return 0.0 if n < 2 else st[1:, 4].max() / m.size

def main():
    global SMIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--out", default="out_blue"); ap.add_argument("--smin", type=int, default=80, help="채도 하한(옅은 파랑이면 30)"); a = ap.parse_args(); SMIN = a.smin
    out = Path(a.out); out.mkdir(exist_ok=True)
    items = collect(a.root, a.step, per_model=10**6)
    if isinstance(items, tuple): items = items[0]
    ng = find_ng(a.root, a.step)
    fr = np.array([blue_frac(it["path"]) for it in items])
    thr = max(np.percentile(fr, 1) * 0.5, 0.02)   # 정상 1% 의 절반, 최소 2%
    print(f"정상 {len(fr)}장  파란 면적 비율: min {fr.min():.3f}  1% {np.percentile(fr,1):.3f}  중앙 {np.median(fr):.3f}")
    print(f"임계값 {thr:.3f} (미만이면 FAIL)  → 정상 오탐 {int((fr<thr).sum())}장 ({(fr<thr).mean()*100:.2f}%)")
    rows = [(it["model"], it["serial"], "normal", f, "FAIL" if f < thr else "OK") for it, f in zip(items, fr)]
    for it in ng:
        f = blue_frac(it["path"]); rows.append((it["model"], it["serial"], "NG", f, "FAIL" if f < thr else "OK"))
        print(f"NG {it['model']}/{it['serial']}: 파란 면적 {f:.3f} → {'FAIL(검출)' if f < thr else 'OK(놓침)'}")
    with open(out / "result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["model", "serial", "kind", "blue_frac", "result"]); w.writerows(rows)
    print("저장:", out / "result.csv")
if __name__ == "__main__":
    main()
