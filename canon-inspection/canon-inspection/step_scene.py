"""구도/장면 일치 점검 (스텝 9, 14 등). 정상 사진들의 '어두운 구조 배치'와 가장 가까운 정상과의 거리를 잰다.
다른 곳을 찍었거나 구도가 크게 틀어진 사진은 거리가 커진다. 임계값 = 정상 leave-one-out 거리의 99.5 백분위.
사용: py step_scene.py --root "<root>" --step 9 --out out_scene9
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from step_dots import collect, find_ng

def desc(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, (128, 96)).astype(np.float32)
    gb = cv2.GaussianBlur(g, (0, 0), 1.5); loc = cv2.GaussianBlur(gb, (0, 0), 12)
    d = np.clip((loc - gb) / 30, 0, 1)
    d = cv2.resize(d, (32, 24), interpolation=cv2.INTER_AREA).ravel()
    return (d - d.mean()) / (d.std() + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", default="out_scene"); ap.add_argument("--min_group", type=int, default=15)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    items = collect(a.root, a.step, per_model=10**6)[0]
    cnt = {}
    for it in items: cnt[it["model"]] = cnt.get(it["model"], 0) + 1
    grp = lambda m: m if cnt.get(m, 0) >= a.min_group else "소수기종묶음"
    X = np.array([desc(i["path"]) for i in items]); G = np.array([grp(i["model"]) for i in items])
    dist = np.zeros(len(items)); thr = {}
    for g in set(G):
        idx = np.where(G == g)[0]; Xg = X[idx]
        D = np.sqrt(((Xg[:, None] - Xg[None]) ** 2).sum(2)); np.fill_diagonal(D, 1e9)
        dist[idx] = D.min(1); thr[g] = float(np.percentile(dist[idx], 99.5))
        print(f"그룹 {g}: 정상 {len(idx)}장  LOO거리 중앙 {np.median(dist[idx]):.1f}  99.5% {thr[g]:.1f}")
    bad = np.array([dist[j] > thr[G[j]] for j in range(len(items))])
    print(f"정상 오탐 {int(bad.sum())}장 ({bad.mean()*100:.2f}%)  ※ 정의상 약 0.5%")
    rows = [(i["model"], i["serial"], "normal", round(dist[j], 1), "FAIL" if bad[j] else "OK") for j, i in enumerate(items)]
    for i in find_ng(a.root, a.step):
        g = grp(i["model"]); d = float(np.sqrt(((X[G == g] - desc(i["path"])) ** 2).sum(1)).min())
        f = d > thr[g]; rows.append((i["model"], i["serial"], "NG", round(d, 1), "FAIL" if f else "OK"))
        print(f"NG {i['model']}/{i['serial']} {Path(i['path']).name[:23]}: 거리 {d:.1f} (임계 {thr[g]:.1f}) → {'FAIL(검출)' if f else 'OK(놓침)'}")
    with open(out / "result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["model", "serial", "kind", "dist", "result"]); w.writerows(rows)
if __name__ == "__main__":
    main()
