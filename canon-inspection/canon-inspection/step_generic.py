"""나머지 스텝(4,5,6,8,9,10~15) 일괄 점검: 정상 사진들과 '장면 구성'이 같은지 본다.
- 사진마다 어두운 구조의 배치를 숫자(32x24)로 만들고, 같은 기종(소수 기종은 묶음)의 정상 중 가장 가까운 것과의 거리를 잰다.
- 임계값 = 정상 leave-one-out 거리의 99.5 백분위. 일치율 = 100*0.5^(거리/임계) (임계에서 50).
- 부품 하나하나의 라벨 오류까지 보는 검사는 아님(→ 스텝 1·2·3 처럼 부품별 기준점 방식이 필요). 부품 누락·엉뚱한 구도·다른 구성 검출용.
- NG 폴더 사진도 같은 기준으로 판정해 함께 기록한다(정답으로 쓰지 않음).
사용: py step_generic.py --root "<root>" --steps 4,5,6,8,9,10,11,12,13,14,15 --out out_gen
출력: gen_result.csv  (스텝,구분번호,기종,제품,출처(정상/NG폴더),일치율,결과,표기)
"""
import argparse, csv
from pathlib import Path
import numpy as np
from step_dots import collect, find_ng
from step_scene import desc

PART = {4: "3102/3699", 5: "1525", 6: "2917", 8: "8588/0000", 9: "3780", 10: "V243/V242/B792", 11: "V243/V242/B792",
        12: "3779/V242", 13: "3779/C160", 14: "3779", 15: "C160"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--steps", default="4,5,6,8,9,10,11,12,13,14,15")
    ap.add_argument("--out", default="out_gen"); ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--margin", type=float, default=1.5, help="임계 배수(정상 99.5백분위 x margin)"); ap.add_argument("--cap", type=int, default=5000, help="기종당 최대 장수(속도)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    rows = []
    for st in [int(s) for s in a.steps.split(",")]:
        try:
            items = collect(a.root, st, per_model=a.cap)[0]
        except Exception as e:
            print(f"스텝 {st}: 수집 실패 {e}"); continue
        if len(items) < 10:
            print(f"스텝 {st}: 사진 {len(items)}장 (너무 적어 건너뜀)"); continue
        cnt = {}
        for it in items: cnt[it["model"]] = cnt.get(it["model"], 0) + 1
        grp = lambda m: m if cnt.get(m, 0) >= a.min_group else "소수기종묶음"
        X = np.array([desc(i["path"]) for i in items]); G = np.array([grp(i["model"]) for i in items])
        thr = {}; dist = np.zeros(len(items))
        for g in set(G):
            idx = np.where(G == g)[0]
            if len(idx) < 3: thr[g] = None; continue
            D = np.sqrt(((X[idx][:, None] - X[idx][None]) ** 2).sum(2)); np.fill_diagonal(D, 1e9)
            dist[idx] = D.min(1); thr[g] = float(np.percentile(dist[idx], 99.5))*a.margin
        def row(it, d, g, src):
            t = thr.get(g)
            if t is None: return (st, PART.get(st, "?"), it["model"], it["serial"], src, 0, "재확인")
            m = round(100 * 0.5 ** (d / t)); return (st, PART.get(st, "?"), it["model"], it["serial"], src, m, "OK" if d <= t else "NG")
        for j, it in enumerate(items): rows.append(row(it, dist[j], G[j], "정상"))
        nbad = sum(1 for r in rows if r[0] == st and r[4] == "정상" and r[6] != "OK")
        print(f"스텝 {st}: 정상 {len(items)}장 중 이상 {nbad}장 ({100*nbad/len(items):.2f}%)")
        for it in find_ng(a.root, st):
            g = grp(it["model"])
            if thr.get(g) is None: continue
            d = float(np.sqrt(((X[G == g] - desc(it["path"])) ** 2).sum(1)).min())
            r = row(it, d, g, "NG폴더"); rows.append(r)
            print(f"   NG폴더 {it['model']}/{it['serial']}: 거리 {d:.1f} (임계 {thr[g]:.1f}) → {r[6]}")
    with open(out / "gen_result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["스텝", "구분번호", "기종", "제품", "출처", "일치율", "결과", "표기"])
        for r in rows: w.writerow([*r, f"{r[1]}:{r[5]}:{r[6]}"])
    print("저장:", out / "gen_result.csv")
if __name__ == "__main__":
    main()
