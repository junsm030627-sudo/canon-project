"""스텝별 결과(out_dots, out_logo5, out_name5, out_blue7)를 하나로 합쳐
  final_result.csv  : 사진 1장 = 1줄 (스텝, 구분번호, 일치율, 결과)
  final_summary.txt : 스텝별 건수
  review_list.csv   : 기준상 FAIL/재확인인 사진 (눈으로 확인할 목록)
을 만든다. 출력 형식:  구분번호:일치율:결과  (예  B793:90:OK)
사용:  py merge_results.py --root "<root>"
"""
import argparse, csv, re
from pathlib import Path

def serial_of(p):
    parts = re.split(r"[\\/]", p)
    return parts[-2] if len(parts) >= 2 else ""

def rd(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--blue_thr", type=float, default=0.071)
    a = ap.parse_args(); R = Path(a.root)
    rows = []
    # 스텝 1  토너 지역 라벨 (C550/C551)
    f = R / "out_dots" / "per_image.csv"
    if f.exists():
        for r in rd(f):
            exp = None
            ok = r["맞음"].strip().lower() == "true"
            pred = r["예측"]; exp = pred if ok else ("C550" if pred == "C551" else "C551")
            sim = float(r["C550 유사도" if exp == "C550" else "C551 유사도"])
            rows.append((1, exp, r["기종"], serial_of(r["파일"]), round(max(0, min(100, sim * 100))), "OK" if ok else "NG"))
    f = R / "out_dots" / "locate_fail.csv"      # 위치를 못 찾은 사진 = 재확인
    if f.exists():
        for r in rd(f):
            rows.append((1, "C550/C551", r["기종"], serial_of(r["파일"]), 0, "재확인"))
    # 스텝 2  좌전카바 각인 B793
    f = R / "out_logo5" / "per_image.csv"
    if f.exists():
        for r in rd(f):
            if not r["일치율"]:      # 학습에 쓰인 사진은 점수를 내지 않음
                continue
            m = int(float(r["일치율"]))
            rows.append((2, "B793", r["기종"], serial_of(r["파일"]), m, "OK" if m >= 50 else "NG"))
    # 스텝 3  제품명 라벨
    f = R / "out_name5" / "per_image.csv"
    if f.exists():
        for r in rd(f):
            rows.append((3, r["기대 라벨"], r["기종"], serial_of(r["파일"]), int(float(r["일치율"])), "OK" if r["판정"].startswith("PASS") else "NG"))
    f = R / "out_name5" / "locate_fail.csv"     # 글자 줄/라벨 막대를 못 찾음 = 재확인
    if f.exists():
        for r in rd(f):
            rows.append((3, r["기대 라벨"], r["기종"], serial_of(r["파일"]), 0, "재확인"))
    # 스텝 7  적재 한도 라벨 부품(1525) 존재
    f = R / "out_blue7" / "result.csv"
    if f.exists():
        for r in rd(f):
            if r["kind"] != "normal": continue
            fr = float(r["blue_frac"]); m = round(100 * (1 - 0.5 ** (fr / a.blue_thr)))   # 임계(blue_thr)에서 50, 크게 넘어도 100으로 포화되지 않음
            rows.append((7, "1525", r["model"], r["serial"], m, "OK" if r["result"] == "OK" else "NG"))
    # 나머지 스텝 (장면 구성 점검)
    f = R / "out_gen" / "gen_result.csv"
    ngc = []
    if f.exists():
        for r in rd(f):
            t = (int(r["스텝"]), r["구분번호"], r["기종"], r["제품"], int(float(r["일치율"])), r["결과"])
            (rows if r["출처"] == "정상" else ngc).append(t)
    if ngc:
        with open(R / "ng_cases.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(["스텝", "구분번호", "기종", "제품", "일치율", "결과"]); w.writerows(ngc)
    # 부품 탐지기(step_part.py)를 돌린 스텝(6,8~14): 장면 거리와 부품 탐지 중 하나라도 이상이면 NG, 일치율은 낮은 쪽
    f = R / "out_part" / "part_result.csv"
    if f.exists():
        pr = {}
        for r in rd(f):
            if r["출처"] == "정상": pr[(int(r["스텝"]), r["기종"], r["제품"])] = (int(float(r["일치율"])), r["결과"], r["위치"])
        n_add = {}
        for i, r in enumerate(rows):
            p = pr.get((r[0], r[2], r[3]))
            if p is None: continue
            res = r[5] if r[5] == "NG" or p[1] == "OK" else p[1]
            if r[5] == "OK" and p[1] != "OK": n_add[r[0]] = n_add.get(r[0], 0) + 1
            if r[5] == "재확인" and p[1] != "재확인": res = p[1]
            rows[i] = (r[0], r[1], r[2], r[3], min(r[4], p[0]), res)
        print("부품 탐지기가 추가로 잡은 제품:", {k: n_add[k] for k in sorted(n_add)})
    # 파란 부품 존재 검사를 추가로 돌린 스텝(4,5,6)은 둘 중 하나라도 이상이면 NG
    for st in (4, 5):   # 스텝 6은 파란 용기가 아니라 흰 부품이 대상이라 제외(오탐 51장 대비 이득 없음)
        f = R / f"out_blue{st}" / "result.csv"
        if not f.exists(): continue
        bad = {(r["model"], r["serial"]): float(r["blue_frac"]) for r in rd(f) if r["kind"] == "normal" and r["result"] != "OK"}
        for i, r in enumerate(rows):
            if r[0] == st and (r[2], r[3]) in bad:
                rows[i] = (r[0], r[1], r[2], r[3], min(r[4], 30), "NG")
        ngc += [(st, r[1], r[2], r[3], 0, "NG") for r in []]
        print(f"스텝 {st}: 파란 부품 면적 부족 {len(bad)}대 추가 NG")
    out = R / "final_result.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["스텝", "구분번호", "기종", "제품", "일치율", "결과", "표기"])
        for s in sorted(rows, key=lambda x: (x[3], x[0])):
            w.writerow([*s, f"{s[1]}:{s[4]}:{s[5]}"])
    with open(R / "review_list.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["스텝", "구분번호", "기종", "제품", "일치율", "결과"])
        w.writerows([s for s in sorted(rows, key=lambda x: (x[0], x[3])) if s[5] != "OK"])
    lines = []
    for st in sorted({r[0] for r in rows}):
        rs = [r for r in rows if r[0] == st]; n = len(rs)
        c = {k: sum(1 for r in rs if r[5] == k) for k in ("OK", "NG", "재확인")}
        lines.append(f"스텝 {st}: {n}건  OK {c['OK']}  NG {c['NG']}  재확인 {c['재확인']}  (이상 {100*(n-c['OK'])/n:.1f}%)")
    prod = {}
    for r in rows: prod.setdefault(r[3], {})[r[0]] = r[5]
    full = {k: v for k, v in prod.items() if all(x in v for x in (1, 2, 3, 7))}     # 스텝 1,2,3,7 모두 판정된 제품
    bad = sum(1 for v in full.values() if any(x != "OK" for x in v.values()))
    lines.append(f"스텝 1·2·3·7 모두 판정된 제품 {len(full)}대 중 한 스텝이라도 이상 {bad}대 ({100*bad/max(1,len(full)):.1f}%)  ← 스텝별 오탐이 누적되는 효과")
    lines.append("※ 스텝 2·3은 학습에 쓰지 않은 사진(검증·시험)만 / 스텝 3 은 학습 사진도 포함(점수가 낙관적일 수 있음)")
    (R / "final_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print("저장:", out)
if __name__ == "__main__":
    main()
