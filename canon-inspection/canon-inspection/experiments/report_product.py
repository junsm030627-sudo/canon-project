"""과제 슬라이드 형식으로 제품 1대의 스텝별 판정을 출력한다.
  STEP 1 PASS(90.5) | STEP 2 FAIL(31.7) | ...
사전: py step_generic.py ... 와 py merge_results.py ... 를 먼저 실행해 final_result.csv 가 있어야 한다.
사용:  py report_product.py --root "<root>" --serial 2EQ16387
       py report_product.py --root "<root>" --model 3029C003AA --n 5     (그 기종의 제품 5대)
"""
import argparse, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--serial"); ap.add_argument("--model"); ap.add_argument("--n", type=int, default=5); ap.add_argument("--full", action="store_true", help="스텝 1~14 모두 판정된 제품만 고른다")
    a = ap.parse_args(); R = Path(a.root)
    rows = list(csv.DictReader(open(R / "final_result.csv", encoding="utf-8-sig")))
    by = {}
    for r in rows: by.setdefault((r["기종"], r["제품"]), {})[int(r["스텝"])] = r
    keys = [k for k in by if (a.serial and k[1] == a.serial) or (a.model and k[0] == a.model and not a.serial)] if not a.full else []
    if a.full:
        keys = [k for k in by if all(s in by[k] for s in range(1, 15)) and (not a.model or k[0] == a.model)]
        print(f"스텝 1~14가 모두 판정된 제품: {len(keys)}대 (앞 {a.n}대만 출력)")
    if not a.serial and (a.model or a.full): keys = keys[: a.n]
    if not keys: print("해당 제품이 없습니다"); return
    for k in keys:
        parts = []
        for s in range(1, 16):
            r = by[k].get(s)
            parts.append(f"STEP {s} 미판정" if r is None else f"STEP {s} {'PASS' if r['결과']=='OK' else 'FAIL' if r['결과']=='NG' else '재확인'}({r['일치율']})")
        print(f"기종 {k[0]}  제품 {k[1]}\n  " + " | ".join(parts))
        print("  구분번호:일치율:결과 → " + "  ".join(by[k][s]["표기"] for s in sorted(by[k])))
if __name__ == "__main__":
    main()
