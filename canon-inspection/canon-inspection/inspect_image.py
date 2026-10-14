"""학습된 모델로 사진을 판독한다 (새 사진 입력용). 이름이 inspect.py 가 아닌 이유: 파이썬 표준 모듈 inspect 와 겹친다.

입력: 기종(--kind), 스텝 번호, 사진.  파일·제품 이름은 판정에 쓰지 않는다.
출력: 구분번호:일치율:결과  (예  2206N:92:OK)  와  PPT 한 줄(STEP 1 PASS(90.5) | STEP 2 FAIL(31.7) | ...)

사용 예
  1) 사진 한 장:   py inspect_image.py --model-dir models --kind 3029C009AA --step 3 --image C:\\사진\\a.jpg
  2) 여러 스텝:    py inspect_image.py --model-dir models --kind 3029C009AA --images 1=a.jpg 2=b.jpg 3=c.jpg
  3) 제품 폴더:    py inspect_image.py --model-dir models --kind 3029C009AA --product "<root>\\3029C009AA\\2EX36524"
                   (폴더 안의 step_N.jpg 를 스텝 N 의 사진으로 쓴다. 스텝 번호는 입력 정보이다)
"""
import argparse, re, sys
from pathlib import Path
import inspector_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models"); ap.add_argument("--kind", required=True, help="기종 (예: 3029C009AA)")
    ap.add_argument("--step", type=int); ap.add_argument("--image")
    ap.add_argument("--images", nargs="*", help="스텝=사진경로 (예: 1=a.jpg 3=c.jpg)")
    ap.add_argument("--product", help="step_N.jpg 가 들어 있는 제품 폴더")
    ap.add_argument("--verbose", action="store_true", help="이상 사유와 부품 위치도 출력")
    a = ap.parse_args()
    if a.kind not in L.sd.ALL_MODELS:
        sys.exit(f"알 수 없는 기종입니다: {a.kind}  (가능: {', '.join(L.sd.ALL_MODELS)})")
    jobs = {}
    if a.image:
        if not a.step:
            sys.exit("--image 는 --step 과 함께 써 주세요.")
        jobs[a.step] = a.image
    for kv in a.images or []:
        k, v = kv.split("=", 1); jobs[int(k)] = v
    if a.product:
        for p in Path(a.product).iterdir():
            m = L._STEP_RE.match(p.name) if p.is_file() else None
            if m and int(m.group(1)) in L.STEPS:
                jobs[int(m.group(1))] = str(p)
    if not jobs:
        sys.exit("판독할 사진이 없습니다 (--image/--images/--product 중 하나).")
    models = L.load_models(a.model_dir)
    results = []
    for st in sorted(jobs):
        r = L.predict_step(models, st, a.kind, jobs[st]); results.append(r)
        line = L.fmt(r)
        if a.verbose or r["res"] != "OK":
            line += ("   " + r["why"]) if r["why"] else ""
            if a.verbose and r.get("pos"):
                line += f"   위치{r['pos']}"
        print(f"STEP {st:>2}  {line}")
    print()
    print(L.ppt_line(results))


if __name__ == "__main__":
    main()
