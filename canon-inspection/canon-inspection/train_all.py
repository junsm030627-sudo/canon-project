"""모든 스텝의 판독 모델을 학습해 models/ 에 저장한다.

시험용 분할: 제품마다 (가장 앞 스텝 사진 내용의 crc32) % 5 == 0 인 제품(약 20%)은 학습에 쓰지 않고 시험 세트로 남긴다.
  파일·제품 이름은 분할에도 쓰지 않는다. NG 폴더 사진은 학습에 쓰지 않는다.
--all 을 주면 시험 세트 없이 전체 제품으로 학습한다(평가를 마친 뒤 최종 배포용).

사용: py train_all.py --root "<root>" --model-dir models
      py train_all.py --root "<root>" --model-dir models --steps 1,2,3        (일부 스텝만 다시 학습 — 기존 모델에 합친다)
"""
import argparse, csv, time
from pathlib import Path
import inspector_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model-dir", default="models")
    ap.add_argument("--steps", default=",".join(map(str, L.STEPS)))
    ap.add_argument("--all", action="store_true", help="시험 세트를 떼지 않고 전체로 학습")
    a = ap.parse_args()
    mdir = Path(a.model_dir); mdir.mkdir(parents=True, exist_ok=True)
    print("제품 목록 읽는 중 ...", flush=True)
    prods = L.list_products(a.root)
    hold = [L.is_holdout(f) for _, _, f in prods]
    print(f"제품 {len(prods)}대 중 시험 세트 {0 if a.all else sum(hold)}대 (학습 {len(prods) - (0 if a.all else sum(hold))}대)")
    with open(mdir / "split.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["기종", "제품", "시험세트"])
        for (m, d, _), h in zip(prods, hold):
            w.writerow([m, d.name, 0 if a.all else int(h)])
    try:
        models = L.load_models(mdir) if (mdir / "model.pkl").exists() else {}
    except Exception:
        models = {}
    models["_meta"] = dict(all=bool(a.all), n_products=len(prods), n_holdout=0 if a.all else int(sum(hold)))
    log = []
    for st in [int(s) for s in a.steps.split(",")]:
        items = []
        for (m, d, files), h in zip(prods, hold):
            if st in files and (a.all or not h):
                items.append(dict(model=m, serial=d.name, path=files[st], variant=L.sd.variant_of(m)))
        t0 = time.time()
        try:
            models[st] = L.fit_step(st, items, a.root)
            print(f"스텝 {st}: 학습 사진 {len(items)}장 -> 완료 ({time.time() - t0:.0f}초)", flush=True)
            log.append((st, len(items), "완료", ""))
        except Exception as e:
            models.pop(st, None)
            print(f"스텝 {st}: 학습 실패 - {e}", flush=True)
            log.append((st, len(items), "실패", str(e)))
        L.save_models(models, mdir)
    with open(mdir / "train_log.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["스텝", "학습 사진", "상태", "비고"]); w.writerows(log)
    print(f"저장: {mdir / 'model.pkl'}")


if __name__ == "__main__":
    main()
