"""학습에 쓰지 않은 사진(시험 세트 제품)과 NG 폴더 사진을 저장된 모델로 판독해 '처음 보는 사진' 성능을 낸다.

출력(out_eval/): holdout_result.csv (시험 제품의 스텝별 판정), ng_result.csv (NG 폴더 사진 판정), summary.txt
사용: py evaluate_holdout.py --root "<root>" --model-dir models --out out_eval
"""
import argparse, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import inspector_lib as L

_M = None


def _init(mdir):
    global _M
    _M = L.load_models(mdir)


def _job(args):
    st, model, serial, path, src = args
    try:
        r = L.predict_step(_M, st, model, path)
    except Exception as e:
        r = L._res(st, model, 0, "재확인", f"판독 오류: {e}")
    return (st, model, serial, src, Path(path).name, L.fmt(r), round(r["rate"]), r["res"], r["why"], r.get("pos"), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="out_eval"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--on", default="holdout", choices=["holdout", "all"], help="holdout=시험 제품만, all=전체 제품(학습 사진 포함)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = L.load_models(a.model_dir).get("_meta", {})
    if meta.get("all") and a.on == "holdout":
        print("주의: 이 모델은 --all 로 학습되어 시험 세트가 없습니다. 전체 제품이 학습에 쓰였으므로 결과는 낙관적입니다.")
    jobs = []
    for m, d, files in L.list_products(a.root):
        if a.on == "holdout" and not L.is_holdout(files):
            continue
        for st, p in files.items():
            jobs.append((st, m, d.name, str(p), "정상"))
    nn = len(jobs)
    ngs = L.ng_photos(a.root)
    for m, serial, st, p in ngs:
        jobs.append((st, m, serial, str(p), "NG폴더"))
    print(f"판독 {nn}장(정상) + {len(ngs)}장(NG 폴더) ...", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.model_dir,)) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    hdr = ["스텝", "기종", "제품", "출처", "파일", "표기", "일치율", "결과", "사유"]
    for name, sel in (("holdout_result.csv", "정상"), ("ng_result.csv", "NG폴더")):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows([r[:9] for r in sorted(res, key=lambda x: x[:9]) if r[3] == sel])
    lines = []
    lines.append(f"[{'시험 세트' if a.on == 'holdout' else '전체'} 정상 사진 판독]  (학습에 쓰지 않은 제품)" if a.on == "holdout" else "[전체 제품 판독]")
    norm = [r for r in res if r[3] == "정상"]
    for st in sorted({r[0] for r in norm}):
        rs = [r for r in norm if r[0] == st]; n = len(rs)
        c = {k: sum(1 for r in rs if r[7] == k) for k in ("OK", "NG", "재확인")}
        lines.append(f"스텝 {st:>2}: {n}장  OK {c['OK']}  NG {c['NG']}  재확인 {c['재확인']}  (이상 {100 * (n - c['OK']) / n:.1f}%)")
    CATS = (("로고 모양", "로고 모양"), ("위치·크기", "로고 위치·크기"), ("종횡비", "로고 종횡비"), ("파란 부품", "파란 부품 면적"),
            ("장면 구성", "장면 구성"), ("확인되지 않음", "부품 탐지기"), ("로 보임", "라벨 종류"), ("와 다름", "라벨 확률 낮음"))
    lines.append("")
    lines.append("[이상 판정 사유 분류] (한 사진에 사유가 여러 개일 수 있음)")
    for st in sorted({r[0] for r in norm}):
        cnt = {}
        for r in norm:
            if r[0] == st and r[7] != "OK":
                key = r[7] if r[7] == "재확인" and not r[8] else None
                for k, nm in CATS:
                    if k in r[8]:
                        cnt[nm] = cnt.get(nm, 0) + 1
                if r[7] == "재확인":
                    cnt["재확인:" + (r[8] or "사유없음")[:20]] = cnt.get("재확인:" + (r[8] or "사유없음")[:20], 0) + 1
        if cnt:
            lines.append(f"스텝 {st:>2}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
    prod = {}
    for r in norm:
        prod.setdefault((r[1], r[2]), []).append(r[7])
    badp = sum(1 for v in prod.values() if any(x != "OK" for x in v))
    lines.append(f"한 스텝이라도 이상인 제품: {badp}/{len(prod)}대 ({100 * badp / max(1, len(prod)):.1f}%)")
    lines.append("")
    lines.append("[NG 폴더 사진 판독]  (학습에 쓰지 않음. 정답이 아니라 사후 확인용)")
    for r in sorted((r for r in res if r[3] == "NG폴더"), key=lambda x: x[:9]):
        lines.append(f"  스텝 {r[0]:>2}  {r[1]}/{r[2]}  {r[4][:23]}  {r[5]}  {r[8]}")
    # 이상 판정된 정상(시험 세트) 사진 모음 그림: 스텝마다 최대 24장, 부품 탐지기 위치는 초록 상자
    import cv2, numpy as np
    for st in sorted({r[0] for r in norm}):
        bad = [r for r in norm if r[0] == st and r[7] != "OK"][:24]
        if not bad:
            continue
        tiles = []
        for r in bad:
            im = cv2.imdecode(np.fromfile(r[10], np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.resize(im, (256, 192), interpolation=cv2.INTER_AREA)
            if r[9]:
                im = cv2.rectangle(im, (int(r[9][0]) - 16, int(r[9][1]) - 16), (int(r[9][0]) + 16, int(r[9][1]) + 16), (0, 255, 0), 1)
            lab = f"{r[2]} {r[6]} {r[7]}"
            cv2.putText(im, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3); cv2.putText(im, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imencode(".png", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))[1].tofile(str(out / f"flagged_step{st}.png"))
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
