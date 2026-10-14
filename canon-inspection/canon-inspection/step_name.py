"""step 3 (제품명 라벨: 2206N / 2206iF / 2206i / 2206 / 2206L / 2006N) - 라벨 끝 정사각형 기준 위치 찾기 + 기종별 정답과 비교해 PASS/FAIL.

기종이 정해지면 기대 라벨이 정해진다(예: 3029C009AA -> 2206N). 사진의 라벨이 기대 라벨과 같은지 본다.
 - 점수: 기대 라벨 기준 이미지와의 유사도(+-4px 위치 탐색)
 - FAIL 조건: (1) 다른 라벨 기준이 기대 라벨보다 더 비슷함(= 다른 기종의 라벨로 잘못 붙음, 예: 2206N 자리에 2006N)
              (2) 기대 라벨 유사도가 정상 검증 사진의 1퍼센타일 기준선보다 낮음(= 글자 깨짐/빠짐/라벨 없음)
              (3) 라벨 위치를 못 찾음(= 라벨 없음/가려짐 가능, 또는 재촬영 필요)
 - 일치율: 기대 라벨 유사도가 기준선과 같으면 50, 높을수록 100에 가까움
사용법 (Windows):
  py step_name.py --root "<데이터 경로>" --step 3 --out out_name
"""
import argparse, csv, random, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import cv2
from step_dots import collect, find_ng, imread_gray, ALL_MODELS
import name_locate as nl

LABEL_OF = {"3029C003AA": "2206N", "3029C004AA": "2206iF", "3029C005AA": "2206N", "3029C006AA": "2206N",
            "3029C009AA": "2206N", "3029C010AA": "2206i", "3030C001AA": "2206", "3030C002AA": "2206",
            "3030C003AA": "2206", "3030C004AA": "2206L", "3031C001AA": "2006N", "3031C002AA": "2006N",
            "3031C003AA": "2006N"}
VARIANTS = ["2206N", "2206iF", "2206i", "2206", "2206L", "2006N"]


def _win(crop):
    return crop[14:66, 90:326]          # 라벨 글자 영역


def f_gray(crop):
    w = cv2.resize(_win(crop), (59, 13), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((w - w.mean()) / (w.std() + 1e-6)).ravel()


def f_hog(crop, cell=(24, 13), nb=9):
    """밝기 대신 윤곽선 방향 분포(셀별) - 그림자·조명 차이와 몇 px의 위치 흔들림에 강하다 (numpy로 직접 구현)"""
    w = cv2.GaussianBlur(_win(crop), (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(w, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(w, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * nb
    H, W = w.shape; cw, ch = cell; out = []
    for y0 in range(0, H - ch + 1, ch):
        for x0 in range(0, W - cw + 1, cw):
            m = mag[y0:y0 + ch, x0:x0 + cw]; an = ang[y0:y0 + ch, x0:x0 + cw].astype(int) % nb
            out.append(np.bincount(an.ravel(), weights=m.ravel(), minlength=nb))
    v = np.array(out).ravel()
    return v / (np.linalg.norm(v) + 1e-6)


FEATS = {"gray": f_gray, "hog": f_hog}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--out", default="out_name")
    ap.add_argument("--search", type=int, default=4)
    ap.add_argument("--per_model", type=int, default=150)
    ap.add_argument("--aug", type=int, default=4, help="학습 사진당 변형 개수(0=끔)")
    ap.add_argument("--margin", type=float, default=0.5, help="기대 라벨 확률 기준선 배수(기준선x배수 미만이면 FAIL)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items, mism = collect(a.root, a.step, a.per_model)
    if not items:
        sys.exit("사진을 찾지 못했습니다.")
    for it in items:
        it["label"] = LABEL_OF[it["model"]]
    print(f"수집 {len(items)}장")
    meta, crops, fails = [], [], []
    for it in items:
        g = imread_gray(it["path"])
        r, why = nl.locate(g) if g is not None else (None, "no_image")
        if r is None:
            fails.append(dict(it, why=why)); continue
        meta.append(dict(it, angle=r["angle"], s=r["s"], mode=r["mode"])); crops.append(r["crop"])
    print(f"라벨 위치 찾기 성공 {len(meta)}/{len(items)}  (정사각형 기준 {sum(m['mode'] == 'square' for m in meta)}, 글자 위치로 추정 {sum(m['mode'] == 'text' for m in meta)})")
    why_cnt = {}
    for i in fails:
        why_cnt[i["why"]] = why_cnt.get(i["why"], 0) + 1
    if why_cnt:
        print("  실패 사유:", why_cnt)
    cnt = {}
    for it in items:
        cnt.setdefault(it["model"], [0, 0])[0] += 1
    for m in meta:
        cnt[m["model"]][1] += 1
    for m, (n, k) in cnt.items():
        print(f"  {m} [{LABEL_OF[m]}] {k}/{n}")
    with open(out / "locate_fail.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["파일", "기종", "기대 라벨", "사유(no_row=imageRUNNER 글자 줄 못 찾음, no_square=라벨 막대 못 찾음)"]] + [[str(i["path"]), i["model"], i["label"], i["why"]] for i in fails])
    if len(meta) < 20:
        sys.exit("위치를 찾은 사진이 너무 적습니다. locate_fail.csv 를 확인하세요.")
    Cr = np.stack(crops)
    lab = np.array([m["label"] for m in meta]); models = np.array([m["model"] for m in meta])
    # 샘플 모음
    tiles = []
    for v in VARIANTS:
        idx = [i for i in range(len(meta)) if lab[i] == v]
        random.Random(1).shuffle(idx)
        cells = [Cr[i] for i in idx[:8]] + [np.zeros((nl.OUT_H, nl.OUT_W), np.uint8)] * max(0, 8 - len(idx))
        row = np.hstack(cells)
        cv2.putText(row, v, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 3); cv2.putText(row, v, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        tiles.append(row)
    cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(out / "crops_by_label.png"))
    # 학습/검증/시험 분할: 파일·제품 이름이 아니라 사진 내용(바이트)의 해시 순서로 기종별 60/20/20 (한 스텝에서 제품당 사진은 한 장이라 사진 단위 분할이 곧 제품 단위 분할)
    from step_dots import content_fold
    hkey = [content_fold(m["path"], 10 ** 9) for m in meta]
    split = np.array([""] * len(meta), dtype=object)
    for mname in sorted(set(models)):
        ids = sorted([i for i in range(len(meta)) if models[i] == mname], key=lambda i: hkey[i])
        n = len(ids); n_tr = max(1, int(0.6 * n)); n_va = max(1, int(0.2 * n)) if n >= 3 else 0
        for k, i in enumerate(ids):
            split[i] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    print("분할:", {k: int((split == k).sum()) for k in ("train", "val", "test")})
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    trm = split == "train"; vam = split == "val"
    # 특징·분류기 조합 4가지를 검증 정상 사진으로 비교해 가장 좋은 것을 쓴다 (모두 정상 사진으로만 학습)
    cands = {}
    # 학습 사진에만 작은 이동·회전·밝기 변형을 4배로 더해 표본이 적은 기종의 접미사(i/iF/L) 구분을 돕는다 (검증·시험에는 쓰지 않음)
    arng = np.random.RandomState(0); idx_tr = np.where(trm)[0]; labA = np.repeat(lab[idx_tr], a.aug); CA = []
    for i in idx_tr:
        for _ in range(a.aug):
            h_, w_ = Cr[i].shape[:2]
            M_ = cv2.getRotationMatrix2D((w_ / 2, h_ / 2), arng.uniform(-2.5, 2.5), arng.uniform(0.96, 1.04)); M_[:, 2] += arng.uniform(-4, 4, 2)
            c_ = cv2.warpAffine(Cr[i], M_, (w_, h_), borderMode=cv2.BORDER_REPLICATE)
            CA.append(np.clip(c_.astype(np.float32) * arng.uniform(0.85, 1.15), 0, 255).astype(np.uint8))
    XA = {fn_: np.stack([f_(c) for c in CA]) for fn_, f_ in FEATS.items()} if a.aug > 0 else None
    for fn, f in FEATS.items():
        X = np.stack([f(c) for c in Cr])
        for cn, mk in (("LR", lambda: LogisticRegression(C=3.0, max_iter=4000)), ("SVM", lambda: SVC(C=10, probability=True, random_state=0))):
            m = mk().fit(np.vstack([X[trm], XA[fn]]), np.concatenate([lab[trm], labA])) if a.aug > 0 else mk().fit(X[trm], lab[trm])
            acc = float((m.predict(X[vam]) == lab[vam]).mean()) if vam.any() else float("nan")
            cands[(fn, cn)] = (acc, m, f)
            print(f"  후보 {fn}+{cn}: 검증 정확도 {acc:.3f}")
    key = max(cands, key=lambda k: cands[k][0])
    print(f"선택: {key[0]}+{key[1]}")
    _, clf, feat = cands[key]
    F = np.stack([feat(c) for c in Cr])
    classes = list(clf.classes_)

    def sims(crop):
        p = clf.predict_proba(feat(crop)[None])[0]
        return {v: float(p[k]) for k, v in enumerate(classes)}

    PR = clf.predict_proba(F)
    S = [{v: float(PR[i][k]) for k, v in enumerate(classes)} for i in range(len(meta))]
    # 학습에 쓴 사진은 분류기가 이미 본 사진이라 점수가 낙관적이다 -> 학습 사진만 제품 단위 3겹 교차검증 점수로 바꾼다(자기 제품이 빠진 분류기로 채점)
    from sklearn.base import clone
    tr_idx = np.where(trm)[0]; fo = {i: hkey[i] % 3 for i in tr_idx}
    for f_ in range(3):
        te_i = [i for i in tr_idx if fo[i] == f_]; fit_i = [i for i in tr_idx if fo[i] != f_]
        if not te_i or len(set(lab[fit_i])) < 2: continue
        if a.aug > 0:                                                                      # 최종 분류기와 같은 조건(증강 포함)으로 맞춰야 공정하다
            pos_ = {i: j for j, i in enumerate(idx_tr)}
            arows = np.concatenate([np.arange(pos_[i] * a.aug, (pos_[i] + 1) * a.aug) for i in fit_i])
            Xf = np.vstack([F[fit_i], XA[key[0]][arows]]); yf = np.concatenate([lab[fit_i], labA[arows]])
        else:
            Xf = F[fit_i]; yf = lab[fit_i]
        m_ = clone(clf).fit(Xf, yf); P_ = m_.predict_proba(F[te_i]); cl_ = list(m_.classes_)
        for r_, i in enumerate(te_i):
            S[i] = {v: (float(P_[r_][cl_.index(v)]) if v in cl_ else 0.0) for v in classes}
    exp_sim = np.array([S[i][lab[i]] for i in range(len(meta))])
    best_other = np.array([max([x for v, x in S[i].items() if v != lab[i]] or [0]) for i in range(len(meta))])
    va = split == "val"
    thr = float(np.percentile(exp_sim[va], 1)) if va.any() else float(np.percentile(exp_sim, 1))
    # 정상 일치율: 기대 라벨 유사도가 기준선과 같으면 50, 기준선 위로 갈수록 100에 가깝게
    def pct(x):
        thr2 = thr * a.margin
        return float(np.clip(50 + 50 * (x - thr2) / max(1.0 - thr2, 1e-3), 0, 100))
    def judge(e, bo):
        if bo > e:
            return "FAIL(다른 라벨로 판단됨)"
        if e < thr * a.margin:
            return "FAIL(기대 라벨과 다름)"
        return "PASS"
    print(f"\n기대 라벨 확률: 검증 정상 중앙값 {np.median(exp_sim[va]) if va.any() else float('nan'):.3f}, 기준선(1퍼센타일) {thr:.3f}")
    te = split == "test"
    res = [judge(exp_sim[i], best_other[i]) for i in range(len(meta))]
    fp = np.mean([res[i] != "PASS" for i in np.where(te)[0]]) if te.any() else float("nan")
    print(f"정상 test 오탐률(PASS 아님): {100 * fp:.1f}% ({sum(res[i] != 'PASS' for i in np.where(te)[0])}/{int(te.sum())})")
    arg = np.array([max(S[i], key=S[i].get) for i in range(len(meta))])
    print("라벨 종류 분류 정확도(test):", f"{np.mean(arg[te] == lab[te]):.3f}" if te.any() else "-")
    print("\n[라벨별 test 분류 결과] 기대 라벨 -> 예측 라벨별 장수")
    for v in VARIANTS:
        m = te & (lab == v)
        if m.any():
            d = {}
            for p in arg[m]:
                d[str(p)] = d.get(str(p), 0) + 1
            print(f"  {v}: {int(m.sum())}장  {d}")
    bad = [i for i in range(len(meta)) if res[i] != "PASS" and split[i] != "train"]
    if bad:
        tl = []
        for i in bad[:30]:
            c = cv2.resize(Cr[i], None, fx=0.5, fy=0.5).copy()
            cv2.putText(c, f"{lab[i]}->{arg[i]} {meta[i]['serial']}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            tl.append(c)
        while len(tl) % 3:
            tl.append(np.zeros_like(tl[0]))
        cv2.imencode(".png", np.vstack([np.hstack(tl[k:k + 3]) for k in range(0, len(tl), 3)]))[1].tofile(str(out / "errors.png"))
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "기종", "기대 라벨", "분할", "기대 유사도", "가장 비슷한 다른 라벨 유사도", "판정", "일치율"])
        for i in range(len(meta)):
            w.writerow([str(meta[i]["path"]), meta[i]["model"], lab[i], split[i], f"{exp_sim[i]:.3f}", f"{best_other[i]:.3f}", res[i], f"{pct(exp_sim[i]):.0f}"])
    # NG 시험
    ng = find_ng(a.root, a.step)
    print(f"\n[NG 시험] step {a.step}의 NG 사진 {len(ng)}장 (기준선 {thr:.3f}, 일치율 50 = 기준선)")
    ngt = []
    for it in ng:
        exp = LABEL_OF[it["model"]]
        g = imread_gray(it["path"])
        r, why = nl.locate(g) if g is not None else (None, "no_image")
        if r is None:
            what = "라벨 막대(글자+정사각형)가 없음 = NG(라벨 없음/다른 모양)" if why == "no_square" else "imageRUNNER 글자 줄을 못 찾음 = 재촬영 필요"
            print(f"  {it['model']} {it['serial']}  기대 {exp}  -> {what}  {it['path'].name}")
            continue
        sm = sims(r["crop"]); e = sm[exp]; bo = max(x for v, x in sm.items() if v != exp)
        pred = max(sm, key=sm.get)
        print(f"  {it['model']} {it['serial']}  기대 {exp}  예측 {pred}  기대 라벨 확률 {e:.3f} / 다른 라벨 최대 {bo:.3f}  일치율 {pct(e):.0f}  -> {judge(e, bo)}  {it['path'].name}")
        ngt.append(r["crop"])
    if ngt:
        cv2.imencode(".png", cv2.resize(np.vstack(ngt), None, fx=3, fy=3))[1].tofile(str(out / "ng_crops.png"))
    print(f"\n결과는 {out.resolve()} 에 저장됨")


if __name__ == "__main__":
    main()
