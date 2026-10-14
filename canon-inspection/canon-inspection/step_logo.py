"""step 2 (좌전카바 각인: Canon 로고) - 정상만으로 학습하는 이상탐지.

13개 기종 모두 같은 로고라서 종류 분류가 아니라 "로고가 정상 모양인가"를 본다.
  1. 로고 찾기: 글자 크기의 어두운 덩어리(C,a,n,o,n)만 골라 묶는다 (모서리 기둥·가장자리 띠는 제외)
  2. 똑바로 세운 표준 crop: 로고 자신의 축(PCA)과 길이로 회전·크기·위치를 정규화
  3. 정상 학습 사진의 로고 마스크 평균 = 기준 모양
  4. 점수 = 기준 모양과 어긋난 면적 (허용 오차 안쪽은 무시). 기준선 = 검증 정상의 99퍼센타일
  5. 로고 전체가 안 보이거나 비율이 이상하면 "판정 불가(재촬영)"로 따로 센다

사용법 (Windows):
  py step_logo.py --root "<데이터 경로>" --step 2 --out out_logo2
"""
import argparse, csv, random, re, sys
from pathlib import Path
import numpy as np
import cv2

ALL_MODELS = ["3029C003AA", "3029C004AA", "3029C005AA", "3029C006AA", "3029C009AA", "3029C010AA",
              "3030C001AA", "3030C002AA", "3030C003AA", "3030C004AA", "3031C001AA", "3031C002AA", "3031C003AA"]
SW = 256                      # 로고 찾기는 가로 256px로 줄여서
CW, CH = 240, 96              # 표준 crop 크기
LOGO_LEN = 216.0              # 표준 crop에서 로고 길이(px)
TOL = 3.0                     # 허용 오차(px)
TARGET_XH = 28.0              # 표준 crop에서 글자(소문자) 높이(px) - 로고 전체 길이가 아니라 글자 높이로 크기를 맞춘다


def imread_gray(p):
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is not None and im.shape[1] != 1024:
        im = cv2.resize(im, (1024, int(im.shape[0] * 1024 / im.shape[1])), interpolation=cv2.INTER_AREA)
    return im


def step_file(d, step):
    pat = re.compile(rf"^step[_\- ]?0*{step}\.(jpg|jpeg|png|bmp)$", re.I)
    for p in d.iterdir():
        if p.is_file() and pat.match(p.name):
            return p
    return None


def collect(root, step):
    items, ng = [], []
    pat_ng = re.compile(rf"step[_\- ]?0*{step}\.(jpg|jpeg|png|bmp)$", re.I)
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir() or mdir.name not in ALL_MODELS:
            continue
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if sdir.name.upper() == "NG":
                continue
            model = mdir.name                  # 기종은 입력으로 주어진 값. 제품번호 접두어로 추정·제외하지 않는다
            f = step_file(sdir, step)
            if f:
                items.append(dict(model=model, serial=sdir.name, path=f))
            nd = sdir / "NG"
            if nd.is_dir():
                for g in sorted(nd.iterdir()):
                    if g.is_file() and pat_ng.search(g.name):
                        ng.append(dict(model=model, serial=sdir.name, path=g))
    return items, ng


def _seg(gs, thr=None, adaptive=None):
    """gs: 가로 256로 줄인 gray. thr: 어두움 기준(없으면 기본), adaptive=(block,C): 국소 적응 임계. 반환: 로고 마스크 또는 None, 사유"""
    b = cv2.GaussianBlur(gs, (0, 0), 1.5)
    if adaptive is not None:
        b2 = cv2.GaussianBlur(gs, (0, 0), 1.2)
        m = (cv2.adaptiveThreshold(b2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   adaptive[0], adaptive[1]) > 0).astype(np.uint8)
    else:
        if thr is None:
            thr = min(70, np.percentile(b, 35))
        m = (b < thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    H, W = gs.shape
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if a > 5000 or a < 60 or max(w, h) > 150:      # 길고 큰 그림자 띠 제외
            continue
        keep[lab == i] = 1
    hs = [st[i][3] for i in range(1, n) if keep[lab == i].any() and st[i][3] >= 8]
    hm = float(np.median(hs)) if hs else 20.0
    kw, kh = int(2.2 * hm) | 1, int(0.7 * hm) | 1
    mc = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(mc)
    best, bs = None, -1
    for i in range(1, n2):
        mass = int(keep[lab2 == i].sum())
        d = np.hypot(cen2[i][0] - W / 2, cen2[i][1] - H / 2)
        sc = mass - 2 * d
        if mass > 700 and sc > bs:
            bs, best = sc, i
    if best is None:
        return None, "로고 덩어리 없음"
    chosen = (keep & (lab2 == best)).astype(np.uint8)
    rk = int(1.3 * hm) | 1
    region = cv2.dilate(chosen, cv2.getStructuringElement(cv2.MORPH_RECT, (rk, rk)))
    excluded = cv2.dilate((m & (1 - keep)).astype(np.uint8), np.ones((7, 7), np.uint8))   # 모서리 기둥·가장자리 띠
    region = region & (1 - excluded) | chosen
    return (chosen, region.astype(np.uint8)), ""


def find_logo(gs):
    """밝기가 사진마다 달라 한 가지 기준으론 글자 일부(큰 C 등)가 빠진다 -> 기준을 바꿔가며 시도하고,
    로고 모양(가로세로비 4.6~5.6)이 나오는 첫 결과를 쓴다. 끝까지 안 나오면 첫 결과를 그대로 돌려줘 게이트가 이상으로 보게 한다."""
    b = cv2.GaussianBlur(gs, (0, 0), 1.5)
    tries = [dict()] + [dict(thr=np.percentile(b, p)) for p in (25, 45, 55)] + \
            [dict(thr=t) for t in (60, 80, 95, 110)] + [dict(adaptive=a) for a in ((61, 16), (101, 12), (61, 8))]
    first = None
    for kw in tries:
        r, why = _seg(gs, **kw)
        if r is None:
            continue
        if first is None:
            first = r
        if 4.6 <= canon_aspect(r) <= 5.6:
            return r, ""
    return (first, "") if first is not None else (None, "로고 덩어리 없음")


def canon_aspect(masks):
    ys, xs = np.nonzero(masks[0])
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    cov = np.cov((P - c).T)
    ang = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    u = np.array([np.cos(ang), np.sin(ang)]); v = np.array([-np.sin(ang), np.cos(ang)])
    pu, pv = (P - c) @ u, (P - c) @ v
    L = np.percentile(pu, 99.5) - np.percentile(pu, 0.5)
    Hh = np.percentile(pv, 99.5) - np.percentile(pv, 0.5)
    return L / max(Hh, 1e-6)


def canon(g, masks):
    """g: 원본(1024) gray, mask_s: 256 기준 로고 마스크. 로고 자신의 축(PCA)으로 회전하고,
    크기는 글자 높이의 중앙값으로 맞춘다(글자가 하나 빠져도 크기가 변하지 않도록).
    반환 (crop, 종횡비, 각도, 글자높이)"""
    mask_s, region_s = masks
    sc = g.shape[1] / float(SW)
    ys, xs = np.nonzero(mask_s)
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    cov = np.cov((P - c).T)
    ang = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    u = np.array([np.cos(ang), np.sin(ang)]); v = np.array([-np.sin(ang), np.cos(ang)])
    pu, pv = (P - c) @ u, (P - c) @ v
    lo, hi = np.percentile(pu, [0.5, 99.5])
    L = hi - lo
    Hh = np.percentile(pv, 99.5) - np.percentile(pv, 0.5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask_s)
    hs = []
    for i in range(1, n):
        if st[i][4] < 60:
            continue
        yy, xx = np.nonzero(lab == i)
        q = (np.stack([xx, yy], 1) - c) @ v
        hs.append(np.percentile(q, 99) - np.percentile(q, 1))
    xh = float(np.median(hs)) if hs else Hh * 0.6
    mid = c * sc                                             # 질량 중심(원본 좌표)
    k = TARGET_XH / (xh * sc)
    R = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), np.degrees(ang), k)
    R[:, 2] += (CW / 2 - mid[0], CH / 2 - mid[1])
    crop = cv2.warpAffine(g, R, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    reg_full = cv2.resize(region_s, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    reg = cv2.warpAffine(reg_full, R, (CW, CH), flags=cv2.INTER_NEAREST, borderValue=0)
    return crop, reg, L / max(Hh, 1e-6), float(np.degrees(ang)), xh * sc, float(mid[0]), float(mid[1])


def binarize(crop, reg):
    """로고 영역(region) 안에서만 어두운 부분을 마스크로 만든다. 모서리 기둥 등은 영역 밖이라 섞이지 않는다."""
    c = cv2.GaussianBlur(crop, (0, 0), 1.2)
    vals = c[reg > 0]
    if vals.size < 50:
        return np.zeros_like(reg)
    t, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((c < t) & (reg > 0)).astype(np.uint8)


def shift(m, dx, dy):
    return cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (CW, CH), flags=cv2.INTER_NEAREST)


def mismatch(M, Rm, dtR=None, search=4):
    """기준 모양 Rm과 마스크 M의 어긋남 점수(작을수록 정상). M을 +-search px 옮겨 가장 좋은 값을 쓴다."""
    dtR = dtR if dtR is not None else cv2.distanceTransform((1 - Rm).astype(np.uint8), cv2.DIST_L2, 3)
    best = 1e9
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            Ms = shift(M, dx, dy)
            dtM = cv2.distanceTransform((1 - Ms).astype(np.uint8), cv2.DIST_L2, 3)
            e1 = np.maximum(0, dtR[Ms > 0] - TOL).sum()        # M에만 있는 부분(번짐·이물)
            e2 = np.maximum(0, dtM[Rm > 0] - TOL).sum()        # 기준에만 있는 부분(빠진 글자·획)
            best = min(best, (e1 + e2) / max(Rm.sum(), 1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default="out_logo")
    ap.add_argument("--per_model", type=int, default=150)
    ap.add_argument("--margin", type=float, default=1.5, help="기준선(검증 정상 99퍼센타일) 배수")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items, ng = collect(a.root, a.step)
    rng = random.Random(0)
    by = {}
    for it in items:
        by.setdefault(it["model"], []).append(it)
    items = []
    for m, l in by.items():
        rng.shuffle(l); items += l[:a.per_model]
    print(f"정상 사진 {len(items)}장, NG 사진 {len(ng)}장 (step {a.step})")

    def process(it):
        g = imread_gray(it["path"])
        if g is None:
            return None, "읽기 실패", None
        gs = cv2.resize(g, (SW, int(g.shape[0] * SW / g.shape[1])), interpolation=cv2.INTER_AREA)
        ms, why = find_logo(gs)
        if ms is None:
            return None, why, None
        crop, reg, asp, ang, xh, cx, cy = canon(g, ms)
        ys_, xs_ = np.nonzero(ms[0])
        Hs, Ws = ms[0].shape
        edge = xs_.min() < 6 or ys_.min() < 6 or xs_.max() > Ws - 7 or ys_.max() > Hs - 7   # 로고가 프레임 가장자리에 걸림
        return crop, "", dict(asp=asp, ang=ang, reg=reg, edge=bool(edge), xh=xh, cx=cx, cy=cy)

    res = []
    for it in items:
        crop, why, info = process(it)
        res.append((it, crop, why, info))
    found_all = [r for r in res if r[1] is not None]
    print(f"로고 찾기 성공 {len(found_all)}/{len(res)}")
    found = found_all
    if len(found) < 20:
        sys.exit("로고를 찾은 사진이 너무 적습니다.")
    asp = np.array([r[3]["asp"] for r in found])
    med = float(np.median(asp)); mad = float(np.median(np.abs(asp - med)))
    tol = max(5 * mad, 0.10 * med)
    lo_a, hi_a = med - tol, med + tol
    print(f"로고 종횡비(길이/높이) 중앙값 {np.median(asp):.2f}, 허용 {lo_a:.2f}~{hi_a:.2f}")

    gate_bad = [r for r in found_all if not (lo_a <= r[3]["asp"] <= hi_a)]
    found = [r for r in found_all if lo_a <= r[3]["asp"] <= hi_a]
    # 로고의 위치·크기 정상 범위(프레임 기준). 표준 crop은 위치 정보를 지우므로 따로 확인한다.
    rngs = {}
    for k_, nm in (("cx", "가로 위치"), ("cy", "세로 위치"), ("xh", "글자 높이")):
        v_ = np.array([r[3][k_] for r in found])
        a1, a99 = np.percentile(v_, [1, 99])
        m_ = 0.25 * (a99 - a1)
        rngs[k_] = (a1 - m_, a99 + m_)
        print(f"로고 {nm}: 정상 1~99퍼센타일 {a1:.0f}~{a99:.0f}, 허용 {a1 - m_:.0f}~{a99 + m_:.0f}")

    def pos_bad(info):
        return [nm for k_, nm in (("cx", "가로위치"), ("cy", "세로위치"), ("xh", "글자크기")) if not (rngs[k_][0] <= info[k_] <= rngs[k_][1])]
    n_pos = sum(1 for r in found_all if pos_bad(r[3]))
    print(f"위치·크기가 정상 범위를 벗어난 정상 사진 {n_pos}/{len(found_all)}장")        # 학습·기준선에는 비율이 정상인 사진만
    # 방향 통일(180도 뒤집힘 대비): 중앙값 crop과 더 닮은 쪽으로
    crops = [r[1] for r in found]
    regs = [r[3]["reg"] for r in found]
    ref = np.median(np.stack(crops), axis=0).astype(np.float32)
    for _ in range(2):
        newc, newr = [], []
        for c, rg in zip(crops, regs):
            cr = c[::-1, ::-1]
            f = lambda x: np.corrcoef(x.ravel().astype(np.float32), ref.ravel())[0, 1]
            if f(c) >= f(cr):
                newc.append(c); newr.append(rg)
            else:
                newc.append(cr); newr.append(rg[::-1, ::-1])
        crops, regs = newc, newr
        ref = np.median(np.stack(crops), axis=0).astype(np.float32)
    flipped = sum(1 for c0, c1 in zip([r[1] for r in found], crops) if c0 is not c1)
    print(f"방향이 뒤집혀 있어 돌린 사진 {flipped}장")

    # 제품(시리얼) 단위 3겹 교차검증: 모든 정상 사진을 '자기 제품이 빠진' 기준 마스크로 채점한다(자기 자신으로 만든 기준과 비교하지 않음)
    NF = 3
    from step_dots import content_fold
    fold_img = [content_fold(r[0]["path"], NF) for r in found]            # 사진 내용 해시로 칸 결정(이름 미사용)
    split = {i: f"교차검증{fold_img[i] + 1}" for i in range(len(found))}
    fold_idx = [[i for i in range(len(found)) if fold_img[i] == f] for f in range(NF)]
    print(f"3겹 교차검증: " + ", ".join(f"{len(x)}장" for x in fold_idx))
    masks = [binarize(c, rg) for c, rg in zip(crops, regs)]
    score = {}
    for f in range(NF):
        tr_i = [i for g_ in range(NF) if g_ != f for i in fold_idx[g_]]
        Rmean = np.mean([masks[i] for i in tr_i], axis=0)
        Rm = (Rmean > 0.5).astype(np.uint8)
        if f == 0:
            cv2.imencode(".png", (255 - 255 * Rmean).astype(np.uint8))[1].tofile(str(out / "ref_mask.png"))
        dtR = cv2.distanceTransform((1 - Rm).astype(np.uint8), cv2.DIST_L2, 3)
        for i in fold_idx[f]:
            score[i] = mismatch(masks[i], Rm, dtR)
    sv = np.array([score[i] for i in sorted(score)]); stt = sv
    thr = float(np.percentile(sv, 99)) * a.margin
    print(f"\n점수(어긋남) 정상 중앙값 {np.median(sv):.3f}, 99퍼센타일 x {a.margin} (기준선) {thr:.3f}")
    print(f"정상 오탐률(기준선 초과, 교차검증): {np.mean(stt > thr):.1%} ({int((stt > thr).sum())}/{len(stt)})")
    g_edge = [r for r in gate_bad if r[3]["edge"]]; g_shape = [r for r in gate_bad if not r[3]["edge"]]
    print(f"로고 못 찾음 {len(res) - len(found_all)}장 / 프레임에 걸려 비율 이상 {len(g_edge)}장(판정 불가, 재촬영) / 프레임 안인데 비율 이상 {len(g_shape)}장(형태 이상, NG 의심)")
    ng_Rm = (np.mean([masks[i] for i in range(len(masks))], axis=0) > 0.5).astype(np.uint8)   # NG 폴더 사진용: 정상 전체 기준 마스크
    Rm = ng_Rm; dtR = cv2.distanceTransform((1 - Rm).astype(np.uint8), cv2.DIST_L2, 3)

    def match_rate(s):
        return 100.0 * 0.5 ** (s / thr)

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "기종", "분할", "점수", "일치율", "종횡비"])
        for i, r in enumerate(found):
            sp = split[i]
            s = score.get(i)
            w.writerow([str(r[0]["path"]), r[0]["model"], sp, "" if s is None else f"{s:.3f}", "" if s is None else f"{match_rate(s):.0f}", f"{r[3]['asp']:.2f}"])
    with open(out / "gate_fail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "기종", "사유"])
        for r in found_all:
            if r not in gate_bad and pos_bad(r[3]):
                w.writerow([str(r[0]["path"]), r[0]["model"], "위치·크기 이상: " + ",".join(pos_bad(r[3]))])
        for r in res:
            if r[1] is None:
                w.writerow([str(r[0]["path"]), r[0]["model"], r[2]])
        for r in gate_bad:
            w.writerow([str(r[0]["path"]), r[0]["model"], f"종횡비 {r[3]['asp']:.2f} / " + ("프레임에 걸림" if r[3]["edge"] else "프레임 안: 형태 이상")])
    # 걸린 정상 사진 모아보기 (로고 못 찾음 / 형태 이상 / 위치·크기 이상): 이 그림을 보내주세요
    def tile_of(it, tag):
        g0 = imread_gray(it["path"])
        if g0 is None:
            return None
        t = cv2.cvtColor(cv2.resize(g0, (256, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        label = f"{tag} {it['model'][-6:]} {it['serial']}"
        cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2)
        cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        return t
    samp = [(r[0], "없음") for r in res if r[1] is None][:8] + [(r[0], "형태") for r in gate_bad][:16] \
        + [(r[0], "위치") for r in found_all if pos_bad(r[3]) and r not in gate_bad][:8]
    tl = [t for t in (tile_of(i, tg) for i, tg in samp) if t is not None]
    if tl:
        while len(tl) % 4:
            tl.append(np.zeros_like(tl[0]))
        cv2.imencode(".png", np.vstack([np.hstack(tl[i:i + 4]) for i in range(0, len(tl), 4)]))[1].tofile(str(out / "gate_samples.png"))
    # 표준 crop 모음
    pick = rng.sample(range(len(crops)), min(40, len(crops)))
    tiles = [cv2.resize(crops[i], (CW // 2, CH // 2)) for i in pick]
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imencode(".png", np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))[1].tofile(str(out / "crops_sample.png"))

    # NG 시험
    print(f"\n[NG 시험] step {a.step} NG 사진 {len(ng)}장 (기준선 {thr:.3f}, 일치율 50 = 기준선)")
    vis = []
    for it in ng:
        crop, why, info = process(it)
        if crop is None:
            print(f"  {it['model']} {it['serial']}  로고를 못 찾음({why}) -> NG 가능(로고 없음/가려짐)  {it['path'].name}")
            continue
        c0 = crop
        cr = c0[::-1, ::-1]
        f = lambda x: np.corrcoef(x.ravel().astype(np.float32), ref.ravel())[0, 1]
        if f(c0) >= f(cr):
            crop, rg = c0, info["reg"]
        else:
            crop, rg = cr, info["reg"][::-1, ::-1]
        M = binarize(crop, rg)
        s = mismatch(M, Rm, dtR)
        bad_asp = not (lo_a <= info["asp"] <= hi_a)
        pb = pos_bad(info)
        flag = "NG(검출)" if (s > thr or pb or (bad_asp and not info["edge"])) else ("판정 불가(프레임에 걸림)" if bad_asp else "OK(통과)")
        gate = ("" if not bad_asp else ("  [종횡비 이상, 프레임에 걸림]" if info["edge"] else "  [종횡비 이상, 프레임 안 = 형태 이상]")) + (f"  [위치·크기 이상: {','.join(pb)}]" if pb else "")
        print(f"  {it['model']} {it['serial']}  점수 {s:.3f}  일치율 {match_rate(s):.0f}  -> {flag}{gate}  {it['path'].name}")
        rgb = np.zeros((CH, CW, 3), np.uint8)
        rgb[..., 1] = 255 * Rm; rgb[..., 2] = 255 * M          # 초록=기준, 빨강=이 사진, 노랑=겹침
        vis.append(np.vstack([cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), rgb]))
    if vis:
        cv2.imencode(".png", np.hstack(vis))[1].tofile(str(out / "ng_overlay.png"))
    print(f"\n결과는 {out.resolve()} 에 저장됨")


if __name__ == "__main__":
    main()
