"""스텝 6, 8~15 부품 단위 판정: 정상 사진에서 스스로 만든 '부품 탐지기'로 부품이 있는지 본다.
왜 이 방식인가: 제품이 놓인 위치·각도가 사진마다 달라서(카메라는 지그에 고정) 한 곳을 고정해 자르면 안 되고,
장면 전체 비교(step_generic)나 조각 비교(step_patch)는 작은 부품 하나의 유무를 정상 편차와 구분하지 못했다.
방법(부품 탐지기를 정상 사진으로 학습, 라벨을 사람이 일일이 달지 않음):
 1) 스텝마다 대표 사진 1장에서 부품 상자 1개만 지정한다(PARTS 설정). 그 상자 모양을 기준 삼아 다른 정상 사진에서
    가장 비슷한 위치를 찾아(마스킹 없는 정규화 상관, 배율 0.9~1.1) 확실하게 찾아진 사진들만 '양성 후보'로 쓴다.
    (이 단계의 템플릿 매칭은 학습용 양성 모으기에만 쓰고 최종 판정에는 쓰지 않는다.)
 2) 양성 조각과 같은 사진의 다른 위치 조각(음성)으로 방향 히스토그램(HOG) + 로지스틱 회귀 탐지기를 학습한다.
    한 번 돌려 가장 헷갈린 음성을 추가해 다시 학습한다.
 3) 사진마다 모든 위치를 훑어 가장 높은 점수(d)를 사진 점수로 한다. 제품 단위로 정상을 두 쪽으로 나눠
    한쪽으로 학습한 탐지기로 다른 쪽을 채점한다(자기 자신으로 학습한 탐지기로 채점하지 않음).
 4) 임계값 d0 = 정상 점수의 1 백분위 - margin x (중앙값 - 1 백분위). 일치율 = 100/(1+exp(-(d-d0)/sc)) (임계에서 50, 중앙값에서 약 95).
 5) 가장 높은 점수의 위치를 '위치'로 저장한다. 이상 판정 사진은 상자를 그린 그림으로 저장한다.
NG 폴더 사진도 같은 기준으로 판정해 기록한다(정답으로 쓰지 않음).
사용: py step_part.py --root "<root>" --steps 9,10,11 --out out_part
출력: part_result.csv (스텝,구분번호,기종,제품,출처,일치율,결과,위치,표기), boxes/*.png
"""
import argparse, csv, zlib
from pathlib import Path
import numpy as np, cv2
from sklearn.linear_model import LogisticRegression
from step_dots import collect, find_ng

# 대표 사진(기종/제품)과 그 사진(1024x768)의 부품 상자 (x, y, w, h). 구분번호는 PPT 기준.
PARTS = {
    6:  dict(part="2917",           ref="3030C001AA/2FA63294", box=(330, 560, 170, 100)),   # 흰색 부품(파란 상자 아래 중앙). 어두운 사진이 많아 불안정할 수 있음
    8:  dict(part="8588/0000",      ref="3030C001AA/2FA63294", box=(250, 288, 462, 266)),   # 덮개판(가운데 홈 있는 둥근 사각형)
    9:  dict(part="3780",           ref="3030C003AA/2FC13350", box=(335, 322, 230, 125)),   # 전원 스위치
    10: dict(part="V243/V242/B792", ref="3030C003AA/2FC13350", box=(510, 160, 130, 100)),   # USB 포트
    11: dict(part="V243/V242/B792", ref="3030C001AA/2FA63294", box=(515, 650, 125, 112)),   # USB 포트(아래쪽)
    12: dict(part="3779/V242",      ref="3030C001AA/2FA63294", box=(250, 290, 100, 120)),   # U자 홈(고리)
    13: dict(part="3779/C160",      ref="3030C001AA/2FA63294", box=(385, 285, 140, 100)),   # 스프링·기어 부분
    14: dict(part="3779",           ref="3030C001AA/2FA63294", box=(610, 335, 130, 100)),   # U자 홈
}
W, H = 256, 192
HOG = None


def gray_small(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def cellhist(g):
    """4x4 픽셀 셀마다 9방향 기울기 히스토그램 -> (48,64,9)"""
    g = g.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); b = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * 9).astype(int), 8)
    cell = (np.arange(H)[:, None] // 4) * (W // 4) + (np.arange(W)[None, :] // 4)
    return np.bincount((cell * 9 + b).ravel(), weights=mag.ravel(), minlength=(H // 4) * (W // 4) * 9).reshape(H // 4, W // 4, 9)


def hog_for(win):
    return win


def windows(g, win, stride=4):
    """모든 창 위치의 특징: (ny, nx, d). 창 = (가로, 세로) 픽셀, 4px 셀, 2x2 셀씩 합쳐 8px 블록 -> L2 정규화. 위치 간격 = 1셀(4px)"""
    ch = cellhist(g); wc, hc = win[0] // 4, win[1] // 4
    v = np.lib.stride_tricks.sliding_window_view(ch, (hc, wc, 9))[:, :, 0]                  # (ny,nx,hc,wc,9)
    ny, nx = v.shape[:2]
    v = v[:, :, :hc // 2 * 2, :wc // 2 * 2].reshape(ny, nx, hc // 2, 2, wc // 2, 2, 9).sum((3, 5)).reshape(ny, nx, -1)
    return (v / (np.linalg.norm(v, axis=2, keepdims=True) + 50.0)).astype(np.float32), nx, ny


def best_match(g, tpl):
    best = (-1, 0, 0)
    for s in (0.9, 1.0, 1.1):
        t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= g.shape[0] or t.shape[1] >= g.shape[1]: continue
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]: best = (mx, loc[0] + t.shape[1] // 2, loc[1] + t.shape[0] // 2)
    return best


def train(Fs, centers, win, stride, rng, neg_per=40, hard=True):
    """Fs: [(ny,nx,d)], centers: [(cx,cy)] 양성 위치(창 중심, 256x192 좌표)"""
    X, y = [], []
    for F, (cx, cy) in zip(Fs, centers):
        ny, nx, _ = F.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ix = int(round((cx - win[0] / 2) / stride)) + dx; iy = int(round((cy - win[1] / 2) / stride)) + dy
                ix = min(max(ix, 0), nx - 1); iy = min(max(iy, 0), ny - 1)             # 화면 가장자리의 부품은 창이 잘리므로 가장 가까운 창으로
                X.append(F[iy, ix]); y.append(1)
        far = [(iy, ix) for iy in range(ny) for ix in range(nx)
               if abs(ix * stride + win[0] / 2 - cx) > 0.8 * win[0] or abs(iy * stride + win[1] / 2 - cy) > 0.8 * win[1]]
        for k in rng.choice(len(far), min(neg_per, len(far)), replace=False):
            X.append(F[far[k]]); y.append(0)
    if len(set(y)) < 2:
        return None
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(np.array(X), np.array(y))
    if hard:                                                                                 # 가장 헷갈린 음성 추가
        X2, y2 = [], []
        for F, (cx, cy) in zip(Fs, centers):
            ny, nx, d = F.shape; s = clf.decision_function(F.reshape(-1, d)).reshape(ny, nx)
            for iy in range(ny):
                for ix in range(nx):
                    if abs(ix * stride + win[0] / 2 - cx) > 0.8 * win[0] or abs(iy * stride + win[1] / 2 - cy) > 0.8 * win[1]:
                        pass
                    else:
                        s[iy, ix] = -1e9
            k = np.argsort(s.ravel())[-5:]
            for q in k: X2.append(F.reshape(-1, d)[q]); y2.append(0)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(np.vstack([X, X2]), np.concatenate([y, y2]))
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--steps", default=",".join(map(str, PARTS)))
    ap.add_argument("--out", default="out_part"); ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--margin", type=float, default=0.5); ap.add_argument("--ncc", type=float, default=0.55, help="양성 후보로 쓸 최소 상관")
    ap.add_argument("--max_train", type=int, default=250, help="한쪽 학습에 쓸 최대 사진 수")
    a = ap.parse_args(); out = Path(a.out); (out / "boxes").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0); rows = []
    for st in [int(s) for s in a.steps.split(",")]:
        cfg = PARTS.get(st)
        if cfg is None: print(f"스텝 {st}: 설정 없음"); continue
        items = collect(a.root, st, per_model=10 ** 6); items = items[0] if isinstance(items, tuple) else items
        if len(items) < 10: print(f"스텝 {st}: 사진 {len(items)}장 (건너뜀)"); continue
        ref = next(Path(a.root).glob(f"{cfg['ref']}/step_{st}.*"), None)
        if ref is None: print(f"스텝 {st}: 대표 사진 {cfg['ref']} 없음"); continue
        rg = cv2.imdecode(np.fromfile(str(ref), np.uint8), cv2.IMREAD_GRAYSCALE)
        x, y, w, h = cfg["box"]; tpl = cv2.resize(rg[y:y + h, x:x + w], None, fx=W / 1024, fy=H / 768, interpolation=cv2.INTER_AREA)
        win = (int(round(tpl.shape[1] / 8) * 8 + 16), int(round(tpl.shape[0] / 8) * 8 + 16)); hog = None; stride = 4
        print(f"스텝 {st}: 사진 {len(items)}장 읽는 중 (창 {win[0]}x{win[1]})", flush=True)
        G = [gray_small(i["path"]) for i in items]; ok = [k for k, g in enumerate(G) if g is not None]
        items = [items[k] for k in ok]; G = [G[k] for k in ok]
        FF = [windows(g, win, stride)[0] for g in G]
        cnt = {}
        for it in items: cnt[it["model"]] = cnt.get(it["model"], 0) + 1
        grp = lambda m: m if cnt[m] >= a.min_group else "소수기종묶음"
        Gm = np.array([grp(it["model"]) for it in items]); fold = np.array([zlib.crc32(Path(it["path"]).read_bytes()) % 2 for it in items])
        mt = [best_match(g, tpl) for g in G]                                                  # 양성 후보 찾기 (학습용)
        used_common = set(); _cc = {}
        def common(f):                                                                        # 같은 쪽(fold) 전체 사진으로 학습, 교차 적합 유지
            if f not in _cc:
                cand = [k for k in range(len(items)) if fold[k] == f and mt[k][0] >= a.ncc]
                if len(cand) > a.max_train: cand = list(rng.choice(cand, a.max_train, replace=False))
                _cc[f] = train([FF[k] for k in cand], [(mt[k][1], mt[k][2]) for k in cand], win, stride, rng) if len(cand) >= 4 else None
            return _cc[f]
        score = np.full(len(items), np.nan); pos = np.zeros((len(items), 2), int); thr = {}; sc = {}; nused = {}
        for g in set(Gm):
            idx = np.where(Gm == g)[0]
            if len(idx) < 8: thr[g] = None; continue
            clfs = []
            for f in (0, 1):
                cand = [k for k in idx if fold[k] == f and mt[k][0] >= a.ncc]
                if len(cand) > a.max_train: cand = list(rng.choice(cand, a.max_train, replace=False))
                c = train([FF[k] for k in cand], [(mt[k][1], mt[k][2]) for k in cand], win, stride, rng) if len(cand) >= 4 else None
                if c is None: c = common(f); used_common.add(g)                                   # 기종 사진이 모자라면 전체 기종 공통 탐지기
                clfs.append(c); nused[(g, f)] = len(cand)
            if g in used_common: print(f"   [알림] {g}: 기종 단독 학습 불가(양성 후보 부족) -> 전체 기종 공통 탐지기 사용")
            for k in idx:
                clf = clfs[1 - fold[k]]
                if clf is None: continue
                F = FF[k]; ny, nx, d = F.shape; s = clf.decision_function(F.reshape(-1, d)); j = int(s.argmax())
                score[k] = s[j]; pos[k] = (j % nx * stride + win[0] // 2, j // nx * stride + win[1] // 2)
            v = score[idx][~np.isnan(score[idx])]
            if len(v) < 8: thr[g] = None; continue
            p1, med = np.percentile(v, 1), np.median(v)
            thr[g] = p1 - a.margin * (med - p1); sc[g] = max((med - thr[g]) / 2.94, 1e-3)
            clfs_all = clfs
            thr[(g, "clf")] = clfs
        def row(it, s, p, g, src):
            t = thr.get(g)
            if t is None or np.isnan(s): return (st, cfg["part"], it["model"], it["serial"], src, 0, "재확인", "")
            m = round(100 / (1 + np.exp(-(s - t) / sc[g]))); return (st, cfg["part"], it["model"], it["serial"], src, m, "OK" if s >= t else "NG", f"({p[0]},{p[1]})")
        for k, it in enumerate(items): rows.append(row(it, score[k], pos[k], Gm[k], "정상"))
        bad = [r for r in rows if r[0] == st and r[4] == "정상" and r[6] != "OK"]
        print(f"스텝 {st}: 정상 {len(items)}장 중 이상 {len(bad)}장 ({100 * len(bad) / len(items):.2f}%)  (양성 후보 상관 {a.ncc} 이상 {sum(m[0] >= a.ncc for m in mt)}장)")
        def draw(it, g, p, name):
            im = cv2.resize(cv2.imdecode(np.fromfile(str(it["path"]), np.uint8), cv2.IMREAD_COLOR), (W * 2, H * 2))
            cv2.rectangle(im, (2 * (p[0] - win[0] // 2), 2 * (p[1] - win[1] // 2)), (2 * (p[0] + win[0] // 2), 2 * (p[1] + win[1] // 2)), (0, 255, 0), 2)
            cv2.putText(im, name, (4, 16), 0, 0.5, (0, 255, 255), 1)
            cv2.imencode(".png", im)[1].tofile(str(out / "boxes" / f"s{st}_{it['serial']}_{name.split()[0]}.png"))
        worst = sorted([k for k in range(len(items)) if not np.isnan(score[k])], key=lambda k: score[k])[:10]
        for k in worst: draw(items[k], Gm[k], pos[k], f"정상 {items[k]['serial']} d={score[k]:.2f}")
        for it in find_ng(a.root, st):
            g = grp(it["model"]) if it["model"] in cnt else "소수기종묶음"; clfs = thr.get((g, "clf"))
            if thr.get(g) is None or not clfs: continue
            gi = gray_small(it["path"])
            if gi is None: continue
            F = windows(gi, win, stride)[0]; ny, nx, d = F.shape
            sv = [c.decision_function(F.reshape(-1, d)) for c in clfs if c is not None]; s_ = np.mean(sv, 0); j = int(s_.argmax())
            p = (j % nx * stride + win[0] // 2, j // nx * stride + win[1] // 2); r = row(it, s_[j], p, g, "NG폴더"); rows.append(r)
            print(f"   NG폴더 {it['model']}/{it['serial']}/{Path(str(it['path'])).name[:23]}: d={s_[j]:.2f} (임계 {thr[g]:.2f}) 일치율 {r[5]} → {r[6]}")
            draw(it, g, p, f"NG {it['serial']} d={s_[j]:.2f}")
        save(out, rows)
    save(out, rows)


def save(out, rows):
    with open(out / "part_result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w_ = csv.writer(fh); w_.writerow(["스텝", "구분번호", "기종", "제품", "출처", "일치율", "결과", "위치", "표기"])
        for r in rows: w_.writerow([*r, f"{r[1]}:{r[5]}:{r[6]}"])
    print("저장:", out / "part_result.csv", flush=True)


if __name__ == "__main__":
    main()
