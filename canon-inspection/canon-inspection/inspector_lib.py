"""판독기 라이브러리: 스텝별 '학습(fit_*)'과 '사진 한 장 판정(pred_*)'을 분리해 모델을 파일로 저장·불러온다.

기존 step_*.py 는 실행할 때마다 전체 사진을 읽어 기준을 만들고 같은 사진을 채점하는 일괄 평가 스크립트였다.
여기서는 같은 방법(점 아이콘 기준 라벨 위치 -> 중심 비교, 로고 마스크 비교, 라벨 분류기, 파란 부품 면적, 장면 거리, 부품 탐지기)을
그대로 쓰되 학습 결과(기준 모양, 분류기, 임계값)를 모델 파일에 담아 새 사진 한 장을 판정한다.

입력 정보: 기종(폴더 이름과 같은 값), 스텝 번호, 사진. 파일·제품 이름은 판정·학습에 쓰지 않는다
(학습/시험 분할도 사진 내용의 crc32 해시로 한다).
"""
import re, zlib, pickle
from pathlib import Path
import numpy as np, cv2

import step_dots as sd
import step_logo as sl
import step_name as sn
import name_locate as nl
import step_scene
import step_part as sp
from step_generic import PART as PART_ID

STEPS = list(range(1, 16))
PART_STEPS = [6, 8, 9, 10, 11, 12, 13, 14]       # 부품 탐지기를 쓰는 스텝
BLUE_SMIN = {4: 30, 5: 30, 7: 80}                  # 파란 부품 검사: 채도 하한
HOLDOUT_MOD = 5                                    # 제품의 약 1/5 를 시험용으로 떼어 둔다
_STEP_RE = re.compile(r"^step[_\- ]?0*(\d+)\.(jpg|jpeg|png|bmp)$", re.I)


# ---------------------------------------------------------------- 데이터 열거
class Files(dict):
    """{스텝: 사진 경로} + 제품 단위 분할 값(fold). fold 는 스텝 0~19 를 포함한 '모든' step 사진 중 가장 앞 번호 사진의 내용 crc32 % 5"""
    fold = -1


def list_products(root):
    """[(기종, 제품 폴더 Path, Files{스텝: 사진 Path})] — NG 폴더 제외, 판정 스텝은 1~15 만"""
    out = []
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir() or mdir.name not in sd.ALL_MODELS:
            continue
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if sdir.name.upper() == "NG":
                continue
            allf = {}
            for p in sdir.iterdir():
                m = _STEP_RE.match(p.name) if p.is_file() else None
                if m:
                    allf[int(m.group(1))] = p
            if allf:
                files = Files({k: v for k, v in allf.items() if k in STEPS})
                files.fold = zlib.crc32(Path(allf[min(allf)]).read_bytes()) % HOLDOUT_MOD
                if files:
                    out.append((mdir.name, sdir, files))
    return out


def product_fold(files):
    return files.fold


def is_holdout(files):
    return files.fold == 0


def ng_photos(root):
    """NG 폴더 사진 [(기종, 제품, 스텝, 경로)] — 평가 전용(학습에 쓰지 않음)"""
    out = []
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir() or mdir.name not in sd.ALL_MODELS:
            continue
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            ngd = sdir / "NG"
            if not ngd.is_dir():
                continue
            for f in sorted(ngd.iterdir()):
                m = re.search(r"step[_\- ]?0*(\d+)\.(jpg|jpeg|png|bmp)$", f.name, re.I)
                if f.is_file() and m and int(m.group(1)) in STEPS:
                    out.append((mdir.name, sdir.name, int(m.group(1)), f))
    return out


def expected_label(step, model):
    if step == 1:
        return sd.variant_of(model)
    if step == 2:
        return "B793"
    if step == 3:
        return sn.LABEL_OF[model]
    if step == 7:
        return "1525"
    return PART_ID.get(step, "?")


def _res(step, model, rate, res, why="", pos=None):
    return dict(step=step, id=expected_label(step, model), rate=float(max(0.0, min(100.0, rate))), res=res, why=why, pos=pos)


# ---------------------------------------------------------------- 스텝 1: 토너 지역 라벨(점 아이콘)
def fit_dots(items, **kw):
    crops, var = [], []
    for it in items:
        g = sd.imread_gray(it["path"])
        r = sd.locate(g) if g is not None else None
        if r is not None:
            crops.append(r["crop"]); var.append(it["variant"])
    if len(crops) < 10:
        raise ValueError(f"위치를 찾은 사진 {len(crops)}장 (너무 적음)")
    F = np.stack([sd.feat(c) for c in crops]); var = np.array(var)
    cen = {v: F[var == v].mean(0) for v in ("C550", "C551") if (var == v).any()}
    return dict(cen=cen, n=len(crops), n_items=len(items), search=4)


def pred_dots(M, model, path):
    exp = sd.variant_of(model)
    g = sd.imread_gray(path)
    r = sd.locate(g) if g is not None else None
    if r is None:
        return _res(1, model, 0, "재확인", "라벨(점 아이콘)을 못 찾음")
    cen, S = M["cen"], M["search"]
    SO = [(sx, sy) for sy in range(-S, S + 1, 2) for sx in range(-S, S + 1, 2)]
    U = np.stack([sd.feat(r["crop"], sx, sy) for sx, sy in SO])
    sc = {v: float((U @ cen[v]).max() / (np.linalg.norm(cen[v]) + 1e-6)) for v in cen}
    pred = max(sc, key=sc.get)
    rate = max(0.0, min(100.0, sc.get(exp, 0.0) * 100))
    if pred == exp:
        return _res(1, model, rate, "OK")
    return _res(1, model, min(rate, 49.0), "NG", f"라벨 종류가 {pred} 로 보임(기대 {exp})")


# ---------------------------------------------------------------- 스텝 2: 좌전카바 로고
def _logo_process(g):
    gs = cv2.resize(g, (sl.SW, int(g.shape[0] * sl.SW / g.shape[1])), interpolation=cv2.INTER_AREA)
    ms, why = sl.find_logo(gs)
    if ms is None:
        return None, why, None
    crop, reg, asp, ang, xh, cx, cy = sl.canon(g, ms)
    ys_, xs_ = np.nonzero(ms[0]); Hs, Ws = ms[0].shape
    edge = xs_.min() < 6 or ys_.min() < 6 or xs_.max() > Ws - 7 or ys_.max() > Hs - 7
    return crop, "", dict(asp=asp, ang=ang, reg=reg, edge=bool(edge), xh=xh, cx=cx, cy=cy)


def fit_logo(items, margin=1.5, **kw):
    res = []
    for it in items:
        g = sl.imread_gray(it["path"])
        crop, why, info = _logo_process(g) if g is not None else (None, "읽기 실패", None)
        if crop is not None:
            res.append((it, crop, info))
    if len(res) < 20:
        raise ValueError(f"로고를 찾은 사진 {len(res)}장 (너무 적음)")
    def widen(v, margin=0.5):
        """학습 사진 분포의 0.5~99.5 퍼센타일 범위를 범위 폭의 margin 배만큼 넓힌 허용 구간 (학습 사진만으로 정함)"""
        p0, p1 = np.percentile(v, [0.5, 99.5]); w = p1 - p0
        return float(p0 - margin * w), float(p1 + margin * w)
    lo_a, hi_a = widen(np.array([r[2]["asp"] for r in res]))
    rngs = {k_: widen(np.array([r[2][k_] for r in res])) for k_ in ("cx", "cy", "xh")}
    inside = lambda i: lo_a <= i["asp"] <= hi_a and all(rngs[k_][0] <= i[k_] <= rngs[k_][1] for k_ in rngs)
    found = [r for r in res if inside(r[2])]
    gate_rate = 1 - len(found) / len(res)
    crops = [r[1] for r in found]; regs = [r[2]["reg"] for r in found]
    ref = np.median(np.stack(crops), axis=0).astype(np.float32)
    f = lambda x, ref_: np.corrcoef(x.ravel().astype(np.float32), ref_.ravel())[0, 1]
    for _ in range(2):
        nc, nr = [], []
        for c, rg in zip(crops, regs):
            cr = c[::-1, ::-1]
            if f(c, ref) >= f(cr, ref):
                nc.append(c); nr.append(rg)
            else:
                nc.append(cr); nr.append(rg[::-1, ::-1])
        crops, regs = nc, nr
        ref = np.median(np.stack(crops), axis=0).astype(np.float32)
    masks = [sl.binarize(c, rg) for c, rg in zip(crops, regs)]
    # 교정용 사진(약 20%, 내용 해시로 선택): 기준 모양을 만들 때 빼고, 이 사진들의 어긋남 점수로 기준선을 정한다(처음 보는 사진과 같은 조건)
    is_cal = [zlib.crc32(Path(r[0]["path"]).read_bytes()) % 5 == 1 for r in found]
    fit_i = [i for i in range(len(found)) if not is_cal[i]]; cal_i = [i for i in range(len(found)) if is_cal[i]]
    if len(cal_i) < 10:                                           # 사진이 너무 적으면 전체로 대신한다
        fit_i = cal_i = list(range(len(found)))
    Rm = (np.mean([masks[i] for i in fit_i], axis=0) > 0.5).astype(np.uint8)
    dtR = cv2.distanceTransform((1 - Rm).astype(np.uint8), cv2.DIST_L2, 3)
    sv = np.array([sl.mismatch(masks[i], Rm, dtR) for i in cal_i])
    thr = float(np.percentile(sv, 99)) * margin
    Rm_all = (np.mean(masks, axis=0) > 0.5).astype(np.uint8)
    return dict(ref=ref, Rm=Rm_all, lo=float(lo_a), hi=float(hi_a), rngs=rngs, thr=thr, n=len(found), n_items=len(items),
                cv_fp=float(np.mean(sv > thr)), gate_rate=float(gate_rate))


def pred_logo(M, model, path):
    g = sl.imread_gray(path)
    if g is None:
        return _res(2, model, 0, "재확인", "사진을 읽지 못함")
    crop, why, info = _logo_process(g)
    if crop is None:
        return _res(2, model, 10, "NG", "로고를 못 찾음(로고 없음 의심)")
    bad_asp = not (M["lo"] <= info["asp"] <= M["hi"])
    pb = [nm for k_, nm in (("cx", "가로위치"), ("cy", "세로위치"), ("xh", "글자크기")) if not (M["rngs"][k_][0] <= info[k_] <= M["rngs"][k_][1])]
    if bad_asp and info["edge"]:
        return _res(2, model, 0, "재확인", "로고가 프레임 가장자리에 걸림(재촬영)")
    ref = M["ref"]
    f = lambda x: np.corrcoef(x.ravel().astype(np.float32), ref.ravel())[0, 1]
    rg = info["reg"]
    if f(crop) < f(crop[::-1, ::-1]):
        crop, rg = crop[::-1, ::-1], rg[::-1, ::-1]
    Mk = sl.binarize(crop, rg)
    Rm = M["Rm"]; dtR = cv2.distanceTransform((1 - Rm).astype(np.uint8), cv2.DIST_L2, 3)
    s = sl.mismatch(Mk, Rm, dtR)
    rate = 100.0 * 0.5 ** (s / M["thr"])
    why = []
    if s > M["thr"]:
        why.append(f"로고 모양이 기준과 다름(점수 {s:.2f} > {M['thr']:.2f})")
    if pb:
        why.append("위치·크기 이상: " + ",".join(pb))
    if bad_asp:
        why.append("종횡비 이상(형태 이상)")
    if why:
        return _res(2, model, min(rate, 49.0), "NG", "; ".join(why))
    return _res(2, model, rate, "OK")


# ---------------------------------------------------------------- 스텝 3: 제품명 라벨
def _name_crop(path):
    g = sd.imread_gray(path)
    if g is None:
        return None, "no_image"
    return nl.locate(g)


def fit_name(items, aug=4, margin=0.5, **kw):
    from sklearn.linear_model import LogisticRegression
    meta, crops = [], []
    for it in items:
        r, why = _name_crop(it["path"])
        if r is not None:
            meta.append(it); crops.append(r["crop"])
    if len(meta) < 30:
        raise ValueError(f"라벨 위치를 찾은 사진 {len(meta)}장 (너무 적음)")
    lab = np.array([sn.LABEL_OF[m["model"]] for m in meta]); models = np.array([m["model"] for m in meta])
    hkey = [sd.content_fold(m["path"], 10 ** 9) for m in meta]
    split = np.array([""] * len(meta), dtype=object)
    for mname in sorted(set(models)):
        ids = sorted([i for i in range(len(meta)) if models[i] == mname], key=lambda i: hkey[i])
        n = len(ids); n_tr = max(1, int(0.75 * n))
        for k, i in enumerate(ids):
            split[i] = "train" if k < n_tr else "val"
    trm = split == "train"; vam = split == "val"
    Cr = np.stack(crops)
    arng = np.random.RandomState(0); idx_tr = np.where(trm)[0]; labA = np.repeat(lab[idx_tr], aug); CA = []
    for i in idx_tr:
        for _ in range(aug):
            h_, w_ = Cr[i].shape[:2]
            M_ = cv2.getRotationMatrix2D((w_ / 2, h_ / 2), arng.uniform(-2.5, 2.5), arng.uniform(0.96, 1.04)); M_[:, 2] += arng.uniform(-4, 4, 2)
            c_ = cv2.warpAffine(Cr[i], M_, (w_, h_), borderMode=cv2.BORDER_REPLICATE)
            CA.append(np.clip(c_.astype(np.float32) * arng.uniform(0.85, 1.15), 0, 255).astype(np.uint8))
    X = np.stack([sn.f_hog(c) for c in Cr])
    XA = np.stack([sn.f_hog(c) for c in CA]) if aug > 0 else np.zeros((0, X.shape[1]))
    clf = LogisticRegression(C=3.0, max_iter=4000).fit(np.vstack([X[trm], XA]) if aug > 0 else X[trm],
                                                       np.concatenate([lab[trm], labA]) if aug > 0 else lab[trm])
    classes = list(clf.classes_)
    P = clf.predict_proba(X)
    exp_sim = np.array([P[i][classes.index(lab[i])] for i in range(len(meta))])
    thr = float(np.percentile(exp_sim[vam], 1)) if vam.any() else float(np.percentile(exp_sim, 1))
    acc = float((np.array(classes)[P[vam].argmax(1)] == lab[vam]).mean()) if vam.any() else float("nan")
    return dict(clf=clf, classes=classes, thr=thr, margin=margin, n=len(meta), n_items=len(items), val_acc=acc)


def pred_name(M, model, path):
    exp = sn.LABEL_OF[model]
    r, why = _name_crop(path)
    if r is None:
        return _res(3, model, 0, "재확인", {"no_square": "라벨 막대(글자+정사각형)를 못 찾음", "no_row": "imageRUNNER 글자 줄을 못 찾음"}.get(why, str(why)))
    p = M["clf"].predict_proba(sn.f_hog(r["crop"])[None])[0]
    S = {v: float(p[k]) for k, v in enumerate(M["classes"])}
    e = S.get(exp, 0.0); others = {v: x for v, x in S.items() if v != exp}
    bo_lab = max(others, key=others.get) if others else None; bo = others[bo_lab] if others else 0.0
    thr2 = M["thr"] * M["margin"]
    rate = float(np.clip(50 + 50 * (e - thr2) / max(1.0 - thr2, 1e-3), 0, 100))
    if bo > e:
        return _res(3, model, min(rate, 49.0), "NG", f"라벨이 {bo_lab} 로 보임(기대 {exp}, 기대 라벨 확률 {e:.3f})")
    if e < thr2:
        return _res(3, model, min(rate, 49.0), "NG", f"기대 라벨 {exp} 와 다름(확률 {e:.3f} < {thr2:.3f})")
    return _res(3, model, rate, "OK")


# ---------------------------------------------------------------- 스텝 4·5·7: 파란 부품 면적
def blue_frac(path, smin):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (256, 192))
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(h, (95, smin, 50), (135, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(m)
    return 0.0 if n < 2 else float(st[1:, 4].max() / m.size)


def fit_blue(items, step=7, **kw):
    smin = BLUE_SMIN[step]
    fr = np.array([f for f in (blue_frac(it["path"], smin) for it in items) if f is not None])
    if len(fr) < 20:
        raise ValueError(f"사진 {len(fr)}장 (너무 적음)")
    thr = max(float(np.percentile(fr, 1)) * 0.5, 0.02)
    return dict(smin=smin, thr=thr, n=len(fr), n_items=len(items), cv_fp=float(np.mean(fr < thr)), med=float(np.median(fr)))


def pred_blue(M, step, model, path):
    f = blue_frac(path, M["smin"])
    if f is None:
        return _res(step, model, 0, "재확인", "사진을 읽지 못함")
    rate = 100.0 * (1 - 0.5 ** (f / M["thr"]))
    if f < M["thr"]:
        return _res(step, model, min(rate, 49.0), "NG", f"파란 부품 면적 {f:.3f} < 기준 {M['thr']:.3f}")
    return _res(step, model, max(rate, 50.0), "OK")


# ---------------------------------------------------------------- 장면 구성(스텝 4,5,6,8~15)
def fit_generic(items, step=None, min_group=15, margin=1.5, **kw):
    if len(items) < 10:
        raise ValueError(f"사진 {len(items)}장 (너무 적음)")
    cnt = {}
    for it in items:
        cnt[it["model"]] = cnt.get(it["model"], 0) + 1
    grp = lambda m: m if cnt.get(m, 0) >= min_group else "소수기종묶음"
    X = np.array([step_scene.desc(i["path"]) for i in items], dtype=np.float32); G = np.array([grp(i["model"]) for i in items])
    bank, thr = {}, {}
    for g in set(G):
        idx = np.where(G == g)[0]
        if len(idx) < 3:
            continue
        Xg = X[idx]
        sq = (Xg ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Xg @ Xg.T, 0)); np.fill_diagonal(D, 1e9)
        thr[g] = float(np.percentile(D.min(1), 99.5)) * margin; bank[g] = Xg
    return dict(bank=bank, thr=thr, counts=cnt, min_group=min_group, n=len(items))


def _gen_group(M, model):
    g = model if M["counts"].get(model, 0) >= M["min_group"] else "소수기종묶음"
    return g if g in M["thr"] else None


def pred_generic(M, step, model, path):
    g = _gen_group(M, model)
    if g is None:
        return _res(step, model, 0, "재확인", "같은 기종의 기준 사진이 부족함")
    d = step_scene.desc(path).astype(np.float32); B = M["bank"][g]
    dist = float(np.sqrt(np.maximum(((B - d) ** 2).sum(1), 0)).min()); t = M["thr"][g]
    rate = 100.0 * 0.5 ** (dist / t)
    if dist > t:
        return _res(step, model, min(rate, 49.0), "NG", f"정상 사진과 장면 구성이 다름(거리 {dist:.1f} > {t:.1f})")
    return _res(step, model, max(rate, 50.0), "OK")


# ---------------------------------------------------------------- 부품 탐지기(스텝 6, 8~14)
def fit_part(items, step, root, min_group=15, margin=0.5, ncc=0.55, max_train=250, **kw):
    cfg = sp.PARTS[step]
    ref = next(Path(root).glob(f"{cfg['ref']}/step_{step}.*"), None)
    if ref is None:
        raise ValueError(f"대표 사진 {cfg['ref']} 없음")
    rg = cv2.imdecode(np.fromfile(str(ref), np.uint8), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = cfg["box"]
    tpl = cv2.resize(rg[y:y + h, x:x + w], None, fx=sp.W / 1024, fy=sp.H / 768, interpolation=cv2.INTER_AREA)
    win = (int(round(tpl.shape[1] / 8) * 8 + 16), int(round(tpl.shape[0] / 8) * 8 + 16)); stride = 4
    G = [sp.gray_small(i["path"]) for i in items]; ok = [k for k, g in enumerate(G) if g is not None]
    items = [items[k] for k in ok]; G = [G[k] for k in ok]
    if len(items) < 10:
        raise ValueError(f"사진 {len(items)}장 (너무 적음)")
    FF = [sp.windows(g, win, stride)[0] for g in G]
    cnt = {}
    for it in items:
        cnt[it["model"]] = cnt.get(it["model"], 0) + 1
    grp = lambda m: m if cnt[m] >= min_group else "소수기종묶음"
    Gm = np.array([grp(it["model"]) for it in items]); fold = np.array([zlib.crc32(Path(it["path"]).read_bytes()) % 2 for it in items])
    mt = [sp.best_match(g, tpl) for g in G]
    rng = np.random.RandomState(0)
    _cc = {}

    def common(f):
        if f not in _cc:
            cand = [k for k in range(len(items)) if fold[k] == f and mt[k][0] >= ncc]
            if len(cand) > max_train:
                cand = list(rng.choice(cand, max_train, replace=False))
            _cc[f] = sp.train([FF[k] for k in cand], [(mt[k][1], mt[k][2]) for k in cand], win, stride, rng) if len(cand) >= 4 else None
        return _cc[f]

    thr, sc, clfs_by, used_common = {}, {}, {}, []
    for g in set(Gm):
        idx = np.where(Gm == g)[0]
        if len(idx) < 8:
            continue
        clfs = []
        for f in (0, 1):
            cand = [k for k in idx if fold[k] == f and mt[k][0] >= ncc]
            if len(cand) > max_train:
                cand = list(rng.choice(cand, max_train, replace=False))
            c = sp.train([FF[k] for k in cand], [(mt[k][1], mt[k][2]) for k in cand], win, stride, rng) if len(cand) >= 4 else None
            if c is None:
                c = common(f); used_common.append(g)
            clfs.append(c)
        score = np.full(len(idx), np.nan)
        for q, k in enumerate(idx):
            clf = clfs[1 - fold[k]]
            if clf is None:
                continue
            F = FF[k]; ny, nx, d = F.shape
            score[q] = clf.decision_function(F.reshape(-1, d)).max()
        v = score[~np.isnan(score)]
        if len(v) < 8 or all(c is None for c in clfs):
            continue
        p1, med = np.percentile(v, 1), np.median(v)
        thr[g] = float(p1 - margin * (med - p1)); sc[g] = float(max((med - thr[g]) / 2.94, 1e-3)); clfs_by[g] = clfs
    return dict(win=win, stride=stride, thr=thr, sc=sc, clfs=clfs_by, counts=cnt, min_group=min_group, n=len(items),
                common_groups=sorted(set(used_common)), part=cfg["part"])


def pred_part(M, step, model, path):
    g = model if M["counts"].get(model, 0) >= M["min_group"] else "소수기종묶음"
    if g not in M["thr"]:
        return _res(step, model, 0, "재확인", "같은 기종의 기준 사진이 부족함")
    gi = sp.gray_small(path)
    if gi is None:
        return _res(step, model, 0, "재확인", "사진을 읽지 못함")
    win, stride = M["win"], M["stride"]
    F = sp.windows(gi, win, stride)[0]; ny, nx, d = F.shape
    sv = [c.decision_function(F.reshape(-1, d)) for c in M["clfs"][g] if c is not None]
    s_ = np.mean(sv, 0); j = int(s_.argmax()); s = float(s_[j])
    pos = (j % nx * stride + win[0] // 2, j // nx * stride + win[1] // 2)
    t, sc = M["thr"][g], M["sc"][g]
    rate = 100.0 / (1.0 + np.exp(-(s - t) / sc))
    if s < t:
        return _res(step, model, min(rate, 49.0), "NG", f"부품({M['part']})이 기대 위치 부근에서 확인되지 않음(점수 {s:.2f} < {t:.2f})", pos)
    return _res(step, model, max(rate, 50.0), "OK", "", pos)


# ---------------------------------------------------------------- 학습 일괄 / 판정 일괄
def fit_step(step, items, root):
    if step == 1:
        return fit_dots(items)
    if step == 2:
        return fit_logo(items)
    if step == 3:
        return fit_name(items)
    out = {}
    if step in BLUE_SMIN:
        out["blue"] = fit_blue(items, step)
    if step != 7:
        out["generic"] = fit_generic(items, step)
    if step in PART_STEPS:
        out["part"] = fit_part(items, step, root)
    return out


def predict_step(models, step, model, path):
    """models: train_all 이 저장한 dict. 반환 dict(step,id,rate,res,why,pos)"""
    M = models.get(step)
    if M is None:
        return _res(step, model, 0, "재확인", "이 스텝의 모델이 없음")
    if step == 1:
        return pred_dots(M, model, path)
    if step == 2:
        return pred_logo(M, model, path)
    if step == 3:
        return pred_name(M, model, path)
    comps = []
    if "blue" in M:
        comps.append(pred_blue(M["blue"], step, model, path))
    if "generic" in M:
        comps.append(pred_generic(M["generic"], step, model, path))
    if "part" in M:
        comps.append(pred_part(M["part"], step, model, path))
    if not comps:
        return _res(step, model, 0, "재확인", "판정기가 없음")
    return combine(step, model, comps)


def combine(step, model, comps):
    """하나라도 NG 면 NG, 아니면 하나라도 재확인이면 재확인, 일치율은 가장 낮은 값"""
    rate = min(c["rate"] for c in comps)
    if any(c["res"] == "NG" for c in comps):
        res = "NG"
    elif any(c["res"] == "재확인" for c in comps):
        res = "재확인"
    else:
        res = "OK"
    why = "; ".join(c["why"] for c in comps if c["why"] and c["res"] != "OK")
    pos = next((c["pos"] for c in comps if c.get("pos")), None)
    return _res(step, model, rate, res, why, pos)


def fmt(r):
    return f"{r['id']}:{round(r['rate'])}:{r['res']}"


def ppt_line(results):
    names = {"OK": "PASS", "NG": "FAIL", "재확인": "확인필요"}
    return " | ".join(f"STEP {r['step']} {names[r['res']]}({r['rate']:.1f})" for r in sorted(results, key=lambda r: r["step"]))


def save_models(models, mdir):
    mdir = Path(mdir); mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / "model.pkl", "wb") as f:
        pickle.dump(models, f, protocol=4)


def load_models(mdir):
    with open(Path(mdir) / "model.pkl", "rb") as f:
        return pickle.load(f)
