"""step 1 (토너 지역 라벨 C550 vs C551) - 점 아이콘 기준 라벨 위치 찾기 + 처음 보는 기종 평가.

전체 사진 정렬(ORB)을 쓰지 않는다. 라벨의 점 6개 아이콘으로 위치·방향·크기를 잡아 똑바로 선 라벨 crop을 만들고,
그 crop으로 C550/C551을 구분한다. (위치 찾기는 전처리이고, 판정은 crop과 라벨별 기준 이미지의 유사도)

사용법 (Windows):
  py step_dots.py --root "<데이터 경로>" --step 1 --out out_dots

출력: crops_by_model.png(기종별 똑바로 선 라벨 샘플), ref_C550.png / ref_C551.png(라벨별 평균), per_image.csv,
      locate_fail.csv(점을 못 찾은 사진), lomo_result.csv, shift_test.csv
"""
import argparse, csv, random, re, sys
from pathlib import Path
import numpy as np
import cv2

C551_MODELS = {"3029C003AA", "3029C004AA", "3030C001AA"}
ALL_MODELS = ["3029C003AA", "3029C004AA", "3029C005AA", "3029C006AA", "3029C009AA", "3029C010AA",
              "3030C001AA", "3030C002AA", "3030C003AA", "3030C004AA", "3031C001AA", "3031C002AA", "3031C003AA"]


def content_fold(path, n):
    """학습/검증 분할용: 파일 이름이 아니라 사진 내용(바이트)의 해시로 칸을 정한다."""
    import zlib
    return zlib.crc32(Path(path).read_bytes()) % n


def variant_of(model):
    return "C551" if model in C551_MODELS else "C550"


def imread_gray(p):
    im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is not None and im.shape[1] != 1024:
        im = cv2.resize(im, (1024, int(im.shape[0] * 1024 / im.shape[1])), interpolation=cv2.INTER_AREA)
    return im


def step_file(serial_dir, step):
    pat = re.compile(rf"^step[_\- ]?0*{step}\.(jpg|jpeg|png|bmp)$", re.I)
    for p in serial_dir.iterdir():
        if p.is_file() and pat.match(p.name):
            return p
    return None


def collect(root, step, per_model, seed=0):
    rng = random.Random(seed)
    by_model, mism = {}, []
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir() or mdir.name not in ALL_MODELS:
            continue
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            if sdir.name.upper() == "NG":
                continue
            f = step_file(sdir, step)
            if not f:
                continue
            model = mdir.name                  # 기종은 '입력으로 주어진 값'(폴더). 제품번호 접두어로 기종을 추정하거나 사진을 거르지 않는다.
            by_model.setdefault(model, []).append(dict(model=model, serial=sdir.name, path=f, variant=variant_of(model)))
    items = []
    for m, lst in by_model.items():
        rng.shuffle(lst)
        items += lst[:per_model]
    return items, mism


def find_ng(root, step):
    """<기종>/<제품>/NG/*_step_N.jpg (앞에 날짜가 붙은 이름) 를 모두 찾는다."""
    pat = re.compile(rf"step[_\- ]?0*{step}\.(jpg|jpeg|png|bmp)$", re.I)
    res = []
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir() or mdir.name not in ALL_MODELS:
            continue
        for sdir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            ngd = sdir / "NG"
            if ngd.is_dir():
                for f in sorted(ngd.iterdir()):
                    if f.is_file() and pat.search(f.name):
                        res.append(dict(model=mdir.name, serial=sdir.name, path=f, variant=variant_of(mdir.name)))
    return res


def norm_local(a):
    a = a.astype(np.float32)
    return (a - cv2.GaussianBlur(a, (0, 0), 12)) / (a.std() + 1e-6)


OUT_W, OUT_H = 150, 64          # 출력 crop 크기
PX_PER_S = 8.0                  # 점 간격 1당 출력 픽셀 수
DOT_X = 22                      # 출력 crop 안에서 점 중심의 x (똑바로 선 라벨: 점이 왼쪽, 글자가 오른쪽)
DOT_Y = OUT_H / 2


def _dark_mask(g):
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 14)


def find_dots(g):
    """g: uint8 grayscale, 가로 약 1024 기준. 반환: (점 좌표 Nx2, 간격 s) 또는 None"""
    m = _dark_mask(g)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (12 <= area <= 500) or w < 3 or h < 3:
            continue
        asp = w / h
        fill = area / float(w * h)
        if 0.65 <= asp <= 1.55 and 0.55 <= fill <= 0.95:
            cand.append((cen[i][0], cen[i][1], np.sqrt(area / np.pi) * 2))   # x, y, 지름
    if len(cand) < 4:
        return None
    P = np.array([[c[0], c[1]] for c in cand]); D = np.array([c[2] for c in cand])
    # 이웃 그래프: 가까우면서(지름의 3배 이내) 크기가 비슷한 후보끼리 연결
    N = len(P)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(P[i] - P[j]); dm = 0.5 * (D[i] + D[j])
            if d < 3.2 * dm and 0.6 < D[i] / D[j] < 1.67:
                adj[i].append(j); adj[j].append(i)
    seen, best = set(), []
    for i in range(N):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            k = stack.pop(); comp.append(k)
            for q in adj[k]:
                if q not in seen:
                    seen.add(q); stack.append(q)
        if len(comp) > len(best):
            best = comp
    if len(best) < 4 or len(best) > 8:
        return None
    pts = P[best]
    nn = []
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1); d[i] = 1e9
        nn.append(d.min())
    return pts, float(np.median(nn))


def locate(g):
    """반환 dict(crop, center, angle, s) 또는 None. g: uint8 gray."""
    r = find_dots(g)
    if r is None:
        return None
    pts, s = r
    cd = pts.mean(0)
    m = _dark_mask(g)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    xs, ys, ws = [], [], []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        mx = max(w, h)
        if area < 10 or not (0.35 * s <= mx <= 3.4 * s):      # 글자 크기만: 길고 큰 그림자·이음선은 제외
            continue
        c = cen[i]
        d = np.linalg.norm(c - cd)
        if 1.6 * s < d < 10 * s and not np.any(np.linalg.norm(pts - c, axis=1) < 0.6 * s):
            xs.append(c[0]); ys.append(c[1]); ws.append(area)
    if len(xs) < 2:
        return None
    ct = np.array([np.average(xs, weights=ws), np.average(ys, weights=ws)])
    v = ct - cd
    ang = np.arctan2(v[1], v[0])            # 점 -> 글자 방향이 +x 가 되도록 돌린다
    k = PX_PER_S / s
    c, si = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -si], [si, c]]) * k
    M = np.hstack([R, (np.array([DOT_X, DOT_Y]) - R @ cd)[:, None]])
    crop = cv2.warpAffine(g, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return dict(crop=crop, center=cd, angle=float(np.degrees(ang)), s=s, n_dots=len(pts))


def feat(crop, dx=0, dy=0):
    if dx or dy:
        crop = cv2.warpAffine(crop, np.float32([[1, 0, dx], [0, 1, dy]]), (OUT_W, OUT_H), borderMode=cv2.BORDER_REPLICATE)
    c = norm_local(crop)
    c = c - c.mean()
    return (c / (np.linalg.norm(c) + 1e-6)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default="out_dots")
    ap.add_argument("--search", type=int, default=4, help="똑바로 선 crop에서 +-S px(2px 간격) 중 가장 잘 맞는 위치로 비교 (0이면 고정)")
    ap.add_argument("--per_model", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items, mism = collect(a.root, a.step, a.per_model)
    if mism:
        print(f"경고: 폴더 기종과 시리얼 접두사가 다른 제품 {len(mism)}건 제외")
    if not items:
        sys.exit("사진을 찾지 못했습니다.")
    print(f"수집 {len(items)}장")
    meta, crops, fails = [], [], []
    for it in items:
        g = imread_gray(it["path"])
        r = locate(g) if g is not None else None
        if r is None:
            fails.append(it); continue
        meta.append(dict(it, angle=r["angle"], s=r["s"], ndots=r["n_dots"])); crops.append(r["crop"])
    stat = {}
    for it in items:
        stat.setdefault(it["model"], [0, 0])[0] += 1
    for it in meta:
        stat[it["model"]][1] += 1
    print(f"라벨 위치 찾기 성공 {len(meta)}/{len(items)}")
    for m, (n, k) in stat.items():
        print(f"  {m} [{variant_of(m)}] {k}/{n}")
    with open(out / "locate_fail.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["파일", "기종"]] + [[str(i["path"]), i["model"]] for i in fails])
    if not meta:
        sys.exit("라벨 위치를 찾은 사진이 없습니다.")
    models = np.array([m["model"] for m in meta]); var = np.array([m["variant"] for m in meta])
    Cr = np.stack(crops)
    ang = np.array([m["angle"] for m in meta])
    print(f"회전 보정량(도): 중앙값 {np.median(ang):.0f}, 5~95퍼센타일 {np.percentile(ang, 5):.0f} ~ {np.percentile(ang, 95):.0f}; "
          f"점 간격 s(px): 5~95퍼센타일 {np.percentile([m['s'] for m in meta], 5):.1f} ~ {np.percentile([m['s'] for m in meta], 95):.1f}")
    for v in ("C550", "C551"):
        if (var == v).any():
            cv2.imencode(".png", cv2.resize(Cr[var == v].mean(0).astype(np.uint8), None, fx=3, fy=3))[1].tofile(str(out / f"ref_{v}.png"))
    tiles = []
    for mname in sorted(set(models)):
        idx = [i for i in range(len(models)) if models[i] == mname]
        random.Random(1).shuffle(idx)
        cells = [Cr[i] for i in idx[:8]] + [np.zeros((OUT_H, OUT_W), np.uint8)] * max(0, 8 - len(idx))
        row = np.hstack(cells)
        cv2.putText(row, f"{mname} {variant_of(mname)}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 3)
        cv2.putText(row, f"{mname} {variant_of(mname)}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        tiles.append(row)
    cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(out / "crops_by_model.png"))

    SO = [(sx, sy) for sy in range(-a.search, a.search + 1, 2) for sx in range(-a.search, a.search + 1, 2)] if a.search else [(0, 0)]

    def scores(crop, cen, dx=0, dy=0):
        U = np.stack([feat(crop, dx + sx, dy + sy) for sx, sy in SO])
        return {v: float((U @ cen[v]).max() / (np.linalg.norm(cen[v]) + 1e-6)) for v in cen}

    rows, recs, tot_ok, tot = [], [], 0, 0
    cens = {}
    F = np.stack([feat(c) for c in Cr])          # 기준 이미지(평균)는 위치 탐색 없이 만든다
    for held in sorted(set(models)):
        tr, te = models != held, models == held
        cen = {v: F[tr & (var == v)].mean(0) for v in ("C550", "C551") if (tr & (var == v)).any()}
        cens[held] = cen
        if len(cen) < 2:
            rows.append([held, variant_of(held), int(te.sum()), "", "한쪽 라벨만 남아 평가 불가", stat[held][0]]); continue
        ok = 0
        for i in np.where(te)[0]:
            sc = scores(Cr[i], cen)
            pred = max(sc, key=sc.get)
            ok += pred == variant_of(held)
            recs.append([str(meta[i]["path"]), held, pred == variant_of(held), pred, f"{sc['C550']:.3f}", f"{sc['C551']:.3f}", f"{meta[i]['angle']:.1f}", f"{meta[i]['s']:.1f}"])
        rows.append([held, variant_of(held), int(te.sum()), f"{ok / te.sum():.3f}", "", stat[held][0]])
        tot_ok += ok; tot += int(te.sum())
    with open(out / "lomo_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["빠진 기종", "정답 라벨", "위치 찾은 장수", "정확도", "비고", "수집 장수"]); w.writerows(rows)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["파일", "기종", "맞음", "예측", "C550 유사도", "C551 유사도", "회전보정(도)", "점간격(px)"]); w.writerows(recs)
    print("\n[처음 보는 기종 평가] 기종 / 정답 / 위치 찾은 장수 / 정확도 / (비고) / 수집 장수")
    for r in rows:
        print("  ", *r)
    if tot:
        print(f"전체 {tot_ok}/{tot} = {tot_ok / tot:.3f}  (위치를 못 찾은 {len(fails)}장은 판정 불가로 따로 센다)")
        print(f"수집 전체 기준 = 맞음 {tot_ok} / {len(items)} = {tot_ok / len(items):.3f}")
    # 위치 민감도: 똑바로 선 crop을 (dx,dy) px 옮겨서 시험 (8px = 점 간격 1개)
    shifts = [-8, -4, 0, 4, 8]
    print(f"\n[위치 민감도] 똑바로 선 crop을 옮겼을 때 정확도, 탐색 +-{a.search}px (가로 dx, 세로 dy, 단위 px; 8px = 점 간격 1개)")
    print("        " + " ".join(f"{d:>6}" for d in shifts))
    grid = np.zeros((len(shifts), len(shifts)))
    for j, dy in enumerate(shifts):
        for i, dx in enumerate(shifts):
            okk, nn = 0, 0
            for k in range(0, len(Cr), 2):
                cen = cens.get(models[k])
                if not cen or len(cen) < 2:
                    continue
                sc = scores(Cr[k], cen, dx, dy)
                okk += max(sc, key=sc.get) == var[k]; nn += 1
            grid[j, i] = okk / max(nn, 1)
        print(f"dy={dy:>3}  " + " ".join(f"{grid[j, i]:6.3f}" for i in range(len(shifts))))
    with open(out / "shift_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["dy\\dx"] + shifts)
        for j, dy in enumerate(shifts):
            w.writerow([dy] + [f"{grid[j, i]:.3f}" for i in range(len(shifts))])
    # ===== 불량(NG) 사진 시험: 모든 정상 사진으로 만든 기준과 비교 =====
    ng = find_ng(a.root, a.step)
    print(f"\n[NG 시험] step {a.step}의 NG 사진 {len(ng)}장")
    if ng:
        cen_all = {v: F[var == v].mean(0) for v in ("C550", "C551") if (var == v).any()}
        ngtiles = []
        for it in ng:
            g = imread_gray(it["path"])
            r = locate(g) if g is not None else None
            if r is None:
                print(f"  {it['model']} {it['serial']}  정답 {it['variant']}  -> 라벨(점 아이콘)을 못 찾음 = NG (라벨 없음/가려짐 가능)  {it['path'].name}")
                continue
            sc = scores(r["crop"], cen_all)
            pred = max(sc, key=sc.get)
            verdict = "OK(통과)" if pred == it["variant"] else "NG(검출)"
            print(f"  {it['model']} {it['serial']}  정답 {it['variant']}  예측 {pred}  C550 {sc['C550']:.2f} / C551 {sc['C551']:.2f}  -> {verdict}  {it['path'].name}")
            ngtiles.append(r["crop"])
        if ngtiles:
            cv2.imencode(".png", cv2.resize(np.vstack(ngtiles), None, fx=3, fy=3))[1].tofile(str(out / "ng_crops.png"))
    print(f"\n결과는 {out.resolve()} 에 저장됨")


if __name__ == "__main__":
    main()
