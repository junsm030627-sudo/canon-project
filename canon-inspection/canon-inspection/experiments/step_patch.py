"""스텝 6, 8~15 부품 단위(조각 단위) 점검 - 정상 사진의 '조각'들과 비교한다.
왜 이 방식인가: 손으로 들고 찍어 부품 위치가 사진마다 달라서 한 곳만 고정해 보는 방식은 안 맞고,
장면 전체 거리(step_generic)는 나사 하나·포트 하나처럼 작은 변화를 희석한다.
방법(PatchCore 를 CPU 용으로 줄인 것, 딥러닝/사전학습 모델 없음):
 1) 사진을 256x192 로 줄이고 32x32 조각을 16칸 간격으로 자른다 (조각 165개).
 2) 조각마다 밝기 모양(8x8, 평균 제거·정규화) + 방향 히스토그램(HOG 36) + 색(Lab a,b) 특징을 만든다.
 3) 같은 기종(정상 15장 미만이면 묶음)의 정상을 제품 단위로 반으로 나누고, 한쪽 정상 조각 은행을 만든다.
    사진마다 '다른 쪽' 은행에서 조각별 최근접 거리를 구하고, 큰 쪽 3개 평균을 사진 점수로 한다 (자기 자신과는 비교하지 않음).
 4) 임계값 = 정상 점수의 99.5 백분위 x margin(기본 1.5). 일치율 = 100 x 0.5^(점수/임계값) (임계에서 50).
 5) 가장 어긋난 조각의 위치(3x3 칸)를 '위치'로 기록하고, 이상 사진에는 히트맵을 저장한다.
NG 폴더 사진도 같은 기준으로 판정해 함께 기록한다(정답으로 쓰지 않음).
사용: py step_patch.py --root "<root>" --steps 6,8,9,10,11,12,13,14,15 --out out_patch
출력: patch_result.csv (스텝,구분번호,기종,제품,출처,일치율,결과,위치,표기), heat/*.png
"""
import argparse, csv, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from step_dots import collect, find_ng

PART = {6: "2917", 8: "8588/0000", 9: "3780", 10: "V243/V242/B792", 11: "V243/V242/B792",
        12: "3779/V242", 13: "3779/C160", 14: "3779", 15: "C160"}
GH, GW = 11, 15          # 조각 격자 (세로, 가로)
POS_W = 1.0
REG_R = ["상", "중", "하"]; REG_C = ["좌", "중", "우"]


def feats(path):
    """사진 -> (165, 102) float32 조각 특징"""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # 밝기 모양: 4x4 블록 평균 영상(48x64) 에서 8x8 창을 4칸(=16px) 간격으로
    blk = g.reshape(48, 4, 64, 4).mean((1, 3))
    win = np.lib.stride_tricks.sliding_window_view(blk, (8, 8))[::4, ::4]            # (11,15,8,8)
    sh = win.reshape(GH, GW, 64); sh = sh - sh.mean(2, keepdims=True)
    sh = sh / (np.linalg.norm(sh, axis=2, keepdims=True) + 8.0)                        # +8: 평평한 곳의 잡음 증폭 방지
    # 방향 히스토그램: 16x16 셀(12x16) 마다 9방향, 조각 = 2x2 셀
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 9
    b = np.minimum(ang.astype(int), 8)
    cell = (np.arange(192)[:, None] // 16) * 16 + (np.arange(256)[None, :] // 16)
    hist = np.bincount((cell * 9 + b).ravel(), weights=mag.ravel(), minlength=12 * 16 * 9).reshape(12, 16, 9)
    hp = np.concatenate([hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]], 2)  # (11,15,36)
    hp = hp / (np.linalg.norm(hp, axis=2, keepdims=True) + 200.0)
    # 색: 조각 평균 Lab a,b (밝기는 제외 - 그림자에 약하게)
    ab = lab[:, :, 1:].reshape(12, 16, 16, 16, 2).mean((1, 3)) - 128.0                  # (12,16,2) 셀 평균
    cp = (ab[:-1, :-1] + ab[:-1, 1:] + ab[1:, :-1] + ab[1:, 1:]) / 4.0 / 40.0
    # 절대 밝기·대비(평평한 어두운 덩어리와 평평한 흰 덩어리를 구분) + 위치(약하게: 손으로 들고 찍어 위치가 흔들리므로)
    mean = win.mean((2, 3)) / 255.0; std = win.std((2, 3)) / 64.0
    yy, xx = np.meshgrid(np.arange(GH) / (GH - 1), np.arange(GW) / (GW - 1), indexing="ij")
    return np.concatenate([sh, hp * 1.5, cp, (mean * 2.0)[..., None], (std * 1.5)[..., None], (yy * POS_W)[..., None], (xx * POS_W)[..., None]], 2).reshape(GH * GW, -1).astype(np.float32)


def nn_dist(Q, bank, bsq):
    """Q (n,d), bank (m,d) -> 각 Q 행의 최근접 거리"""
    d2 = (Q ** 2).sum(1)[:, None] + bsq[None, :] - 2 * Q @ bank.T
    return np.sqrt(np.maximum(d2.min(1), 0))


def img_score(dd):
    return float(np.sort(dd)[-3:].mean())


def region(k):
    i, j = divmod(int(k), GW)
    return REG_R[min(2, i * 3 // GH)] + REG_C[min(2, j * 3 // GW)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--steps", default="6,8,9,10,11,12,13,14,15")
    ap.add_argument("--out", default="out_patch"); ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--margin", type=float, default=1.5); ap.add_argument("--bank_n", type=int, default=120, help="한쪽 은행에 쓸 최대 사진 수")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--heat", type=int, default=12, help="히트맵 저장 최대 장수(스텝당)")
    a = ap.parse_args(); out = Path(a.out); (out / "heat").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0); rows = []
    for st in [int(s) for s in a.steps.split(",")]:
        try:
            items = collect(a.root, st, per_model=10 ** 6)
            items = items[0] if isinstance(items, tuple) else items
        except Exception as e:
            print(f"스텝 {st}: 수집 실패 {e}"); continue
        if len(items) < 10:
            print(f"스텝 {st}: 사진 {len(items)}장 (너무 적어 건너뜀)"); continue
        print(f"스텝 {st}: 특징 추출 {len(items)}장 ...", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            F = list(ex.map(feats, [i["path"] for i in items], chunksize=16))
        ok = [k for k, f in enumerate(F) if f is not None]; items = [items[k] for k in ok]; F = np.stack([F[k] for k in ok])
        cnt = {}
        for it in items: cnt[it["model"]] = cnt.get(it["model"], 0) + 1
        G = np.array([it["model"] if cnt[it["model"]] >= a.min_group else "소수기종묶음" for it in items])
        fold = np.array([zlib.crc32(Path(it["path"]).read_bytes()) % 2 for it in items])           # 제품 단위 두 쪽
        score = np.full(len(items), np.nan); loc = np.zeros(len(items), int); thr = {}; banks = {}
        for g in set(G):
            idx = np.where(G == g)[0]
            if len(idx) < 6: thr[g] = None; continue
            bk = []
            for f in (0, 1):
                m = idx[fold[idx] == f]
                if len(m) > a.bank_n: m = rng.choice(m, a.bank_n, replace=False)
                B = F[m].reshape(-1, F.shape[2]); bk.append((B, (B ** 2).sum(1)))
            banks[g] = bk
            for k in idx:
                B, bsq = bk[1 - fold[k]]                                                      # 다른 쪽 은행과 비교
                dd = nn_dist(F[k], B, bsq); score[k] = img_score(dd); loc[k] = int(dd.argmax())
            thr[g] = float(np.percentile(score[idx], 99.5)) * a.margin
        def row(it, s, k, g, src):
            t = thr.get(g)
            if t is None or np.isnan(s): return (st, PART.get(st, "?"), it["model"], it["serial"], src, 0, "재확인", "")
            return (st, PART.get(st, "?"), it["model"], it["serial"], src, round(100 * 0.5 ** (s / t)), "OK" if s <= t else "NG", region(k))
        for k, it in enumerate(items): rows.append(row(it, score[k], loc[k], G[k], "정상"))
        bad = [r for r in rows if r[0] == st and r[4] == "정상" and r[6] != "OK"]
        print(f"스텝 {st}: 정상 {len(items)}장 중 이상 {len(bad)}장 ({100 * len(bad) / len(items):.2f}%)")
        # 히트맵: 일치율이 낮은 정상 사진 + NG 폴더 사진
        def heat(it, dd, name):
            im = cv2.resize(cv2.imdecode(np.fromfile(str(it["path"]), np.uint8), cv2.IMREAD_COLOR), (256, 192))
            h = cv2.resize(dd.reshape(GH, GW), (256, 192), interpolation=cv2.INTER_CUBIC)
            h = np.clip(h / (max(thr[G_of(it)], 1e-6)), 0, 2) / 2 * 255
            col = cv2.applyColorMap(h.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.putText(im, name, (4, 14), 0, 0.45, (0, 255, 255), 1)
            cv2.imencode(".png", np.hstack([im, cv2.addWeighted(im, 0.5, col, 0.5, 0)]))[1].tofile(str(out / "heat" / f"s{st}_{it['serial']}_{name.split()[-1]}.png"))
        def G_of(it): return it["model"] if cnt[it["model"]] >= a.min_group else "소수기종묶음"
        low = sorted(range(len(items)), key=lambda k: score[k] if not np.isnan(score[k]) else -1, reverse=True)[:a.heat]
        for k in low:
            if thr.get(G[k]) is None: continue
            B, bsq = banks[G[k]][1 - fold[k]]; heat(items[k], nn_dist(F[k], B, bsq), f"정상 {items[k]['serial']}")
        for it in find_ng(a.root, st):
            g = G_of(it); f = feats(it["path"])
            if thr.get(g) is None or f is None: continue
            dds = [nn_dist(f, B, bsq) for B, bsq in banks[g]]; dd = (dds[0] + dds[1]) / 2
            s = img_score(dd); r = row(it, s, int(dd.argmax()), g, "NG폴더"); rows.append(r)
            print(f"   NG폴더 {it['model']}/{it['serial']}: 점수 {s:.2f} (임계 {thr[g]:.2f}) 일치율 {r[5]} → {r[6]} {r[7]}")
            heat(it, dd, f"NG {it['serial']}")
    with open(out / "patch_result.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["스텝", "구분번호", "기종", "제품", "출처", "일치율", "결과", "위치", "표기"])
        for r in rows: w.writerow([*r, f"{r[1]}:{r[5]}:{r[6]}"])
    print("저장:", out / "patch_result.csv")


if __name__ == "__main__":
    main()
