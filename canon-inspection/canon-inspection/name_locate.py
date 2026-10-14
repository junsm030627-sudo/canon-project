"""스텝 3(제품명 라벨 '2206N' 등) 위치 찾기 v2.
1) 라벨 바로 옆의 큰 글자 줄 'imageRUNNER'(11~12덩어리, 모든 기종 공통)를 찾아 방향·크기를 잡는다.
   글자 줄 위에서 소문자 쪽(imag e)보다 대문자 쪽(RUNNER)이 크다는 점으로 방향(좌우)을 정한다.
2) 그 줄 아래쪽(예상 위치) 영역만 똑바로 세운 뒤, 그 안에서 라벨 끝의 진한 정사각형을 찾는다.
   (큰 그림자·기둥이 없는 좁은 영역에서만 찾으므로 연한 회색 정사각형도 잡힌다)
3) 정사각형이 오른쪽, 글자가 왼쪽이 되도록 돌려 최종 crop을 만든다.
OCR/템플릿 매칭을 쓰지 않는다: 덩어리 크기·간격·위치 관계만 사용. 판정은 crop과 라벨별 기준 이미지의 유사도.
locate(g) 는 (결과 dict 또는 None, 실패 사유) 를 돌려준다. 사유: 'no_row'(글자 줄 못 찾음) / 'no_square'(라벨 막대·정사각형 못 찾음)
"""
import cv2
import numpy as np

ROI_PPC = 40.0           # ROI 단계에서 글자 높이(cap) 1당 픽셀 (원본 해상도에 가깝게 유지: 글자 모양 구분에 필요)
PX_PER_S = 45.0          # 정사각형(=1.5cap) 한 변당 출력 픽셀
OUT_W, OUT_H = 380, 80
SQ_X = 340
SQ_Y = OUT_H / 2


def _dark(g, block, C):
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)


def find_row(g):
    """'imageRUNNER' 글자 줄. 반환 (중심, 방향 u(대문자 쪽이 +), cap높이) 또는 None"""
    m = _dark(g, 41, 14)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    P, Hh, A = [], [], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 120 or not (14 <= max(w, h) <= 90) or x <= 1 or y <= 1 or x + w >= g.shape[1] - 1 or y + h >= g.shape[0] - 1:
            continue
        if a / float(w * h) > 0.97:      # 순수 직사각 덩어리(기둥 모서리 등) 제외
            continue
        P.append(cen[i]); Hh.append(max(w, h)); A.append(a)
    if len(P) < 9:
        return None
    P = np.array(P); Hh = np.array(Hh, float)
    best = None
    for deg in range(-60, 61, 2):
        th = np.deg2rad(deg)
        u = np.array([np.cos(th), np.sin(th)]); p = np.array([-u[1], u[0]])
        perp = P @ p
        order = np.argsort(perp)
        ps = perp[order]
        j = 0
        for i in range(len(ps)):
            while ps[i] - ps[j] > 14:
                j += 1
            k = i - j + 1
            if k >= 9 and (best is None or k > best[0]):
                best = (k, u, order[j:i + 1])
    if best is None:
        return None
    k, u, idx = best
    pts = P[idx]
    # 선 맞춤(주성분)으로 각도 다듬기
    c0 = pts.mean(0); cov = np.cov((pts - c0).T)
    ang = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    u = np.array([np.cos(ang), np.sin(ang)])
    along = (pts - c0) @ u
    span = along.max() - along.min()
    if span < 8 * np.median(Hh[idx]):          # 글자 줄이 충분히 길어야 한다
        return None
    order = np.argsort(along)
    hs = Hh[idx][order]
    half = len(hs) // 2
    if np.median(hs[half:]) < np.median(hs[:half]):    # 대문자(RUNNER)가 +u 쪽이 되도록
        u = -u
    return c0, u, float(np.percentile(Hh[idx], 75))


def _find_square_t(roi, thr):
    """roi: 똑바로 세운 영역(uint8). 진한 정사각형 후보 중 가장 그럴듯한 하나 -> (cx, cy, side) 또는 None"""
    b = cv2.GaussianBlur(roi, (0, 0), 1.5)
    bh = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))) - b
    m = (bh > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    found = []
    side0 = 1.5 * ROI_PPC
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (0.45 * side0 ** 2 <= a <= 2.2 * side0 ** 2):
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        if rw < 5 or rh < 5 or not (0.72 <= rw / rh <= 1.4) or a / float(rw * rh) < 0.78:
            continue
        ring = b[max(y - 13, 0):y + h + 13, max(x - 13, 0):x + w + 13].astype(np.float32)
        inner = b[y:y + h, x:x + w].astype(np.float32)
        contrast = np.percentile(ring, 75) - inner.mean()
        if contrast < 25:
            continue
        # 가장자리가 또렷해야 한다(그림자 띠는 경계가 흐리다) + 크기가 기대 크기(1.5cap)에 가까워야 한다
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        edge = cv2.morphologyEx(comp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        gx_ = cv2.Sobel(b[y:y + h, x:x + w], cv2.CV_32F, 1, 0, ksize=3); gy_ = cv2.Sobel(b[y:y + h, x:x + w], cv2.CV_32F, 0, 1, ksize=3)
        sharp = float(np.hypot(gx_, gy_)[edge].mean()) if edge.any() else 0.0
        size_pen = min(a, side0 ** 2) / max(a, side0 ** 2)
        sc = contrast * sharp * size_pen
        found.append((sc, cen[i][0], cen[i][1], float(np.sqrt(a))))
    found.sort(key=lambda t: -t[0])
    return [f[1:] for f in found]


def _find_square(roi):
    """임계값을 올려가며 시도: 그림자 띠가 정사각형과 붙어 한 덩어리가 되면 낮은 임계값에서는 모양이 깨진다 (처음 찾아지는 임계값 결과를 사용)."""
    for thr in (38, 52, 68, 85):
        r = _find_square_t(roi, thr)
        if r:
            return r
    return []


def _side_counts(st, cen, n, sx, sy, side):
    """정사각형 (sx,sy) 왼쪽/오른쪽으로 바로 이어진 글자 덩어리 수. 가장 가까운 글자가 2.4변 밖이면 0으로 친다"""
    out = []
    for sign in (-1, +1):
        xs = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 30 or not (0.3 * side <= max(w, h) <= 1.3 * side):
                continue
            dx, dy = cen[i][0] - sx, cen[i][1] - sy
            if abs(dy) < 0.6 * side and sign * dx > 0.8 * side and abs(dx) < 8 * side:
                xs.append(abs(dx))
        xs.sort()
        k = 0
        if xs and xs[0] <= 2.4 * side:
            k = 1
            while k < len(xs) and xs[k] - xs[k - 1] < 1.6 * side:
                k += 1
        out.append(k)
    return out[0], out[1]


def _fallback_from_text(roi):
    """정사각형 대비가 약할 때: 띠 안에서 한 줄로 늘어선 글자 덩어리(>=4)를 찾고, 그 왼쪽 끝에서 1.5변 떨어진 곳을 정사각형 중심으로 추정"""
    side0 = 1.5 * ROI_PPC
    y0, y1 = int(0.9 * ROI_PPC), int(4.2 * ROI_PPC)
    m = _dark(roi[y0:y1], 51, 12)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 30 and 0.3 * side0 <= max(w, h) <= 0.85 * side0 and x > 2 and x + w < roi.shape[1] - 2:
            comps.append((cen[i][0], cen[i][1] + y0, x))
    best = None
    for cx, cy, _ in comps:
        grp = [c for c in comps if abs(c[1] - cy) < 0.5 * side0]
        if len(grp) >= 4 and (best is None or len(grp) > len(best)):
            best = grp
    if best is None:
        return None
    xmin = min(c[2] for c in best)
    return xmin - 1.5 * side0, float(np.median([c[1] for c in best])), side0


def locate(g):
    r = find_row(g)
    if r is None:
        return None, "no_row"
    c0, u, cap = r
    v = np.array([-u[1], u[0]])          # u를 +x로 돌렸을 때 +y 방향 (라벨은 이 방향 아래쪽)
    # 글자 줄 위에서 라벨이 어느 쪽인지는 모른다: 위/아래 양쪽 영역을 모두 만들어 본다
    k = ROI_PPC / cap
    W = int(20 * ROI_PPC); H = int(6.5 * ROI_PPC)
    for dy in (0.2, 1.7):      # 글자 줄 중심 추정이 어긋난 사진은 창을 줄 쪽으로 더 내려 본다 (처음 것이 실패할 때만)
        for sgn in (+1, -1):
            # sgn=+1: 줄 아래쪽(v 방향) 영역, sgn=-1: 위쪽(180도 돌려서 같은 꼴로)
            uu = u * sgn; vv = v * sgn
            R = np.array([uu, vv]) * k              # 이미지 -> ROI (x축=uu, y축=vv)
            # ROI 좌표: 줄 중심이 (W/2, 0.2cap) 위치, 아래로 6.5cap
            t = np.array([W / 2.0, dy * ROI_PPC]) - R @ c0
            M = np.hstack([R, t[:, None]])
            roi = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cands = _find_square(roi[int(0.9 * ROI_PPC):int(4.2 * ROI_PPC), :])       # 글자 줄 자체(0.9cap 이내)는 제외
            m = _dark(roi, 51, 12)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m)
            mode = "square"
            picked = None
            for (sx, sy, side_) in cands:
                sy += int(0.9 * ROI_PPC)
                side = 1.5 * ROI_PPC              # 크기는 글자 높이로 고정(덩어리 크기는 흔들린다)
                L, R_ = _side_counts(st, cen, n, sx, sy, side)
                # 진짜 정사각형은 글자가 한쪽에만, 그리고 바로 옆(2.4변 이내)부터 이어져 있다 (그림자 덩어리는 멀거나 양쪽이다)
                if max(L, R_) >= 3 and min(L, R_) <= 1:
                    picked = (sx, sy, side, L, R_)
                    break
            if picked is None:
                sq = _fallback_from_text(roi) if dy == 0.2 else None   # 내려 본 창에서는 글자 추정 대신 사각형만 인정                  # 정사각형이 흐리면 글자 덩어리 위치로 대신 추정
                mode = "text"
                if sq is None:
                    continue
                sx, sy, side = sq
                L, R_ = _side_counts(st, cen, n, sx, sy, side)
                if max(L, R_) < 3:
                    continue
            else:
                sx, sy, side, L, R_ = picked
            flip = R_ > L                            # 글자가 오른쪽이면 180도 돌려 정사각형이 오른쪽에 오게 한다
            s2 = PX_PER_S / side
            rot = np.pi if flip else 0.0
            c, s_ = np.cos(rot), np.sin(rot)
            R2 = np.array([[c, -s_], [s_, c]]) * s2
            M2 = np.hstack([R2, (np.array([SQ_X, SQ_Y]) - R2 @ np.array([sx, sy]))[:, None]])
            crop = cv2.warpAffine(roi, M2, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return dict(crop=crop, center=c0, angle=float(np.degrees(np.arctan2(uu[1], uu[0]))), s=side / k,
                        n_letters=max(L, R_), mode=mode), ""
    return None, "no_square"
