"""라벨 위치 찾기: 라벨에 공통으로 있는 점 6개 아이콘(삼각형 배열)을 기준으로 위치·방향·크기를 구해 똑바로 선 라벨 crop을 만든다.

전체 사진 정렬(ORB)에 의존하지 않는다. 판정(C550/C551)에는 쓰지 않고, 라벨 영역을 같은 모양으로 잘라내는 전처리다.
 - 점 아이콘: 어두운 작은 원 덩어리 중 간격이 규칙적인 무리(4개 이상)
 - 방향: 점 무리의 중심 -> 글자 덩어리의 중심 으로 향하는 벡터가 '오른쪽'(글자를 읽는 방향). 점 아이콘은 글자 왼쪽에 있다
 - 크기: 점 사이 간격
"""
import cv2
import numpy as np

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
