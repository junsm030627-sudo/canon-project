"""스텝 사진 모아보기 + NG 폴더 위치 찾기.

사용법 (Windows):
  py contact_sheet.py --root "<데이터 경로>" --step 2 --out sheet_step2

출력: sheet_step2\\contact_sheet.png  (기종마다 한 줄, 사진 N장을 줄여서 나열) -> 이 한 장을 보내주세요.
      터미널에 NG 폴더 위치와 그 안의 파일 이름 예시를 출력합니다. -> 이 출력도 같이 보내주세요.
"""
import argparse, os, random, re
from pathlib import Path
import numpy as np
import cv2

MODELS = ["3029C003AA", "3029C004AA", "3029C005AA", "3029C006AA", "3029C009AA", "3029C010AA",
          "3030C001AA", "3030C002AA", "3030C003AA", "3030C004AA", "3031C001AA", "3031C002AA", "3031C003AA"]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def step_file(d, step):
    pat = re.compile(rf"^step[_\- ]?0*{step}\.(jpg|jpeg|png|bmp)$", re.I)
    for p in d.iterdir():
        if p.is_file() and pat.match(p.name):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", default="sheet")
    ap.add_argument("--n", type=int, default=4, help="기종당 사진 수")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    rng = random.Random(0)
    CW, CH = 256, 192
    rows = []
    for m in MODELS:
        md = root / m
        if not md.is_dir():
            continue
        files = []
        for sd in md.iterdir():
            if sd.is_dir() and sd.name.upper() != "NG":
                f = step_file(sd, a.step)
                if f:
                    files.append(f)
        rng.shuffle(files)
        cells = []
        for f in files[:a.n]:
            im = imread(f)
            if im is None:
                continue
            cells.append(cv2.resize(im, (CW, CH), interpolation=cv2.INTER_AREA))
        while len(cells) < a.n:
            cells.append(np.zeros((CH, CW, 3), np.uint8))
        row = np.hstack(cells)
        txt = f"{m} ({len(files)} files)"
        cv2.putText(row, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(row, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rows.append(row)
    if rows:
        sheet = np.vstack(rows)
        cv2.imencode(".png", sheet)[1].tofile(str(out / "contact_sheet.png"))
        print(f"저장: {(out / 'contact_sheet.png').resolve()}  ({len(rows)}개 기종 x {a.n}장)")
    else:
        print("사진을 찾지 못했습니다. --root 와 --step 을 확인하세요.")

    # NG 폴더 찾기
    print("\n[NG 폴더/파일 찾기]")
    found = 0
    for dp, dn, fn in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        if depth > 4:
            dn[:] = []
            continue
        name = Path(dp).name
        ngfiles = [f for f in fn if "ng" in f.lower()]
        if name.upper() == "NG" or ngfiles:
            found += 1
            if found <= 15:
                shown = sorted(os.listdir(dp))[:6] if name.upper() == "NG" else ngfiles[:6]
                print(f"  {dp}\n     -> {shown}")
    print(f"  NG로 보이는 위치 {found}곳" if found else "  NG 폴더/파일을 찾지 못했습니다 (깊이 4단계까지 검색)")


if __name__ == "__main__":
    main()
