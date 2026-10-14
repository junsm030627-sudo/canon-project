# 캐논 프린터 조립 이미지 판독 (스텝별 PASS/FAIL)

기종과 스텝 번호, 사진을 입력하면 스텝마다 `구분번호:일치율:결과`(예 `C550:90:OK`)와
PPT 한 줄(`STEP 1 PASS(90.5) | STEP 2 FAIL(31.7) | ...`)을 출력합니다.
정상 사진만으로 기준을 만들고 벗어나면 NG로 보는 방식이며, 파일·제품 이름은 판정·학습·분할에 쓰지 않습니다.

## 설치
```
pip install -r requirements.txt
```

## 사용
데이터는 `canon_data/<기종>/<제품번호>/step_N.jpg` 구조입니다(저장소에 포함하지 않음).

```
# 학습: 제품의 약 20%를 시험 세트로 떼고 models/ 에 저장
py train_all.py --root "<데이터 경로>" --model-dir models
# 최종 배포용: 전체 제품으로 학습
py train_all.py --root "<데이터 경로>" --model-dir models_final --all
# 시험 세트(학습에 쓰지 않은 제품) 평가
py evaluate_holdout.py --root "<데이터 경로>" --model-dir models --out out_eval
# 새 사진 판독
py inspect_image.py --model-dir models_final --kind 3029C009AA --product "<제품 폴더>" --verbose
```

## 구성
- `inspector_lib.py`: 스텝별 학습·판독 함수와 결합 규칙
- `step_*.py`, `name_locate.py`: 스텝별 위치 탐색과 특징 계산
- `train_all.py`, `inspect_image.py`, `evaluate_holdout.py`: 학습, 판독, 평가
- `experiments/`: 실험과 진단용 스크립트
- `docs/report_draft.md`: 보고서 초안

## 주의
- 이 코드는 과제용이며 `docs/report_draft.md`에 방법, 근거, 한계를 적었습니다.
- AI(Claude)의 도움을 받아 작성했습니다.
