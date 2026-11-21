# training/drone_boxes — 드론 map 그림의 상자 찾기 학습

라벨링 웹 화면(`../../tools/label_drone_frames.py`)에서 내보낸 데이터셋으로 **YOLO OBB**(Ultralytics) 모델을 학습한다.
그림은 전처리 `map`(경기장 테두리로 잘라 지도 좌표와 같게 편 612 × 500 px, 1 cm/px)이라, 모델이 찾은 상자 가운데가 곧 지도(`FieldMap`, frame `arena`)의 상자 자리다.
조명에 흔들리지 않게 학습 그림마다 밝기 · 색 · 그림자 · 연막 · 화질 변형을 여러 장 만들어 섞고, 평가도 조명 조건별로 따로 잰다.

이 폴더는 혼자 돈다 (ROS · perception 패키지를 import 하지 않는다) → 폴더째 학습 서버에 올린다.

| 파일 | 하는 일 |
|---|---|
| `push.sh` | (내 PC) 코드 + 데이터셋을 서버로 rsync |
| `setup_server.sh` | (서버, 처음 한 번) 가상환경 + torch(CUDA) + ultralytics |
| `run.sh` | (서버) 변형 → 학습 → 평가 → 내보내기를 뒤에서(nohup) 한 번에 |
| `pull.sh` | (내 PC) 결과(가중치 · 평가 · 그래프)를 `perception/data/drone_models/<이름>` 으로 |
| `train.py` | 학습 본체 (`run.sh` 가 부른다) |
| `augment.py` | 조명 강건성용 오프라인 변형 — 학습 그림만 늘리고 검증은 그대로. `--preview` 로 예시 그림 |
| `edge_cut.py` | 시야 끝 자르기 — 카메라 시야 끝(검정)을 안쪽으로 밀어 상자를 가로지르게 한다. 잘려도 상자 전체 라벨 |
| `photometric.py` | 변형 연산들 · 평가용 조명 조건(`STRESS`) |
| `evaluate.py` | 경기장 좌표(cm) 평가 + 조명 조건별 평가 |
| `config/augment.yaml` | 변형 종류 · 세기 · 확률 · 장수 |
| `config/train.yaml` | 모델 · 에폭 · 학습 중 변형 (Ultralytics `train()` 인자) |

## 데이터가 다 모이면 — 순서

```bash
# 0) 라벨링 웹 화면 → 「내보내기」 → perception/data/drone_datasets/<시각>_undist_map  (경기장 좌표 정답 fieldmaps/ 도 같이)

# 1) (내 PC, WSL) 변형이 너무 세거나 약하지 않은지 먼저 본다 — preview.jpg
~/.venvs/maicon_perception/bin/python augment.py <데이터셋> --preview 6 --out /tmp/aug_preview

# 2) (내 PC) 서버로 올리기 — ssh 대상은 user@호스트 또는 ~/.ssh/config 의 이름
bash push.sh <ssh 대상> <데이터셋>

# 3) (서버) 처음 한 번
ssh <ssh 대상>
bash ~/maicon_drone_train/code/setup_server.sh

# 4) (서버) 학습 — 뒤에서 돈다 (ssh 가 끊겨도 계속)
bash ~/maicon_drone_train/code/run.sh <데이터셋 이름>                 # 실행 이름은 시각
bash ~/maicon_drone_train/code/run.sh <데이터셋 이름> v1 epochs=200   # 이름 · 설정 덮어쓰기
tail -f ~/maicon_drone_train/code/runs/<실행 이름>.log

# 5) (내 PC) 결과 받기 — eval.md 가 바로 출력된다
bash pull.sh <ssh 대상> <실행 이름>
```

**팀 학습 PC (`Admin@<학습PC IP>`, RTX 5070 12 GB, Windows 의 WSL2 Ubuntu 26.04)**: setup 은 끝나 있다 (`~/.venvs/maicon_drone`, torch 2.15 cu132).
ssh 로 들어가면 시작 폴더가 `/mnt/c/Users/Admin` 이라 서버 폴더는 `~/maicon_drone_train` (Linux 쪽)으로 둔다 (push.sh · pull.sh 기본값).
**ssh 가 끊기면 WSL 이 통째로 꺼져 nohup · tmux 로 띄운 학습도 죽는다** → `FOREGROUND=1 bash run.sh ...` 로 ssh 를 열어 둔 채 돌린다.

서버에 인터넷이 없으면: 사전학습 가중치(`yolo11n-obb.pt`)를 PC 에서 받아 `WEIGHTS=yolo11n-obb.pt bash push.sh ...` 로 같이 올린다
(setup_server.sh 의 pip 설치도 인터넷이 필요하다 — 안 되면 휠을 따로 올려야 한다 [확인 필요]).
GPU 가 여럿이면 `run.sh ... device=1`. 메모리가 모자라면 `batch=8`.

## 조명 강건성

1. **오프라인 변형** (`augment.py`, `config/augment.yaml`) — 학습 그림 1장마다 `copies`(기본 4)장을 더 만든다. 연산마다 확률 `p` 로 섞는다.
   - 조명 모양: 한쪽 어두운 조명 · 둥근 밝은/어두운 자리 · 흐릿한 그림자 · 반사 얼룩
   - 노출 · 색: 노출 ×0.45 ~ 1.6 · 밝기 ± · 대비 · 감마 · 색온도(채널 배율) · 색상/채도/명도 흔들기
   - 연막: 한쪽에서 퍼지는 회백색 (중앙 측면 포그머신)
   - 화질: 흐림 · 잡음 · JPEG 재압축
   - 기하는 건드리지 않아 라벨을 그대로 쓴다. 시야 밖 검정 테두리는 검정으로 둔다. 검증 그림은 변형하지 않는다.
   - **시야 끝 자르기** (`edge_cut.py`, 설정 `edge_cut:`) — 변형 1장마다 확률 0.5 로, 그림 가장자리 30 % 안쪽 상자 하나를 골라 카메라 시야 끝
     (조금 기울고 휜 검정 경계)이 그 상자를 20 ~ 85 % 남기고 가로지르게 한다 (위 · 아래 0.35 씩, 좌 · 우 0.15 씩, 0.2 확률로 한쪽 더).
     map 그림은 경기장 좌표에 고정이라 그림을 잘라 옮기지(crop) 않고 시야 끝을 민다. 20 % 이상 남은 상자는 **상자 전체** 라벨 그대로
     (잘려도 가운데 = 지도 자리를 내게), 그보다 적게 남은 것은 그 자리를 마저 검게 칠하고 라벨을 뺀다. 조명 변형보다 먼저 건다.
     까닭: 2026-11-19 본선 항공뷰(0162.jpg)에서 화면 위쪽 끝에 윗면이 잘린 상자 3개를 v1 이 놓쳤다 — 학습 라벨에 절반 넘게 잘린 상자가 없었다.
   - 변형 데이터셋 폴더는 `<데이터셋><out_suffix>` (지금 `_aug_cut`). train.py 는 그 폴더가 있으면 다시 쓰므로 설정을 바꾸면 이름도 바꾼다.
2. **학습 중 변형** (`config/train.yaml`) — 색 흔들기(`hsv_*`), 뒤집기, 작은 회전(±5°) · 이동, 모자이크. 크기 변형(`scale`)은 작게 —
   map 그림에서 상자 윗면은 늘 40 × 40 px 라 그것이 강한 단서다.
3. **조명 조건별 평가** (`evaluate.py`, `photometric.STRESS`) — 검증 그림을 어둡게 · 밝게 · 한쪽 어둡게 · 그림자 · 반사 · 연막 · 따뜻한/차가운 조명 ·
   대비 낮음 · 흐림+잡음+압축으로 바꿔 각각 recall · precision 을 잰다. 「원본」 대비 많이 떨어지는 조건이 있으면 `augment.yaml` 에서 그 변형의
   `p` · 범위를 키워 다시 학습한다.

평가는 픽셀이 아니라 **경기장 좌표**로 한다: 예측 가운데가 정답 가운데에서 10 cm 안이면 맞음 (`--match-cm`), 가운데 오차를 cm 로.

## 결과 (`runs/<이름>/`)

- `export/drone_boxes_<이름>.pt` · `.onnx` — 쓸 가중치
- `eval_cutval/eval.md` — 시야 끝을 자른 검증 그림(`<데이터셋>_cutval`, 검증 1장마다 2장 · 조명 그대로)으로 잰 표 (edge_cut 설정이 있을 때).
  따로 만들기: `python augment.py <데이터셋> --cut-val <폴더>` → `python evaluate.py --weights ... --data <폴더> --no-stress`
- `eval_val/eval.md` — 조건별 · 클래스별 표, `eval_val/vis/` — 틀린 것이 많은 그림 (정답 초록 · 예측 빨강)
- `meta.json` — 데이터 장수 · 설정 · 변형 설정 · torch/ultralytics 판 · GPU · git 커밋 · 평가 요약
- `results.csv` · `*.png` — Ultralytics 학습 곡선

## 로봇(Jetson Orin Nano)에서 쓰기

- TensorRT `.engine` 은 GPU 마다 따로 만들어야 해서 **로봇에서** 만든다: `yolo export model=drone_boxes_<이름>.pt format=engine half=True imgsz=640`
  (perception 의 그림 모델 `pictures.engine` 과 같은 방식).
- 경기 중 입력도 같은 `map` 그림이어야 한다 — 드론 화면 → `drone_boundary` 경계 → `undistort_drone_frames.py` 의 `Warper`(map, 같은 카메라 높이) →
  모델 → 상자 가운데 (u/100, 5.00 − v/100). 지도 노드(`perception/map_maker.py`)에 붙이는 일은 아직 하지 않았다 (perception 담당과 정할 것).
- 카메라 높이: 학습 그림은 `data/drone_camera_heights.json` 의 카메라별 중앙값으로 폈다. 경기용 `main` 카메라는 아직 못 쟀다 (우리 경기 중에만 나온다).

## 확인한 것 (2026-11-19)

- 로컬 CPU(WSL, Python 3.14 · torch 2.15 CPU · ultralytics 8.4.215)에서 끝까지: 실제 연습 화면 36장(test1 · test2, 배치 1개)에 `drone_boxes` 로 만든
  가짜 라벨 → 내보내기 → 변형(×5) → 2 에폭 학습 → 평가 · 조명 조건 11개 → `.pt` · `.onnx`. 파이프라인이 도는지만 본 것이라 점수는 의미 없다.
- 시험: `python -m unittest discover -s test -p "test_drone_training.py"` (perception 폴더에서) — 변형 · 데이터셋 늘리기 · 좌표 평가.
- **GPU 서버에서는 아직 돌려 보지 않았다** (서버 주소 · GPU · CUDA 판 확인 필요).

## 학습 기록

| 실행 | 데이터 | 결과 (검증, 원본 조명) | 조명 조건 10개 | 비고 |
|---|---|---|---|---|
| `v1_yolo11n` (2026-11-19) | 라벨 59 장 (세션 5, test2 55 · test1 4) → 학습 44 장 × 변형 7 = 308 · 검증 15 장 (세션 3). 시야 밖 라벨 285 개 뺌 | 상자 recall 98.1 % · precision 98.6 % · 가운데 오차 평균 2.4 cm (최대 9.9). 줄무늬 블록 85.7 % · 75.8 % | recall 92.2 ~ 95.2 % (원본 94.6 %) — 대비 낮음이 −2.4 로 가장 큼 | yolo11n-obb, 174 에폭(최고 114), RTX 5070 에서 4.6 분. 상자 오답은 사람이 가린 장면, 줄무늬 오답은 그림 가장자리의 반쯤 보이는 블록 |
| `v2_edgecut` (2026-11-19) | v1 과 같은 라벨 · 나눔. 변형 308 장 중 118 장에 시야 끝 자르기 | 상자 recall 97.6 % · precision 97.2 % · 가운데 오차 평균 2.2 cm (최대 9.1). 줄무늬 블록 88.1 % · 78.7 %. **시야 끝 자른 검증 30 장**: 상자 recall 97.6 % (v1 91.0 %) — 잘린 상자만 보면 남은 비율 40 % 미만 14/16 (v1 0/16) · 40 ~ 70 % 45/45 (v1 31/45) | recall 94.2 ~ 95.2 % (원본 94.9 %) | yolo11n-obb, 174 에폭, RTX 5070 에서 6.2 분. 본선 항공뷰 0162.jpg (학습에 안 넣음): 상자 16 개 모두 · 헛것 0 (v1 13 개, 규칙 17 개 중 헛것 2 · 놓침 1). 잘리지 않은 검증 상자 점수는 v1 보다 조금 낮다 (98.1 → 97.6 · 98.6 → 97.2, 15 장이라 상자 한두 개 차이). 가중치 `export/drone_boxes_v2_edgecut.pt` sha256 0d9bf24e6075… |

검증이 15 장 · 연습 카메라 2 대뿐이고 조명 조건은 학습 변형과 같은 종류의 합성이다 → 본선 카메라(`main`) · 실제 다른 조명에서 다시 재야 한다.

## 라이선스

Ultralytics(코드 · 사전학습 가중치)는 AGPL-3.0 이다. perception 의 그림 모델도 이미 쓰고 있다. 발표 평가 때 GitHub 에 낼 때 출처 · 라이선스를 적는다.
