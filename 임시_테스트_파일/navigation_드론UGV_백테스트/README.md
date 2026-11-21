# 드론 UGV 자리 — 공식 테스트 경기 기록으로 되짚어 보기 (2026-11-19)

드론이 본 UGV 자리를 주행에 쓰는 코드(`navigation/drive/ugvs.py`, `perception/drone_ugvs.py`)를 2026-11-19 16:55 ~ 18:06 공식 테스트 경기 6판의 기록에 대 본 것.
결과는 [navigation README](../../개발/src/navigation/README.md) 「드론이 본 UGV 자리 쓰기」 에 적었다.

| 파일 | 하는 일 |
|---|---|
| `bt_detect.py <카메라 높이>` | 경기별 본선 카메라 드론 사진(라벨링 세션 `perception/data/drone_sessions/match_*`)에 실제 모델을 돌려 `bt/det_<경기>.json` 으로 (바닥 좌표 · 4.80 지도) |
| `bt_replay.py <UGV 높이> [-v]` | 그 자리 + 그 경기의 `navigation/debug.jsonl`(기록 사이트 `/files/<경기>/…`) → 우리 차 고르기 · BLOCKED 마다 막은 것이 무엇이었나 |
| `bt_self.py <UGV 높이>` | 우리가 화면 안에 있을 때 가장 가까운 드론 UGV 까지의 거리 분포 |

- 일회용이다: 경로가 만든 PC 의 것으로 박혀 있다 (드론 사진 폴더 · `runs/<경기>/navigation_debug.jsonl` · ultralytics 를 깐 가상환경). 다시 돌리려면 경로를 고친다.
- 그 판들은 그때의 코드로 돌았다. 판마다 지도 세로(4.80 · 5.00)를 기록에서 읽어 맞추고, 옛 빌드의 「우리 몸에 찍힌 점」 은 지금 빌드처럼 뺀다.
  걸린 시간 · 기억 칸 · 헛돎 · 지도 고치기의 결과는 되돌려 보지 않는다 (그때의 코드에 묶여 있다).
