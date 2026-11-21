# 2026-11-20 거점 앞 우회 (03:28 판) — 기록 재생

서 있는 UGV 곁의 묵은 라이다 칸(`obstacles.sync_ugvs`)을 고치기 전 · 후로 견준 것. 설명은 `supervisor/docs/대기_자리_앞_우회_답_2026-11-20.md`.

1. 기록 받기: `bt/<판 id>/navigation.jsonl` · `navigation_slam.jsonl` (기록 사이트의 `navigation/debug.jsonl` · `navigation_slam/debug.jsonl`)
2. `NAV=<navigation 패키지 폴더> python replay_plan.py bt/<판 id> bt/<판 id>/old.json` — 고치기 전 코드의 폴더로 한 번, 고친 코드의 폴더로 `new.json` 한 번
3. `python compare.py` — 판마다 길이 달라진 순간 · 묵은 칸 수 · 서 있는 UGV 까지의 틈
4. `python box_faces.py` — 서서 본 스캔으로 잰 상자 면의 어긋남 (안의 navigation 경로는 고쳐 쓸 것)

스캔이 1 Hz 로만 남아 실제(10 Hz)보다 묵은 칸이 적게 쌓인다.

## 「서 있는 물체」 기준 (같은 날 04:15)

- `NAV=<navigation 폴더> python standing_rules.py` — 기준 후보마다 서 있는 물체로 치는 칸의 비율 (드론이 본 UGV 를 정답으로)
- `NAV=<navigation 폴더> python standing_decisions.py` — 목표 칸이 서 있는 물체에 덮였다고 보는 순간 · 좁은 틈 길에서 작게 보는 칸 수
