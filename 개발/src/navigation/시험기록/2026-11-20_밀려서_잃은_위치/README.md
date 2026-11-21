# 2026-11-20 06:17 판: 밀려서 잃은 위치

기록(경기 보기 사이트의 navigation/debug.jsonl · navigation_slam/debug.jsonl)을 한 폴더에 navigation.jsonl · navigation_slam.jsonl 로 받아 두고 돌린다.
환경 변수 NAV = navigation 패키지 폴더. 설명과 결과는 navigation/README.md 의 「밀려서 방향을 잃은 뒤」.

- best_fit_pose.py <폴더> <경기 시작 t> <부터 s> <까지 s> [반경] — 스캔마다 마지막 자리 둘레에서 가장 잘 맞는 자세 셋
- search_all_headings.py <폴더> <t0> <부터> <까지> <반 폭> <몇 장마다> <x,y | logged> — 한 자리 둘레를 모든 방향으로 (slot_search)
- replay_relocate.py <폴더> <t0> [--no-drone] — 기록의 relocate_lost 를 새 규칙으로
- drone_map_share.py — 스캔이 드론 지도에 맞는 비율 (폴더 · 시각은 파일 안에 적혀 있다)
- replay_node_relocate.py lost|normal <폴더> <t0> ... — 진짜 노드 코드(_relocate · _fits_drone_map)에 기록의 스캔을 넣는다 (파일 머리의 설명)
