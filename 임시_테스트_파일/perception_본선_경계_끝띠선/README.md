# 본선 카메라: 드론 UGV 자리 · 경계 찾기 점검 (2026-11-20)

「드론이 본 UGV 자리가 본선 드론캠에서도 강건한가」 를 2026-11-19 16:55 ~ 18:06 공식 테스트 경기 여섯 판(본선 카메라 사진 819장)으로 되돌려 본 것.
결과는 [perception README](../../개발/src/perception/README.md) 「본선 카메라에서 끝 띠 선 대신 거점 테두리 선을 고르던 것」 과
[navigation README](../../개발/src/navigation/README.md) 「본선 카메라에서도 드론 UGV 자리가 연습 경기장만큼 맞는가」 에 적었다.

| 파일 | 하는 일 |
|---|---|
| `all_photos.py <perception 패키지 폴더> <출력.json>` | 사진 819장 모두에 `detect_boundary` → 사진마다 고른 선 · 렌즈 · 걸린 시간 |
| `summary.py <패키지 폴더> <json...>` | 위 결과의 요약: 통과 · 렌즈 · 끝 띠 선 · 시간, 같은 화면 픽셀을 옮긴 자리의 퍼짐 |
| `line_lengths.py <패키지 폴더>` | 끝 띠 선(u ≈ 349 · 952)과 거점 테두리 선(u ≈ 972)의 따라간 점 수 |
| `maps.py <패키지 폴더> <출력.json>` · `map_compare.py` | 경기 첫 사진 4장씩의 지도(상자 모델) → 상자 수 · 좌우 맞보는 상자의 어긋남 · 사진 사이 퍼짐 (전 · 후) |
| `ugv_detect.py <출력 폴더>` | 경기 첫 사진 4장 각각의 경계로 사진 모두에서 UGV 찾기 (`perception.drone_ugvs`, 실제 모델) |
| `ugv_replay.py <경계 이름> [점수] [폴더]` | 그 자리를 지금의 `UgvTracks` · `slam_node._drone_self` 에 넣기 + 라이다 원본(`scan_raw`)과 대조 |
| `ugv_still_offset.py` · `ugv_start_photo_spread.py` | 서 있을 때 드론 자리 − 라이다 위치 (자리별) · 시작 사진에 따른 좌표 차이 |
| `practice_runs_to_same_format.py <폴더>` | 연습 경기장 실차 여섯 판의 `perception/ugvs.jsonl` 을 같은 꼴로 (견줄 기준) |
| `photo_intervals.py <matches.json>` · `thin_to_5s.py` | 판마다 사진 간격이 바뀐 때 · 받은 시각 − 찍은 시각, 1 s 판을 2 s · 5 s 로 솎아 넣기 |

- 일회용이다: 경로가 만든 PC 의 것으로 박혀 있다 (드론 사진 `perception/data/drone_sessions/match_*` — 저장소에 없다, 경기 기록 `logs/<경기>/…` — 기록 사이트의
  `/files/<경기>/navigation/debug.jsonl` 등, ultralytics 를 깐 가상환경). 다시 돌리려면 경로를 고친다.
- 사진 목록(`index.jsonl`)은 시간순이 아니다 — `wall`(로봇이 받은 시각)로 정렬해 쓴다. `match_t` 는 몇 판에서 틀리다.
- 그 판들은 그때의 코드로 돌았다 (자세 · 지도가 지금과 다르다). 견준 것은 빌드와 무관한 양뿐이다: 드론 좌표, 사진 간격, 라이다 원본.
