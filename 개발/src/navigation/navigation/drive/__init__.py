"""주행 노드(navigation)의 ROS 비의존 핵심 로직.

- ``geometry``     자세 계산 공용 함수
- ``params``       튜닝값 한곳 (ROS 파라미터로 그대로 노출)
- ``fieldmap``     FieldMap 메시지에서 꺼낸 경기장 기하(벽·박스·다른 UGV)
- ``grid``         5 cm 점유 격자, 부풀리기, A*, 임시 장애물
- ``pose_tracker`` SLAM 위치 + /odom 거리 + /imu/data 방향으로 사이를 메움, 위치 상태 단계
- ``obstacles``    /scan 에서 지도에 없는 물체를 뽑고 임시 장애물·충돌 정지 판단
- ``executor``     목표 하나를 실행하는 상태 기계 (경로 따라가기·회전·직진·돌진·막힘)
- ``trail``        지나온 길(로컬 궤적) — 막혔을 때 돌지 않고 뒤로 되짚어 물러난다 · 「뒤가 막혔다」 는 기억

rclpy 없이 단위 시험이 돈다. ROS 연결은 ``navigation/driving_node.py`` 에만 있다.
"""
