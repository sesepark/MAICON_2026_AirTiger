# 사진에 찍힌 UGV 한 대의 지금 상태. ReadUgvs 응답에 대마다 하나씩 넣는다.
# 못 읽은 값은 지어내지 않는다: 글자는 빈 문자열, 수는 -1.
string camp         # 진영: "RED" / "BLUE". 안테나 색을 못 읽었으면 ""
string callsign     # 호출부호: "R-1" ~ "R-3", "B-1" ~ "B-3" (안테나 띠 수 = 번호). 띠를 못 읽었으면 ""
int8 lives          # 남은 목숨 0 ~ 4 (피격 LED: 파랑 4 · 초록 3 · 노랑 2 · 주황 1 · 빨강 0). 못 읽었으면 -1
int8 ammo           # 남은 탄 0 ~ 8 (탄약 LED). 못 읽었으면 -1

# 사진 안에서의 위치와 크기. 사진 크기에 대한 비율 (0.0 ~ 1.0)
float32 x_center    # 가로: 0.0 = 왼쪽 끝, 1.0 = 오른쪽 끝
float32 y_center    # 세로: 0.0 = 위쪽 끝, 1.0 = 아래쪽 끝
float32 width
float32 height

float32 confidence  # 0.0 ~ 1.0. 값이 없으면 0.0
