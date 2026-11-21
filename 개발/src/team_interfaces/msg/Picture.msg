# 사진 속 그림(상자 옆면에 붙은 그림) 하나. ClassifyImage 응답에서 왼쪽부터 나열한다.
#
# kind 는 아래 11가지 중 하나를 글자 그대로 쓴다.
#   객체 6종 (M3 보고값과 같은 글자)
string MILITARY_TRUCK="military_truck"
string MORTAR="mortar"
string TANK="tank"
string MISSILE="missile"
string AMMO_BOX="ammo_box"
string DRUM="drum"
#   지정표적 3종 (통신시설 · 감시초소 · 레이더시설). 어느 진영 것인지는 적지 않는다 — 면의 위치로 supervisor 가 정한다
string COMMS="comms"
string OUTPOST="outpost"
string RADAR="radar"
#   위 9가지가 아닌 그림 (나무 · 벙커 · 파괴건물 · 방호벽 등). 종류를 나누지 않고 전부 이 값
string NON_OBJECT="non_object"
#   그림은 보이는데 무엇인지 모름 (가장자리에서 잘림 · 가림 · 너무 비스듬함 · 점수가 낮음). 빼지 말고 이 값으로 한 칸을 채운다
string UNKNOWN="unknown"

string kind

# 사진 안에서의 위치와 크기. 사진 크기에 대한 비율 (0.0 ~ 1.0)
float32 x_center    # 가로: 0.0 = 왼쪽 끝, 1.0 = 오른쪽 끝
float32 y_center    # 세로: 0.0 = 위쪽 끝, 1.0 = 아래쪽 끝
float32 width       # 그림 가로 폭 / 사진 가로 폭
float32 height      # 그림 세로 높이 / 사진 세로 높이

float32 confidence  # 0.0 ~ 1.0. 값이 없으면 0.0
bool cut_off        # 그림이 사진 가장자리에 걸려 일부가 잘렸으면 true
