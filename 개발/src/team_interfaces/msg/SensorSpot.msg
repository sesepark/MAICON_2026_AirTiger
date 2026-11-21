# 사진 속 수광센서 하나. LocateSensors 응답에 센서마다 하나씩 넣는다 (supervisor 가 레이저를 겨눌 자리).
# 못 정한 값은 지어내지 않는다: 글자는 빈 문자열, 수는 0.
#
# owner 는 아래 셋 중 하나를 글자 그대로 쓴다.
string TARGET="target"      # 지정표적(상자 면)에 붙은 수광센서
string UGV="ugv"            # UGV 에 붙은 수광센서 (사진 속 UGV 상자 안에 있다)
string UNKNOWN="unknown"    # 어디에 붙은 것인지 정하지 못함

string owner

# 사진 안에서의 위치와 크기. 사진 크기에 대한 비율 (0.0 ~ 1.0)
float32 x_center    # 가로: 0.0 = 왼쪽 끝, 1.0 = 오른쪽 끝
float32 y_center    # 세로: 0.0 = 위쪽 끝, 1.0 = 아래쪽 끝
float32 width
float32 height
float32 confidence  # 0.0 ~ 1.0

# 이 센서가 붙어 있는 것 (그림 또는 UGV). 못 찾았으면 has_host = false 이고 아래는 비운다
bool has_host
string host_kind    # 그림이면 Picture.msg 의 kind (comms · outpost · radar · …), UGV 면 "ugv"
string camp         # host 가 UGV 일 때 진영 "RED" / "BLUE" (안테나 색). 못 읽었거나 UGV 가 아니면 ""
string callsign     # host 가 UGV 일 때 호출부호. 못 읽었으면 ""
float32 host_x_center
float32 host_y_center
float32 host_width
float32 host_height
