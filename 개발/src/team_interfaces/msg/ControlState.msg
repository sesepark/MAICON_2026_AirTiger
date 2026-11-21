# 동작 허용 임대. true 값의 영구 보존·재사용 금지.
std_msgs/Header header
string run_id
uint64 sequence
bool motion_allowed
bool aiming_allowed
bool firing_allowed
uint32 lease_ms
string reason
