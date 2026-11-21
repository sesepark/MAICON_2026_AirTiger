# header.stamp는 결과 생성 시각. 각 관측의 원본 시각은 observations[].observed_at.
std_msgs/Header header
string run_id
bool processing_healthy
string detail
team_interfaces/Observation[] observations
