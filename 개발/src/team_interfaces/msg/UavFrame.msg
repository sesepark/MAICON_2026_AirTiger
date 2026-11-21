# header.stamp는 ROS 수신 시각. 원본 HTTP 촬영 시각은 따로 보존.
std_msgs/Header header
string run_id
string captured_at_utc
bool capture_time_known
sensor_msgs/CompressedImage image
