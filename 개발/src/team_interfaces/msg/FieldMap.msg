# Field frame and observation time. Coordinates and dimensions are in metres.
std_msgs/Header header
float64 width
float64 height
float64 box_size
# Box centres. The z coordinate is unused.
geometry_msgs/Point[] boxes
team_interfaces/FieldRegion[] bases
team_interfaces/FieldRegion[] sectors
float64 beam_depth 0.20
team_interfaces/FieldFace[] faces
team_interfaces/FieldVehicle[] vehicles
