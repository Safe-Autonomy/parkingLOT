
FLAG_GAZEBO = True

# ---- GEM ----
if not FLAG_GAZEBO:

  # topics 
  CAMERA_TOPIC = '/zed2/zed_node/rgb/image_rect_color'
  CONTROL_TOPIC = '/gem/stanley_gnss_cmd'

  # GNSS <-> XY origin
  LAT_ORIGIN = 40.092722 
  LONG_ORIGIN = -88.236365

# ---- GAZEBO ----
else:

  # topics
  CAMERA_TOPIC = '/front_single_camera/image_raw'
  CONTROL_TOPIC = '/ackermann_cmd'

  # GNSS <-> XY origin
  LAT_ORIGIN = 0 
  LONG_ORIGIN = 0

MAP_IMG_WIDTH     = 2107   
MAP_IMG_HEIGHT    = 1313
MAP_IMG_LAT_SCALE = 0.00062 
MAP_IMG_LON_SCALE = 0.00136
