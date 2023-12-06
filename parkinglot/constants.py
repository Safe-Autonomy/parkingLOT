
FLAG_GAZEBO = False

# ---- GEM ----
if not FLAG_GAZEBO:

  # topics 
  CAMERA_TOPIC = '/zed2/zed_node/rgb/image_rect_color'
  CONTROL_TOPIC = '/gem/stanley_gnss_cmd'
  LIDAR_TOPIC = '/lidar1/velodyne_points'

  # GNSS <-> XY origin
  LAT_ORIGIN = 40.092722 
  LONG_ORIGIN = -88.236365

  MAP_IMG_WIDTH     = 2107   
  MAP_IMG_HEIGHT    = 1313
  MAP_IMG_LAT_SCALE = 0.00062 
  MAP_IMG_LON_SCALE = 0.00136

# ---- GAZEBO ----
else:

  # topics
  CAMERA_TOPIC = '/front_single_camera/image_raw'
  CONTROL_TOPIC = '/ackermann_cmd'
  LIDAR_TOPIC = '/velodyne_points'

  # GNSS <-> XY origin
  LAT_ORIGIN = -30 
  LONG_ORIGIN = -40

  MAP_IMG_WIDTH     = 800   
  MAP_IMG_HEIGHT    = 300
  MAP_IMG_LAT_SCALE = 0.1 
  MAP_IMG_LON_SCALE = 0.1


