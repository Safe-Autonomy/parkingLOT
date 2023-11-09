################################################################
## Demo Code Only.
## Last update: 02/03/2023
## Auther: Hang Cui (hangcui3@illinois.edu)
################################################################

### Enable/Disable CAN Bus

$ sudo bash ~/Desktop/can_start.bash  
$ sudo bash ~/Desktop/can_stop.bash  

### GNSS-based waypoint follower with Stanley controller and RTK enabled  

$ cd ~/demo_ws/ && catkin_make  

$ source devel/setup.bash  
$ roslaunch basic_launch sensor_init.launch  

$ source devel/setup.bash  
$ roslaunch basic_launch visualization.launch  

$ source devel/setup.bash  
$ roslaunch basic_launch dbw_joystick.launch  

$ source devel/setup.bash  
$ roslaunch basic_launch gem_pacmod_control.launch  

$ source devel/setup.bash  
$ rosrun gem_gnss_control gem_gnss_tracker_stanley_rtk.py  

### GNSS-based waypoints follower with Pure Pursuit controller  

$ cd ~/demo_ws/ && catkin_make  

$ source devel/setup.bash  
$ roslaunch basic_launch sensor_init.launch  

$ source devel/setup.bash  
$ roslaunch basic_launch visualization.launch  

$ source devel/setup.bash  
$ roslaunch basic_launch dbw_joystick.launch  

$ source devel/setup.bash  
$ rosrun gem_gnss_control gem_gnss_tracker_pp.py  






