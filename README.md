# parkingLOT: Parking with Lidar Optimal Trajectory

### Problem Statement

Parking is hard for human, so why don't we build a robot that does parking for us? In this project, we implement a parking system that ensures safety for both the vehicle, and the pedestrians around it.

### Install:

Clone the repo recursively to a catkin workspace (named `GEM-01` for our case):

```
cd GEM-01/src
git clone --recursive-submodules https://github.com/Safe-Autonomy/parkingLOT.git
cd parkingLOT/
pip install -v -e .
```

Build it!

```
cd .. # make sure to be in the src/ folder, not sure if needed
catkin build
```

Make sure to download the pre-trained YoloPv2 weights from their git repo into `YOLOPv2/data/weights`

### Usage: 

#### Real-world GEM Vehicle

Run these (on separate terminals, make sure to first `source devel/setup.bash` in each)

```
roslaunch basic_launch sensor_init.launch
roslaunch basic_launch visualization.launch
```

On another terminal, run `parkingLOT`:

```
roslaunch basic_launch lane_following.launch
```

#### Gazebo Simulation 

TODO

### Credit:

This project is the final project of group GEM-1 in [ECE484: Principle of Safe Autonomy](https://publish.illinois.edu/safe-autonomy/) (Fall 2023). Developers are (of equal contribution): [Zirui Wang](https://github.com/Ziruiwang409), [Anuj Sesha](https://github.com/a-sesha), [Henry Che](https://github.com/hungdche).