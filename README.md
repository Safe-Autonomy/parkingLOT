# parkingLOT: Parking with Lidar Optimal Trajectory

### Problem Statement

Parking is hard for human, so why don't we build a robot that does parking for us? In this project, we implement a parking system that ensures safety for both the vehicle, and the pedestrians around it.

### Install:

```
git clone --recursive-submodules https://github.com/Safe-Autonomy/parkingLOT.git
cd parkingLOT/
pip install -v -e .
```

### Usage: 

Clone this repo into `vehicle_drivers` and either the [hardware_drivers](https://github.com/Safe-Autonomy/hardware-drivers) or the [simulator](https://github.com/Safe-Autonomy/simulator) repo depends on use cases. 

For real-world hardware drivers run, run these (on separate terminals, make sure to first `source devel/setup.bash` in each)

```
roslaunch basic_launch sensor_init.launch
roslaunch basic_launch visualization.launch
```

For simulators run, run these (on separate terminals, make sure to first `source devel/setup.bash` in each)

```
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=-1.5 y:=-21 yaw:=3.1416
```

On another terminal, run `parkingLOT`:

```
python runner/lane_following.py [--gem]
```

### Credit:

This project is the final project of group GEM-1 in [ECE484: Principle of Safe Autonomy](https://publish.illinois.edu/safe-autonomy/) (Fall 2023). Developers are (of equal contribution): [Zirui Wang](https://github.com/Ziruiwang409), [Anuj Sesha](https://github.com/a-sesha), [Henry Che](https://github.com/hungdche).