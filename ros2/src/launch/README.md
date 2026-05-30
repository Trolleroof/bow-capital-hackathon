# OAK-D Optimized Launch Files for VSLAM

## Problem
The standard `calibration.launch.py` uses `RGBStereo` pipeline which publishes RGB + left + right stereo images, consuming excessive PoE bandwidth and resulting in very slow frame rates (~0.4 Hz).

## Solution
These launch files use RGB-only pipeline to minimize bandwidth usage for monocular SLAM.

## Usage

### Option 1: Custom Launch File (Recommended)
```bash
ros2 launch /home/dronelab/vikram/vslam-testing/src/launch/oak_rgb_only.launch.py
```

### Option 2: Standard Launch with Custom Config
```bash
# Standard bandwidth (720p @ 30fps)
ros2 launch depthai_ros_driver camera.launch.py \
    params_file:=/home/dronelab/vikram/vslam-testing/src/config/oak_rgb_minimal.yaml \
    rectify_rgb:=true

# Low bandwidth (480p @ 15fps)
ros2 launch depthai_ros_driver camera.launch.py \
    params_file:=/home/dronelab/vikram/vslam-testing/src/config/oak_rgb_low_bandwidth.yaml \
    rectify_rgb:=true
```

## Testing Frame Rate
After launching, check the publishing rate:
```bash
ros2 topic hz /oak/rgb/image_rect
```

Expected rates:
- **Before optimization**: ~0.4 Hz (very slow!)
- **After optimization**: 15-30 Hz (should be much better!)

## Published Topics
- `/oak/rgb/image_raw` - Raw RGB image
- `/oak/rgb/image_rect` - Rectified RGB image (use this for SLAM)
- `/oak/rgb/camera_info` - Camera calibration info

## Running with ORB-SLAM3
After launching the optimized camera node:
```bash
ros2 run ros2_orb_slam3 mono_node_cpp \
    --ros-args \
    -p image_topic:=/oak/rgb/image_rect \
    -p settings_name:=OAK
```

## Troubleshooting

### Still too slow?
1. Try the low bandwidth config (480p @ 15fps)
2. Reduce FPS further in the YAML config (try 10 fps)
3. Check PoE power budget and network switch capabilities

### No image output?
1. Check that camera is detected: `ros2 topic list | grep oak`
2. View raw image: `ros2 run rqt_image_view rqt_image_view /oak/rgb/image_raw`
3. Check camera permissions and PoE connection

## Key Optimizations
- **Pipeline**: RGB only (not RGBStereo) - saves 2 camera streams
- **No IMU**: Disabled for monocular SLAM
- **No IR**: Disabled IR illuminator
- **No NN**: No neural network processing
- **Rectification**: Only RGB rectification needed for SLAM
