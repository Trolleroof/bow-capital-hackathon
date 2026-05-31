# Testing the YOLOX Perception Node

This guide tests `combatos_perception/yolox_node.py` by itself on the Jetson, before involving the orchestrator or frontend.

## Prerequisites

- ROS2 Humble is installed and sourced.
- The OAK-D camera can publish an image topic such as `/oak/rgb/image_rect`.
- The `combatos_perception` package has been built in the ROS2 workspace.
- Python dependencies are installed in the environment used by ROS2:

```bash
pip install ultralytics opencv-python
```

For real YOLOX mode, install YOLOX and the Jetson-compatible PyTorch/CUDA stack in the same Python environment. For TensorRT deployment, provide a `.engine` model and use `model_type:=ultralytics`.

## Build the Package

From the repo on the Jetson:

```bash
cd /path/to/bow-capital-hackathon/ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select combatos_perception
source install/setup.bash
```

Verify ROS can see the executable:

```bash
ros2 run combatos_perception yolox_node --help
```

## Start an Image Source

In one terminal, start the OAK-D RGB stream:

```bash
cd /path/to/bow-capital-hackathon/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch src/launch/oak_rgb_only.launch.py
```

Confirm the image topic exists and is publishing:

```bash
ros2 topic list | grep oak
ros2 topic hz /oak/rgb/image_rect
```

You should see a steady frame rate. If `/oak/rgb/image_rect` is not present, use the actual image topic in the node commands below.

## Run the Node With the Checked-In YOLO Model

In a second terminal:

```bash
cd /path/to/bow-capital-hackathon/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run combatos_perception yolox_node --ros-args \
  -p image_topic:=/oak/rgb/image_rect \
  -p detections_topic:=/perception/detections \
  -p annotated_topic:=/perception/annotated_image \
  -p model_type:=ultralytics \
  -p model_path:=/path/to/bow-capital-hackathon/perception/yolo11n.pt \
  -p device:=0 \
  -p confidence:=0.40 \
  -p iou:=0.45 \
  -p fp16:=true \
  -p max_fps:=15.0
```

Expected startup log:

```text
subscribed /oak/rgb/image_rect; publishing /perception/detections and /perception/annotated_image
```

## Run the Node With a TensorRT Engine

Use this for Jetson deployment after exporting the model to TensorRT:

```bash
ros2 run combatos_perception yolox_node --ros-args \
  -p image_topic:=/oak/rgb/image_rect \
  -p model_type:=ultralytics \
  -p model_path:=/path/to/yolo11n.engine \
  -p device:=0 \
  -p confidence:=0.40 \
  -p fp16:=true \
  -p max_fps:=15.0
```

## Run the Node With Real YOLOX Code

Use this only when the YOLOX repo/package, checkpoint, and experiment file are present on the Jetson:

```bash
ros2 run combatos_perception yolox_node --ros-args \
  -p image_topic:=/oak/rgb/image_rect \
  -p model_type:=yolox \
  -p model_path:=/path/to/yolox_checkpoint.pth \
  -p yolox_exp_file:=/path/to/yolox_exp.py \
  -p device:=0 \
  -p confidence:=0.40 \
  -p iou:=0.45 \
  -p fp16:=true \
  -p max_fps:=15.0
```

If `model_type:=yolox` is used without `yolox_exp_file`, the node should fail fast because YOLOX needs the experiment file to construct the model.

## Verify Detection Output

In a third terminal:

```bash
source /opt/ros/humble/setup.bash
source /path/to/bow-capital-hackathon/ros2/install/setup.bash

ros2 topic echo /perception/detections
```

Expected message shape:

```json
{
  "schema": "combatos.perception.v1",
  "seq": 12,
  "frame_id": "oak_rgb_camera_optical_frame",
  "source": "/oak/rgb/image_rect",
  "width": 1280,
  "height": 720,
  "latency_ms": 22.4,
  "objects": [
    {
      "id": 1,
      "cls": "troop",
      "raw_cls": "person",
      "conf": 0.8123,
      "bbox": [0.12, 0.18, 0.22, 0.41],
      "bbox_px": [154, 130, 282, 295],
      "is_primary": false,
      "is_candidate": true,
      "confirmed": false
    }
  ]
}
```

The `bbox` field is normalized `[x, y, w, h]` in the `0.0-1.0` range. The `bbox_px` field is pixel `[x, y, w, h]` for debugging.

## Verify Annotated Image Output

Check publishing rate:

```bash
ros2 topic hz /perception/annotated_image
```

View the annotated image:

```bash
ros2 run rqt_image_view rqt_image_view /perception/annotated_image
```

You should see the OAK frame with bounding boxes and a small `YOLOX ... ms` status label at the bottom.

## Test Without the OAK Camera

If the OAK camera is unavailable, publish one static image repeatedly using any ROS2 image publisher available in your environment, then point `image_topic` at that topic. The node only requires `sensor_msgs/msg/Image` with an encoding `cv_bridge` can convert to `bgr8`.

One quick option is to run a small OpenCV image publisher separately and publish to `/test/image_raw`, then start the node with:

```bash
ros2 run combatos_perception yolox_node --ros-args \
  -p image_topic:=/test/image_raw \
  -p model_type:=ultralytics \
  -p model_path:=/path/to/bow-capital-hackathon/perception/yolo11n.pt
```

## Performance Checks

Use these commands while the node is running:

```bash
ros2 topic hz /perception/detections
ros2 topic hz /perception/annotated_image
tegrastats
```

Expected behavior:

- Detection topic rate should be near `max_fps` if inference is fast enough.
- Annotated image topic should match detection rate.
- CUDA/GPU activity should increase when `device:=0` and a CUDA-capable backend is working.

## Common Failures

`No module named ultralytics`:

Install dependencies in the Python environment used by ROS2.

`No module named yolox`:

Use `model_type:=ultralytics`, or install YOLOX and provide `yolox_exp_file`.

`cv_bridge decode failed`:

The image topic encoding is not convertible to `bgr8`. Check the topic with:

```bash
ros2 topic echo /oak/rgb/image_rect --once
```

No detections:

- Lower `confidence`, for example `-p confidence:=0.25`.
- Confirm the camera frame contains COCO-detectable objects.
- Verify the model path exists on the Jetson.

CUDA not being used:

- Confirm Jetson PyTorch or TensorRT is installed with CUDA support.
- Use `device:=0`, not `device:=cpu`.
- Watch `tegrastats` while inference is running.

