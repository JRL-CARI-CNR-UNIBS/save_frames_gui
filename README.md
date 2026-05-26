# save_frames_gui

ROS 2 Python package for real-time RGB/depth preview and manual frame capture from multiple cameras in parallel.

## Features

- Modern Qt GUI with dark theme.
- Top tabs for `Camera rs1`, `Camera rs2`, `Camera rs3`, etc.
- Real-time RGB and depth preview for each configured camera.
- Runtime dataset directory selection.
- RGB formats: `png`, `jpg`, `bmp`, `tiff`.
- Depth formats: `npy`, `png16`, `tiff32`, `exr`.
- Save modes: `RGB only`, `RGB + depth_m`, `Depth only`.
- Button to save the active camera frame.
- Button to save all currently available camera frames.
- Best-effort all-camera saving: unavailable cameras or unavailable requested modalities are skipped and reported, while available data is still saved.
- JSON metadata saved with topic names, ROS header stamps (`sec`, `nanosec`, `float_sec`), ROS `frame_id`, encodings, camera info, requested mode, actual saved modalities, and warnings.

## Dependencies

Example for ROS 2 Humble/Iron/Jazzy on Ubuntu:

```bash
sudo apt update
sudo apt install \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-message-filters \
  python3-pyqt5 \
  python3-opencv \
  python3-numpy \
  python3-yaml
```

## Installation

```bash
cd ~/ros2_ws/src
unzip /path/to/save_frames_gui.zip
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch save_frames_gui save_frames_gui.launch.py
```

With a custom config file:

```bash
ros2 launch save_frames_gui save_frames_gui.launch.py config_file:=/absolute/path/to/config.yaml
```

## Configuration

The default configuration is in `config/save_frames_gui.yaml`.

Important note: the YAML file is read directly by the node as a regular YAML file. The launch file only passes the `config_file` parameter to ROS. This allows camera sections such as `rs1:`, `rs2:`, and `rs3:` to stay at the YAML top level, as requested, without forcing ROS 2 `ros__parameters` syntax for every camera block.

```yaml
save_frames_gui_node:
  ros__parameters:
    cameras: [rs1, rs2, rs3]
    dataset_dir: "/tmp/vision_system_dataset"
    ui_refresh_rate_hz: 30.0
    sync_queue_size: 10
    sync_slop_sec: 0.05

rs1:
  color_image_topic: "/rs1/rs1/color/image_raw"
  depth_image_topic: "/rs1/rs1/aligned_depth_to_color/image_raw"
  camera_info_topic: "/rs1/rs1/aligned_depth_to_color/camera_info"
  frames_approx_sync: false
  depth_frame_encoding: "32FC1"
  depth_unit_in_meters: true
```

## Synchronization

- `frames_approx_sync: false` uses `message_filters.TimeSynchronizer`, so RGB and depth timestamps must match exactly.
- If the GUI stays on "Waiting for frames" with RealSense cameras or drivers that are not perfectly synchronized, set `frames_approx_sync: true`.
- `sync_slop_sec` is used only with approximate synchronization.

## Saved dataset structure

Each camera gets its own directory. RGB, depth, and metadata are separated into `color/`, `depth/`, and `meta/` subdirectories.

Example:

```text
/tmp/vision_system_dataset/
  rs1/
    color/
      rs1_20260525_153012_421_rgb.png
    depth/
      rs1_20260525_153012_421_depth_m.npy
    meta/
      rs1_20260525_153012_421_meta.json
  rs2/
    color/
      rs2_20260525_153012_427_rgb.png
    depth/
      rs2_20260525_153012_427_depth_m.npy
    meta/
      rs2_20260525_153012_427_meta.json
```

The timestamp in the filename is the local save time with millisecond precision: `YYYYMMDD_HHMMSS_mmm`. Exact ROS message timestamps are stored in the metadata.

## Metadata timestamp format

Each saved sample writes ROS header timestamps without reducing them to a float-only representation.
For both color and depth, the metadata contains only these timestamp fields:

```json
"color_stamp": {
  "sec": 1779714743,
  "nanosec": 16073680,
  "float_sec": 1779714743.0160737
},
"depth_stamp": {
  "sec": 1779714743,
  "nanosec": 16891234,
  "float_sec": 1779714743.0168912
},
"color_frame_id": "rs1_color_optical_frame",
"depth_frame_id": "rs1_color_optical_frame"
```

Use `(sec, nanosec)` for exact matching against rosbag messages. `float_sec` is included only for convenience in plotting, sorting, or approximate matching.

## All-camera save behavior

`Save all available frames` is intentionally best-effort:

- If a camera has the requested frame data, it is saved.
- If a camera has no requested data yet, it is skipped.
- If `RGB + depth_m` is selected and only RGB or only depth is available for a camera, the available requested modality is saved and the metadata reports the missing one.
- A warning dialog summarizes skipped cameras and missing modalities.

The active-camera save button remains strict: if the selected mode requires RGB/depth and that modality is missing, it reports the error instead of silently saving a partial sample.

## Depth formats

- `npy`: saves `float32` in meters, recommended for numeric datasets.
- `png16`: saves `uint16` in millimeters, compatible with many RGB-D tools.
- `tiff32`: saves `float32` in meters.
- `exr`: saves `float32` in meters, but requires OpenCV built with OpenEXR enabled.


## Qt/OpenCV note

For pip-based environments, prefer `opencv-python-headless` instead of `opencv-python`. The GUI uses PyQt for display, and OpenCV is only used for image conversion/saving. This avoids OpenCV's bundled Qt plugin path from overriding PyQt's platform plugin. The node also resets the Qt platform plugin path before creating `QApplication` as a defensive measure.

## Use with vision_system

This package does not require `vision_system`: it directly consumes standard ROS 2 `sensor_msgs/Image` and `sensor_msgs/CameraInfo` topics.

`vision_system` can still be used upstream for camera acquisition or camera management. This GUI can subscribe to the topics published/configured by that pipeline. The dependency is kept optional so the tool remains generic and compatible with RealSense, simulators, rosbag playback, and other ROS 2 drivers.


## Trigger service

The node exposes a `std_srvs/srv/Trigger` service that performs the same action as the **Save all available frames** button.

Default service name:

```bash
/save_frames_gui_node/save_all_frames
```

Call it with:

```bash
ros2 service call /save_frames_gui_node/save_all_frames std_srvs/srv/Trigger {}
```

The service uses the current save settings mirrored from the GUI: dataset directory, RGB format, depth format, and save mode. Its behavior is best-effort: it saves all cameras/modalities currently available and reports skipped cameras or missing frames in the response message.

You can change the service name in `config/save_frames_gui.yaml`:

```yaml
save_frames_gui_node:
  ros__parameters:
    save_all_service_name: "~/save_all_frames"
```
