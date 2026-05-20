# Camera calibration and 3D reconstruction (calib3d module)

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Create calibration pattern](camera_calibration_pattern.md)
  - Catalogue of OpenCV-supported calibration patterns — chessboard, circles grid (symmetric and asymmetric), and ChAruco — with strengths, pitfalls, and detection notes.
* - [Camera calibration with square chessboard](camera_calibration_square_chess.md)
  - Camera calibration from chessboard images and pose estimation against any object with known 3D geometry.
* - [Camera calibration With OpenCV](camera_calibration.md)
  - Camera-matrix and distortion-coefficient recovery with `cv::calibrateCameraRO`, including radial/tangential distortion theory and XML/YAML output.
* - [Real Time pose estimation of a textured object](real_time_pose.md)
  - Real-time 6-DoF pose tracking of a textured object using ORB features, FLANN matching, PnP + RANSAC, and a Kalman filter.
* - [Interactive camera calibration application](interactive_calibration.md)
  - Interactive sample app with auto pattern capture, live re-projection error, and adaptive flag tuning.
* - [Multi-view camera calibration tutorial](multiview_calibration.md)
  - Intrinsics and pairwise-extrinsics calibration for a set of synchronized cameras observing a shared calibration pattern.
* - [USAC: Improvement of Random Sample Consensus in OpenCV](usac.md)
  - USAC RANSAC framework in the `3d` module — sampling (PROSAC, NAPSAC), verification, and local-optimisation strategies.
```

```{toctree}
:hidden:
:maxdepth: 1

camera_calibration_pattern
camera_calibration_square_chess
camera_calibration
real_time_pose
interactive_calibration
multiview_calibration
usac
```
