Multi-view Camera Calibration Tutorial {#tutorial_multiview_camera_calibration}
==========================

@tableofcontents

@prev_tutorial{tutorial_interactive_calibration}
@next_tutorial{tutorial_usac}

|    |    |
| -: | :- |
| Original author | Maksym Ivashechkin |
| Compatibility | OpenCV >= 5.0 |

Structure:
----
This tutorial consists of the following sections:
* Introduction
* Briefly
* How to run
* Python example
* Python visualization
* Details Of The Algorithm
* Method Input
* Method Output
* Method Input
* Pseudocode
* Python sample API
* C++ sample API

Introduction
----
Multiview calibration is a very important task in computer vision. It is widely used in 3D reconstruction, structure from motion, autonomous driving etc. The calibration procedure is often the first step for any vision task that must be done to obtain intrinsics and extrinsics parameters of the cameras. The accuracy of camera calibration parameters directly influence all further computations and results, hence, estimating precise intrinsincs and extrinsics is crucial.

The calibration algorithms require a set of images for each camera, where on the images a calibration pattern (e.g., checkerboard, aruco etc) is visible and detected. Additionally, to get results with a real scale, the 3D distance between two neighbor points of the calibration pattern grid should be measured. For extrinsics calibration, images must share the calibration pattern obtained from different views, i.e., overlap of cameras' field of view. Moreover, images that share the pattern grid have to be taken at the same moment of time, or in other words, cameras must be synchronized. Otherwise, the extrinsics calibration will fail.

The intrinsics calibration incorporates estimation of focal lengths, skew, and principal point of the camera; these parameters are combined in the intrinsic upper triangular matrix of size 3x3. Additionally, intrinsic calibration includes finding distortion parameters of the camera. The extrinsics parameters represent a relative rotation and translation between two cameras. Therefore, for \f$N\f$ cameras, a sufficient amount of correctly selected pairs of estimated relative rotations and translations is \f$N-1\f$, while extrinsics parameters for all possible pairs \f$N^2 = N * (N-1) / 2\f$ could be derived from those that are estimated. More details about intrinsics calibration could be found in this tutorial @ref tutorial_camera_calibration_pattern, and its implementation @ref cv::calibrateCamera.

After intrinsics and extrinsics calibration, the projection matrices of cameras are found by combing intrinsic, rotation matrices and translation. The projection matrices enable doing triangulation (3D reconstruction), rectification, finding epipolar geometry etc.

The following sections describes the individual algorithmic steps of the overall multi-camera calibration pipeline:

Briefly
----
The algorithm consists of three major steps that could be enumerated as follows:

1. Calibrate intrinsics parameters (intrinsic matrix and distortion coefficients) for each camera independently.
2. Calibrate pairwise cameras (using stereo calibration) using intrinsics parameters from the step 1.
3. Do global optimization using all cameras simultaneously to refine extrinsic parameters.


How to run:
====

Assume we have `N` camera views, for each `i`-th view there are `M` images containing pattern points (e.g., checkerboard).

Python example
--
There are two options to run the sample code in Python (`opencv/samples/python/multiview_calibration.py`) either with raw images or provided points.
The first option is to prepare `N` files where each file has path to image per line (images of a specific camera of the corresponding file). For example, a file for camera `i` should look like (`file_i.txt`):
```
/path/to/image_1_of_camera_i
...
/path/to/image_M_of_camera_i
```

Then sample program could be run via command line as follows:
```console
$ python3 multiview_calibration.py --pattern_size W,H --pattern_type TYPE --fisheye IS_FISHEYE_1,...,IS_FISHEYE_N \
--pattern_distance DIST --filenames /path/to/file_1.txt,...,/path/to/file_N.txt
```

Replace `W` and `H` with size of the pattern points, `TYPE` with name of a type of the calibration grid (supported patterns: `checkerboard`, `circles`, `acircles`), `IS_FISHEYE` corresponds to the camera type (1 - is fisheye, 0 - pinhole), `DIST` is pattern distance (i.e., distance between two cells of checkerboard).
The sample script automatically detects image points accordingly to the specified pattern type. By default detection is done in parallel, but this option could be turned off.

Additional (optional) flags to Python sample that could be used are as follows:
* `--winsize` - pass values `H,W` to define window size for corners detection (default is 5,5).
* `--debug_corners` - pass `True` or `False`. If `True` program shows several random images with detected corners for user to manually verify the detection (default is `False`).
* `--points_json_file` - pass name of JSON file where image and pattern points could be saved after detection. Later this file could be used to run sample code. Default value is '' (nothing is saved).
* `--find_intrinsics_in_python` - pass `0` or `1`. If `1` then the Python sample automatically calibrates intrinsics parameters and reports reprojection errors. The multiview calibration is done only for extrinsics parameters. This flag aims to separate calibration process and make it easier to debug what goes wrong.
* `--path_to_save` - path to save results in pickle file
* `--path_to_visualize` - path to results pickle file needed to run visualization
* `--visualize` - visualization flag (True or False), if True only runs visualization but path_to_visualize must be provided
* `--resize_image_detection` - True / False, if True an image will be resized to speed-up corners detection

Alternatively, the Python sample could be run from JSON file that should contain image points, pattern points, and boolean indicator whether a camera is fisheye.
The example of JSON file is in `opencv_extra/testdata/python/multiview_calibration_data.json` (currently under pull request 1001 in `opencv_extra`). Its format should be dictionary with the following items:
* `object_points` - list of lists of pattern (object) points (size NUM_POINTS x 3).
* `image_points` - list of lists of lists of lists of image points (size NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2).
* `image_sizes` - list of tuples (width x height) of image size.
* `is_fisheye` - list of boolean values (true - fisheye camera, false - otherwise).
Optionally:
* `Ks` and `distortions` - intrinsics parameters. If they are provided in JSON file then the proposed method does not estimate intrinsics parameters. `Ks` (intrinsic matrices) is list of lists of lists (NUM_CAMERAS x 3 x 3), `distortions` is list of lists (NUM_CAMERAS x NUM_VALUES) of distortion parameters.
* `images_names` - list of lists (NUM_CAMERAS x NUM_FRAMES x string) of image filenames for visualization of points after calibration.

```console
$ python3 multiview_calibration.py --json_file /path/to/json
```

The description of flags could be found directly by running the sample script with `help` option:
```console
python3 multiview_calibration.py --help
```

The expected output in Linux terminal for `multiview_calibration_images` data (from `opencv_extra/testdata/python/` generated in Blender) should be the following:
![](camera_multiview_calibration/images/terminal-demo.jpg)

The expected output for real-life calibration images in `opencv_extra/testdata/python/real_multiview_calibration_images` is the following:
![](camera_multiview_calibration/images/terminal-real-demo.jpg)


Python visualization
----

Apart from estimated extrinsics / intrinsics, the python sample provides a comprehensive visualization.
Firstly, the sample shows positions of cameras, checkerboard (of a random frame), and pairs of cameras connected by black lines explicitly demonstrating tuples that were used in the initial stage of stereo calibration.
If images are not known, then a simple plot with arrows (from given point to the back-projected one) visualizing errors is shown. The color of arrows highlights the error values. Additionally, the title reports mean error on this frame, and its accuracy among other frames used in calibration.
The following test instances were synthetically generated (see `opencv/apps/python-calibration-generator/calibration_generator.py`):

![](camera_multiview_calibration/images/1.jpg)

![](camera_multiview_calibration/images/2.jpg)

This instance has large Gaussian points noise.

![](camera_multiview_calibration/images/3.jpg)

Another example, with more complex tree structure is here, it shows a weak connection between two groups of cameras.

![](camera_multiview_calibration/images/4.jpg)

If files to images are provided, then the output is an image with plotted arrows:

![](camera_multiview_calibration/images/checkerboard.png)


Details Of The Algorithm
----
1. If the intrinsics are not provided, the calibration procedure starts intrinsics calibration independently for each camera using OpenCV function @ref cv::calibrateCamera.
* a. If input is a combination of fisheye and pinhole cameras, then fisheye images are calibrated with the default OpenCV calibrate function. The reason is that stereo calibration in OpenCV does not support a mix of fisheye and pinhole cameras. The following flags are used in this scenario;
* * i. @ref cv::CALIB_RATIONAL_MODEL - it extends default (5 coefficients) distortion model and returns more parameters.
* * ii. @ref cv::CALIB_ZERO_TANGENT_DIST - it zeroes out tangential distortion coefficients, since the fisheye model does not have them.
* * iii. @ref cv::CALIB_FIX_K5, @ref cv::CALIB_FIX_K6 - it zeroes out the fifth and sixth parameter, so in total 4 parameters are returned.
* b. Output of intrinsic calibration is also rotation, translation vectors (transform of pattern points to camera frame), and errors per frame.
* * i. For each frame, the index of the camera with the lowest error among all cameras is saved.
2. Otherwise, if intrinsics are known, then the proposed algorithm runs perspective-n-point estimation (@ref cv::solvePnP) to estimate rotation and translation vectors, and reprojection error for each frame.
3. Assume that cameras can be represented as nodes of a connected graph. An edge between two cameras is created if there is any image overlap over all frames. If the graph does not connect all cameras (i.e., exists a camera that has no overlap with other cameras) then calibration is not possible. Otherwise, the next step consists of finding the [maximum spanning tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree) (MST) of this graph. The MST captures all best pairwise camera connections. The weight of edges across all frames is a weighted combination of multiple factors:
* a. The main contribution is a number of pattern points detected in both images (cameras).
* b. Ratio of area of convex hull of projected points in the image to the image resolution.
* c. Angle between cameras' optical axes (found from rotation vectors).
* d. Angle between the camera's optical axis and the pattern's normal vector (found from 3 non-collinear pattern's points).
4. The initial estimate of cameras' extrinsics is found by pairwise stereo calibration (see @ref cv::stereoCalibrate). Without loss of generality, the 0-th camera’s rotation is fixed to identity and translation to zero vector, and the 0-th node becomes the root of the MST. The order of stereo calibration is selected by traversing MST in breadth first search, starting from the root. The total number of pairs (also number of edges of tree) is NUM_CAMERAS - 1, which is property of a tree graph.
5. Given the initial estimate of extrinsics the aim is to polish results using global optimization (via Levenberq-Marquardt method, see @ref cv::LevMarq class).
* a. To reduce the total number of parameters, all rotation and translation vectors estimated in the first step from intrinsics calibration with the lowest error are transformed to be relative with respect to the root camera.
* b. The total number of parameters is (NUM_CAMERAS - 1) x (3 + 3) + NUM_FRAMES x (3 + 3), where 3 stands for a rotation vector and 3 for a translation vector. The first part of parameters are extrinsics, and the second part is for rotation and translation vectors per frame.
* c. Robust function is additionally applied to mitigate impact of outlier points during the optimization. The function has the shape of derivative of Gaussian, or it is $x * exp(-x/s)$ (efficiently implemented by approximation of the `exp`), where `x` is a square pixel error, and `s` is manually pre-defined scale. The choice of this function is that it is increasing on the interval of `0` to `y` pixel error, and it’s decreasing thereafter. The idea is that the function slightly decreases errors until it reaches `y`, and if error is too high (more than `y`) then its robust value limits to `0`. The value of scale factor was found by exhaustive evaluation that forces robust function to almost linearly increase until the robust value of an error is 10 px and decrease afterwards (see graph of the function below). The value itself is equal to 30, but could be modified in OpenCV source code.
![](camera_multiview_calibration/images/exp.jpg)

Method Input
----
The high-level input of the proposed method is as follows:

* Pattern (object) points. (NUM_FRAMES x) NUM_PATTERN_POINTS x 3. Points may contain a copy of pattern points along frames.
* Image points: NUM_CAMERAS x NUM_FRAMES x NUM_PATTERN_POINTS x 2.
* Image sizes: NUM_CAMERAS x 2 (width and height).
* Detection mask matrix of size NUM_CAMERAS x NUM_FRAMES that indicates whether pattern points are detected for specific camera and frame index.
* Ks (optional) - intrinsic matrices per camera.
* Distortions (optional).
* use_intrinsics_guess - indicates whether intrinsics are provided.
* Flags_intrinsics - flag for intrinsics estimation.

Method Output
----
The high-level output of the proposed method is the following:

* Boolean indicator of success
* Rotation and translation vectors of extrinsics parameters with respect to camera (relative) 0. Number of vectors is `NUM_CAMERAS-1`, for the first camera rotation and translation vectors are zero.
* Intrinsic matrix for each camera.
* Distortion coefficients for each camera.
* Rotation and translation vectors of each frame pattern with respect to camera 0. The combination of rotation and translation is able to tranform the pattern points to the camera coordinate space, and hence with intrinsics parameters project 3D points to image.
* Matrix of reprojection errors of size NUM_CAMERAS x NUM_FRAMES
* Output pairs used for initial estimation of extrinsics, number of pairs is `NUM_CAMERAS-1`.

Pseudocode
----
The idea of the method could be demonstrated in a high-level pseudocode whereas the whole C++ implementation of the proposed approach is implemented in `opencv/modules/calib/src/multiview_calibration.cpp` file.

```python
def mutiviewCalibration (pattern_points, image_points, detection_mask):
  for cam_i = 1,…,NUMBER_CAMERAS:
    if CALIBRATE_INTRINSICS:
      K_i, distortion_i, rvecs_i, tvecs_i = calibrateCamera(pattern_points, image_points[cam_i])
    else:
      rvecs_i, tvecs_i = solvePnP(pattern_points, image_points[cam_i], K_i, distortion_i)
    # Select best rvecs, tvecs based on reprojection errors. Process data:
    pattern_img_area[cam_i][frame] = area(convexHull(image_points[cam_i][frame]
    angle_to_board[cam_i][frame] = arccos(pattern_normal_frame * optical_axis_cam_i)
    angle_cam_to_cam[cam_i][cam_j] = arccos(optical_axis_cam_i * optical_axis_cam_j)
  graph = maximumSpanningTree(detection_mask, pattern_img_area, angle_to_board, angle_cam_to_cam)
  camera_pairs = bread_first_search(graph, root_camera=0)
  for pair in camera_pairs:
    # find relative rotation, translation from camera i to j
    R_ij, t_ij = stereoCalibrate(pattern_points, image_points[i], image_points[j])
  R*, t* = optimizeLevenbergMarquardt(R, t, pattern_points, image_points, K, distortion)
```

Python sample API
----

To run the calibration procedure in Python follow the following steps (see sample code in `samples/python/multiview_calibration.py`):

1. Prepare data:

@snippet samples/python/multiview_calibration.py calib_init

The detection mask matrix is later built by checking the size of image points after detection:

3. Detect pattern points on images:

@snippet samples/python/multiview_calibration.py detect_pattern

4. Build detection mask matrix:

@snippet samples/python/multiview_calibration.py detection_matrix

5. Finally, the calibration function is run as follows:

@snippet samples/python/multiview_calibration.py multiview_calib


C++ sample API
----

To run the calibration procedure in C++ follow the following steps (see sample code in `opencv/samples/cpp/multiview_calibration_sample.cpp`):

1. Prepare data similarly to Python sample, ie., pattern size and scale, fisheye camera mask, files containing image filenames, and pass them to function:

@snippet samples/cpp/multiview_calibration_sample.cpp detectPointsAndCalibrate_signature

2. Initialize data:

@snippet samples/cpp/multiview_calibration_sample.cpp calib_init

3. Detect pattern points on images:

@snippet samples/cpp/multiview_calibration_sample.cpp detect_pattern

4. Build detection mask matrix:

@snippet samples/cpp/multiview_calibration_sample.cpp detection_matrix

5. Run calibration:

@snippet samples/cpp/multiview_calibration_sample.cpp multiview_calib
