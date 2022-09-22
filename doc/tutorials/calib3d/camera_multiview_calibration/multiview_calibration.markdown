Multi-view Camera Calibration Tutorial
==================

The following sections describes the individual algorithmic steps of the overall multi-camera calibration pipeline:

Briefly:
----
1. Calibrate intrinsics parameters (intrinsic matrix and distortion coefficients) for each camera independently.
2. Calibrate pairwise cameras (using stereo calibration) via intrinsics parameters from step 1.
3. Do global optimization using all cameras simultaneously to refine extrinsic parameters.

Steps in detail:
----
1. If the intrinsics are not provided, the calibration procedure starts calibrating them independently for each camera.
* a\. If input is a combination of fisheye and pinhole cameras, then fisheye images are calibrated with the default OpenCV calibrate function. The reason is that stereo calibration in OpenCV does not support a mix of fisheye and pinhole cameras. The following flags are used;
* * i\. CALIB_RATIONAL_MODEL - it extends default (5 coefficients) distortion model and returns more parameters.
* * ii\. CALIB_ZERO_TANGENT_DIST - it zeroes out tangential distortion coefficients, since the fisheye model does not have them.
* * iii\. CALIB_FIX_K5, CALIB_FIX_K6 - it zeroes out the fifth and sixth parameter, so in total 4 parameters are returned.
* b\. Output of intrinsic calibration is also rotation, translation vectors, and errors per frame.
* * i\. For each frame, the index of the camera with the lowest error among all cameras is saved.
2. Otherwise, if intrinsics are known, then the algorithm runs perspective-n-point estimation to estimate rotation / translation vector, and reprojection error for each frame.
3. Assume that cameras can be represented as nodes of a connected graph. An edge between two cameras is created if there is any image overlap over all frames. If the graph does not connect all cameras (i.e., no overlap) then calibration is not possible. The next step consists of finding the maximum spanning tree (MST) of this graph, the MST captures all the best pairwise camera connections. The weight of edges across all frames is a weighted combination of multiple factors:
* a\. The main contribution is a number of pattern points visible in both images (cameras).
* b\. Ratio of area of convex hull of projected points in the image to the image resolution.
* c\. Angle between cameras’ optical axes (found from rotation vectors).
* d\. Angles between the camera's optical axis and the pattern's normal vector (found from 3 non-collinear pattern’s points).
4. The initial estimate of cameras’ extrinsics is found by pairwise stereo calibration. Without loss of generality, the 0-th camera’s rotation is fixed to identity and translation to zero vector, and the 0-th node becomes the root of the MST. The order of stereo calibration is selected by traversing MST in breadth first search, starting from the root, total number of pairs (also number of edges of tree) is NUM_CAMERAS - 1.
5. Given the initial estimate of extrinsics the aim is to polish results using global optimization (via Levenberq-Marquardt method).
* a\. To reduce the total number of parameters, all rotation / translation vectors estimated in the first step from intrinsics calibration with the lowest error are transformed to be relative with respect to the root camera.
* b\. The total number of parameters is (NUM_CAMERAS - 1) x (3 + 3) + NUM_FRAMES x (3 + 3), where 3 is for rotation vector and 3 for translation vector. The first part of parameters are for extrinsics, and the second part is for rotation / translation vectors per frame.
* c\. Robust function is additionally applied to mitigate impact of outlier points during the optimization. The function has the shape of derivative of Gaussian, or it is x*exp(-x/s) (efficiently implemented by its approximation), where x is a square pixel error, and s is manually defined scale. The choice of this function is that it is increasing on the interval of 0 to y (e.g., 30) error px, and it’s decreasing after. The idea is that the function slightly decreases errors until it reaches y, and if error is too high (more than y) then its robust value limits to 0.

Input:
----
* Pattern (object) points. (NUM_FRAMES x) NUM_PATTERN_POINTS x 3. Points may contain a copy of pattern points along frames.
* Image points: NUM_CAMERAS x NUM_FRAMES x NUM_PATTERN_POINTS x 2.
* Image sizes: NUM_CAMERAS x 2
* Visibility matrix of size NUM_CAMERAS x NUM_FRAMES that indicates whether pattern points are visible for specific camera and frame index.
* Ks (optional) - intrinsic matrices per camera.
* Distortions (optional).
* USE_INTRINSICS_GUESS - indicates whether intrinsics are provided.
* Flags_intrinsics - flag for intrinsics estimation.

Output:
----
* Boolean indicator of success
* Rotation / Translation vectors of extrinsics with respect to camera 0. Number of vectors NUM_CAMERAS - 1, for the first camera rotation / translation vector is zero.
* Intrinsic matrix for each camera.
* Distortion coefficients for each camera.
* Rotation / Translation vector of each frame pattern with respect to camera 0.
* Matrix of reprojection errors of size NUM_CAMERAS x NUM_FRAMES
* Output pairs used for initial estimation of extrinsics. NUM_CAMERAS - 1.

Pseudocode:
----
```{r, eval = FALSE}
def mutiviewCalibration (pattern_points, image_points, visibility_matrix):
  for cam_i = 1,…,NUMBER_CAMERAS:
    if CALIBRATE_INTRINSICS:
      K_i, distortion_i, rvecs_i, tvecs_i = calibrateCamera(pattern_points, image_points[cam_i])
    else:
      rvecs_i, tvecs_i = solvePnP(pattern_points, image_points[cam_i], K_i, distortion_i)
  Select best rvecs, tvecs based on reprojection errors.
  Process data:
    pattern_img_area[cam_i][frame] = area(convexHull(image_points[cam_i][frame]
    angle_to_board[cam_i][frame] = arccos(pattern_normal_frame * optical_axis_cam_i)
    angle_cam_to_cam[cam_i][cam_j] = arccos(optical_axis_cam_i * optical_axis_cam_j)
  graph = maximumSpanningTree(visibility_mat, pattern_img_area, angle_to_board, angle_cam_to_cam)
  camera_pairs = bread_first_search(graph, root_camera=0)
  for pair in camera_pairs:
    # find relative rotation, translation from camera i to j
    R_ij, t_ij = stereoCalibrate(pattern_points, image_points[i], image_points[j])
  R*, t* = optimizeLevenbergMarquardt(R, t, pattern_points, image_points, K, distortion)
```

Python samples:
----
To demonstrate functionally of the proposed method, the corresponding sample file is created in Python.
Its arguments are either a path to a JSON file already containing all image and pattern points, together with camera information; or files containing image names, and camera information passed through a command line.
If the arguments are files containing images, the function automatically does detection (the pattern type has to be specified, e.g., checkerboard).
Apart from estimated extrinsics / intrinsics, the python sample provides a comprehensive visualization.

Firstly, the sample shows positions of cameras, checkerboard (of a random frame), and pairs of cameras connected by black lines explicitly demonstrate tuples used in the initial stage of stereo calibration.

If images are not known, then a simple plot with arrows (from given point to the back-projected one) visualizing errors are shown. The color of arrows highlights the error values. Additionally, the title reports mean error on this frame, and its accuracy among other frames used in calibration.

The following test instances were synthetically generated (see `opencv/apps/python-calibration-generator/calibration_generator.py`):

<p align="center">
  <img src="images/1a.png" width="400" />
  <img src="images/1b.png" width="286" />
</p>

<p align="center">
  <img src="images/4a.png" width="340" />
  <img src="images/4b.png" width="350" />
</p>

This instance has large Gaussian points noise.

<p align="center">
  <img src="images/2a.png" width="261" />
  <img src="images/2b.png" width="320" />
</p>

Another example, with more complex tree structure is here, it shows a weak connection between two groups of cameras.

<p align="center">
  <img src="images/3a.png" width="345" />
  <img src="images/3b.png" width="300" />
</p>


If files to images are provided, then the output is an image with plotted arrows:

<p align="center">
  <img src="images/checkerboard.png" width="340" />
</p>
