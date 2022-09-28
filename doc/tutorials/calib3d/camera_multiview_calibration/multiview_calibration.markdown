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

How to run:
----

Assume we have `N` camera views, for each `i`-th view there are `M` images containg pattern points (e.g., checkeboard).

Python
--

There are two options to run the sample code in Python (`opencv/samples/python/multiview_calibration.py`).
The first option is to prepare `N` files where each file has path to image per line (images of a specific camera of the corresponding file). For example, a file for camera `i` should look like (`file_i.txt`):
```
/path/to/image_1
...
/path/to/image_M
```

Then sample program could be run via command line as follows:
```console
$ python3 multiview_calibration.py --pattern_size W,H --pattern_type TYPE --fisheye IS_FISHEYE_1,...,IS_FISHEYE_N \
--pattern_distance DIST --filenames /path/to/file_1.txt,...,/path/to/file_N.txt
```

Replace `W` and `H` with size of the pattern points, `TYPE` with name of a type of the calibration grid (supported patterns: `checkerboard`, `circles`, `acircles`), `IS_FISHEYE` corresponds to the camera type (1 - is fisheye, 0 - pinhole), `DIST` is pattern distance (i.e., distance between two cells of checkerboard).

Then the sample code automatically detects pattern points on images in parallel (beware of window size and criteria for detection that may are needed to be adjusted manually):

```{r, eval = FALSE}
if pattern_type.lower() == 'checkerboard':
    ret, corners = cv.findChessboardCorners(img_detection, grid_size, None)
elif pattern_type.lower() == 'circles':
    ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_SYMMETRIC_GRID)
elif pattern_type.lower() == 'acircles':
    ret, corners = cv.findCirclesGrid(img_detection, patternSize=grid_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID)
```

The visibility matrix is later built by checking the size of image points after detection:

```{r, eval = FALSE}
for i in range(num_cameras):
    for j in range(num_frames):
        visibility[i,j] = int(len(image_points[i][j]) != 0)
```

Finally, the calibration function is run as follows:
```{r, eval = FALSE}
success, rvecs, Ts, Ks, distortions, rvecs0, tvecs0, errors_per_frame, output_pairs = \
    cv.calibrateMultiview(objPoints=pattern_points_all,
                          imagePoints=image_points,
                          imageSize=image_sizes,
                          visibility=visibility,
                          Ks=Ks,
                          distortions=distortions,
                          is_fisheye=np.array(is_fisheye, dtype=int),
                          USE_INTRINSICS_GUESS=USE_INTRINSICS_GUESS,
                          flags_intrinsics=0)
```

Alternatively, the Python sample could be run from JSON file that should contain image points, pattern points, and the same information about pattern distance, grid type etc.
```console
$ python3 multiview_calibration.py --json_file /path/to/json
```

C++
--

To run the calibration procedure in C++ follow the steps:
1. Initialize data.

```{r, eval = FALSE}
cv::Mat visibility = cv::Mat_<int>(NUM_CAMERAS, NUM_FRAMES);
std::vector<std::vector<cv::Mat>> image_points(NUM_CAMERAS, std::vector<cv::Mat>(NUM_FRAMES));
std::vector<Size> image_sizes(NUM_CAMERAS);
std::vector<bool> is_fisheye(NUM_CAMERAS);
std::vector<std::vector<cv::Point3f>> objPoints;
// todo: init object points accordingly to pattern (e.g., checkeboard)
// output data:
std::vector<Mat> Rs, Ts, Ks, distortions, rvecs0, tvecs0;
cv::Mat output_pairs, errors_mat;
```

2. Detect pattern points on images.
3. Build visibility matrix.

```{r, eval = FALSE}
for (int c = 0; c < NUM_CAMERAS; c++) {
  // image_sizes[c] = ... ; // todo: initialize image size
  // is_fisheye[c] = ....; // todo: true if fisheye, false otherwise
  for (int f = 0; f < NUM_FRAMES; f++) {
    // read image
    cv::Mat gray, corners, img = cv::imread(filenames[c][f]);
    // convert image to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // find pattern points on image, e.g., for checkerboard
    bool success = cv::findChessboardCorners(gray, pattern_size, corners);
    // save
    visibility.at<int>(c, f) = success;
    if (success) corners.copyTo(image_points[c][f]);
  }
}
```

4. Run calibration.

```{r, eval = FALSE}
bool ret = cv::calibrateMultiview (objPoints, image_points, image_sizes, visibility, Rs, Ts, Ks,
distortions, rvecs0, tvecs0, is_fisheye, errors_mat, output_pairs, false/*use_intrinsics_guess*/);
```
