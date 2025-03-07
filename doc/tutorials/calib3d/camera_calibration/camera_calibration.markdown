Camera calibration With OpenCV {#tutorial_camera_calibration}
==============================

@tableofcontents

@prev_tutorial{tutorial_camera_calibration_square_chess}
@next_tutorial{tutorial_real_time_pose}

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 4.0 |


Cameras have been around for a long-long time. However, with the introduction of the cheap *pinhole*
cameras in the late 20th century, they became a common occurrence in our everyday life.
Unfortunately, this cheapness comes with its price: significant distortion. Luckily, these are
constants and with a calibration and some remapping we can correct this. Furthermore, with
calibration you may also determine the relation between the camera's natural units (pixels) and the
real world units (for example millimeters).

Theory
------

For the distortion OpenCV takes into account the radial and tangential factors. For the radial
factor one uses the following formula:

\f[x_{distorted} = x( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
y_{distorted} = y( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6)\f]

So for an undistorted pixel point at \f$(x,y)\f$ coordinates, its position on the distorted image
will be \f$(x_{distorted} y_{distorted})\f$. The presence of the radial distortion manifests in form
of the "barrel" or "fish-eye" effect.

Tangential distortion occurs because the image taking lenses are not perfectly parallel to the
imaging plane. It can be represented via the formulas:

\f[x_{distorted} = x + [ 2p_1xy + p_2(r^2+2x^2)] \\
y_{distorted} = y + [ p_1(r^2+ 2y^2)+ 2p_2xy]\f]

So we have five distortion parameters which in OpenCV are presented as one row matrix with 5
columns:

\f[distortion\_coefficients=(k_1 \hspace{10pt} k_2 \hspace{10pt} p_1 \hspace{10pt} p_2 \hspace{10pt} k_3)\f]

Now for the unit conversion we use the following formula:

\f[\left [  \begin{matrix}   x \\   y \\  w \end{matrix} \right ] = \left [ \begin{matrix}   f_x & 0 & c_x \\  0 & f_y & c_y \\   0 & 0 & 1 \end{matrix} \right ] \left [ \begin{matrix}  X \\  Y \\   Z \end{matrix} \right ]\f]

Here the presence of \f$w\f$ is explained by the use of homography coordinate system (and \f$w=Z\f$). The
unknown parameters are \f$f_x\f$ and \f$f_y\f$ (camera focal lengths) and \f$(c_x, c_y)\f$ which are the optical
centers expressed in pixels coordinates. If for both axes a common focal length is used with a given
\f$a\f$ aspect ratio (usually 1), then \f$f_y=f_x*a\f$ and in the upper formula we will have a single focal
length \f$f\f$. The matrix containing these four parameters is referred to as the *camera matrix*. While
the distortion coefficients are the same regardless of the camera resolutions used, these should be
scaled along with the current resolution from the calibrated resolution.

The process of determining these two matrices is the calibration. Calculation of these parameters is
done through basic geometrical equations. The equations used depend on the chosen calibrating
objects. Currently OpenCV supports three types of objects for calibration:

-   Classical black-white chessboard
-   ChArUco board pattern
-   Symmetrical circle pattern
-   Asymmetrical circle pattern

Basically, you need to take snapshots of these patterns with your camera and let OpenCV find them.
Each found pattern results in a new equation. To solve the equation you need at least a
predetermined number of pattern snapshots to form a well-posed equation system. This number is
higher for the chessboard pattern and less for the circle ones. For example, in theory the
chessboard pattern requires at least two snapshots. However, in practice we have a good amount of
noise present in our input images, so for good results you will probably need at least 10 good
snapshots of the input pattern in different positions.

Goal
----

The sample application will:

-   Determine the distortion matrix
-   Determine the camera matrix
-   Take input from Camera, Video and Image file list
-   Read configuration from XML/YAML file
-   Save the results into XML/YAML file
-   Calculate re-projection error

Source code
-----------

You may also find the source code in the `samples/cpp/tutorial_code/calib3d/camera_calibration/`
folder of the OpenCV source library or [download it from here
](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp).
For the usage of the program, run it with `-h` argument. The program has an
essential argument: the name of its configuration file. If none is given then it will try to open the
one named "default.xml". [Here's a sample configuration file
](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/calib3d/camera_calibration/in_VID5.xml) in XML format. In the
configuration file you may choose to use camera as an input, a video file or an image list. If you
opt for the last one, you will need to create a configuration file where you enumerate the images to
use. Here's [an example of this ](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/calib3d/camera_calibration/VID5.xml).
The important part to remember is that the images need to be specified using the absolute path or
the relative one from your application's working directory. You may find all this in the samples
directory mentioned above.

The application starts up with reading the settings from the configuration file. Although, this is
an important part of it, it has nothing to do with the subject of this tutorial: *camera
calibration*. Therefore, I've chosen not to post the code for that part here. Technical background
on how to do this you can find in the @ref tutorial_file_input_output_with_xml_yml tutorial.

Explanation
-----------

-#  **Read the settings**
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp file_read

    For this I've used simple OpenCV class input operation. After reading the file I've an
    additional post-processing function that checks validity of the input. Only if all inputs are
    good then *goodInput* variable will be true.

-#  **Get next input, if it fails or we have enough of them - calibrate**

    After this we have a big
    loop where we do the following operations: get the next image from the image list, camera or
    video file. If this fails or we have enough images then we run the calibration process. In case
    of image we step out of the loop and otherwise the remaining frames will be undistorted (if the
    option is set) via changing from *DETECTION* mode to the *CALIBRATED* one.
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp get_input
    For some cameras we may need to flip the input image. Here we do this too.

-#  **Find the pattern in the current input**

    The formation of the equations I mentioned above aims
    to finding major patterns in the input: in case of the chessboard this are corners of the
    squares and for the circles, well, the circles themselves. ChArUco board is equivalent to
    chessboard, but corners are matched by ArUco markers. The position of these will form the
    result which will be written into the *pointBuf* vector.
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp find_pattern
    Depending on the type of the input pattern you use either the @ref cv::findChessboardCorners or
    the @ref cv::findCirclesGrid function or @ref cv::aruco::CharucoDetector::detectBoard method.
    For all of them you pass the current image and the size of the board and you'll get the positions
    of the patterns. cv::findChessboardCorners and cv::findCirclesGrid return a boolean variable
    which states if the pattern was found in the input (we only need to take into account
    those images where this is true!). `CharucoDetector::detectBoard` may detect partially visible
    pattern and returns coordunates and ids of visible inner corners.

    @note Board size and amount of matched points is different for chessboard, circles grid and ChArUco.
    All chessboard related algorithm expects amount of inner corners as board width and height.
    Board size of circles grid is just amount of circles by both grid dimensions. ChArUco board size
    is defined in squares, but detection result is list of inner corners and that's why is smaller
    by 1 in both dimensions.

    Then again in case of cameras we only take camera images when an input delay time is passed.
    This is done in order to allow user moving the chessboard around and getting different images.
    Similar images result in similar equations, and similar equations at the calibration step will
    form an ill-posed problem, so the calibration will fail. For square images the positions of the
    corners are only approximate. We may improve this by calling the @ref cv::cornerSubPix function.
    (`winSize` is used to control the side length of the search window. Its default value is 11.
    `winSize` may be changed by command line parameter `--winSize=<number>`.)
    It will produce better calibration result. After this we add a valid inputs result to the
    *imagePoints* vector to collect all of the equations into a single container. Finally, for
    visualization feedback purposes we will draw the found points on the input image using @ref
    cv::findChessboardCorners function.
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp pattern_found
-#  **Show state and result to the user, plus command line control of the application**

    This part shows text output on the image.
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp output_text
    If we ran calibration and got camera's matrix with the distortion coefficients we may want to
    correct the image using @ref cv::undistort function:
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp output_undistorted
    Then we show the image and wait for an input key and if this is *u* we toggle the distortion removal,
    if it is *g* we start again the detection process, and finally for the *ESC* key we quit the application:
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp await_input
-#  **Show the distortion removal for the images too**

    When you work with an image list it is not
    possible to remove the distortion inside the loop. Therefore, you must do this after the loop.
    Taking advantage of this now I'll expand the @ref cv::undistort function, which is in fact first
    calls @ref cv::initUndistortRectifyMap to find transformation matrices and then performs
    transformation using @ref cv::remap function. Because, after successful calibration map
    calculation needs to be done only once, by using this expanded form you may speed up your
    application:
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp show_results

The calibration and save
------------------------

Because the calibration needs to be done only once per camera, it makes sense to save it after a
successful calibration. This way later on you can just load these values into your program. Due to
this we first make the calibration, and if it succeeds we save the result into an OpenCV style XML
or YAML file, depending on the extension you give in the configuration file.

Therefore in the first function we just split up these two processes. Because we want to save many
of the calibration variables we'll create these variables here and pass on both of them to the
calibration and saving function. Again, I'll not show the saving part as that has little in common
with the calibration. Explore the source file in order to find out how and what:
@snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp run_and_save
We do the calibration with the help of the @ref cv::calibrateCameraRO function. It has the following
parameters:

-   The object points. This is a vector of *Point3f* vector that for each input image describes how
    should the pattern look. If we have a planar pattern (like a chessboard) then we can simply set
    all Z coordinates to zero. This is a collection of the points where these important points are
    present. Because, we use a single pattern for all the input images we can calculate this just
    once and multiply it for all the other input views. We calculate the corner points with the
    *calcBoardCornerPositions* function as:
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp board_corners
    And then multiply it as:
    @code{.cpp}
    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
    objectPoints[0][s.boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(),objectPoints[0]);
    @endcode
    @note If your calibration board is inaccurate, unmeasured, roughly planar targets (Checkerboard
    patterns on paper using off-the-shelf printers are the most convenient calibration targets and
    most of them are not accurate enough.), a method from @cite strobl2011iccv can be utilized to
    dramatically improve the accuracies of the estimated camera intrinsic parameters. This new
    calibration method will be called if command line parameter `-d=<number>` is provided. In the
    above code snippet, `grid_width` is actually the value set by `-d=<number>`. It's the measured
    distance between top-left (0, 0, 0) and top-right (s.squareSize*(s.boardSize.width-1), 0, 0)
    corners of the pattern grid points. It should be measured precisely with rulers or vernier calipers.
    After calibration, newObjPoints will be updated with refined 3D coordinates of object points.
-   The image points. This is a vector of *Point2f* vector which for each input image contains
    coordinates of the important points (corners for chessboard and centers of the circles for the
    circle pattern). We have already collected this from @ref cv::findChessboardCorners or @ref
    cv::findCirclesGrid function. We just need to pass it on.
-   The size of the image acquired from the camera, video file or the images.
-   The index of the object point to be fixed. We set it to -1 to request standard calibration method.
    If the new object-releasing method to be used, set it to the index of the top-right corner point
    of the calibration board grid. See cv::calibrateCameraRO for detailed explanation.
    @code{.cpp}
    int iFixedPoint = -1;
    if (release_object)
        iFixedPoint = s.boardSize.width - 1;
    @endcode
-   The camera matrix. If we used the fixed aspect ratio option we need to set \f$f_x\f$:
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp fixed_aspect
-   The distortion coefficient matrix. Initialize with zero.
    @code{.cpp}
    distCoeffs = Mat::zeros(8, 1, CV_64F);
    @endcode
-   For all the views the function will calculate rotation and translation vectors which transform
    the object points (given in the model coordinate space) to the image points (given in the world
    coordinate space). The 7-th and 8-th parameters are the output vector of matrices containing in
    the i-th position the rotation and translation vector for the i-th object point to the i-th
    image point.
-   The updated output vector of calibration pattern points. This parameter is ignored with standard
    calibration method.
-   The final argument is the flag. You need to specify here options like fix the aspect ratio for
    the focal length, assume zero tangential distortion or to fix the principal point. Here we use
    CALIB_USE_LU to get faster calibration speed.
@code{.cpp}
rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
                        cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                        s.flag | CALIB_USE_LU);
@endcode
-   The function returns the average re-projection error. This number gives a good estimation of
    precision of the found parameters. This should be as close to zero as possible. Given the
    intrinsic, distortion, rotation and translation matrices we may calculate the error for one view
    by using the @ref cv::projectPoints to first transform the object point to image point. Then we
    calculate the absolute norm between what we got with our transformation and the corner/circle
    finding algorithm. To find the average error we calculate the arithmetical mean of the errors
    calculated for all the calibration images.
    @snippet samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp compute_errors

Results
-------

Let there be [this input chessboard pattern ](pattern.png) which has a size of 9 X 6. I've used an
AXIS IP camera to create a couple of snapshots of the board and saved it into VID5 directory. I've
put this inside the `images/CameraCalibration` folder of my working directory and created the
following `VID5.XML` file that describes which images to use:
@code{.xml}
<?xml version="1.0"?>
<opencv_storage>
<images>
images/CameraCalibration/VID5/xx1.jpg
images/CameraCalibration/VID5/xx2.jpg
images/CameraCalibration/VID5/xx3.jpg
images/CameraCalibration/VID5/xx4.jpg
images/CameraCalibration/VID5/xx5.jpg
images/CameraCalibration/VID5/xx6.jpg
images/CameraCalibration/VID5/xx7.jpg
images/CameraCalibration/VID5/xx8.jpg
</images>
</opencv_storage>
@endcode
Then passed `images/CameraCalibration/VID5/VID5.XML` as an input in the configuration file. Here's a
chessboard pattern found during the runtime of the application:

![](images/fileListImage.jpg)

After applying the distortion removal we get:

![](images/fileListImageUnDist.jpg)

The same works for [this asymmetrical circle pattern ](acircles_pattern.png) by setting the input
width to 4 and height to 11. This time I've used a live camera feed by specifying its ID ("1") for
the input. Here's, how a detected pattern should look:

![](images/asymetricalPattern.jpg)

In both cases in the specified output XML/YAML file you'll find the camera and distortion
coefficients matrices:
@code{.xml}
<camera_matrix type_id="opencv-matrix">
<rows>3</rows>
<cols>3</cols>
<dt>d</dt>
<data>
 6.5746697944293521e+002 0. 3.1950000000000000e+002 0.
 6.5746697944293521e+002 2.3950000000000000e+002 0. 0. 1.</data></camera_matrix>
<distortion_coefficients type_id="opencv-matrix">
<rows>5</rows>
<cols>1</cols>
<dt>d</dt>
<data>
 -4.1802327176423804e-001 5.0715244063187526e-001 0. 0.
 -5.7843597214487474e-001</data></distortion_coefficients>
@endcode
Add these values as constants to your program, call the @ref cv::initUndistortRectifyMap and the
@ref cv::remap function to remove distortion and enjoy distortion free inputs for cheap and low
quality cameras.

You may observe a runtime instance of this on the [YouTube
here](https://www.youtube.com/watch?v=ViPN810E0SU).

@youtube{ViPN810E0SU}
