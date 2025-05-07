Camera calibration with square chessboard {#tutorial_camera_calibration_square_chess}
=========================================

@tableofcontents

@prev_tutorial{tutorial_camera_calibration_pattern}
@next_tutorial{tutorial_camera_calibration}

|    |    |
| -: | :- |
| Original author | Victor Eruhimov |
| Compatibility | OpenCV >= 4.0 |


The goal of this tutorial is to learn how to calibrate a camera given a set of chessboard images.

*Test data*: use images in your data/chess folder.

-   Compile OpenCV with samples by setting BUILD_EXAMPLES to ON in cmake configuration.

-   Go to bin folder and use imagelist_creator to create an XML/YAML list of your images.

-   Then, run calibration sample to get camera parameters. Use square size equal to 3cm.

Pose estimation
---------------

Now, let us write code that detects a chessboard in an image and finds its distance from the
camera. You can apply this method to any object with known 3D geometry; which you detect in an
image.

*Test data*: use chess_test\*.jpg images from your data folder.

-   Create an empty console project. Load a test image :

        Mat img = imread(argv[1], IMREAD_GRAYSCALE);

-   Detect a chessboard in this image using findChessboard function :

        bool found = findChessboardCorners( img, boardSize, ptvec, CALIB_CB_ADAPTIVE_THRESH );

-   Now, write a function that generates a vector\<Point3f\> array of 3d coordinates of a chessboard
    in any coordinate system. For simplicity, let us choose a system such that one of the chessboard
    corners is in the origin and the board is in the plane *z = 0*

-   Read camera parameters from XML/YAML file :

        FileStorage fs( filename, FileStorage::READ );
        Mat intrinsics, distortion;
        fs["camera_matrix"] >> intrinsics;
        fs["distortion_coefficients"] >> distortion;

-   Now we are ready to find a chessboard pose by running \`solvePnP\` :

        vector<Point3f> boardPoints;
        // fill the array
        ...

        solvePnP(Mat(boardPoints), Mat(foundBoardCorners), cameraMatrix,
                             distCoeffs, rvec, tvec, false);

-   Calculate reprojection error like it is done in calibration sample (see
    opencv/samples/cpp/calibration.cpp, function computeReprojectionErrors).

Question: how would you calculate distance from the camera origin to any one of the corners?
Answer: After obtaining the camera pose using solvePnP, the rotation (rvec) and translation (tvec) vectors define the transformation between the world (chessboard) coordinates and the camera coordinate system. To calculate the distance from the camera’s origin to any chessboard corner, first transform the 3D point from the chessboard coordinate system to the camera coordinate system (if not already done) and then compute its Euclidean distance using the L2 norm, for example:

        // assuming 'point' is the 3D position of a chessboard corner in the camera coordinate system
        double distance = norm(point);

This is equivalent to applying the L2 norm on the 3D point’s coordinates (x, y, z).