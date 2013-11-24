.. _CameraCalibrationSquareChessBoardTutorial:

Camera calibration with square chessboard
*****************************************

.. highlight:: cpp

The goal of this tutorial is to learn how to calibrate a camera given a set of chessboard images.

*Test data*: use images in your data/chess folder.

#.
    Compile opencv with samples by setting ``BUILD_EXAMPLES`` to ``ON`` in cmake configuration.

#.
    Go to ``bin`` folder and use ``imagelist_creator`` to create an ``XML/YAML`` list of your images.

#.
    Then, run ``calibration`` sample to get camera parameters. Use square size equal to 3cm.

Pose estimation
===============

Now, let us write a code that detects a chessboard in a new image and finds its distance from the camera. You can apply the same method to any object with known 3D geometry that you can detect in an image.

*Test data*: use chess_test*.jpg images from your data folder.

#.
    Create an empty console project. Load a test image: ::

        Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

#.
    Detect a chessboard in this image using findChessboard function. ::

        bool found = findChessboardCorners( img, boardSize, ptvec, CV_CALIB_CB_ADAPTIVE_THRESH );

#.
    Now, write a function that generates a ``vector<Point3f>`` array of 3d coordinates of a chessboard in any coordinate system. For simplicity, let us choose a system such that one of the chessboard corners is in the origin and the board is in the plane *z = 0*.

#.
    Read camera parameters from XML/YAML file: ::

        FileStorage fs(filename, FileStorage::READ);
        Mat intrinsics, distortion;
        fs["camera_matrix"] >> intrinsics;
        fs["distortion_coefficients"] >> distortion;

#.
    Now we are ready to find chessboard pose by running ``solvePnP``: ::

        vector<Point3f> boardPoints;
        // fill the array
        ...

        solvePnP(Mat(boardPoints), Mat(foundBoardCorners), cameraMatrix,
                             distCoeffs, rvec, tvec, false);

#.
    Calculate reprojection error like it is done in ``calibration`` sample (see ``opencv/samples/cpp/calibration.cpp``, function ``computeReprojectionErrors``).

Question: how to calculate the distance from the camera origin to any of the corners?
