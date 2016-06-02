Camera calibration and 3D reconstruction (calib3d module) {#tutorial_table_of_content_calib3d}
==========================================================

Although we got most of our images in a 2D format they do come from a 3D world. Here you will learn
how to find out from the 2D images information about the 3D world.

-   @subpage tutorial_camera_calibration_square_chess

    *Compatibility:* \> OpenCV 2.0

    *Author:* Victor Eruhimov

    You will use some chessboard images to calibrate your camera.

-   @subpage tutorial_camera_calibration

    *Compatibility:* \> OpenCV 2.0

    *Author:* Bernát Gábor

    Camera calibration by using either the chessboard, circle or the asymmetrical circle
    pattern. Get the images either from a camera attached, a video file or from an image
    collection.

-   @subpage tutorial_real_time_pose

    *Compatibility:* \> OpenCV 2.0

    *Author:* Edgar Riba

    Real time pose estimation of a textured object using ORB features, FlannBased matcher, PnP
    approach plus Ransac and Linear Kalman Filter to reject possible bad poses.

-   @subpage tutorial_interactive_calibration

    *Compatibility:* \> OpenCV 3.1

    *Author:* Vladislav Sovrasov

    Camera calibration by using either the chessboard, chAruco, asymmetrical circle or dual asymmetrical circle
    pattern. Calibration process is continious, so you can see results after each new pattern shot.
    As an output you get average reprojection error, intrinsic camera parameters, distortion coefficients and
     confidence intervals for all of evaluated variables.
