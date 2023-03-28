ArUco marker detection (aruco module) {#tutorial_table_of_content_aruco}
==========================================================

ArUco markers are binary square fiducial markers that can be used for camera pose estimation.
Their main benefit is that their detection is robust, fast and simple.

The aruco module includes the detection of these types of markers and the tools to employ them
for pose estimation and camera calibration.

Also, the ChArUco functionalities combine ArUco markers with traditional chessboards to allow
an easy and versatile corner detection. The module also includes the functions to detect
ChArUco corners and use them for pose estimation and camera calibration.

If you are going to print out the markers, an useful script/GUI tool is place at
opencv_contrib/modules/aruco/misc/pattern_generator/ that can generate vector graphics
of ArUco, ArUcoGrid and ChArUco boards. It can help you to print out the pattern with real size
and without artifacts.

-   @subpage tutorial_aruco_detection

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido, Steve Nicholson

    Basic detection and pose estimation from single ArUco markers.

-   @subpage tutorial_aruco_board_detection

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido

    Detection and pose estimation using a Board of markers

-   @subpage tutorial_charuco_detection

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido

    Basic detection using ChArUco corners

-   @subpage tutorial_charuco_diamond_detection

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido

    Detection and pose estimation using ChArUco markers

-   @subpage tutorial_aruco_calibration

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido

    Camera Calibration using ArUco and ChArUco boards

-   @subpage tutorial_aruco_faq

    *Compatibility:* \> OpenCV 3.0

    *Author:* Sergio Garrido

    General and useful questions about the aruco module
