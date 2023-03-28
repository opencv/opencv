set(the_description "ArUco Marker Detection")
ocv_define_module(aruco opencv_core opencv_imgproc opencv_calib3d opencv_objdetect WRAP python java objc js)
ocv_include_directories(${CMAKE_CURRENT_BINARY_DIR})

ocv_add_testdata(samples/ contrib/aruco
    FILES_MATCHING PATTERN "*yml"
)

ocv_add_testdata(tutorials/aruco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "singlemarkersoriginal.jpg"
)

ocv_add_testdata(tutorials/aruco_board_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "gboriginal.png"
)

ocv_add_testdata(tutorials/charuco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "choriginal.jpg"
)

ocv_add_testdata(tutorials/charuco_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "chocclusion_original.jpg"
)

ocv_add_testdata(tutorials/charuco_diamond_detection/images/ contrib/aruco
    FILES_MATCHING PATTERN "diamondmarkers.png"
)

ocv_add_testdata(tutorials/aruco_calibration/images/ contrib/aruco
    FILES_MATCHING REGEX "img_[0-9]+.jpg"
)
