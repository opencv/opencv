Create calibration pattern {#tutorial_camera_calibration_pattern}
=========================================

@tableofcontents

@next_tutorial{tutorial_camera_calibration_square_chess}

|    |    |
| -: | :- |
| Original author | Laurent Berger |
| Compatibility | OpenCV >= 3.0 |


The goal of this tutorial is to learn how to create calibration pattern.

You can find a chessboard pattern in https://github.com/opencv/opencv/blob/master/doc/pattern.png

You can find a circleboard pattern in https://github.com/opencv/opencv/blob/master/doc/acircles_pattern.png

Create your own pattern
---------------

Now, if you want to create your own pattern, you will need python to use https://github.com/opencv/opencv/blob/master/doc/pattern_tools/gen_pattern.py

Example

create a checkerboard pattern in file chessboard.svg with 9 rows, 6 columns and a square size of 20mm:

        python gen_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20

create a circle board pattern in file circleboard.svg with 7 rows, 5 columns and a radius of 15mm:

        python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15

create a circle board pattern in file acircleboard.svg with 7 rows, 5 columns and a square size of 10mm and less spacing between circle:

        python gen_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2

create a radon checkerboard for findChessboardCornersSB() with markers in (7 4), (7 5), (8 5) cells:

        python gen_pattern.py -o radon_checkerboard.svg --rows 10 --columns 15 --type radon_checkerboard -s 12.1 -m 7 4 7 5 8 5

If you want to change unit use -u option (mm inches, px, m)

If you want to change page size use -w and -h options

@cond HAVE_opencv_aruco
If you want to create a ChArUco board read @ref tutorial_charuco_detection "tutorial Detection of ChArUco Corners" in opencv_contrib tutorial.
@endcond
@cond !HAVE_opencv_aruco
If you want to create a ChArUco board read tutorial Detection of ChArUco Corners in opencv_contrib tutorial.
@endcond
