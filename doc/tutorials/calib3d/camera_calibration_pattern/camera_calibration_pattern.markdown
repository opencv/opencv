Create calibration pattern {#tutorial_camera_calibration_pattern}
=========================================

@tableofcontents

@next_tutorial{tutorial_camera_calibration_square_chess}

|    |    |
| -: | :- |
| Original author | Laurent Berger |
| Compatibility | OpenCV >= 3.0 |


The goal of this tutorial is to learn how to create a calibration pattern.

You can find a chessboard pattern in https://github.com/opencv/opencv/blob/4.x/doc/pattern.png

You can find a circleboard pattern in https://github.com/opencv/opencv/blob/4.x/doc/acircles_pattern.png

You can find a ChAruco board pattern in https://github.com/opencv/opencv/blob/4.x/doc/charuco_board_pattern.png
(7X5 ChAruco board, square size: 30 mm , marker size: 15 mm, aruco dict: DICT_5X5_100, page width: 210 mm, page height: 297 mm)

Create your own pattern
---------------

Now, if you want to create your own pattern, you will need python to use https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py

Example

create a checkerboard pattern in file chessboard.svg with 9 rows, 6 columns and a square size of 20mm:

        python gen_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20

create a circle board pattern in file circleboard.svg with 7 rows, 5 columns and a radius of 15 mm:

        python gen_pattern.py -o circleboard.svg --rows 7 --columns 5 --type circles --square_size 15

create a circle board pattern in file acircleboard.svg with 7 rows, 5 columns and a square size of 10mm and less spacing between circle:

        python gen_pattern.py -o acircleboard.svg --rows 7 --columns 5 --type acircles --square_size 10 --radius_rate 2

create a radon checkerboard for findChessboardCornersSB() with markers in (7 4), (7 5), (8 5) cells:

        python gen_pattern.py -o radon_checkerboard.svg --rows 10 --columns 15 --type radon_checkerboard -s 12.1 -m 7 4 7 5 8 5

create a ChAruco board pattern in charuco_board.svg with 7 rows, 5 columns, square size 30 mm, aruco marker size 15 mm and using DICT_5X5_100 as dictionary for aruco markers (it contains in DICT_ARUCO.json file):

        python gen_pattern.py -o charuco_board.svg --rows 7 --columns 5 -T charuco_board --square_size 30 --marker_size 15 -f DICT_5X5_100.json.gz

If you want to change the unit of measurement, use the -u option (e.g. mm, inches, px, m)

If you want to change the page size, use the -w (width) and -h (height) options

If you want to use your own dictionary for the ChAruco board, specify the name of your dictionary file. For example

        python gen_pattern.py -o charuco_board.svg --rows 7 --columns 5 -T charuco_board -f my_dictionary.json

You can generate your dictionary in the file my_dictionary.json with 30 markers and a marker size of 5 bits using the utility provided in opencv/samples/cpp/aruco_dict_utils.cpp.

        bin/example_cpp_aruco_dict_utils.exe my_dict.json -nMarkers=30 -markerSize=5
