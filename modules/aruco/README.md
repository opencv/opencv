ArUco Marker Detection
======================

**ArUco**

ArUco markers are easy to detect pattern grids that yield up to 1024 different patterns. They were built for augmented reality and later used for camera calibration. Since the grid uniquely orients the square, the detection algorithm can determing the pose of the grid.

**ChArUco**

ArUco markers were improved by interspersing them inside a checkerboard called ChArUco. Checkerboard corner intersections provide more stable corners because the edge location bias on one square is countered by the opposite edge orientation in the connecting square. By interspersing ArUco markers inside the checkerboard, each checkerboard corner gets a label which enables it to be used in complex calibration or pose scenarios where you cannot see all the corners of the checkerboard.

The smallest ChArUco board is 5 checkers and 4 markers called a "Diamond Marker".
