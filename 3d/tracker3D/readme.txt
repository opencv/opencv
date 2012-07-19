-------------------------------------------
Introduction
-------------------------------------------
tracker3D application was developed to obtain automatically
training images from video or camera only by selecting an object
in one frame. To do so you need only a pre-recored video or
live input from camera.
The object view must include a chessboard.

-------------------------------------------
How to launch application
-------------------------------------------
You must have OpenCV library installed on your computer.
To run the application, just unpack the archive,
enter to the directory where it was extracted and
either run "make" (if on Linux, MacOSX or other Unix system),
or use CMake to generate makefiles and then run make -
there must be OpenCVConfig.cmake somewhere
where CMake could find it.

After that the application can be run from the console with one
required command-line parameter - the configuration file name, for example:

createsamples args.yml

-------------------------------------------
Configuration file description
-------------------------------------------
Configuration file is YAML file. You can see the example below

%YAML:1.0
USE_3D: 1
USE_DEINTERLACING: 1
USE_UNDISTORTION: 1
CALIBRATE_CAMERA: 0
DRAW_CHESSBOARD: 0
DRAW_CORNERS: 1
AUTOFIND_CHESSBOARD: 1
IS_VIDEO_CAPTURE: 1
SAVE_DRAWINGS: 1
SAVE_SAMPLES: 1
SAVE_ALL_FRAMES: 0
SHOW_TEST_SQUARE: 1
CHESSBOARD_WIDTH: 8
CHESSBOARD_HEIGHT: 6
INPUT_VIDEO: "/../../../../../DATA/DoorHandle1_xvid.AVI"
OUTPUT_DIRECTORY: "/../../../../../DoorHandleOutput/"
SAMPLES_PATH: "/../../../../../DoorHandleOutput/rectified/"
CAMERA_PARAMETERS_PATH: "/../../../../../_camera.yml"


Fields:
IS_VIDEO_CAPTURE: defines whether we work with video(1) or web-camera(0).

AUTOFIND_CHESSBOARD: If it is enabled(1) application will try to find the first video frame with chessboard and show it. Only for 2D version.

USE_DEINTERLACING: enables(1) or disables(0) deinterlacing algorithm

USE_UNDISTORTION: enables(1) or disables(0) undistortion. If enabled we must have yml file with camera parameters (in CAMERA_PARAMETERS_PATH field) or CALIBRATE_CAMERA option enabled.

CALIBRATE_CAMERA: enables(1) or disables(0) camera calibration when USE_UNDISTORTION is enabled

DRAW_CHESSBOARD: if it is enabled(1), application will draw chessboard corners on each frame.

DRAW_POINTS: if it is enabled(1), application will draw points around founded region of interest

SAVE_DRAWINGS: if it is enabled(1), application will save all extra information
  (like chessboard corners) on the frame into OUTPUT_DIRECTORY.
  Otherwise only original frames will be saved.

SAVE_SAMPLES: if it is enabled(1), application will save automatically
  the selected part of each frame with interested region into
  SAMPLES_PATH folder with corrected perspective transformation
  (transformation which makes chessboard rectangular)

SAVE_ALL_FRAMES: if it is enabled(1), application will save automatically all video frames to the output folder

SHOW_TEST_SQUARE: if it is enabled(1), application will show test square on the chessboard: the last square which was founded by the same algorithm as region of interests. Uses to estimate algorithm's quality. Only for 2D version.

CHESSBOARD_WIDTH: chessboard inner corners count (horizontal)

CHESSBOARD_HEIGHT: chessboard inner corners count (vertical)

INPUT_VIDEO: path to the input video file (uses only when isVideoCapture == 1)

OUTPUT_DIRECTORY: path to the output directory which contains frames
             with the object and frames.txt file with object coordinates
             for each frame (frames.txt structure see below)

CAMERA_PARAMETERS_PATH: path to camera parameters (uses for undistortion)

!NOTE! All the paths must end with slash (/)

-------------------------------------------
How to work with the application
-------------------------------------------
2D version:
On the first frame with chessboard found you should select quadrangle
(select vertices by clicking on the image) with interested object.
Then the application works automatically. During the process you can break the process
with Esc key. Also if you work with video file you can go to the next or previous
frame before region selection with a help of Space and Backspace key respectively.
If OpenCV will not be able to find chessboard on the frame then application
will be closed.

3D version: you have to select vertices. Then you have to set depth of box around the object and its position using two trackbars in the upper part of the window.

If USE_UNDISTORTION and CALIBRATE_CAMERA options are both enabled we should calibrate camera in the way like in OpenCV calibration sample.

  
-------------------------------------------
frames.txt structure
-------------------------------------------

The structure is following:

IMAGE_PATH,x1,y1,x2,y2,x3,y3,x4,y4[,x5,y5,x6,y6,x7,y7,x8,y8]

where xi,yi are coordinates of vertices of the selected quadrangle on the IMAGE_PATH image

-------------------------------------------
How to run the sample application 
-------------------------------------------
You can see "data" directory inside your tracker3D directory. To launch the sample just change all paths in args.yml file to your paths and type "tracker3D <path_to_args.yml>" command