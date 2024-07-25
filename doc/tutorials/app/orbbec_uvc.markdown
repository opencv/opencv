Using Orbbec 3D cameras (UVC) {#tutorial_orbbec_uvc}
====================================================

@tableofcontents

@prev_tutorial{tutorial_orbbec_astra_openni}
@next_tutorial{tutorial_intelperc}

|    |    |
| -: | :- |
| Original author | Jinyue Chen |
| Compatibility | OpenCV >= 4.10 |

### Introduction
This tutorial is devoted to the Orbbec 3D cameras based on UVC protocol. For the use of the older
Orbbec 3D cameras which depends on OpenNI, please refer to the
[previous tutorial](https://github.com/opencv/opencv/blob/5.x/doc/tutorials/app/orbbec_astra_openni.markdown).

Unlike working with the OpenNI based Astra 3D cameras which requires OpenCV built with OpenNI2 SDK,
Orbbec SDK is not required to be installed for accessing Orbbec UVC 3D cameras via OpenCV. By using
`cv::VideoCapture` class, users get the stream data from 3D cameras, similar to working with USB
cameras. The calibration and alignment of the depth map and color image are done internally.

### Instructions

In order to use the 3D cameras with OpenCV. You can refer to [Get Started](https://opencv.org/get-started/)
to install OpenCV.

Note since 4.11 on, Mac OS users need to compile OpenCV from source with flag
`-DOBSENSOR_USE_ORBBEC_SDK=ON` in order to use the cameras:
```bash
cmake -DOBSENSOR_USE_ORBBEC_SDK=ON ..
make
sudo make install
```

Code
----

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/blob/5.x/samples/python/videocapture_obsensor.py)
@include samples/python/videocapture_obsensor.py
@end_toggle

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/blob/5.x/samples/cpp/videocapture_obsensor.cpp)
@include samples/cpp/videocapture_obsensor.cpp
@end_toggle

### Code Explanation

#### Python

- **Open Orbbec Depth Sensor**:
  Using `cv.VideoCapture(0, cv.CAP_OBSENSOR)` to attempt to open the first Orbbec depth sensor device.
  If the camera fails to open, the program will exit and display an error message.

- **Loop to Grab and Process Data**:
  In an infinite loop, the code continuously grabs data from the camera. The `orbbec_cap.grab()`
  method attempts to grab a frame.

- **Process BGR Image**:
  Using `orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)` to retrieve the BGR image data.
  If successfully retrieved, the BGR image is displayed in a window using `cv.imshow("BGR", bgr_image)`.

- **Process Depth Image**:
  Using `orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)` to retrieve the depth image data.
  If successfully retrieved, the depth image is first normalized to a range of 0 to 255, then a
  false color image is applied, and the result is displayed in a window using `cv.imshow("DEPTH", color_depth_map)`.

- **Keyboard Interrupt**:
  Using `cv.pollKey()` to detect keyboard events. If a key is pressed, the loop breaks and
  the program ends.

- **Release Resources**:
  After exiting the loop, the camera resources are released using `orbbec_cap.release()`.

#### C++

- **Open Orbbec Depth Sensor**:
  Using `VideoCapture obsensorCapture(0, CAP_OBSENSOR)` to attempt to open the first Orbbec depth
  sensor device. If the camera fails to open, an error message is displayed, and the program exits.

- **Retrieve Camera Intrinsic Parameters**:
  Using `obsensorCapture.get()` to retrieve the intrinsic parameters of the camera, including focal
  lengths (`fx`, `fy`) and principal points (`cx`, `cy`).

- **Loop to Grab and Process Data**:
  In an infinite loop, the code continuously grabs data from the camera. The `obsensorCapture.grab()`
  method attempts to grab a frame.

- **Process BGR Image**:
  Using `obsensorCapture.retrieve(image, CAP_OBSENSOR_BGR_IMAGE)` to retrieve the BGR image data.
  If successfully retrieved, the BGR image is displayed in a window using `imshow("BGR", image)`.

- **Process Depth Image**:
  Using `obsensorCapture.retrieve(depthMap, CAP_OBSENSOR_DEPTH_MAP)` to retrieve the depth image data.
  If successfully retrieved, the depth image is normalized and a false color image is applied, then
  the result is displayed in a window using `imshow("DEPTH", adjDepthMap)`. The retrieved depth
  values are in millimeters and are truncated to a range between 300 and 5000 (millimeter).
  This fixed range can be interpreted as a truncation based on the depth camera's depth range,
  removing invalid pixels on the depth map.

- **Overlay Depth Map on BGR Image**:
  Convert the depth map to an 8-bit image, resize it to match the BGR image size, and overlay it
  on the BGR image with a specified transparency (`alpha`). The overlaid image is displayed in
  a window using `imshow("DepthToColor", image)`.

- **Keyboard Interrupt**:
  Using `pollKey()` to detect keyboard events. If a key is pressed, the loop breaks and the program ends.

- **Release Resources**:
  After exiting the loop, the camera resources are released.
### Results
#### Python

![BGR And DEPTH frame](images/orbbec_uvc_python.jpg)

#### C++

![BGR And DEPTH And DepthToColor frame](images/orbbec_uvc_cpp.jpg)

### Note
Mac users need sudo privileges to execute the code.
