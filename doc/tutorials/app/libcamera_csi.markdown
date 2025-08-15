Using CSI Cameras with libcamera backend {#tutorial_libcamera_csi}
==================================================================

@tableofcontents

Introduction
------------
`libcamera` is an open-source software library aimed at supporting camera systems directly from the Linux operating system on Arm processors. This tutorial is based on the Raspberry Pi OS Bookworm. It is applicable to other OS like Ubuntu but the behavior in other OS is not garanteed. In this tutorial, you will learn how to:

- Install OpenCV with `libcamera` backend enabled
- Capture video frames with CSI cameras on Raspberry Pi

Steps
-----
## Install libcamera development pack

In terminal, run:

```bash
sudo apt update && sudo apt upgrade
sudo apt install libcamera-dev

# The following package may not be found in official source of Ubuntu. Try to download the disk image from Raspberry Pi Imager, or you can ignore it.
sudo apt install rpicam-apps
```

## Configure your CSI camera

First please check the sensor of your CSI camera, Raspberry Pi’s implementation of libcamera supports the following cameras:

- Official cameras:
  - OV5647 (V1)
  - IMX219 (V2)
  - IMX708 (V3)
  - IMX477 (HQ)
  - IMX500 (AI)
  - IMX296 (GS)
- Third-party sensors:
  - IMX290
  - IMX327
  - IMX378
  - IMX519
  - OV9281

If you are unsure about the model of the sensor, please consult your camera manufacturer. Do not connect your camera to DSI port.

For example, if you have plugged in 2 cameras with ov5647 and imx219, make sure these lines have been added to `/boot/firmware/config.txt`:

```bash
dtoverlay=ov5647
dtoverlay=imx219
```

Reboot to take effects. If you RaspberryPi cannot reboot successfully and throw errors such as `device descriptor read/64 error -71` or `cannot enumerate USB device`, please check if your power adapter support 5.1V/5A output. If it does not help, replace your disk drive and other power-consuming USB devices, or consider active USB devices.

In terminal, run:

```bash
# If you did not run `sudo apt install rpicam-apps` successfully, skip this command.
rpicam-hello --list-cameras
```

The output should be like this:

```bash
Available cameras
-----------------
0 : ov5647 [2592x1944 10-bit GBRG] (/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36)
    Modes: 'SGBRG10_CSI2P' : 640x480 [58.92 fps - (16, 0)/2560x1920 crop]
                             1296x972 [46.34 fps - (0, 0)/2592x1944 crop]
                             1920x1080 [32.81 fps - (348, 434)/1928x1080 crop]
                             2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]

1 : imx219 [3280x2464 10-bit RGGB] (/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10)
    Modes: 'SRGGB10_CSI2P' : 640x480 [103.33 fps - (1000, 752)/1280x960 crop]
                             1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
                             1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                             3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
           'SRGGB8' : 640x480 [103.33 fps - (1000, 752)/1280x960 crop]
                      1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
                      1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                      3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
```

## OpenCV configuration

To enable libcamera support, please install OpenCV from source.

run `git clone https://github.com/opencv/opencv.git` and go to the source directory.

Afterwards, run:

```bash
cmake . -B build \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=RelWithDebug \
    -DBUILD_LIST=videoio,highgui,imgcodecs,imgproc \
    -DWITH_GTK=ON \
    -DWITH_GTK_2_X=ON \
    -DWITH_PNG=ON \
    -DWITH_WEBP=OFF \
    -DWITH_SUNRASTER=OFF \
    -DWITH_PXM=OFF \
    -DWITH_PFM=OFF \
    -DWITH_AVIF=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_OpenEXR=OFF \
    -DBUILD_TESTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DVIDEOIO_ENABLE_PLUGINS=ON \
    -DVIDEOIO_PLUGIN_LIST="libcamera" \
    -DWITH_LIBCAMERA=ON

cd ./build
sudo make && sudo make install
```

You can check the index of CSI camera by running `rpicam-hello --list-cameras`. In the previous example, `cv::VideoCapture cap(0);` can create a VideoCapture instance of the ov5647 camera. A minimum code example is like this:

```cpp
int main() {
    cv::VideoCapture cap(0); // or cv::VideoCapture cap(0, cv::CAP_LIBCAMERA); if there is a USB camera plugged in.
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: No frame captured" << std::endl;
            break;
        } else {
            cv::imshow("Video", frame);
        }
        if (cv::waitKey(30) >= 0) break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
```

Reference
---------
https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-apps
