# Building OpenCV from Source for Windows Embedded Compact (WINCE/WEC)

## Requirements
CMake 3.1.0 or higher
Windows Embedded Compact SDK

## Configuring
To configure CMake for Windows Embedded, specify Visual Studio 2013 as generator and the name of your installed SDK:

`cmake -G "Visual Studio 12 2013" -A "MySDK WEC2013" -DCMAKE_TOOLCHAIN_FILE:FILEPATH=../platforms/wince/arm-wince.toolchain.cmake`

If you are building for a headless WINCE, specify `-DBUILD_HEADLESS=ON` when configuring. This will remove the `commctrl.lib` dependency.

If you are building for anything else than WINCE800, you need to specify that in the configuration step. Example:

```
-DCMAKE_SYSTEM_VERSION=7.0 -DCMAKE_GENERATOR_TOOLSET=CE700 -DCMAKE_SYSTEM_PROCESSOR=arm-v4
```

For headless WEC2013, this configuration may not be limited to but is known to work:

```
-DBUILD_EXAMPLES=OFF `
-DBUILD_opencv_apps=OFF `
-DBUILD_opencv_calib3d=OFF `
-DBUILD_opencv_highgui=OFF `
-DBUILD_opencv_features2d=OFF `
-DBUILD_opencv_flann=OFF `
-DBUILD_opencv_ml=OFF `
-DBUILD_opencv_objdetect=OFF `
-DBUILD_opencv_photo=OFF `
-DBUILD_opencv_shape=OFF `
-DBUILD_opencv_stitching=OFF `
-DBUILD_opencv_superres=OFF `
-DBUILD_opencv_ts=OFF `
-DBUILD_opencv_video=OFF `
-DBUILD_opencv_videoio=OFF `
-DBUILD_opencv_videostab=OFF `
-DBUILD_opencv_dnn=OFF `
-DBUILD_opencv_java=OFF `
-DBUILD_opencv_python2=OFF `
-DBUILD_opencv_python3=OFF `
-DBUILD_opencv_java_bindings_generator=OFF `
-DBUILD_opencv_python_bindings_generator=OFF `
-DBUILD_TIFF=OFF `
-DCV_TRACE=OFF `
-DWITH_OPENCL=OFF `
-DHAVE_OPENCL=OFF `
-DWITH_QT=OFF `
-DWITH_GTK=OFF `
-DWITH_QUIRC=OFF `
-DWITH_JASPER=OFF `
-DWITH_WEBP=OFF `
-DWITH_PROTOBUF=OFF `
-DBUILD_SHARED_LIBS=OFF `
-DWITH_OPENEXR=OFF `
-DWITH_TIFF=OFF `
```

## Configuring to build as shared
Building OpenCV as shared libraries is as easy as appending
```
-DBUILD_SHARED_LIBS=ON `
-DBUILD_ZLIB=ON
```
to the build configuration.

## Building
You are required to build using Unicode:
`cmake --build . -- /p:CharacterSet=Unicode`
