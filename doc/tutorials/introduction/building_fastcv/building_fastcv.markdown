Building OpenCV with FastCV {#tutorial_building_fastcv}
===========================

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.11.0 |

Enable OpenCV with FastCV for Qualcomm Chipsets
-----------------------------------------------

This document scope is to guide the Developers to enable OpenCV Acceleration with FastCV for the
Qualcomm chipsets with ARM64 architecture. Entablement of OpenCV with FastCV back-end on non-Qualcomm
chipsets or Linux platforms other than [Qualcomm Linux](https://www.qualcomm.com/developer/software/qualcomm-linux)
is currently out of scope.

About FastCV
------------

FastCV provides two main features to computer vision application developers:

- A library of frequently used computer vision (CV) functions, optimized to run efficiently on a wide variety of Qualcomm’s Snapdragon devices.
- A clean processor-agnostic hardware acceleration API, under which chipset vendors can hardware accelerate FastCV functions on Qualcomm’s Snapdragon hardware.

FastCV is released as a unified binary, a single binary containing two implementations of the algorithms:

- Generic implementation runs on Arm® architecture and is referred to as FastCV for Arm architecture.
- Implementation runs only on Qualcomm® Snapdragon™ chipsets and is called FastCV for Snapdragon.

FastCV library is Qualcomm proprietary and provides faster implementation of CV algorithms on various hardware as compared to other CV libraries.

OpenCV Acceleration with FastCV HAL and Extensions
--------------------------------------------------

OpenCV and FastCV integration is implemented in two ways:

1. FastCV-based HAL for basic computer vision and arithmetic algorithms acceleration.
2. FastCV module in opencv_contrib with custom algorithms and FastCV function wrappers that do not fit generic OpenCV interface or behaviour.


![](fastcv_hal_extns.png)

Supported Platforms
-------------------

1. Android : Qualcomm Chipsets with the Android from Snapdragon 8 Gen 1 onwards(https://www.qualcomm.com/products/mobile/snapdragon/smartphones#product-list)
2. Linux   : Qualcomm Linux Program related boards mentioned in [Hardware](https://www.qualcomm.com/developer/software/qualcomm-linux/hardware)

Compiling OpenCV with FastCV for Android
----------------------------------------

1. **Follow Wiki page for OpenCV Compilation** : https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build

 Once the OpenCV repository code is cloned into the workspace, please add `-DWITH_FASTCV=ON` flag to cmake vars as below to arm64 entry
 in `opencv/platforms/android/default.config.py` or create new one with the option to enable FastCV HAL and/or extenstions compilation:

 ```
  ABI("3", "arm64-v8a", None, 24, cmake_vars=dict(WITH_FASTCV='ON')),
 ```

2. Remaining steps can be followed as mentioned in [the wiki page](https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build)

Compiling OpenCV with FastCV for Qualcomm Linux
-----------------------------------------------

@note: Only Ubuntu 22.04 is supported as host platform for eSDK deployment.

1. Install eSDK by following [Qualcomm® Linux Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-51/install-sdk.html)

2. After installing the eSDK, set the ESDK_ROOT:

  ```
  export ESDK_ROOT=<eSDK install location>
  ```

3.  Add SDK tools and libraries to your environment:

  ```
  source environment-setup-armv8-2a-qcom-linux
  ```

  If you encounter the following message:
  ```
  Your environment is misconfigured, you probably need to 'unset LD_LIBRARY_PATH'
  but please check why this was set in the first place and that it's safe to unset.
  The SDK will not operate correctly in most cases when LD_LIBRARY_PATH is set.
  ```
  just unset your host `LD_LIBRARY_PATH` environment variable: `unset LD_LIBRARY_PATH`.

4. Clone OpenCV Repositories:

  Clone the OpenCV main and optionally opencv_contrib repositories into any directory
  (it does not need to be inside the SDK directory).

  ```
  git clone https://github.com/opencv/opencv.git
  git clone https://github.com/opencv/opencv_contrib.git
  ```

5. Build OpenCV

  Create a build directory, navigate into it and build the project with CMake there:

  ```
  mkdir build
  cd build
  cmake -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWITH_FASTCV=ON -DBUILD_SHARED_LIBS=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/fastcv/ ../opencv
  make -j$(nproc)
  ```

  If the FastCV library is updated, please replace the old FastCV libraries located at:
  ```
  <ESDK_PATH>\qcom-wayland_sdk\tmp\sysroots\qcs6490-rb3gen2-vision-kit\usr\lib
  ```
  with the latest FastCV libraries downloaded in:
  ```
  build\3rdparty\fastcv\libs
  ```

6. Validate

  Push the OpenCV libraries, test binaries and test data on to the target. Execute the OpenCV conformance or performance tests.
  During runtime, If libwebp.so.7 lib is missing, find the lib in the below Path and push it on the target
  ```
  <ESDK_PATH>\qcom-wayland_sdk\tmp\sysroots\qcs6490-rb3gen2-vision-kit\usr\lib\libwebp.so.7
  ```

HAL and Extension list of APIs
------------------------------

**FastCV based OpenCV HAL APIs list :**

|OpenCV module  |OpenCV API        | Underlying FastCV API for OpenCV acceleration |
|---------------|------------------|-----------------------------------------------|
|IMGPROC        |medianBlur        |fcvFilterMedian3x3u8_v3                        |
|               |sobel             |fcvFilterSobel3x3u8s16                         |
|               |                  |fcvFilterSobel5x5u8s16                         |
|               |                  |fcvFilterSobel7x7u8s16                         |
|               |boxFilter         |fcvBoxFilter3x3u8_v3                           |
|               |                  |fcvBoxFilter5x5u8_v2                           |
|               |                  |fcvBoxFilterNxNf32                             |
|               |adaptiveThreshold |fcvAdaptiveThresholdGaussian3x3u8_v2           |
|               |                  |fcvAdaptiveThresholdGaussian5x5u8_v2           |
|               |                  |fcvAdaptiveThresholdMean3x3u8_v2               |
|               |                  |fcvAdaptiveThresholdMean5x5u8_v2               |
|               |pyrDown           |fcvPyramidCreateu8_v4                          |
|               |cvtColor          |fcvColorRGB888toYCrCbu8_v3                     |
|               |                  |fcvColorRGB888ToHSV888u8                       |
|               |gaussianBlur      |fcvFilterGaussian5x5u8_v3                      |
|               |                  |fcvFilterGaussian3x3u8_v4                      |
|               |warpPerspective   |fcvWarpPerspectiveu8_v5                        |
|               |Canny             |fcvFilterCannyu8                               |
|               |                  |                                               |
|CORE           |lut               | fcvTableLookupu8                              |
|               |norm              |fcvHammingDistanceu8                           |
|               |multiply          |fcvElementMultiplyu8u16_v2                     |
|               |transpose         |fcvTransposeu8_v2                              |
|               |                  |fcvTransposeu16_v2                             |
|               |                  |fcvTransposef32_v2                             |
|               |meanStdDev        |fcvImageIntensityStats_v2                      |
|               |flip              |fcvFlipu8                                      |
|               |                  |fcvFlipu16                                     |
|               |                  |fcvFlipRGB888u8                                |
|               |rotate            |fcvRotateImageu8                               |
|               |                  |fcvRotateImageInterleavedu8                    |
|               |multiply          |fcvElementMultiplyu8                           |
|               |                  |fcvElementMultiplys16                          |
|               |                  |fcvElementMultiplyf32                          |
|               |addWeighted       |fcvAddWeightedu8_v2                            |
|               |subtract          |fcvImageDiffu8f32_v2                           |
|               |SVD & solve       |fcvSVDf32_v2                                   |
|               |gemm              |fcvMatrixMultiplyf32_v2                        |
|               |                  |fcvMultiplyScalarf32                           |
|               |                  |fcvAddf32_v2                                   |


**FastCV based OpenCV Extensions APIs list :**

These OpenCV extension APIs are implemented under the **cv::fastcv** namespace. 

|OpenCV Extension APIs |Underlying FastCV API for OpenCV acceleration |
|----------------------|----------------------------------------------|
|matmuls8s32           |fcvMatrixMultiplys8s32                        |
|clusterEuclidean      |fcvClusterEuclideanu8                         |
|FAST10                |fcvCornerFast10InMaskScoreu8                  |
|                      |fcvCornerFast10InMasku8                       |
|                      |fcvCornerFast10Scoreu8                        |
|                      |fcvCornerFast10u8                             |
|FFT                   |fcvFFTu8                                      |
|IFFT                  |fcvIFFTf32                                    |
|fillConvexPoly        |fcvFillConvexPolyu8                           |
|houghLines            |fcvHoughLineu8                                |
|moments               |fcvImageMomentsu8                             |
|                      |fcvImageMomentss32                            |
|                      |fcvImageMomentsf32                            |
|runMSER               |fcvMserInit                                   |
|                      |fcvMserNN8Init                                |
|                      |fcvMserExtu8_v3                               |
|                      |fcvMserExtNN8u8                               |
|                      |fcvMserNN8u8                                  |
|                      |fcvMserRelease                                |
|remap                 |fcvRemapu8_v2                                 |
|remapRGBA             |fcvRemapRGBA8888BLu8                          |
|                      |fcvRemapRGBA8888NNu8                          |
|resizeDown            |fcvScaleDownBy2u8_v2                          |
|                      |fcvScaleDownBy4u8_v2                          |
|                      |fcvScaleDownMNInterleaveu8                    |
|                      |fcvScaleDownMNu8                              |
|meanShift             |fcvMeanShiftu8                                |
|                      |fcvMeanShifts32                               |
|                      |fcvMeanShiftf32                               |
|bilateralRecursive    |fcvBilateralFilterRecursiveu8                 |
|thresholdRange        |fcvFilterThresholdRangeu8_v2                  |
|bilateralFilter       |fcvBilateralFilter5x5u8_v3                    |
|                      |fcvBilateralFilter7x7u8_v3                    |
|                      |fcvBilateralFilter9x9u8_v3                    |
|calcHist              |fcvImageIntensityHistogram                    |
|gaussianBlur          |fcvFilterGaussian3x3u8_v4                     |
|                      |fcvFilterGaussian5x5u8_v3                     |
|                      |fcvFilterGaussian5x5s16_v3                    |
|                      |fcvFilterGaussian5x5s32_v3                    |
|                      |fcvFilterGaussian11x11u8_v2                   |
|filter2D              |fcvFilterCorrNxNu8                            |
|                      |fcvFilterCorrNxNu8s16                         |
|                      |fcvFilterCorrNxNu8f32                         |
|sepFilter2D           |fcvFilterCorrSepMxNu8                         |
|                      |fcvFilterCorrSep9x9s16_v2                     |
|                      |fcvFilterCorrSep11x11s16_v2                   |
|                      |fcvFilterCorrSep13x13s16_v2                   |
|                      |fcvFilterCorrSep15x15s16_v2                   |
|                      |fcvFilterCorrSep17x17s16_v2                   |
|                      |fcvFilterCorrSepNxNs16                        |
|sobel3x3u8            |fcvImageGradientSobelPlanars8_v2              |
|sobel3x3u8            |fcvImageGradientSobelPlanars16_v2             |
|sobel3x3u8            |fcvImageGradientSobelPlanars16_v3             |
|sobel3x3u8            |fcvImageGradientSobelPlanarf32_v2             |
|sobel3x3u8            |fcvImageGradientSobelPlanarf32_v3             |
|sobel                 |fcvFilterSobel3x3u8_v2                        |
|                      |fcvFilterSobel3x3u8s16                        |
|                      |fcvFilterSobel5x5u8s16                        |
|                      |fcvFilterSobel7x7u8s16                        |
|DCT                   |fcvDCTu8                                      |
|iDCT                  |fcvIDCTs16                                    |
|sobelPyramid          |fcvPyramidAllocate                            |
|                      |fcvPyramidAllocate_v2                         |
|                      |fcvPyramidAllocate_v3                         |
|                      |fcvPyramidSobelGradientCreatei8               |
|                      |fcvPyramidSobelGradientCreatei16              |
|                      |fcvPyramidSobelGradientCreatef32              |
|                      |fcvPyramidDelete                              |
|                      |fcvPyramidDelete_v2                           |
|                      |fcvPyramidCreatef32_v2                        |
|                      |fcvPyramidCreateu8_v4                         |
|trackOpticalFlowLK    |fcvTrackLKOpticalFlowu8_v3                    |
|                      |fcvTrackLKOpticalFlowu8                       |
|warpPerspective2Plane |fcv2PlaneWarpPerspectiveu8                    |
|warpPerspective       |fcvWarpPerspectiveu8_v5                       |
|arithmetic_op         |fcvAddu8                                      |
|                      |fcvAdds16_v2                                  |
|                      |fcvAddf32                                     |
|                      |fcvSubtractu8                                 |
|                      |fcvSubtracts16                                |
|integrateYUV          |fcvIntegrateImageYCbCr420PseudoPlanaru8       |
|normalizeLocalBox     |fcvNormalizeLocalBoxu8                        |
|                      |fcvNormalizeLocalBoxf32                       |
|merge                 |fcvChannelCombine2Planesu8                    |
|                      |fcvChannelCombine3Planesu8                    |
|                      |fcvChannelCombine4Planesu8                    |
|split                 |fcvDeinterleaveu8                             |
|                      |fcvChannelExtractu8                           |
|warpAffine            |fcvTransformAffineu8_v2                       |
|                      |fcvTransformAffineClippedu8_v3                |
|                      |fcv3ChannelTransformAffineClippedBCu8         |


**FastCV QDSP based OpenCV Extension APIs list :**
These OpenCV extension APIs are implemented under the **cv::fastcv::dsp** namespace.
This namespace provides optimized implementations that leverage QDSP (**Qualcomm's Digital Signal Processor**) acceleration using FastCV's Q-suffixed APIs. These functions require DSP initialization (fcvQ6Init).

|OpenCV Extension APIs |Underlying FastCV API for OpenCV acceleration |
|----------------------|----------------------------------------------|
|filter2D              |fcvFilterCorr3x3s8_v2Q                        |
|                      |fcvFilterCorrNxNu8Q                           |
|                      |fcvFilterCorrNxNu8s16Q                        |
|                      |fcvFilterCorrNxNu8f32Q                        |
|FFT                   |fcvFFTu8Q                                     |
|IFFT                  |fcvIFFTf32Q                                   |
|fcvdspinit            |fcvQ6Init                                     |
|fcvdspdeinit          |fcvQ6DeInit                                   |
|Canny                 |fcvFilterCannyu8Q                             |
|sumOfAbsoluteDiffs    |fcvSumOfAbsoluteDiffs8x8u8_v2Q                |
|thresholdOtsu         |fcvFilterThresholdOtsuu8Q                     |

**How to Use FastCV QDSP based OpenCV Extension APIs**

This section outlines the essential steps required to use OpenCV Extension APIs that are accelerated using FastCV on QDSP(**Qualcomm's Digital Signal Processor**).

1. Initialize QDSP:
    - Call **cv::fastcv::dsp::fcvdspinit()** to initialize the QDSP.

2. Allocate memory using **Qualcomm's memory allocator** for all buffers that are being fed to the OpenCV extension API.:
    - Use **cv::fastcv::getQcAllocator()** to assign the allocator to the buffers.
    - Example:
      cv::Mat src;
      src.allocator = cv::fastcv::getQcAllocator(); **// Set Qualcomm's memory allocator**
      \
      After setting Qualcomm's memory allocator, any buffer created using methods like src.create(...), cv::imread(...) etc., will have its memory allocated using Qualcomm's memory allocator.

3. Call the OpenCV extension API from 'cv::fastcv::dsp':
    - Example: **cv::fastcv::dsp::thresholdOtsu(src, dst, binaryType);**
      where 'src' and 'dst' are 'cv::Mat' objects with the Qualcomm's memory allocator,
      and 'binaryType' is a boolean indicating the thresholding mode.

4. Deinitialize QDSP:
    - Call **cv::fastcv::dsp::fcvdspdeinit()** to deinitialize the QDSP.


**Reference Example**:
Refer to a working test case using the OpenCV Extension APIs in the opencv_contrib repository:[opencv_contrib/modules/fastcv/test/test_thresh_dsp.cpp](https://github.com/opencv/opencv_contrib/blob/4.x/modules/fastcv/test/test_thresh_dsp.cpp)