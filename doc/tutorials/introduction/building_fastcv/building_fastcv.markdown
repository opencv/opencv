Building Opencv with Fastcv {#tutorial_building_fastcv}
===================================

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.11.0 |

Enable Opencv with Fastcv for Qcom Chipsets
-------------------------------------------
This Document scope is to guide the Developers to enable Opencv Acceleration with Fastcv for the Qualcomm chipsets on arm64 architecture.
Enablement of  Opencv with Fastcv backend on Non-Qualcomm chipsets or Linux platforms other than [Qualcomm Linux](https://www.qualcomm.com/developer/software/qualcomm-linux) is currently out of scope 

About Fastcv :
-------------------------------------------
FastCV provides two main features to computer vision application developers: 
- First, it provides a library of frequently used computer vision (CV) functions, optimized to run efficiently on a wide variety of Qualcomm’s Snapdragon devices. 
- Second, it provides a clean processor-agnostic hardware acceleration API, under which chipset vendors can hardware accelerate FastCV functions on Qualcomm’s Snapdragon hardware.

FastCV is released as a unified binary, a single binary containing two implementations of the library. 
- The first implementation runs on Arm® architecture and is referred to as FastCV for Arm architecture. 
- The second implementation runs only on Qualcomm® Snapdragon™ chipsets and is called FastCV for Snapdragon. 
FastCV library is Qualcomm proprietary and provides faster implementation of CV algorithms on various HW as compared to other CV libraries. 

Opencv Acceleration with Fastcv HAL/ Extensions :
-------------------------------------------
1.	Accelerates OpenCV APIs using FastCV as the backend, resulting in improved performance.
2.	Enhances OpenCV functionality with FastCV-based HAL and Extension APIs, which were not available in earlier versions that only included the default OpenCV library.
3.	CV applications can be developed using the standard OpenCV APIs. The OpenCV library invokes FastCV HAL APIs, which in turn call FastCV algorithms.


![](fastcv_hal_extns.png)

Supported Platforms : 
-------------------------------------------
+ Android : Qualcomm Chipsets with the Android from Snapdragon 8 Gen 1 onwards(https://www.qualcomm.com/products/mobile/snapdragon/smartphones#product-list)
+ Linux   : Qualcomm Linux Program related boards mentioned in Hardware

Compiling Opencv with Fastcv for Android :
-------------------------------------------
1.	**Follow Wiki page for Opencv Compilation** : https://github.com/opencv/opencv/wiki/Custom-OpenCV-Android-SDK-and-AAR-package-build

 Once the Opencv repository code is cloned into the workspace , Please add **WITH_FASTCV** flag to cmake vars as below to arm64 entry in **opencv\platforms\android\ndk-18-api-level-24.config.py**  to enable Fastcv HAL/Extenstions Compilation

 ```
  ABI("3", "arm64-v8a", None, 24, cmake_vars=dict(WITH_FASTCV='ON')),
 ```
2.	Remaining steps can be followed as mentioned in the above wiki page 

Compiling Opencv with FastCV for Qualcomm Linux :
-------------------------------------------
1.	Install eSDK by following [Qualcomm® Linux Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-51/install-sdk.html?vproduct=1601111740013072&version=1.3&facet=Qualcomm%20Intelligent%20Multimedia%20Product%20(QIMP)%20SDK&state=preview)

2.	After installing the eSDK, set the ESDK_ROOT:
```
export ESDK_ROOT=/local/mnt/workspace/<PATH>
 ```
Go to the directory where the SDK was installed:
```
cd $ESDK_ROOT
```
3.  Environment Setup

During the execution of the command
```
source environment-setup-armv8-2a-qcom-linux
```
if you encounter the following message:
```
Your environment is misconfigured, you probably need to 'unset LD_LIBRARY_PATH'
but please check why this was set in the first place and that it's safe to unset.
The SDK will not operate correctly in most cases when LD_LIBRARY_PATH is set.
```
Then Follow these steps:
 + Unset LD_LIBRARY_PATH:unset LD_LIBRARY_PATH

4. Clone OpenCV Repositories

 You can clone the OpenCV repositories in any directory (it does not need to be inside the SDK directory).
 + Clone the main OpenCV repository:
```
git clone https://github.com/opencv/opencv.git
```

 + Clone the OpenCV contrib repository:
```
git clone https://github.com/opencv/opencv_contrib.git
```
5. Build OpenCV
 + Create a build directory and navigate into it:
```
mkdir build
cd build
```
 + Configure the build with CMake:
```
cmake -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWITH_FASTCV=ON -DBUILD_SHARED_LIBS=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/fastcv/ ../opencv
```
 + Compile the code:
```
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
6. Validation
Push the Opencv libraries , test binaries and test data on to the target . Execute the OpenCV Conformance or Performance tests
During runtime, If libwebp.so.7 lib is missing, find the lib in the below Path and push it on the target
```
<ESDK_PATH>\qcom-wayland_sdk\tmp\sysroots\qcs6490-rb3gen2-vision-kit\usr\lib\libwebp.so.7
```

Sample Apps for Fastcv based Extension APIs :
-------------------------------------------
Sample application examples for calling the Fastcv based Extension APIs are covered in this link : https://github.com/opencv/opencv_contrib/tree/4.x/modules/fastcv/test


HAL and Extension list of APIs :
-------------------------------------------

**Fastcv based Opencv HAL APIs list :**

|OpenCV module	|OpenCV API	        | Underlying FastCV API for OpenCV acceleration|
| --------------| ------------------|--------------------------------------------- |
|IMGPROC	    |medianBlur		 	|fcvFilterMedian3x3u8_v3                       |
|	            |sobel	 	    	|fcvFilterSobel3x3u8s16                        |
|				|					|fcvFilterSobel5x5u8s16                        |
|				|					|fcvFilterSobel7x7u8s16                        |
|	            |boxFilter		 	|fcvBoxFilter3x3u8_v3                          |
|				|					|fcvBoxFilter5x5u8_v2                          |
|	            |adaptiveThreshold	|fcvAdaptiveThresholdGaussian3x3u8_v2          |
|				|					|fcvAdaptiveThresholdGaussian5x5u8_v2          |
|				|					|fcvAdaptiveThresholdMean3x3u8_v2              |
|				|					|fcvAdaptiveThresholdMean5x5u8_v2              |
|				|pyrUp & pyrDown	|fcvPyramidCreateu8_v4      	               |
|				|cvtColor			|fcvColorRGB888toYCrCbu8_v3                    |
|				|					|fcvColorRGB888ToHSV888u8                      |
|				|GaussianBlur		|fcvFilterGaussian5x5u8_v3                     |
|				|					|fcvFilterGaussian3x3u8_v4                     |
|				|cvWarpPerspective	|fcvWarpPerspectiveu8_v5	                   |
|				|Canny				|fcvFilterCannyu8                              |
|               |                   |                                              |
|CORE			|lut				|	fcvTableLookupu8                           |
|				|norm				|fcvHammingDistanceu8                          |
|				|multiply			|fcvElementMultiplyu8u16_v2                    |
|				|transpose			|fcvTransposeu8_v2                             |
|				|					|fcvTransposeu16_v2                            |
|				|					|fcvTransposef32_v2                            |
|				|meanStdDev			|fcvImageIntensityStats_v2                     |
|				|flip				|fcvFlipu8                                     |
|				|					|fcvFlipu16                                    |
|				|					|fcvFlipRGB888u8                               |
|				|rotate				|fcvRotateImageu8                              |
|				|					|fcvRotateImageInterleavedu8                   |
|				|multiply			|fcvElementMultiplyu8                          |
|				|					|fcvElementMultiplys16                         |
|				|					|fcvElementMultiplyf32                         |
|				|addWeighted		|	fcvAddWeightedu8_v2                        |


**Fastcv based Opencv Extensions APIs list :**

|OpenCV Extension APIs  |Underlying FastCV API for OpenCV acceleration|
| --------------------  |---------------------------------------------|
|matmuls8s32			|fcvMatrixMultiplys8s32                       |
|clusterEuclidean		|fcvClusterEuclideanu8                        |
|FAST10					|fcvCornerFast10InMaskScoreu8                 |
|						|fcvCornerFast10InMasku8                      |
|						|fcvCornerFast10Scoreu8                       |
|						|fcvCornerFast10u8                            |
|FFT					|fcvFFTu8                                     |
|IFFT					|fcvIFFTf32                                   |
|fillConvexPoly			|fcvFillConvexPolyu8                          |
|houghLines				|fcvHoughLineu8                               |
|moments				|fcvImageMomentsu8                            |
|						|fcvImageMomentss32                           |
|						|fcvImageMomentsf32                           |
|runMSER				|fcvMserInit                                  |
|						|fcvMserNN8Init                               |
|						|fcvMserExtu8_v3                              |
|						|fcvMserExtNN8u8                              |
|						|fcvMserNN8u8                                 |
|						|fcvMserRelease                               |
|remap					|fcvRemapu8_v2                                |
|remapRGBA				|fcvRemapRGBA8888BLu8                         |
|						|fcvRemapRGBA8888NNu8                         |
|resizeDownBy2			|fcvScaleDownBy2u8_v2                         |
|resizeDownBy4			|fcvScaleDownBy4u8_v2                         |
|meanShift				|fcvMeanShiftu8                               |
|						|fcvMeanShifts32                              |
|						|fcvMeanShiftf32                              |
|bilateralRecursive		|fcvBilateralFilterRecursiveu8                |
|thresholdRange			|fcvFilterThresholdRangeu8_v2                 |
|bilateralFilter		|fcvBilateralFilter5x5u8_v3                   |
|						|fcvBilateralFilter7x7u8_v3                   |
|						|fcvBilateralFilter9x9u8_v3                   |
|calcHist				|fcvImageIntensityHistogram                   |
|gaussianBlur			|fcvFilterGaussian3x3u8_v4                    |
|						|fcvFilterGaussian5x5u8_v3                    |
|						|fcvFilterGaussian5x5s16_v3                   |
|						|fcvFilterGaussian5x5s32_v3                   |
|						|fcvFilterGaussian11x11u8_v2                  |
|filter2D				|fcvFilterCorrNxNu8                           |
|						|fcvFilterCorrNxNu8s16                        |
|						|fcvFilterCorrNxNu8f32                        |
|sepFilter2D			|fcvFilterCorrSepMxNu8                        |
|						|fcvFilterCorrSep9x9s16_v2                    |
|						|fcvFilterCorrSep11x11s16_v2                  |
|						|fcvFilterCorrSep13x13s16_v2                  |
|						|fcvFilterCorrSep15x15s16_v2                  |
|						|fcvFilterCorrSep17x17s16_v2                  |
|						|fcvFilterCorrSepNxNs16                       |
|sobel3x3u8				|fcvImageGradientSobelPlanars8_v2             |
|sobel3x3u9				|fcvImageGradientSobelPlanars16_v2            |
|sobel3x3u10			|fcvImageGradientSobelPlanars16_v3            |
|sobel3x3u11			|fcvImageGradientSobelPlanarf32_v2            |
|sobel3x3u12			|fcvImageGradientSobelPlanarf32_v3            |
|sobel					|fcvFilterSobel3x3u8_v2                       |
|						|fcvFilterSobel3x3u8s16                       |
|						|fcvFilterSobel5x5u8s16                       |
|						|fcvFilterSobel7x7u8s16                       |
|DCT					|fcvDCTu8                                     |
|iDCT					|fcvIDCTs16                                   |
|sobelPyramid			|fcvPyramidAllocate                           |
|						|fcvPyramidAllocate_v2                        |
|						|fcvPyramidAllocate_v3                        |
|						|fcvPyramidSobelGradientCreatei8              |
|						|fcvPyramidSobelGradientCreatei16             |
|						|fcvPyramidSobelGradientCreatef32             |
|						|fcvPyramidDelete                             |
|						|fcvPyramidDelete_v2                          |
|						|fcvPyramidCreatef32_v2                       |
|						|fcvPyramidCreateu8_v4                        |
|trackOpticalFlowLK		|fcvTrackLKOpticalFlowu8_v3                   |
|						|fcvTrackLKOpticalFlowu8                      |
|warpPerspective2Plane	|fcv2PlaneWarpPerspectiveu8                   |