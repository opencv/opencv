Intel® IPP Asynchronous C/C++ library in OpenCV {#tutorial_how_to_use_ippa_conversion}
===============================================

Goal
----

The tutorial demonstrates the [Intel® IPP Asynchronous
C/C++](http://software.intel.com/en-us/intel-ipp-preview) library usage with OpenCV. The code
example below illustrates implementation of the Sobel operation, accelerated with Intel® IPP
Asynchronous C/C++ functions. In this code example, @ref cv::hpp::getMat and @ref cv::hpp::getHpp
functions are used for data conversion between
[hppiMatrix](http://software.intel.com/en-us/node/501660) and Mat matrices.

Code
----

You may also find the source code in the
`samples/cpp/tutorial_code/core/ippasync/ippasync_sample.cpp` file of the OpenCV source library or
download it from [here](https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/core/ippasync/ippasync_sample.cpp).

@include cpp/tutorial_code/core/ippasync/ippasync_sample.cpp

Explanation
-----------

-#  Create parameters for OpenCV:
    @code{.cpp}
    VideoCapture cap;
    Mat image, gray, result;
    @endcode
    and IPP Async:
    @code{.cpp}
    hppiMatrix* src,* dst;
    hppAccel accel = 0;
    hppAccelType accelType;
    hppStatus sts;
    hppiVirtualMatrix * virtMatrix;
    @endcode
-#  Load input image or video. How to open and read video stream you can see in the
    @ref tutorial_video_input_psnr_ssim tutorial.
    @code{.cpp}
    if( useCamera )
    {
       printf("used camera\n");
       cap.open(0);
    }
    else
    {
       printf("used image %s\n", file.c_str());
       cap.open(file.c_str());
    }

    if( !cap.isOpened() )
    {
       printf("can not open camera or video file\n");
       return -1;
    }
    @endcode
-#  Create accelerator instance using
    [hppCreateInstance](http://software.intel.com/en-us/node/501686):
    @code{.cpp}
    accelType = sAccel == "cpu" ? HPP_ACCEL_TYPE_CPU:
                sAccel == "gpu" ? HPP_ACCEL_TYPE_GPU:
                                  HPP_ACCEL_TYPE_ANY;

    //Create accelerator instance
    sts = hppCreateInstance(accelType, 0, &accel);
    CHECK_STATUS(sts, "hppCreateInstance");
    @endcode
-#  Create an array of virtual matrices using
    [hppiCreateVirtualMatrices](http://software.intel.com/en-us/node/501700) function.
    @code{.cpp}
    virtMatrix = hppiCreateVirtualMatrices(accel, 1);
    @endcode
-#  Prepare a matrix for input and output data:
    @code{.cpp}
    cap >> image;
    if(image.empty())
       break;

    cvtColor( image, gray, COLOR_BGR2GRAY );

    result.create( image.rows, image.cols, CV_8U);
    @endcode
-#  Convert Mat to [hppiMatrix](http://software.intel.com/en-us/node/501660) using @ref cv::hpp::getHpp
    and call [hppiSobel](http://software.intel.com/en-us/node/474701) function.
    @code{.cpp}
    //convert Mat to hppiMatrix
    src = getHpp(gray, accel);
    dst = getHpp(result, accel);

    sts = hppiSobel(accel,src, HPP_MASK_SIZE_3X3,HPP_NORM_L1,virtMatrix[0]);
    CHECK_STATUS(sts,"hppiSobel");

    sts = hppiConvert(accel, virtMatrix[0], 0, HPP_RND_MODE_NEAR, dst, HPP_DATA_TYPE_8U);
    CHECK_STATUS(sts,"hppiConvert");

    // Wait for tasks to complete
    sts = hppWait(accel, HPP_TIME_OUT_INFINITE);
    CHECK_STATUS(sts, "hppWait");
    @endcode
    We use [hppiConvert](http://software.intel.com/en-us/node/501746) because
    [hppiSobel](http://software.intel.com/en-us/node/474701) returns destination matrix with
    HPP_DATA_TYPE_16S data type for source matrix with HPP_DATA_TYPE_8U type. You should check
    hppStatus after each call IPP Async function.

-#  Create windows and show the images, the usual way.
    @code{.cpp}
    imshow("image", image);
    imshow("rez", result);

    waitKey(15);
    @endcode
-#  Delete hpp matrices.
    @code{.cpp}
    sts =  hppiFreeMatrix(src);
    CHECK_DEL_STATUS(sts,"hppiFreeMatrix");

    sts =  hppiFreeMatrix(dst);
    CHECK_DEL_STATUS(sts,"hppiFreeMatrix");
    @endcode
-#  Delete virtual matrices and accelerator instance.
    @code{.cpp}
    if (virtMatrix)
    {
       sts = hppiDeleteVirtualMatrices(accel, virtMatrix);
       CHECK_DEL_STATUS(sts,"hppiDeleteVirtualMatrices");
    }

    if (accel)
    {
       sts = hppDeleteInstance(accel);
       CHECK_DEL_STATUS(sts, "hppDeleteInstance");
    }
    @endcode

Result
------

After compiling the code above we can execute it giving an image or video path and accelerator type
as an argument. For this tutorial we use baboon.png image as input. The result is below.

![](images/How_To_Use_IPPA_Result.jpg)
