#include <stdio.h>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "cvconfig.h"

using namespace std;
using namespace cv;

#ifdef HAVE_IPP_A
#include "opencv2/core/ippasync.hpp"

#define CHECK_STATUS(STATUS, NAME)\
    if(STATUS!=HPP_STATUS_NO_ERROR){ printf("%s error %d\n", NAME, STATUS);\
    if (virtMatrix) {hppStatus delSts = hppiDeleteVirtualMatrices(accel, virtMatrix); CHECK_DEL_STATUS(delSts,"hppiDeleteVirtualMatrices");}\
    if (accel)      {hppStatus delSts = hppDeleteInstance(accel); CHECK_DEL_STATUS(delSts, "hppDeleteInstance");}\
    return -1;}

#define CHECK_DEL_STATUS(STATUS, NAME)\
    if(STATUS!=HPP_STATUS_NO_ERROR){ printf("%s error %d\n", NAME, STATUS); return -1;}

#endif

static void help()
{
 printf("\nThis program shows how to use the conversion for IPP Async.\n"
"This example uses the Sobel filter.\n"
"You can use cv::Sobel or hppiSobel.\n"
"Usage: \n"
"./ipp_async_sobel [--camera]=<use camera,if this key is present>, \n"
"                  [--file_name]=<path to movie or image file>\n"
"                  [--accel]=<accelerator type: auto (default), cpu, gpu>\n\n");
}

const char* keys =
{
    "{c  camera   |           | use camera or not}"
    "{fn file_name|../data/baboon.jpg | image file       }"
    "{a accel     |auto       | accelerator type: auto (default), cpu, gpu}"
};

//this is a sample for hppiSobel functions
int main(int argc, const char** argv)
{
    help();

    VideoCapture cap;
    CommandLineParser parser(argc, argv, keys);
    Mat image, gray, result;

#ifdef HAVE_IPP_A

    hppiMatrix* src,* dst;
    hppAccel accel = 0;
    hppAccelType accelType;
    hppStatus sts;
    hppiVirtualMatrix * virtMatrix;

    bool useCamera = parser.has("camera");
    string file = parser.get<string>("file_name");
    string sAccel = parser.get<string>("accel");

    parser.printMessage();

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

    accelType = sAccel == "cpu" ? HPP_ACCEL_TYPE_CPU:
                sAccel == "gpu" ? HPP_ACCEL_TYPE_GPU:
                                  HPP_ACCEL_TYPE_ANY;

    //Create accelerator instance
    sts = hppCreateInstance(accelType, 0, &accel);
    CHECK_STATUS(sts, "hppCreateInstance");

    accelType = hppQueryAccelType(accel);

    sAccel = accelType == HPP_ACCEL_TYPE_CPU ? "cpu":
             accelType == HPP_ACCEL_TYPE_GPU ? "gpu":
             accelType == HPP_ACCEL_TYPE_GPU_VIA_DX9 ? "gpu dx9": "?";

    printf("accelType %s\n", sAccel.c_str());

    virtMatrix = hppiCreateVirtualMatrices(accel, 1);

    for(;;)
    {
        cap >> image;
        if(image.empty())
            break;

        cvtColor( image, gray, COLOR_BGR2GRAY );

        result.create( image.rows, image.cols, CV_8U);

        double execTime = (double)getTickCount();

        //convert Mat to hppiMatrix
        src = hpp::getHpp(gray,accel);
        dst = hpp::getHpp(result,accel);

        sts = hppiSobel(accel,src, HPP_MASK_SIZE_3X3,HPP_NORM_L1,virtMatrix[0]);
        CHECK_STATUS(sts,"hppiSobel");

        sts = hppiConvert(accel, virtMatrix[0], 0, HPP_RND_MODE_NEAR, dst, HPP_DATA_TYPE_8U);
        CHECK_STATUS(sts,"hppiConvert");

        // Wait for tasks to complete
        sts = hppWait(accel, HPP_TIME_OUT_INFINITE);
        CHECK_STATUS(sts, "hppWait");

        execTime = ((double)getTickCount() - execTime)*1000./getTickFrequency();

        printf("Time : %0.3fms\n", execTime);

        imshow("image", image);
        imshow("rez", result);

        waitKey(15);

        sts =  hppiFreeMatrix(src);
        CHECK_DEL_STATUS(sts,"hppiFreeMatrix");

        sts =  hppiFreeMatrix(dst);
        CHECK_DEL_STATUS(sts,"hppiFreeMatrix");

    }

    if (!useCamera)
        waitKey(0);

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

    printf("SUCCESS\n");

#else

    printf("IPP Async not supported\n");

#endif

    return 0;
}
