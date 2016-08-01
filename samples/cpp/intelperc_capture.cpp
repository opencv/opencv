// testOpenCVCam.cpp : Defines the entry point for the console application.
//

#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static bool g_printStreamSetting;
static int g_imageStreamProfileIdx;
static int g_depthStreamProfileIdx;
static bool g_irStreamShow;
static double g_imageBrightness;
static double g_imageContrast;
static bool g_printTiming;
static bool g_showClosedPoint;


static int g_closedDepthPoint[2];

static void printUsage(const char *arg0)
{
    const char *filename = arg0;
    while (*filename)
        filename++;
    while ((arg0 <= filename) && ('\\' != *filename) && ('/' != *filename))
        filename--;
    filename++;

    cout << "This program demonstrates usage of camera supported\nby Intel Perceptual computing SDK." << endl << endl;
    cout << "usage: " << filename << "[-ps] [-isp=IDX] [-dsp=IDX]\n [-ir] [-imb=VAL] [-imc=VAL]" << endl << endl;
    cout << "   -ps,            print streams setting and profiles" << endl;
    cout << "   -isp=IDX,       set profile index of the image stream" << endl;
    cout << "   -dsp=IDX,       set profile index of the depth stream" << endl;
    cout << "   -ir,            show data from IR stream" << endl;
    cout << "   -imb=VAL,       set brighness value for a image stream" << endl;
    cout << "   -imc=VAL,       set contrast value for a image stream" << endl;
    cout << "   -pts,           print frame index and frame time" << endl;
    cout << "   --show-closed,  print frame index and frame time" << endl;
    cout <<  endl;
}

static void parseCMDLine(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
        "{ h help | | }"
        "{ ps print-streams | | }"
        "{ isp image-stream-prof | -1 | }"
        "{ dsp depth-stream-prof | -1 | }"
        "{ir||}{imb||}{imc||}{pts||}{show-closed||}");
    if (parser.has("h"))
    {
        printUsage(argv[0]);
        exit(0);
    }
    g_printStreamSetting = parser.has("ps");
    g_imageStreamProfileIdx = parser.get<int>("isp");
    g_depthStreamProfileIdx = parser.get<int>("dsp");
    g_irStreamShow = parser.has("ir");
    if (parser.has("imb"))
        g_imageBrightness = parser.get<double>("imb");
    else
        g_imageBrightness = -DBL_MAX;
    if (parser.has("imc"))
        g_imageContrast = parser.get<double>("imc");
    else
        g_imageContrast = -DBL_MAX;
    g_printTiming = parser.has("pts");
    g_showClosedPoint = parser.has("show-closed");
    if (!parser.check())
    {
        parser.printErrors();
        exit(-1);
    }
    if (g_showClosedPoint && (-1 == g_depthStreamProfileIdx))
    {
        cerr << "For --show-closed depth profile has be selected" << endl;
        exit(-1);
    }
}

static void printStreamProperties(VideoCapture &capture)
{
    size_t profilesCount = (size_t)capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_INTELPERC_PROFILE_COUNT);
    cout << "Image stream." << endl;
    cout << "  Brightness = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_BRIGHTNESS) << endl;
    cout << "  Contrast = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_CONTRAST) << endl;
    cout << "  Saturation = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_SATURATION) << endl;
    cout << "  Hue = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_HUE) << endl;
    cout << "  Gamma = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_GAMMA) << endl;
    cout << "  Sharpness = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_SHARPNESS) << endl;
    cout << "  Gain = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_GAIN) << endl;
    cout << "  Backligh = " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_BACKLIGHT) << endl;
    cout << "Image streams profiles:" << endl;
    for (size_t i = 0; i < profilesCount; i++)
    {
        capture.set(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, (double)i);
        cout << "  Profile[" << i << "]: ";
        cout << "width = " <<
            (int)capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_FRAME_WIDTH);
        cout << ", height = " <<
            (int)capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_FRAME_HEIGHT);
        cout << ", fps = " <<
            capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_FPS);
        cout << endl;
    }

    profilesCount = (size_t)capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_COUNT);
    cout << "Depth stream." << endl;
    cout << "  Low confidence value = " << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE) << endl;
    cout << "  Saturation value = " << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE) << endl;
    cout << "  Confidence threshold = " << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD) << endl;
    cout << "  Focal length = (" << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ) << ", "
        << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT) << ")" << endl;
    cout << "Depth streams profiles:" << endl;
    for (size_t i = 0; i < profilesCount; i++)
    {
        capture.set(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, (double)i);
        cout << "  Profile[" << i << "]: ";
        cout << "width = " <<
            (int)capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_FRAME_WIDTH);
        cout << ", height = " <<
            (int)capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_FRAME_HEIGHT);
        cout << ", fps = " <<
            capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_FPS);
        cout << endl;
    }
}

static void imshowImage(const char *winname, Mat &image, VideoCapture &capture)
{
    if (g_showClosedPoint)
    {
        Mat uvMap;
        if (capture.retrieve(uvMap, CAP_INTELPERC_UVDEPTH_MAP))
        {
            float *uvmap = (float *)uvMap.ptr() + 2 * (g_closedDepthPoint[0] * uvMap.cols + g_closedDepthPoint[1]);
            int x = (int)((*uvmap) * image.cols); uvmap++;
            int y = (int)((*uvmap) * image.rows);

            if ((0 <= x) && (0 <= y))
            {
                static const int pointSize = 4;
                for (int row = y; row < min(y + pointSize, image.rows); row++)
                {
                    uchar* ptrDst = image.ptr(row) + x * 3 + 2;//+2 -> Red
                    for (int col = 0; col < min(pointSize, image.cols - x); col++, ptrDst+=3)
                    {
                        *ptrDst = 255;
                    }
                }
            }
        }
    }
    imshow(winname, image);
}
static void imshowIR(const char *winname, Mat &ir)
{
    Mat image;
    if (g_showClosedPoint)
    {
        image.create(ir.rows, ir.cols, CV_8UC3);
        for (int row = 0; row < ir.rows; row++)
        {
            uchar* ptrDst = image.ptr(row);
            short* ptrSrc = (short*)ir.ptr(row);
            for (int col = 0; col < ir.cols; col++, ptrSrc++)
            {
                uchar val = (uchar) ((*ptrSrc) >> 2);
                *ptrDst = val;  ptrDst++;
                *ptrDst = val;  ptrDst++;
                *ptrDst = val;  ptrDst++;
            }
        }

        static const int pointSize = 4;
        for (int row = g_closedDepthPoint[0]; row < min(g_closedDepthPoint[0] + pointSize, image.rows); row++)
        {
            uchar* ptrDst = image.ptr(row) + g_closedDepthPoint[1] * 3 + 2;//+2 -> Red
            for (int col = 0; col < min(pointSize, image.cols - g_closedDepthPoint[1]); col++, ptrDst+=3)
            {
                *ptrDst = 255;
            }
        }
    }
    else
    {
        image.create(ir.rows, ir.cols, CV_8UC1);
        for (int row = 0; row < ir.rows; row++)
        {
            uchar* ptrDst = image.ptr(row);
            short* ptrSrc = (short*)ir.ptr(row);
            for (int col = 0; col < ir.cols; col++, ptrSrc++, ptrDst++)
            {
                *ptrDst = (uchar) ((*ptrSrc) >> 2);
            }
        }
    }

    imshow(winname, image);
}
static void imshowDepth(const char *winname, Mat &depth, VideoCapture &capture)
{
    short lowValue = (short)capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE);
    short saturationValue = (short)capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE);

    Mat image;
    if (g_showClosedPoint)
    {
        image.create(depth.rows, depth.cols, CV_8UC3);
        for (int row = 0; row < depth.rows; row++)
        {
            uchar* ptrDst = image.ptr(row);
            short* ptrSrc = (short*)depth.ptr(row);
            for (int col = 0; col < depth.cols; col++, ptrSrc++)
            {
                if ((lowValue == (*ptrSrc)) || (saturationValue == (*ptrSrc)))
                {
                    *ptrDst = 0; ptrDst++;
                    *ptrDst = 0; ptrDst++;
                    *ptrDst = 0; ptrDst++;
                }
                else
                {
                    uchar val = (uchar) ((*ptrSrc) >> 2);
                    *ptrDst = val;  ptrDst++;
                    *ptrDst = val;  ptrDst++;
                    *ptrDst = val;  ptrDst++;
                }
            }
        }

        static const int pointSize = 4;
        for (int row = g_closedDepthPoint[0]; row < min(g_closedDepthPoint[0] + pointSize, image.rows); row++)
        {
            uchar* ptrDst = image.ptr(row) + g_closedDepthPoint[1] * 3 + 2;//+2 -> Red
            for (int col = 0; col < min(pointSize, image.cols - g_closedDepthPoint[1]); col++, ptrDst+=3)
            {
                *ptrDst = 255;
            }
        }
    }
    else
    {
        image.create(depth.rows, depth.cols, CV_8UC1);
        for (int row = 0; row < depth.rows; row++)
        {
            uchar* ptrDst = image.ptr(row);
            short* ptrSrc = (short*)depth.ptr(row);
            for (int col = 0; col < depth.cols; col++, ptrSrc++, ptrDst++)
            {
                if ((lowValue == (*ptrSrc)) || (saturationValue == (*ptrSrc)))
                    *ptrDst = 0;
                else
                    *ptrDst = (uchar) ((*ptrSrc) >> 2);
            }
        }
    }
    imshow(winname, image);
}

int main(int argc, char* argv[])
{
    parseCMDLine(argc, argv);

    VideoCapture capture;
    capture.open(CAP_INTELPERC);
    if (!capture.isOpened())
    {
        cerr << "Can not open a capture object." << endl;
        return -1;
    }

    if (g_printStreamSetting)
        printStreamProperties(capture);

    if (-1 != g_imageStreamProfileIdx)
    {
        if (!capture.set(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, (double)g_imageStreamProfileIdx))
        {
            cerr << "Can not setup a image stream." << endl;
            return -1;
        }
    }
    if (-1 != g_depthStreamProfileIdx)
    {
        if (!capture.set(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, (double)g_depthStreamProfileIdx))
        {
            cerr << "Can not setup a depth stream." << endl;
            return -1;
        }
    }
    else if (g_irStreamShow)
    {
        if (!capture.set(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, 0.0))
        {
            cerr << "Can not setup a IR stream." << endl;
            return -1;
        }
    }
    else
    {
        cout << "Streams not selected" << endl;
        return 0;
    }

    //Setup additional properies only after set profile of the stream
    if ( (-10000.0 < g_imageBrightness) && (g_imageBrightness < 10000.0))
        capture.set(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_BRIGHTNESS, g_imageBrightness);
    if ( (0 < g_imageContrast) && (g_imageContrast < 10000.0))
        capture.set(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_BRIGHTNESS, g_imageContrast);

    int frame = 0;
    for(;;frame++)
    {
        Mat bgrImage;
        Mat depthImage;
        Mat irImage;

        if (!capture.grab())
        {
            cout << "Can not grab images." << endl;
            return -1;
        }

        if ((-1 != g_depthStreamProfileIdx) && (capture.retrieve(depthImage, CAP_INTELPERC_DEPTH_MAP)))
        {
            if (g_showClosedPoint)
            {
                double minVal = 0.0; double maxVal = 0.0;
                minMaxIdx(depthImage, &minVal, &maxVal, g_closedDepthPoint);
            }
            imshowDepth("depth image", depthImage, capture);
        }
        if ((g_irStreamShow) && (capture.retrieve(irImage, CAP_INTELPERC_IR_MAP)))
            imshowIR("ir image", irImage);
        if ((-1 != g_imageStreamProfileIdx) && (capture.retrieve(bgrImage, CAP_INTELPERC_IMAGE)))
            imshowImage("color image", bgrImage, capture);

        if (g_printTiming)
        {
            cout << "Image frame: " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_POS_FRAMES)
                 << ", Depth(IR) frame: " << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_POS_FRAMES) << endl;
            cout << "Image frame: " << capture.get(CAP_INTELPERC_IMAGE_GENERATOR | CAP_PROP_POS_MSEC)
                 << ", Depth(IR) frame: " << capture.get(CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_POS_MSEC) << endl;
        }
        if( waitKey(30) >= 0 )
            break;
    }

    return 0;
}
