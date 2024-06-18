#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <list>
#include <iostream>

#if defined(HAVE_THREADS)
#define USE_THREADS 1
#endif

#ifdef USE_THREADS
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#endif

using namespace cv;
using namespace std;

// Stores frames along with their timestamps
struct Frame
{
    int64 timestamp;
    Mat frame;
};

static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp=-1.f)
{
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1);

    if(maxDisp <= 0)
    {
        maxDisp = 0;
        minMaxLoc(gray, 0, &maxDisp);
    }

    rgb.create(gray.size(), CV_8UC3);
    rgb = Scalar::all(0);
    if(maxDisp < 1)
        return;

    Mat tmp;
    convertScaleAbs(gray, tmp, 255.f / maxDisp);
    applyColorMap(tmp, rgb, COLORMAP_JET);
}

static float getMaxDisparity(VideoCapture& capture)
{
    const int minDistance = 400; // mm
    float b = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_BASELINE); // mm
    float F = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH); // pixels
    return b * F / minDistance;
}

static int camOpenni()
{
    VideoCapture capture(CAP_OPENNI2);
   if (!capture.isOpened())
   {
        cerr << "ERROR: Failed to open a capture object." << endl;
        return -1;
   }

    capture.set(CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ);
    capture.set(CAP_OPENNI_DEPTH_GENERATOR_PRESENT, true);
    capture.set(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, true);
    capture.set(CAP_OPENNI_IR_GENERATOR_PRESENT, true);

    Mat depthMap;
    Mat bgrImage;
    Mat irImage;
    Mat disparityMap;

    for(;;)
    {
        if (!capture.grab())
        {
            cerr << "ERROR: Can not grab images." << endl;
            return -1;
        }
        else
        {
            if(capture.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP))
            {
                const float scaleFactor = 0.05f;
                Mat show;
                depthMap.convertTo(show, CV_8UC1, scaleFactor);
                imshow("DEPTH", show);
            }

            if(capture.retrieve(bgrImage, CAP_OPENNI_BGR_IMAGE))
                imshow("RGB", bgrImage);

            if(capture.retrieve(irImage, CAP_OPENNI_IR_IMAGE))
            {
                const float scaleFactor = 256.0f / 3500;
                Mat ir8;
                irImage.convertTo(ir8, CV_8U, scaleFactor);
                imshow("IR", ir8);
            }

            if(capture.retrieve(disparityMap, CAP_OPENNI_DISPARITY_MAP))
            {
                Mat colorDisparityMap;
                colorizeDisparity(disparityMap, colorDisparityMap, getMaxDisparity(capture));
                Mat validColorDisparityMap;
                colorDisparityMap.copyTo(validColorDisparityMap, disparityMap != 0);
                imshow("Colorized Disparity Map", validColorDisparityMap);
            }
        }

        if(waitKey(30) >= 0)
            break;
    }

    return 0;
}

static int camOpenniOrbbec()
{
    //! [Open streams]
    // Open depth stream
    VideoCapture depthStream(CAP_OPENNI2_ASTRA);
    // Open color stream
    VideoCapture colorStream(0, CAP_V4L2);
    //! [Open streams]

    // Check that stream has opened
    if (!colorStream.isOpened())
    {
        cerr << "ERROR: Unable to open color stream" << endl;
        return -1;
    }

    // Check that stream has opened
    if (!depthStream.isOpened())
    {
        cerr << "ERROR: Unable to open depth stream" << endl;
        return -1;
    }

    //! [Setup streams]
    // Set color and depth stream parameters
    colorStream.set(CAP_PROP_FRAME_WIDTH,  640);
    colorStream.set(CAP_PROP_FRAME_HEIGHT, 480);
    depthStream.set(CAP_PROP_FRAME_WIDTH,  640);
    depthStream.set(CAP_PROP_FRAME_HEIGHT, 480);
    depthStream.set(CAP_PROP_OPENNI2_MIRROR, 0);
    //! [Setup streams]

    // Print color stream parameters
    cout << "Color stream: "
         << colorStream.get(CAP_PROP_FRAME_WIDTH) << "x" << colorStream.get(CAP_PROP_FRAME_HEIGHT)
         << " @" << colorStream.get(CAP_PROP_FPS) << " fps" << endl;

    //! [Get properties]
    // Print depth stream parameters
    cout << "Depth stream: "
         << depthStream.get(CAP_PROP_FRAME_WIDTH) << "x" << depthStream.get(CAP_PROP_FRAME_HEIGHT)
         << " @" << depthStream.get(CAP_PROP_FPS) << " fps" << endl;
    //! [Get properties]

    //! [Read streams]
    // Create two lists to store frames
    std::list<Frame> depthFrames, colorFrames;
    const std::size_t maxFrames = 64;

    // Synchronization objects
    std::mutex mtx;
    std::condition_variable dataReady;
    std::atomic<bool> isFinish;

    isFinish = false;

    // Start depth reading thread
    std::thread depthReader([&]
    {
        while (!isFinish)
        {
            // Grab and decode new frame
            if (depthStream.grab())
            {
                Frame f;
                f.timestamp = cv::getTickCount();
                depthStream.retrieve(f.frame, CAP_OPENNI_DEPTH_MAP);
                if (f.frame.empty())
                {
                    cerr << "ERROR: Failed to decode frame from depth stream" << endl;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (depthFrames.size() >= maxFrames)
                        depthFrames.pop_front();
                    depthFrames.push_back(f);
                }
                dataReady.notify_one();
            }
        }
    });

    // Start color reading thread
    std::thread colorReader([&]
    {
        while (!isFinish)
        {
            // Grab and decode new frame
            if (colorStream.grab())
            {
                Frame f;
                f.timestamp = cv::getTickCount();
                colorStream.retrieve(f.frame);
                if (f.frame.empty())
                {
                    cerr << "ERROR: Failed to decode frame from color stream" << endl;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (colorFrames.size() >= maxFrames)
                        colorFrames.pop_front();
                    colorFrames.push_back(f);
                }
                dataReady.notify_one();
            }
        }
    });
    //! [Read streams]

    //! [Pair frames]
    // Pair depth and color frames
    while (!isFinish)
    {
        std::unique_lock<std::mutex> lk(mtx);
        while (!isFinish && (depthFrames.empty() || colorFrames.empty()))
            dataReady.wait(lk);

        while (!depthFrames.empty() && !colorFrames.empty())
        {
            if (!lk.owns_lock())
                lk.lock();

            // Get a frame from the list
            Frame depthFrame = depthFrames.front();
            int64 depthT = depthFrame.timestamp;

            // Get a frame from the list
            Frame colorFrame = colorFrames.front();
            int64 colorT = colorFrame.timestamp;

            // Half of frame period is a maximum time diff between frames
            const int64 maxTdiff = int64(1000000000 / (2 * colorStream.get(CAP_PROP_FPS)));
            if (depthT + maxTdiff < colorT)
            {
                depthFrames.pop_front();
                continue;
            }
            else if (colorT + maxTdiff < depthT)
            {
                colorFrames.pop_front();
                continue;
            }
            depthFrames.pop_front();
            colorFrames.pop_front();
            lk.unlock();

            //! [Show frames]
            // Show depth frame
            Mat d8, dColor;
            depthFrame.frame.convertTo(d8, CV_8U, 255.0 / 2500);
            applyColorMap(d8, dColor, COLORMAP_OCEAN);
            imshow("Depth (colored)", dColor);

            // Show color frame
            imshow("Color", colorFrame.frame);
            //! [Show frames]

            // Exit on Esc key press
            int key = waitKey(1);
            if (key == 27) // ESC
            {
                isFinish = true;
                break;
            }
        }
    }
    //! [Pair frames]

    dataReady.notify_one();
    depthReader.join();
    colorReader.join();

    return 0;
}

static int camRealsense()
{
   VideoCapture capture(CAP_INTELPERC);
   if (!capture.isOpened())
   {
        cerr << "ERROR: Failed to open RealSense camera." << endl;
        return -1;
   }

    Mat depthMap;
    Mat bgrImage;
    Mat irImage;

    for(;;)
    {
        if(capture.grab())
        {
            if(capture.retrieve(depthMap,CAP_INTELPERC_DEPTH_MAP))
            {
                Mat adjMap;
                normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjMap, adjMap, COLORMAP_JET);
                imshow("DEPTH", adjMap);
            }

            if(capture.retrieve(bgrImage,CAP_INTELPERC_IMAGE))
                imshow("RGB", bgrImage);

            if(capture.retrieve(irImage,CAP_INTELPERC_IR_MAP))
                imshow("IR", irImage);
        }

        if(waitKey(30) >= 0)
            break;
    }

    return 0;
}

static int camOrbbec()
{
    VideoCapture capture(0, CAP_OBSENSOR);
    if(!capture.isOpened())
    {
        cerr << "ERROR: Failed to open Orbbec camera."  << endl;
        return -1;
    }

    // get the intrinsic parameters of Orbbec camera
    double fx = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_FX);
    double fy = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_FY);
    double cx = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_CX);
    double cy = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_CY);
    cout << "Camera intrinsic params: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << endl;

    Mat depthMap;
    Mat bgrImage;

    for(;;)
    {
        // Get the depth map:
        // obsensorCapture >> depthMap;

        // Another way to get the depth map:
        if (capture.grab())
        {
            if (capture.retrieve(depthMap, CAP_OBSENSOR_DEPTH_MAP))
            {
                Mat adjMap;
                normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjMap, adjMap, COLORMAP_JET);
                imshow("DEPTH", adjMap);
            }

            if (capture.retrieve(bgrImage, CAP_OBSENSOR_BGR_IMAGE))
                imshow("RGB", bgrImage);
        }

        if(waitKey(30) >= 0)
            break;
    }

    return 0;
}

const char* keys = "{camType | | Camera Type: OpenNI, OpenNI-Orbbec, RealSense, Orbbec}";

const char* about =
            "\nThis example demostrates how to get data from 3D cameras via OpenCV.\n"
            "Currently OpenCV supports 3 types of 3D cameras:\n"
            "1. Depth sensors compatible with OpenNI:\n"
            "\t - Kinect, XtionPRO. Users must install OpenNI library and PrimeSensorModule for OpenNI and configure OpenCV with WITH_OPENNI flag ON in CMake.\n"
            "\t - Orbbec Astra Series. Users must install Orbbec OpenNI SDK and configure accordingly.\n"
            "2. Depth sensors compatible with Intel® RealSense SDK (RealSense). "
            "Users must install Intel RealSense SDK 2.0 and configure OpenCV with WITH_LIBREALSENSE flag ON in CMake.\n"
            "3. Orbbec depth sensors.\n";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if(argc < 2)
    {
        parser.printMessage();
        return 0;
    }

    String camType = parser.get<String>("camType");
    if (camType == "OpenNI")
    {
        camOpenni();
    }
    else if (camType == "OpenNI-Orbbec")
    {
#ifdef USE_THREADS
        camOpenniOrbbec();
#else
        cerr << "ERROR: No threading support. Sample code is disabled." << endl;
        return -1;
#endif
    }
    else if (camType == "RealSense")
    {
        camRealsense();
    }
    else if (camType == "Orbbec")
    {
        camOrbbec();
    }
    else
    {
        cerr << "ERROR: Invalid input." << endl;
        return -1;
    }

    return 0;
}
