#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <list>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;


// Stores frames along with their timestamps
struct Frame
{
    int64 timestamp;
    Mat frame;
};

int main()
{
    //! [Open streams]
    // Open color stream
    VideoCapture colorStream(CAP_V4L2);
    // Open depth stream
    VideoCapture depthStream(CAP_OPENNI2_ASTRA);
    //! [Open streams]

    // Check that stream has opened
    if (!colorStream.isOpened())
    {
        cerr << "ERROR: Unable to open color stream" << endl;
        return 1;
    }

    // Check that stream has opened
    if (!depthStream.isOpened())
    {
        cerr << "ERROR: Unable to open depth stream" << endl;
        return 1;
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
    std::mutex depthFramesMtx, colorFramesMtx;
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
                //depthStream.retrieve(f.frame, CAP_OPENNI_DISPARITY_MAP);
                //depthStream.retrieve(f.frame, CAP_OPENNI_IR_IMAGE);
                if (f.frame.empty())
                {
                    cerr << "ERROR: Failed to decode frame from depth stream" << endl;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(depthFramesMtx);
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
                    std::lock_guard<std::mutex> lk(colorFramesMtx);
                    if (colorFrames.size() >= maxFrames)
                        colorFrames.pop_front();
                    colorFrames.push_back(f);
                }
                dataReady.notify_one();
            }
        }
    });
    //! [Read streams]

    while (true)
    {
        std::unique_lock<std::mutex> lk(mtx);
        while (depthFrames.empty() && colorFrames.empty())
            dataReady.wait(lk);

        depthFramesMtx.lock();
        if (depthFrames.empty())
        {
            depthFramesMtx.unlock();
        }
        else
        {
            // Get a frame from the list
            Mat depthMap = depthFrames.front().frame;
            depthFrames.pop_front();
            depthFramesMtx.unlock();

            // Show depth frame
            Mat d8, dColor;
            depthMap.convertTo(d8, CV_8U, 255.0 / 2500);
            applyColorMap(d8, dColor, COLORMAP_OCEAN);
            imshow("Depth (colored)", dColor);
        }

        //! [Show color frame]
        colorFramesMtx.lock();
        if (colorFrames.empty())
        {
            colorFramesMtx.unlock();
        }
        else
        {
            // Get a frame from the list
            Mat colorFrame = colorFrames.front().frame;
            colorFrames.pop_front();
            colorFramesMtx.unlock();

            // Show color frame
            imshow("Color", colorFrame);
        }
        //! [Show color frame]

        // Exit on Esc key press
        int key = waitKey(1);
        if (key == 27) // ESC
            break;
    }

    isFinish = true;
    depthReader.join();
    colorReader.join();

    return 0;
}
