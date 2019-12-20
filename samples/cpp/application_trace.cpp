/* OpenCV Application Tracing support demo. */
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/trace.hpp>

using namespace cv;
using namespace std;

static void process_frame(const cv::UMat& frame)
{
    CV_TRACE_FUNCTION(); // OpenCV Trace macro for function

    imshow("Live", frame);

    UMat gray, processed;
    cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
    Canny(gray, processed, 32, 64, 3);
    imshow("Processed", processed);
}

int main(int argc, char** argv)
{
    CV_TRACE_FUNCTION();

    cv::CommandLineParser parser(argc, argv,
        "{help h ? |     | help message}"
        "{n        | 100 | number of frames to process }"
        "{@video   | 0   | video filename or cameraID }"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    VideoCapture capture;
    std::string video = parser.get<string>("@video");
    if (video.size() == 1 && isdigit(video[0]))
        capture.open(parser.get<int>("@video"));
    else
        capture.open(samples::findFileOrKeep(video));  // keep GStreamer pipelines
    int nframes = 0;
    if (capture.isOpened())
    {
        nframes = (int)capture.get(CAP_PROP_FRAME_COUNT);
        cout << "Video " << video <<
            ": width=" << capture.get(CAP_PROP_FRAME_WIDTH) <<
            ", height=" << capture.get(CAP_PROP_FRAME_HEIGHT) <<
            ", nframes=" << nframes << endl;
    }
    else
    {
        cout << "Could not initialize video capturing...\n";
        return -1;
    }

    int N = parser.get<int>("n");
    if (nframes > 0 && N > nframes)
        N = nframes;

    cout << "Start processing..." << endl
        << "Press ESC key to terminate" << endl;

    UMat frame;
    for (int i = 0; N > 0 ? (i < N) : true; i++)
    {
        CV_TRACE_REGION("FRAME"); // OpenCV Trace macro for named "scope" region
        {
            CV_TRACE_REGION("read");
            capture.read(frame);

            if (frame.empty())
            {
                cerr << "Can't capture frame: " << i << std::endl;
                break;
            }

            // OpenCV Trace macro for NEXT named region in the same C++ scope
            // Previous "read" region will be marked complete on this line.
            // Use this to eliminate unnecessary curly braces.
            CV_TRACE_REGION_NEXT("process");
            process_frame(frame);

            CV_TRACE_REGION_NEXT("delay");
            if (waitKey(1) == 27/*ESC*/)
                break;
        }
    }

    return 0;
}
