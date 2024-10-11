/*
 * This sample demonstrates how to read and display frames from a video file, a camera device or an image sequence.
 * The sample also demonstrates extracting audio buffers from a video file.

 * Capturing video and audio simultaneously from camera device is not supported by OpenCV.
*/


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string keys =
"{input i | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{help  h | | print usage}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("This sample demonstrates how to read and display frames from a video file, a camera device or an image sequence \n"
                 "Usage:\n --input/-i=<video file, image sequence> Skip this argument to capture frames from a camera\n"
                 "q,Q,esc -- quit\n"
                 "You can also pass the path to an image sequence and OpenCV will treat the sequence just like a video\n"
                 "example: --input=right%02d.jpg\n");

    // Check for 'help' argument to display usage information
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file = parser.get<string>("input");

    Mat videoFrame;
    Mat audioFrame;
    vector<vector<Mat>> audioData;
    VideoCapture cap;

    if (file.empty()) {
        // No input file, capture video from default camera
        cap.open(0, CAP_ANY);
        if (!cap.isOpened()) {
            cerr << "ERROR! Unable to open camera" << endl;
            return -1;
        }
    } else {
        // Open input video file
        cap.open(file, CAP_ANY);
        if (!cap.isOpened()) {
            cerr << "ERROR! Unable to open file: " + file << endl;
            return -1;
        }
    }
    cout << "Frame width: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "      height: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Capturing FPS: " << cap.get(CAP_PROP_FPS) << endl;

    int numberOfFrames = 0;

    // Main loop for capturing and displaying frames
    for (;;)
    {
        if (!cap.grab()) {
            cerr << "Error during frame capture" << endl;
            break;
        }

        bool videoFrameRetrieved = cap.retrieve(videoFrame);
        if (videoFrameRetrieved) {
            numberOfFrames++;
            imshow("Video | q or esc to quit", videoFrame);
            if (waitKey(30) >= 0)
                break;
            cout << "Video frame retrieved successfully. Frame count: " << numberOfFrames << endl;
        }
    }

    // Output the count of captured samples and frames
    cout << "Number of video frames: " << numberOfFrames << endl;
    return 0;
}
