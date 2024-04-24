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
"{with_audio wt | | set --with_audio if you want to extract audio buffers. If no input video file is provided only audio samples will be extracted}"
"{help h | | print usage}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("This sample demonstrates how to read and display frames from a video file, a camera device or an image sequence \n"
                 "The sample also demonstrates extracting audio buffers from a video file\n"
                 "Capturing video and audio simultaneously from camera device is not supported by OpenCV\n"
                 "Usage:\n --input/-i=<video file, image sequence> Skip this argument to capture frames from a camera\n"
                 "--with_audio/-wt, set --with_audio if you want to extract audio buffers. If no input video file is provided only audio samples will be extracted\n"
                 "q,Q,esc -- quit\n"
                 "You can also pass the path to an image sequence and OpenCV will treat the sequence just like a video\n"
                 "example: --input=right%02d.jpg\n");

    // Check for 'help' argument to display usage information
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Get input file path and check if audio extraction is required
    string file = parser.get<string>("input");
    bool isWithAudio = parser.has("with_audio");

    Mat videoFrame;
    Mat audioFrame;
    vector<vector<Mat>> audioData;
    VideoCapture cap;

    // Setup VideoCapture based on user input
    if (isWithAudio) {
        vector<int> videoParams{ CAP_PROP_AUDIO_STREAM, 0, CAP_PROP_VIDEO_STREAM, 0, CAP_PROP_AUDIO_DATA_DEPTH, CV_16S };
        if (file.empty()) {
            // No input file, extract audio from default device
            cout << "Extracting audio samples from default device is only supported via MSMF backend" << endl;
            cap.open(0, CAP_MSMF, videoParams);
            if (!cap.isOpened()) {
                cerr << "ERROR! Unable to open camera and microphone" << endl;
                return -1;
            }
        } else {
            // Open input file for both video and audio
            cap.open(file, CAP_ANY, videoParams);
            if (!cap.isOpened()) {
                cerr << "ERROR! Unable to open file" << endl;
                return -1;
            }
        }
    } else {
        // Setup VideoCapture for video only
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
    }

    // Audio settings and info
    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "Audio Data Depth: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    cout << "Audio Samples Per Second: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "Total Audio Channels: " << numberOfChannels << endl;
    cout << "Total Audio Streams: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    int numberOfFrames = 0;
    int numberOfSamples = 0;
    if (numberOfChannels > 0){
        audioData.resize(numberOfChannels);
    }

    // Timing for sample and frame capture
    const double cvTickFreq = getTickFrequency();
    int64 sysTimeCurr = getTickCount();
    int64 sysTimePrev = sysTimeCurr;

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
        // Retrieve audio frames for all channels
        for (int nCh = 0; nCh < numberOfChannels; nCh++) {
            if (cap.retrieve(audioFrame, audioBaseIndex + nCh)) {
                if (!audioFrame.empty()) {
                    audioData[nCh].push_back(audioFrame);
                    numberOfSamples += audioFrame.cols;
                    sysTimeCurr = getTickCount();
                    cout << "Audio frame retrieved successfully. Channel: " << nCh + 1 << " Sample count: " << numberOfSamples << endl;
                }
            }
        }

        // Break after 10 seconds of audio capture
        if ((sysTimeCurr - sysTimePrev) / cvTickFreq >= 10) {
            cout << "Stream closed after 10 seconds timeout" << endl;
            break;
        }
    }

    // Output the count of captured samples and frames
    cout << "Number of audio samples: " << numberOfSamples << endl
         << "Number of video frames: " << numberOfFrames << endl;
    return 0;
}
