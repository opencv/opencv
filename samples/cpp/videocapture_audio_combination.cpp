#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{@audio||}");
    string file = parser.get<string>("@audio");

    if (file.empty())
    {
        return 1;
    }

    Mat videoFrame;
    Mat audioFrame;
    vector<vector<Mat>> audioData;
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, 0,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

    cap.open(file, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open file: " + file << endl;
        return -1;
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    int numberOfSamples = 0;
    int numberOfFrames = 0;
    audioData.resize(numberOfChannels);
    for (;;)
    {
        if (cap.grab())
        {
            cap.retrieve(videoFrame);
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                cap.retrieve(audioFrame, audioBaseIndex+nCh);
                if (!audioFrame.empty())
                    audioData[nCh].push_back(audioFrame);
                numberOfSamples+=audioFrame.cols;
            }
            if (!videoFrame.empty())
            {
                numberOfFrames++;
                imshow("Live", videoFrame);
                if (waitKey(5) >= 0)
                    break;
            }
        } else { break; }
    }

    cout << "Number of audio samples: " << numberOfSamples << endl
         << "Number of video frames: " << numberOfFrames << endl;
    return 0;
}
