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
    vector<Mat> audioData;
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, 0,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

    cap.open(file, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open file" << endl;
        return -1;
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    for (;;)
    {
        if (cap.grab())
        {
            cap.retrieve(videoFrame);
            cap.retrieve(audioFrame, audioBaseIndex);
            if (!videoFrame.empty())
            {
                imshow("Live", videoFrame);
                if (waitKey(5) >= 0)
                    break;
            }
            if (!audioFrame.empty())
            {
                audioData.push_back(audioFrame);
            }
        }
        else
        {
            int numberOfSamles = 0;
            for (auto item : audioData)
                numberOfSamles+=item.cols;
            cout << "Number of samples: " << numberOfSamles << endl;
            break;
        }
    }
    return 0;
}
