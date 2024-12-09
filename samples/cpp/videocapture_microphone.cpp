#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
    Mat frame;
    vector<Mat> audioData;
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, -1   };

    cap.open(0, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open microphone" << endl;
        return -1;
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    const double cvTickFreq = getTickFrequency();
    int64 sysTimeCurr = getTickCount();
    int64 sysTimePrev = sysTimeCurr;
    while ((sysTimeCurr-sysTimePrev)/cvTickFreq < 10)
    {
        if (cap.grab())
        {
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                cap.retrieve(frame, audioBaseIndex+nCh);
                audioData.push_back(frame);
                sysTimeCurr = getTickCount();
            }
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }
    int numberOfSamles = 0;
    for (auto item : audioData)
        numberOfSamles+=item.cols;
    cout << "Number of samples: " << numberOfSamles << endl;

    return 0;
}
