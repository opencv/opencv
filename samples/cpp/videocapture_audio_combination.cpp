#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

string depthToStringCustom(int depth);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@audio||}");
    string file = parser.get<string>("@audio");

    if (file.empty())
    {
        cerr << "No audio file provided" << endl;
        return 1;
    }

    Mat videoFrame;
    Mat audioFrame;
    vector<vector<Mat>> audioData;
    VideoCapture cap;

    // 尝试使用不同的后端打开文件，不传递参数
    vector<int> backends = {CAP_ANY, CAP_FFMPEG, CAP_GSTREAMER, CAP_MSMF};
    bool opened = false;
    for (int backend : backends)
    {
        cap.open(file, backend);
        if (cap.isOpened())
        {
            cout << "Successfully opened file with backend: " << backend << endl;
            opened = true;
            break;
        }
        else
        {
            cerr << "Failed to open file with backend: " << backend << endl;
        }
    }

    if (!opened)
    {
        cerr << "ERROR! Can't open file: " << file << endl;
        cerr << "Supported backends: " << getBuildInformation() << endl;
        return -1;
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToStringCustom((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
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
                cap.retrieve(audioFrame, audioBaseIndex + nCh);
                if (!audioFrame.empty())
                    audioData[nCh].push_back(audioFrame);
                numberOfSamples += audioFrame.cols;
            }
            if (!videoFrame.empty())
            {
                numberOfFrames++;
                // imshow("Live", videoFrame); // 注释掉
                // if (waitKey(5) >= 0) // 注释掉
                //     break;
            }
        }
        else
        {
            break;
        }
    }

    cout << "Number of audio samples: " << numberOfSamples << endl
         << "Number of video frames: " << numberOfFrames << endl;

    // Create directory if it doesn't exist
    system("mkdir -p videocapture_audio_combination");

    // Save audio data to files
    for (int nCh = 0; nCh < numberOfChannels; nCh++)
    {
        stringstream ss;
        ss << "videocapture_audio_combination/audio_channel_" << nCh << ".txt";
        string save_path = ss.str();

        ofstream outFile(save_path);
        if (outFile.is_open())
        {
            for (size_t i = 0; i < audioData[nCh].size(); i++)
            {
                for (int j = 0; j < audioData[nCh][i].cols; j++)
                {
                    outFile << audioData[nCh][i].at<short>(0, j) << " ";
                }
                outFile << endl;
            }
            outFile.close();
        }
        cout << "Audio data for channel " << nCh << " saved to " << save_path << endl;
    }

    return 0;
}

string depthToStringCustom(int depth)
{
    switch (depth)
    {
    case CV_8U: return "8U";
    case CV_8S: return "8S";
    case CV_16U: return "16U";
    case CV_16S: return "16S";
    case CV_32S: return "32S";
    case CV_32F: return "32F";
    case CV_64F: return "64F";
    default: return "Unknown";
    }
}

