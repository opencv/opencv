#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID) || (defined(_MSC_VER) && _MSC_VER>=1800)
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
        IDetector(),
        Detector(detector)
    {
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) CV_OVERRIDE
    {
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }

    virtual ~CascadeDetectorAdapter() CV_OVERRIDE
    {}

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};

int main(int , char** )
{
    // 使用官方提供的视频文件
    std::string videoFile = samples::findFile("samples/data/vtest.avi");
    VideoCapture VideoStream(videoFile);

    if (!VideoStream.isOpened())
    {
        printf("Error: Cannot open video file\n");
        return 1;
    }

    std::string cascadeFrontalfilename = samples::findFile("data/lbpcascades/lbpcascade_frontalface.xml");
    cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if (cascade->empty())
    {
        printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
        return 2;
    }

    cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
    cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
    if (cascade->empty())
    {
        printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
        return 2;
    }

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

    if (!Detector.run())
    {
        printf("Error: Detector initialization failed\n");
        return 2;
    }

    // 准备保存处理结果的视频文件
    int frame_width = static_cast<int>(VideoStream.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(VideoStream.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outputVideo("output.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width, frame_height));

    if (!outputVideo.isOpened())
    {
        printf("Error: Cannot open video writer\n");
        return 3;
    }

    Mat ReferenceFrame;
    Mat GrayFrame;
    vector<Rect> Faces;

    do
    {
        VideoStream >> ReferenceFrame;
        if (ReferenceFrame.empty()) break;  // 检查是否到达视频结尾

        cvtColor(ReferenceFrame, GrayFrame, COLOR_BGR2GRAY);
        Detector.process(GrayFrame);
        Detector.getObjects(Faces);

        for (size_t i = 0; i < Faces.size(); i++)
        {
            rectangle(ReferenceFrame, Faces[i], Scalar(0,255,0));
        }

        // 将处理后的帧保存到视频文件
        outputVideo.write(ReferenceFrame);
    } while (VideoStream.grab());

    Detector.stop();
    outputVideo.release();  // 关闭视频文件

    return 0;
}

#else

#include <stdio.h>
int main()
{
    printf("This sample works for UNIX or ANDROID or Visual Studio 2013+ only\n");
    return 0;
}
#endif
