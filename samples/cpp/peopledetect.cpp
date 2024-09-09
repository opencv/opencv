#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

class Detector
{
    enum Mode { Default, Daimler } m;
    HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    vector<Rect> detect(InputArray img)
    {
        vector<Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, true);
        return found;
    }
    void adjustRect(Rect & r) const
    {
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};

static const string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }"
                           "{ output o | peopledetect/output.avi | output video file name }";

bool createDirectory(const string& dir)
{
    struct stat info;
    if (stat(dir.c_str(), &info) != 0)
    {
        // Directory does not exist, attempt to create it
        if (mkdir(dir.c_str(), 0777) != 0)
        {
            return false;
        }
    }
    else if (info.st_mode & S_IFDIR)
    {
        // Directory exists
        return true;
    }
    else
    {
        // Path exists but is not a directory
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use of the HoG descriptor.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");
    string outputFileName = parser.get<string>("output");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Ensure the output directory exists
    size_t pos = outputFileName.find_last_of('/');
    if (pos != string::npos)
    {
        string outputDir = outputFileName.substr(0, pos);
        if (!createDirectory(outputDir))
        {
            cout << "Could not create output directory: " << outputDir << endl;
            return 3;
        }
    }

    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter outputVideo(outputFileName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

    if (!outputVideo.isOpened())
    {
        cout << "Could not open the output video file for write: " << outputFileName << endl;
        return 3;
    }

    cout << "Processing video, output will be saved to: " << outputFileName << endl;
    Detector detector;
    Mat frame;
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        int64 t = getTickCount();
        vector<Rect> found = detector.detect(frame);
        t = getTickCount() - t;

        ostringstream buf;
        buf << "Mode: " << detector.modeName() << " ||| "
            << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
        putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        
        for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            Rect &r = *i;
            detector.adjustRect(r);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
        }
        
        outputVideo.write(frame);
    }

    cout << "Processing complete. Video saved to: " << outputFileName << endl;
    return 0;
}

