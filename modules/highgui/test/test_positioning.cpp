#include "test_precomp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

/* #include <cv.h>
#include <cxcore.h>

    #include <iostream>
    #include <sstream>
    #include <string> */

using namespace cv;
using namespace std;

//ticket #1497

#define HIGHGUI_POSITIONING_ERROR_OPEN 0

#define MESSAGE_ERROR_CONTENT "Cannot read source video file."

class CV_VideoPositioningTest: public cvtest::BaseTest
{
public:
    void run(int);

};

void CV_VideoPositioningTest::run(int)
{
    const string& src_dir = ts->get_data_path();

    std::cout << src_dir.c_str() << endl;

    string file_path = "/home/reshetnikov/SVN_Projects/OpenCV/opencv_extra/testdata/perf/video/sample_sorenson.mov";

    std::cout << file_path.c_str() << endl;

    cv::VideoCapture cap(file_path);

    // CvCapture* cap = cvCreateFileCapture(file_path.c_str());
     if (!cap.isOpened())
    {
        printf("Error!");
        return;
    }

    std::cout << "Frame pos: " << cap.get(CV_CAP_PROP_POS_FRAMES) << std::endl;

    // IplImage* frame = cvQueryFrame(cap);

    Mat frame; cap >> frame;

    /* if (!frame)
    {

            return;
    } */

    std::cout << "Frames number: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
    
    int step = 20;
    int frameCount = 1;
    while (frameCount < 100)
    {
            std::cout << "Frame count: " << frameCount << "\tActual frame pos: " << cap.get(CV_CAP_PROP_POS_FRAMES) << std::endl;

            // Save the frame
            std::stringstream ss;
            ss << frameCount;
            std::string filename = ss.str() + ".png";
            imwrite(file_path, frame, vector<int>(1));
            // Advance by step frames
            frameCount += step;
            std::cout << "cvSetCaptureProperty result: " << cap.set(CV_CAP_PROP_POS_FRAMES, frameCount) << std::endl;;
            // frame = cvQueryFrame(cap);
    }

    // cvReleaseCapture(&cap);
    cap.release();
}





    /*
47	NOTES
48
49	Output:
50	Frame pos: 0
51	Frame count: 1  Actual frame pos: -1.84467e+017
52	cvSetCaptureProperty result: 1
53	Frame count: 21 Actual frame pos: -1.84467e+017
54	cvSetCaptureProperty result: 1
55	Frame count: 41 Actual frame pos: -1.84467e+017
56	cvSetCaptureProperty result: 1
57	Frame count: 61 Actual frame pos: -1.84467e+017
58	cvSetCaptureProperty result: 1
59	Frame count: 81 Actual frame pos: -1.84467e+017
60	cvSetCaptureProperty result: 1
61
62	Expected:
63	Frame pos: 0
64	Frame count: 1  Actual frame pos: 1
65	cvSetCaptureProperty result: 1
66	Frame count: 21 Actual frame pos: 21
67	cvSetCaptureProperty result: 1
68	Frame count: 41 Actual frame pos: 41
69	cvSetCaptureProperty result: 1
70	Frame count: 61 Actual frame pos: 61
71	cvSetCaptureProperty result: 1
72	Frame count: 81 Actual frame pos: 81
73	cvSetCaptureProperty result: 1
74
75	In addition, the frame retrieved from cvQueryFrame was not the correct frame
76	*/

TEST (HighguiPositioning, regression) { CV_VideoPositioningTest test; test.safe_run(); }


