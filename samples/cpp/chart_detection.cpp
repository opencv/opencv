#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/mcc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mcc;

const char *about = "Basic chart detection";
const char *keys = {
    "{ help h usage ? |    | show this message }"
    "{t        |      |  chartType: 0-Standard, 1-DigitalSG, 2-Vinyl }"
    "{v        |      | Input from video file, if ommited, input comes from camera }"
    "{ci       | 0    | Camera id if input doesnt come from video (-v) }"
    "{nc       | 1    | Maximum number of charts in the image }"};

int main(int argc, char *argv[])
{

    // ----------------------------------------------------------
    // Scroll down a bit (~40 lines) to find actual relevant code
    // ----------------------------------------------------------

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    int t = parser.get<int>("t");

    CV_Assert(0 <= t && t <= 2);
    TYPECHART chartType = TYPECHART(t);

    int camId = parser.get<int>("ci");
    int nc = parser.get<int>("nc");
    String video;
    if (parser.has("v"))
        video = parser.get<String>("v");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if (!video.empty())
    {
        inputVideo.open(video);
        waitTime = 10;
    }
    else
    {
        inputVideo.open(camId);
        waitTime = 10;
    }

    //--------------------------------------------------------------------------
    //-------------------------Actual Relevant Code-----------------------------
    //--------------------------------------------------------------------------

    while (inputVideo.grab())
    {

        Mat image, imageCopy;
        inputVideo.retrieve(image);
        imageCopy = image.clone();
        Ptr<CCheckerDetector> detector = CCheckerDetector::create();
        // Marker type to detect
        if (!detector->process(image, chartType, nc))
        {
            printf("ChartColor not detected \n");
        }
        else
        {

            // get checker
            std::vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();

            for (Ptr<mcc::CChecker> checker : checkers)
            {
                // current checker
                Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
                cdraw->draw(image);
            }
        }
        imshow("image result | q or esc to quit", image);
        imshow("original", imageCopy);
        char key = (char)waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
