#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
        "{ i | lena_tmpl.jpg |image name }"
        "{ t | tmpl.png |template name }"
        "{ m | mask.png |mask name }"
        "{ cm| 3 |comparison method }");

    cout << "This program demonstrates the use of template matching with mask." << endl
         << endl
         << "Available methods: https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d" << endl
         << "    TM_SQDIFF = " << (int)TM_SQDIFF << endl
         << "    TM_SQDIFF_NORMED = " << (int)TM_SQDIFF_NORMED << endl
         << "    TM_CCORR = " << (int)TM_CCORR << endl
         << "    TM_CCORR_NORMED = " << (int)TM_CCORR_NORMED << endl
         << "    TM_CCOEFF = " << (int)TM_CCOEFF << endl
         << "    TM_CCOEFF_NORMED = " << (int)TM_CCOEFF_NORMED << endl
         << endl;

    parser.printMessage();

    string filename = samples::findFile(parser.get<string>("i"));
    string tmplname = samples::findFile(parser.get<string>("t"));
    string maskname = samples::findFile(parser.get<string>("m"));
    Mat img = imread(filename);
    Mat tmpl = imread(tmplname);
    Mat mask = imread(maskname);
    Mat res;

    if(img.empty())
    {
        cout << "can not open " << filename << endl;
        return -1;
    }

    if(tmpl.empty())
    {
        cout << "can not open " << tmplname << endl;
        return -1;
    }

    if(mask.empty())
    {
        cout << "can not open " << maskname << endl;
        return -1;
    }

    int method = parser.get<int>("cm"); // default 3 (cv::TM_CCORR_NORMED)
    matchTemplate(img, tmpl, res, method, mask);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    Rect rect;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    if(method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
        rect = Rect(minLoc, tmpl.size());
    else
        rect = Rect(maxLoc, tmpl.size());

    rectangle(img, rect, Scalar(0, 255, 0), 2);

    imshow("detected template", img);
    waitKey();

    return 0;
}
