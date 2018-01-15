#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
        "{ i | ../data/lena_tmpl.jpg |image name }"
        "{ t | ../data/tmpl.png |template name }"
        "{ m | ../data/mask.png |mask name }"
        "{ cm| 3 |comparison method }");

    cout << "This program demonstrates the use of template matching with mask.\n\n";
    parser.printMessage();

    string filename = parser.get<string>("i");
    string tmplname = parser.get<string>("t");
    string maskname = parser.get<string>("m");
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

    int method = parser.get<int>("cm"); // default 3 (CV_TM_CCORR_NORMED)
    matchTemplate(img, tmpl, res, method, mask);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    Rect rect;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    if(method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
        rect = Rect(minLoc, tmpl.size());
    else
        rect = Rect(maxLoc, tmpl.size());

    rectangle(img, rect, Scalar(0, 255, 0), 2);

    imshow("detected template", img);
    waitKey();

    return 0;
}
