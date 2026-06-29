// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const char* keys =
        "{ width  | 640 | input width }"
        "{ height | 480 | input height }"
        "{ help h |     | print help message }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << "Usage: " << argv[0] << " [--width=<width>] [--height=<height>]" << endl;
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    const int width = parser.get<int>("width");
    const int height = parser.get<int>("height");
    if (!parser.check() || width <= 0 || height <= 0)
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }

    Mat src1(height, width, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expectedAdd;
    add(src1, src2, expectedAdd);

    UMat usrc1, usrc2, usum, ugray, uthresh, ufloat;
    src1.copyTo(usrc1);
    src2.copyTo(usrc2);

    add(usrc1, usrc2, usum);
    cvtColor(usum, ugray, COLOR_BGR2GRAY);
    threshold(ugray, uthresh, 127, 255, THRESH_BINARY);
    uthresh.convertTo(ufloat, CV_32F, 1.0 / 255.0);

    Mat actualAdd;
    usum.copyTo(actualAdd);

    cout << "UMat backend sample completed" << endl;
    cout << "add() max difference against CPU Mat path: "
         << norm(actualAdd, expectedAdd, NORM_INF) << endl;
    cout << "final output type: " << ufloat.type() << ", size: "
         << ufloat.cols << "x" << ufloat.rows << endl;

    return EXIT_SUCCESS;
}
