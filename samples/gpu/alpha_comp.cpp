#include <iostream>

#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{
    cout << "This program demonstrates using alphaComp" << endl;
    cout << "Press SPACE to change compositing operation" << endl;
    cout << "Press ESC to exit" << endl;

    namedWindow("First Image", WINDOW_NORMAL);
    namedWindow("Second Image", WINDOW_NORMAL);
    namedWindow("Result", WINDOW_OPENGL);

    setGlDevice();

    Mat src1(640, 480, CV_8UC4, Scalar::all(0));
    Mat src2(640, 480, CV_8UC4, Scalar::all(0));

    rectangle(src1, Rect(50, 50, 200, 200), Scalar(0, 0, 255, 128), 30);
    rectangle(src2, Rect(100, 100, 200, 200), Scalar(255, 0, 0, 128), 30);

    GpuMat d_src1(src1);
    GpuMat d_src2(src2);

    GpuMat d_res;

    imshow("First Image", src1);
    imshow("Second Image", src2);

    int alpha_op = ALPHA_OVER;

    const char* op_names[] =
    {
        "ALPHA_OVER", "ALPHA_IN", "ALPHA_OUT", "ALPHA_ATOP", "ALPHA_XOR", "ALPHA_PLUS", "ALPHA_OVER_PREMUL", "ALPHA_IN_PREMUL", "ALPHA_OUT_PREMUL",
        "ALPHA_ATOP_PREMUL", "ALPHA_XOR_PREMUL", "ALPHA_PLUS_PREMUL", "ALPHA_PREMUL"
    };

    for(;;)
    {
        cout << op_names[alpha_op] << endl;

        alphaComp(d_src1, d_src2, d_res, alpha_op);

        imshow("Result", d_res);

        char key = static_cast<char>(waitKey());

        if (key == 27)
            break;

        if (key == 32)
        {
            ++alpha_op;

            if (alpha_op > ALPHA_PREMUL)
                alpha_op = ALPHA_OVER;
        }
    }

    return 0;
}
