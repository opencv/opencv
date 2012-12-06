#include <cmath>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

static void help()
{
    cout << "This program demonstrates line finding with the Hough transform." << endl;
    cout << "Usage:" << endl;
    cout << "./gpu-example-houghlines <image_name>, Default is pic1.png\n" << endl;
}

int main(int argc, const char* argv[])
{
    const string filename = argc >= 2 ? argv[1] : "pic1.png";

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    if (src.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    Mat mask;
    Canny(src, mask, 50, 200, 3);

    Mat dst_cpu;
    cvtColor(mask, dst_cpu, CV_GRAY2BGR);
    Mat dst_gpu = dst_cpu.clone();

    vector<Vec4i> lines_cpu;
    HoughLinesP(mask, lines_cpu, 1, CV_PI / 180, 50, 50, 5);
    cout << lines_cpu.size() << endl;

    for (size_t i = 0; i < lines_cpu.size(); ++i)
    {
        Vec4i l = lines_cpu[i];
        line(dst_cpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
    }

    GpuMat d_src(src);
    GpuMat d_lines;
    CannyBuf d_buf;
    gpu::HoughLinesP(d_src, d_lines, d_buf, 50, 5);
    vector<Vec4i> lines_gpu;
    if (!d_lines.empty())
    {
        lines_gpu.resize(d_lines.cols);
        Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
        d_lines.download(h_lines);
    }
    cout << lines_gpu.size() << endl;

    for (size_t i = 0; i < lines_gpu.size(); ++i)
    {
        Vec4i l = lines_gpu[i];
        line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
    }

    imshow("source", src);
    imshow("detected lines [CPU]", dst_cpu);
    imshow("detected lines [GPU]", dst_gpu);
    waitKey();

    return 0;
}

