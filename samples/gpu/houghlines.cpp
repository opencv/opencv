#include <cmath>
#include <iostream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "This program demonstrates line finding with the Hough transform." << endl;
    cout << "Usage:" << endl;
    cout << "./gpu-example-houghlines <image_name>, Default is ../data/pic1.png\n" << endl;
}

int main(int argc, const char* argv[])
{
    const string filename = argc >= 2 ? argv[1] : "../data/pic1.png";

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    if (src.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    Mat mask;
    cv::Canny(src, mask, 100, 200, 3);

    Mat dst_cpu;
    cv::cvtColor(mask, dst_cpu, COLOR_GRAY2BGR);
    Mat dst_gpu = dst_cpu.clone();

    vector<Vec4i> lines_cpu;
    {
        const int64 start = getTickCount();

        cv::HoughLinesP(mask, lines_cpu, 1, CV_PI / 180, 50, 60, 5);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
        cout << "CPU Found : " << lines_cpu.size() << endl;
    }

    for (size_t i = 0; i < lines_cpu.size(); ++i)
    {
        Vec4i l = lines_cpu[i];
        line(dst_cpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    GpuMat d_src(mask);
    GpuMat d_lines;
    {
        const int64 start = getTickCount();

        Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);

        hough->detect(d_src, d_lines);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
        cout << "GPU Found : " << d_lines.cols << endl;
    }
    vector<Vec4i> lines_gpu;
    if (!d_lines.empty())
    {
        lines_gpu.resize(d_lines.cols);
        Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
        d_lines.download(h_lines);
    }

    for (size_t i = 0; i < lines_gpu.size(); ++i)
    {
        Vec4i l = lines_gpu[i];
        line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("source", src);
    imshow("detected lines [CPU]", dst_cpu);
    imshow("detected lines [GPU]", dst_gpu);
    waitKey();

    return 0;
}
