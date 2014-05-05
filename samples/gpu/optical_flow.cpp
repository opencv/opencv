#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flowx, const GpuMat& d_flowy)
{
    Mat flowx(d_flowx);
    Mat flowy(d_flowy);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
}

int main(int argc, const char* argv[])
{
    if (argc < 3)
    {
        cerr << "Usage : " << argv[0] << "<frame0> <frame1>" << endl;
        return -1;
    }

    Mat frame0 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat frame1 = imread(argv[2], IMREAD_GRAYSCALE);

    if (frame0.empty())
    {
        cerr << "Can't open image ["  << argv[1] << "]" << endl;
        return -1;
    }
    if (frame1.empty())
    {
        cerr << "Can't open image ["  << argv[2] << "]" << endl;
        return -1;
    }

    if (frame1.size() != frame0.size())
    {
        cerr << "Images should be of equal sizes" << endl;
        return -1;
    }

    GpuMat d_frame0(frame0);
    GpuMat d_frame1(frame1);

    GpuMat d_flowx(frame0.size(), CV_32FC1);
    GpuMat d_flowy(frame0.size(), CV_32FC1);

    BroxOpticalFlow brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
    PyrLKOpticalFlow lk; lk.winSize = Size(7, 7);
    FarnebackOpticalFlow farn;
    OpticalFlowDual_TVL1_CUDA tvl1;
    FastOpticalFlowBM fastBM;

    {
        GpuMat d_frame0f;
        GpuMat d_frame1f;

        d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
        d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

        const int64 start = getTickCount();

        brox(d_frame0f, d_frame1f, d_flowx, d_flowy);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Brox : " << timeSec << " sec" << endl;

        showFlow("Brox", d_flowx, d_flowy);
    }

    {
        const int64 start = getTickCount();

        lk.dense(d_frame0, d_frame1, d_flowx, d_flowy);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "LK : " << timeSec << " sec" << endl;

        showFlow("LK", d_flowx, d_flowy);
    }

    {
        const int64 start = getTickCount();

        farn(d_frame0, d_frame1, d_flowx, d_flowy);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Farn : " << timeSec << " sec" << endl;

        showFlow("Farn", d_flowx, d_flowy);
    }

    {
        const int64 start = getTickCount();

        tvl1(d_frame0, d_frame1, d_flowx, d_flowy);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "TVL1 : " << timeSec << " sec" << endl;

        showFlow("TVL1", d_flowx, d_flowy);
    }

    {
        const int64 start = getTickCount();

        GpuMat buf;
        calcOpticalFlowBM(d_frame0, d_frame1, Size(7, 7), Size(1, 1), Size(21, 21), false, d_flowx, d_flowy, buf);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "BM : " << timeSec << " sec" << endl;

        showFlow("BM", d_flowx, d_flowy);
    }

    {
        const int64 start = getTickCount();

        fastBM(d_frame0, d_frame1, d_flowx, d_flowy);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Fast BM : " << timeSec << " sec" << endl;

        showFlow("Fast BM", d_flowx, d_flowy);
    }

    imshow("Frame 0", frame0);
    imshow("Frame 1", frame1);
    waitKey();

    return 0;
}
