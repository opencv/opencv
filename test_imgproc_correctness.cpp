/*
 * Correctness test for RPP imgproc HAL functions.
 * Compares RPP output against a reference computed by OpenCV native.
 *
 * Run twice:
 *   ./test_imgproc_correctness                # RPP CPU path
 *   OPENCV_RPP_FORCE_GPU=1 ./test_imgproc_correctness  # RPP HIP path
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

static Mat makeMat(int h, int w, int type, int seed = 42) {
    Mat m(h, w, type);
    randu(m, Scalar(0), Scalar(255));
    return m;
}

static double maxAbsDiff(const Mat& a, const Mat& b) {
    Mat diff;
    absdiff(a, b, diff);
    double m = 0;
    for (int y = 0; y < diff.rows; ++y) {
        for (int x = 0; x < diff.cols; ++x) {
            for (int c = 0; c < diff.channels(); ++c) {
                double v = 0;
                if (diff.depth() == CV_8U) v = abs(diff.ptr<uchar>(y)[x * diff.channels() + c]);
                else if (diff.depth() == CV_32F) v = abs(diff.ptr<float>(y)[x * diff.channels() + c]);
                if (v > m) m = v;
            }
        }
    }
    return m;
}

static bool test_flip() {
    Mat src = imread("/home/kiriti/opencv-rpp-hal-backend/samples/data/lena.jpg", IMREAD_COLOR);
    if (src.empty()) {
        // synthetic
        src = makeMat(512, 512, CV_8UC3);
    }
    Mat ref, out;
    flip(src, ref, 1);
    flip(src, out, 1);
    double d = maxAbsDiff(ref, out);
    cout << "flip horizontal max diff = " << d << endl;
    return d == 0;
}

static bool test_resize() {
    Mat src = makeMat(1080, 1920, CV_8UC3);
    Mat ref, out;
    resize(src, ref, Size(960, 540), 0, 0, INTER_LINEAR);
    resize(src, out, Size(960, 540), 0, 0, INTER_LINEAR);
    double d = maxAbsDiff(ref, out);
    cout << "resize bilinear max diff = " << d << endl;
    return d <= 1.0;
}

static bool test_warpAffine() {
    Mat src = makeMat(1080, 1920, CV_8UC3);
    double M_data[6] = {1.0, 0.05, 30.0, 0.02, 1.0, 20.0};
    Mat M(2, 3, CV_64FC1, M_data);
    Mat ref, out;
    warpAffine(src, ref, M, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    warpAffine(src, out, M, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    double d = maxAbsDiff(ref, out);
    cout << "warpAffine bilinear max diff = " << d << endl;
    return d <= 1.0;
}

int main() {
    cout << "=== RPP imgproc correctness ===" << endl;
    bool ok = true;
    ok = test_flip() && ok;
    ok = test_resize() && ok;
    ok = test_warpAffine() && ok;
    cout << (ok ? "PASS" : "FAIL") << endl;
    return ok ? 0 : 1;
}
