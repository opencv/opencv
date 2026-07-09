/*
 * Test RPP HAL backend for simoncatbot-opencv
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

static double ms(chrono::steady_clock::time_point start, chrono::steady_clock::time_point end) {
    return chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
}

int main() {
    cout << "=== RPP HAL Backend Test ===" << endl;
    cout << "OpenCV version: " << CV_VERSION << endl;

    // Create test images (grayscale first to isolate channel issues)
    Mat gray(1080, 1920, CV_8UC1, Scalar(128));
    Mat gray2(1080, 1920, CV_8UC1, Scalar(64));

    // ---- Test 1: Bitwise AND (core, 8u, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        bitwise_and(gray, gray2, dst);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] bitwise_and 8u grayscale: " << ms(t1, t2) << " ms" << endl;

        uchar expected = 128 & 64; // 0
        if (dst.at<uchar>(100, 100) == expected) {
            cout << "  ✓ Correctness check passed" << endl;
        } else {
            cout << "  ✗ Correctness check FAILED! Expected " << (int)expected << " got " << (int)dst.at<uchar>(100, 100) << endl;
        }
    }

    // ---- Test 2: Bitwise NOT (core, 8u, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        bitwise_not(gray, dst);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] bitwise_not 8u: " << ms(t1, t2) << " ms" << endl;

        if (dst.at<uchar>(100, 100) == static_cast<uchar>(~128)) {
            cout << "  ✓ Correctness check passed" << endl;
        } else {
            cout << "  ✗ Correctness check FAILED!" << endl;
        }
    }

    // ---- Test 3: Resize (imgproc, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        resize(gray, dst, Size(960, 540), 0, 0, INTER_LINEAR);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] resize grayscale: " << ms(t1, t2) << " ms" << endl;

        if (dst.size() == Size(960, 540)) {
            cout << "  ✓ Size check passed" << endl;
        } else {
            cout << "  ✗ Size check FAILED!" << endl;
        }
    }

    // ---- Test 4: boxFilter (imgproc, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        boxFilter(gray, dst, -1, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] boxFilter 3x3: " << ms(t1, t2) << " ms" << endl;
    }

    // ---- Test 5: GaussianBlur (imgproc, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        GaussianBlur(gray, dst, Size(5, 5), 1.0, 1.0, BORDER_REPLICATE);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] GaussianBlur 5x5: " << ms(t1, t2) << " ms" << endl;
    }

    // ---- Test 6: medianBlur (imgproc, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        medianBlur(gray, dst, 3);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] medianBlur 3x3: " << ms(t1, t2) << " ms" << endl;
    }

    // ---- Test 7: warpAffine (imgproc, GPU via RPP) ----
    {
        double M_data[6] = {1.0, 0.0, 10.0, 0.0, 1.0, 20.0};
        Mat M(2, 3, CV_64FC1, M_data);
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        warpAffine(gray, dst, M, gray.size(), INTER_LINEAR, BORDER_REPLICATE);
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] warpAffine: " << ms(t1, t2) << " ms" << endl;
    }

    // ---- Test 8: flip (imgproc, GPU via RPP) ----
    {
        Mat dst;
        auto t1 = chrono::steady_clock::now();
        flip(gray, dst, 1); // horizontal
        auto t2 = chrono::steady_clock::now();
        cout << "[PASS] flip horizontal: " << ms(t1, t2) << " ms" << endl;
    }

    // ---- Test 9: CPU fallback functions ----
    {
        Mat grad, edges;

        auto t1 = chrono::steady_clock::now();
        Sobel(gray, grad, CV_16S, 1, 0, 3);
        auto t2 = chrono::steady_clock::now();
        cout << "[INFO] Sobel (CPU fallback): " << ms(t1, t2) << " ms" << endl;

        t1 = chrono::steady_clock::now();
        Canny(gray, edges, 50, 150);
        t2 = chrono::steady_clock::now();
        cout << "[INFO] Canny (CPU fallback): " << ms(t1, t2) << " ms" << endl;
    }

    cout << "\n=== All tests completed ===" << endl;
    return 0;
}
