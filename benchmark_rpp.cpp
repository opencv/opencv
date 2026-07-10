/*
 * Benchmark RPP HAL backend vs OpenCV native for core/imgproc operations.
 *
 * Modes:
 *   Native:  run with OPENCV_RPP_DISABLE=1 (if supported) or by linking a
 *            build without the RPP HAL. For convenience we compare against
 *            the RPP path that returns NOT_IMPLEMENTED for everything except
 *            bitwise; that is effectively native for imgproc.
 *   RPP CPU: default path on this system.
 *   RPP HIP: OPENCV_RPP_FORCE_GPU=1.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>

using namespace cv;
using namespace std;

static double ms(chrono::steady_clock::time_point start,
                 chrono::steady_clock::time_point end) {
    return chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
}

struct BenchResult {
    string name;
    double totalMs;
    int iterations;
    bool ok;
};

static Mat makeMat(int h, int w, int type, int seed = 42) {
    Mat m(h, w, type);
    randu(m, Scalar(0), Scalar(255));
    return m;
}

static BenchResult bench_resize(int w, int h, int iterations) {
    Mat src = makeMat(h, w, CV_8UC3);
    Mat dst;
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        resize(src, dst, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
    }
    auto t2 = chrono::steady_clock::now();
    return {"resize 8UC3 " + to_string(w) + "x" + to_string(h) + " -> " +
            to_string(w / 2) + "x" + to_string(h / 2),
            ms(t1, t2), iterations, !dst.empty()};
}

static BenchResult bench_boxFilter(int w, int h, int iterations) {
    Mat src = makeMat(h, w, CV_8UC3);
    Mat dst;
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        boxFilter(src, dst, -1, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
    }
    auto t2 = chrono::steady_clock::now();
    return {"boxFilter 3x3 8UC3 " + to_string(w) + "x" + to_string(h),
            ms(t1, t2), iterations, !dst.empty()};
}

static BenchResult bench_warpAffine(int w, int h, int iterations) {
    Mat src = makeMat(h, w, CV_8UC3);
    double M_data[6] = {1.0, 0.1, 10.0, 0.05, 1.0, 20.0};
    Mat M(2, 3, CV_64FC1, M_data);
    Mat dst;
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        warpAffine(src, dst, M, src.size(), INTER_LINEAR, BORDER_REPLICATE);
    }
    auto t2 = chrono::steady_clock::now();
    return {"warpAffine 8UC3 " + to_string(w) + "x" + to_string(h),
            ms(t1, t2), iterations, !dst.empty()};
}

static BenchResult bench_flip(int w, int h, int iterations) {
    Mat src = makeMat(h, w, CV_8UC3);
    Mat dst;
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        flip(src, dst, 1);
    }
    auto t2 = chrono::steady_clock::now();
    return {"flip 8UC3 " + to_string(w) + "x" + to_string(h),
            ms(t1, t2), iterations, !dst.empty()};
}

static BenchResult bench_bitwise_and(int w, int h, int iterations) {
    Mat a = makeMat(h, w, CV_8UC1);
    Mat b = makeMat(h, w, CV_8UC1);
    Mat dst;
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bitwise_and(a, b, dst);
    }
    auto t2 = chrono::steady_clock::now();
    return {"bitwise_and 8UC1 " + to_string(w) + "x" + to_string(h),
            ms(t1, t2), iterations, !dst.empty()};
}

static void runSet(const string& label, int w, int h, int iterResize, int iterWarp, int iterFlip, int iterBitwise) {
    cout << "\n=== " << label << " ===" << endl;
    vector<BenchResult> results;
    results.push_back(bench_resize(w, h, iterResize));
    results.push_back(bench_warpAffine(w, h, iterWarp));
    results.push_back(bench_flip(w, h, iterFlip));
    results.push_back(bench_boxFilter(w, h, iterBitwise)); // reuse iter count
    results.push_back(bench_bitwise_and(w, h, iterBitwise));

    for (const auto& r : results) {
        if (r.ok) {
            cout << "[" << label << "] " << r.name
                 << ": " << (r.totalMs / r.iterations) << " ms/op  ("
                 << r.iterations << " iters, " << r.totalMs << " ms total)" << endl;
        } else {
            cout << "[" << label << "] " << r.name << ": FAILED" << endl;
        }
    }
}

int main() {
    cout << "=== RPP HAL Benchmark ===" << endl;
    cout << "OpenCV version: " << CV_VERSION << endl;

    // Small sizes: many iterations. Large sizes: fewer iterations.
    runSet("HD", 1920, 1080, 200, 100, 500, 200);
    runSet("4K", 3840, 2160, 50, 30, 150, 50);

    return 0;
}
