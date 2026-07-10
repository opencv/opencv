/*
 * Comprehensive RPP HAL benchmark.
 *
 * Runs every implemented RPP HAL function at multiple resolutions,
 * with warm-up and measured iterations, for CPU and forced HIP.
 *
 * Run:
 *   ./benchmark_rpp_full              # RPP CPU (or probe-selected path)
 *   OPENCV_RPP_FORCE_GPU=1 ./benchmark_rpp_full  # force HIP
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

using namespace cv;
using namespace std;

static int WARMUP = 20;
static int ITERS  = 1000;
static int MAX_FAILURES = 3;

struct BenchCase {
    string name;
    int width;
    int height;
    function<void()> fn;
};

static double ms(chrono::steady_clock::time_point a,
                 chrono::steady_clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() / 1000.0;
}

static Mat makeMat(int h, int w, int type) {
    Mat m(h, w, type);
    randu(m, Scalar(0), Scalar(255));
    return m;
}

static void addBitwiseCases(vector<BenchCase>& cases, int w, int h) {
    Mat a = makeMat(h, w, CV_8UC1);
    Mat b = makeMat(h, w, CV_8UC1);
    auto dst = make_shared<Mat>();
    cases.push_back({"bitwise_and 8UC1", w, h, [a, b, dst]{ bitwise_and(a, b, *dst); }});
    cases.push_back({"bitwise_or 8UC1",  w, h, [a, b, dst]{ bitwise_or(a, b, *dst); }});
    cases.push_back({"bitwise_xor 8UC1", w, h, [a, b, dst]{ bitwise_xor(a, b, *dst); }});
    cases.push_back({"bitwise_not 8UC1", w, h, [a, dst]{ bitwise_not(a, *dst); }});
}

static void addImgprocCases(vector<BenchCase>& cases, int w, int h) {
    Mat src8u3 = makeMat(h, w, CV_8UC3);
    Mat src8u1 = makeMat(h, w, CV_8UC1);
    auto dst = make_shared<Mat>();

    cases.push_back({"resize 8UC3 bilinear down2x", w, h, [src8u3, dst]{
        resize(src8u3, *dst, Size(src8u3.cols / 2, src8u3.rows / 2), 0, 0, INTER_LINEAR);
    }});
    cases.push_back({"resize 8UC3 bilinear up2x", w, h, [src8u3, dst]{
        resize(src8u3, *dst, Size(src8u3.cols * 2, src8u3.rows * 2), 0, 0, INTER_LINEAR);
    }});
    cases.push_back({"flip 8UC3 horizontal", w, h, [src8u3, dst]{
        flip(src8u3, *dst, 1);
    }});
    cases.push_back({"flip 8UC3 vertical", w, h, [src8u3, dst]{
        flip(src8u3, *dst, 0);
    }});
    cases.push_back({"flip 8UC3 both", w, h, [src8u3, dst]{
        flip(src8u3, *dst, -1);
    }});

    double M_data[6] = {1.0, 0.05, 30.0, 0.02, 1.0, 20.0};
    Mat M(2, 3, CV_64FC1, M_data);
    cases.push_back({"warpAffine 8UC3 bilinear", w, h, [src8u3, M, dst]{
        warpAffine(src8u3, *dst, M, src8u3.size(), INTER_LINEAR, BORDER_REPLICATE);
    }});

    cases.push_back({"boxFilter 3x3 8UC3 replicate", w, h, [src8u3, dst]{
        boxFilter(src8u3, *dst, -1, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
    }});
    cases.push_back({"boxFilter 5x5 8UC3 replicate", w, h, [src8u3, dst]{
        boxFilter(src8u3, *dst, -1, Size(5, 5), Point(-1, -1), true, BORDER_REPLICATE);
    }});
    cases.push_back({"boxFilter 3x3 8UC1 replicate", w, h, [src8u1, dst]{
        boxFilter(src8u1, *dst, -1, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
    }});
}

static vector<BenchCase> buildCases(const vector<pair<int,int>>& sizes) {
    vector<BenchCase> cases;
    for (auto& sz : sizes) {
        int w = sz.first, h = sz.second;
        addBitwiseCases(cases, w, h);
        addImgprocCases(cases, w, h);
    }
    return cases;
}

static void runBenchmark(const vector<pair<int,int>>& sizes) {
    vector<BenchCase> cases = buildCases(sizes);

    cout << "\n" << string(90, '=') << "\n";
    cout << "RPP HAL full benchmark | warmups=" << WARMUP << " | iters=" << ITERS << "\n";
    cout << string(90, '=') << "\n";
    cout << left << setw(38) << "function"
         << setw(12) << "resolution"
         << right << setw(14) << "ms/op"
         << setw(14) << "total_ms"
         << setw(10) << "status" << "\n";
    cout << string(90, '-') << "\n";

    for (auto& c : cases) {
        bool failed = false;
        string failMsg;
        // warm-up
        try {
            for (int i = 0; i < WARMUP; ++i) c.fn();
        } catch (const std::exception& e) {
            failed = true; failMsg = string("WARMUP_ERR:") + e.what();
        } catch (...) {
            failed = true; failMsg = "WARMUP_ERR";
        }

        double total = 0, perOp = 0;
        if (!failed) {
            auto t1 = chrono::steady_clock::now();
            for (int i = 0; i < ITERS; ++i) {
                c.fn();
            }
            auto t2 = chrono::steady_clock::now();
            total = ms(t1, t2);
            perOp = total / ITERS;
        }

        cout << left << setw(38) << c.name
             << setw(12) << (to_string(c.width) + "x" + to_string(c.height))
             << right << fixed << setprecision(4) << setw(14) << perOp
             << setw(14) << total
             << setw(10) << (failed ? failMsg : "OK") << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        try { WARMUP = stoi(argv[1]); } catch (...) {}
    }
    if (argc > 2) {
        try { ITERS = stoi(argv[2]); } catch (...) {}
    }

    cout << "OpenCV version: " << CV_VERSION << "\n";

    vector<pair<int,int>> sizes = {
        {640, 480},    // VGA
        {1280, 720},   // HD
        {1920, 1080},  // FHD
        {2560, 1440},  // QHD
        {3840, 2160},  // 4K
    };

    runBenchmark(sizes);
    return 0;
}
