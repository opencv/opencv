/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ocl.hpp"
#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "opencv2/ts/ts_gtest.h"

#include "opencv2/core/utility.hpp"

#define Min_Size 1000
#define Max_Size 4000
#define Multiple 2
#define TAB "    "

using namespace std;
using namespace cv;

void gen(Mat &mat, int rows, int cols, int type, Scalar low, Scalar high);
void gen(Mat &mat, int rows, int cols, int type, int low, int high, int n);

string abspath(const string &relpath);

typedef struct
{
    short x;
    short y;
} COOR;
COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep,
                  cv::Size size, int sp, int sr, int maxIter, float eps, int *tab);
void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi,
                    int sp, int sr, cv::TermCriteria crit);


template<class T1, class T2>
int ExpectedEQ(T1 expected, T2 actual)
{
    if(expected == actual)
        return 1;

    return 0;
}

template<class T1>
int EeceptDoubleEQ(T1 expected, T1 actual)
{
    testing::internal::Double lhs(expected);
    testing::internal::Double rhs(actual);

    if (lhs.AlmostEquals(rhs))
    {
        return 1;
    }

    return 0;
}

template<class T>
int AssertEQ(T expected, T actual)
{
    if(expected == actual)
    {
        return 1;
    }
    return 0;
}

int ExceptDoubleNear(double val1, double val2, double abs_error);
bool match_rect(cv::Rect r1, cv::Rect r2, int threshold);

double checkNorm(const cv::Mat &m);
double checkNorm(const cv::Mat &m1, const cv::Mat &m2);
double checkSimilarity(const cv::Mat &m1, const cv::Mat &m2);

int ExpectedMatNear(cv::Mat dst, cv::Mat cpu_dst, double eps);
int ExceptedMatSimilar(cv::Mat dst, cv::Mat cpu_dst, double eps);

class Runnable
{
public:
    explicit Runnable(const std::string &runname): name_(runname) {}
    virtual ~Runnable() {}

    const std::string &name() const
    {
        return name_;
    }

    virtual void run() = 0;

private:
    std::string name_;
};

class TestSystem
{
public:
    static TestSystem &instance()
    {
        static TestSystem me;
        return me;
    }

    void setWorkingDir(const std::string &val)
    {
        working_dir_ = val;
    }
    const std::string &workingDir() const
    {
        return working_dir_;
    }

    void setTestFilter(const std::string &val)
    {
        test_filter_ = val;
    }
    const std::string &testFilter() const
    {
        return test_filter_;
    }

    void setNumIters(int num_iters)
    {
        num_iters_ = num_iters;
    }
    void setGPUWarmupIters(int num_iters)
    {
        gpu_warmup_iters_ = num_iters;
    }
    void setCPUIters(int num_iters)
    {
        cpu_num_iters_ = num_iters;
    }

    void setTopThreshold(double top)
    {
        top_ = top;
    }
    void setBottomThreshold(double bottom)
    {
        bottom_ = bottom;
    }

    void addInit(Runnable *init)
    {
        inits_.push_back(init);
    }
    void addTest(Runnable *test)
    {
        tests_.push_back(test);
    }
    void run();

    // It's public because OpenCV callback uses it
    void printError(const std::string &msg);

    std::stringstream &startNewSubtest()
    {
        finishCurrentSubtest();
        return cur_subtest_description_;
    }

    bool stop() const
    {
        return cur_iter_idx_ >= num_iters_;
    }

    bool cpu_stop() const
    {
        return cur_iter_idx_ >= cpu_num_iters_;
    }

    int get_cur_iter_idx()
    {
        return cur_iter_idx_;
    }

    int get_cpu_num_iters()
    {
        return cpu_num_iters_;
    }

    bool warmupStop()
    {
        return cur_warmup_idx_++ >= gpu_warmup_iters_;
    }

    void warmupComplete()
    {
        cur_warmup_idx_ = 0;
    }

    void cpuOn()
    {
        cpu_started_ = cv::getTickCount();
    }
    void cpuOff()
    {
        int64 delta = cv::getTickCount() - cpu_started_;
        cpu_times_.push_back(delta);
        ++cur_iter_idx_;
    }
    void cpuComplete()
    {
        cpu_elapsed_ += meanTime(cpu_times_);
        cur_subtest_is_empty_ = false;
        cur_iter_idx_ = 0;
    }

    void gpuOn()
    {
        gpu_started_ = cv::getTickCount();
    }
    void gpuOff()
    {
        int64 delta = cv::getTickCount() - gpu_started_;
        gpu_times_.push_back(delta);
        ++cur_iter_idx_;
    }
    void gpuComplete()
    {
        gpu_elapsed_ += meanTime(gpu_times_);
        cur_subtest_is_empty_ = false;
        cur_iter_idx_ = 0;
    }

    void gpufullOn()
    {
        gpu_full_started_ = cv::getTickCount();
    }
    void gpufullOff()
    {
        int64 delta = cv::getTickCount() - gpu_full_started_;
        gpu_full_times_.push_back(delta);
        ++cur_iter_idx_;
    }
    void gpufullComplete()
    {
        gpu_full_elapsed_ += meanTime(gpu_full_times_);
        cur_subtest_is_empty_ = false;
        cur_iter_idx_ = 0;
    }

    bool isListMode() const
    {
        return is_list_mode_;
    }
    void setListMode(bool value)
    {
        is_list_mode_ = value;
    }

    void setRecordName(const std::string &name)
    {
        recordname_ = name;
    }

    void setCurrentTest(const std::string &name)
    {
        itname_ = name;
        itname_changed_ = true;
    }

    void setAccurate(int accurate, double diff)
    {
        is_accurate_ = accurate;
        accurate_diff_ = diff;
    }

    void ExpectMatsNear(vector<Mat>& dst, vector<Mat>& cpu_dst, vector<double>& eps)
    {
        assert(dst.size() == cpu_dst.size());
        assert(cpu_dst.size() == eps.size());
        is_accurate_ = 1;
        for(size_t i=0; i<dst.size(); i++)
        {
            double cur_diff = checkNorm(dst[i], cpu_dst[i]);
            accurate_diff_ = max(accurate_diff_, cur_diff);
            if(cur_diff > eps[i])
                is_accurate_ = 0;
        }
    }

    void ExpectedMatNear(cv::Mat& dst, cv::Mat& cpu_dst, double eps)
    {
        assert(dst.type() == cpu_dst.type());
        assert(dst.size() == cpu_dst.size());
        accurate_diff_ = checkNorm(dst, cpu_dst);
        if(accurate_diff_ <= eps)
            is_accurate_ = 1;
        else
            is_accurate_ = 0;
    }

    void ExceptedMatSimilar(cv::Mat& dst, cv::Mat& cpu_dst, double eps)
    {
        assert(dst.type() == cpu_dst.type());
        assert(dst.size() == cpu_dst.size());
        accurate_diff_ = checkSimilarity(cpu_dst, dst);
        if(accurate_diff_ <= eps)
            is_accurate_ = 1;
        else
            is_accurate_ = 0;
    }

    std::stringstream &getCurSubtestDescription()
    {
        return cur_subtest_description_;
    }

private:
    TestSystem():
        cur_subtest_is_empty_(true), cpu_elapsed_(0),
        gpu_elapsed_(0), gpu_full_elapsed_(0), speedup_total_(0.0),
        num_subtests_called_(0),
        speedup_faster_count_(0), speedup_slower_count_(0), speedup_equal_count_(0),
        speedup_full_faster_count_(0), speedup_full_slower_count_(0), speedup_full_equal_count_(0), is_list_mode_(false),
        num_iters_(10), cpu_num_iters_(2),
        gpu_warmup_iters_(1), cur_iter_idx_(0), cur_warmup_idx_(0),
        record_(0), recordname_("performance"), itname_changed_(true),
        is_accurate_(-1), accurate_diff_(0.)
    {
        cpu_times_.reserve(num_iters_);
        gpu_times_.reserve(num_iters_);
        gpu_full_times_.reserve(num_iters_);
    }

    void finishCurrentSubtest();
    void resetCurrentSubtest()
    {
        cpu_elapsed_ = 0;
        gpu_elapsed_ = 0;
        gpu_full_elapsed_ = 0;
        cur_subtest_description_.str("");
        cur_subtest_is_empty_ = true;
        cur_iter_idx_ = 0;
        cur_warmup_idx_ = 0;
        cpu_times_.clear();
        gpu_times_.clear();
        gpu_full_times_.clear();
        is_accurate_ = -1;
        accurate_diff_ = 0.;
    }

    double meanTime(const std::vector<int64> &samples);

    void printHeading();
    void printSummary();
    void printMetrics(int is_accurate, double cpu_time, double gpu_time = 0.0f, double gpu_full_time = 0.0f, double speedup = 0.0f, double fullspeedup = 0.0f);

    void writeHeading();
    void writeSummary();
    void writeMetrics(double cpu_time, double gpu_time = 0.0f, double gpu_full_time = 0.0f,
                      double speedup = 0.0f, double fullspeedup = 0.0f,
                      double gpu_min = 0.0f, double gpu_max = 0.0f, double std_dev = 0.0f);

    std::string working_dir_;
    std::string test_filter_;

    std::vector<Runnable *> inits_;
    std::vector<Runnable *> tests_;

    std::stringstream cur_subtest_description_;
    bool cur_subtest_is_empty_;

    int64 cpu_started_;
    int64 gpu_started_;
    int64 gpu_full_started_;
    double cpu_elapsed_;
    double gpu_elapsed_;
    double gpu_full_elapsed_;

    double speedup_total_;
    double speedup_full_total_;
    int num_subtests_called_;

    int speedup_faster_count_;
    int speedup_slower_count_;
    int speedup_equal_count_;

    int speedup_full_faster_count_;
    int speedup_full_slower_count_;
    int speedup_full_equal_count_;

    bool is_list_mode_;

    double top_;
    double bottom_;

    int num_iters_;
    int cpu_num_iters_;     //there's no need to set cpu running same times with gpu
    int gpu_warmup_iters_;  //gpu warm up times, default is 1
    int cur_iter_idx_;
    int cur_warmup_idx_;    //current gpu warm up times
    std::vector<int64> cpu_times_;
    std::vector<int64> gpu_times_;
    std::vector<int64> gpu_full_times_;

    FILE *record_;
    std::string recordname_;
    std::string itname_;
    bool itname_changed_;

    int is_accurate_;
    double accurate_diff_;
};


#define GLOBAL_INIT(name) \
struct name##_init: Runnable { \
    name##_init(): Runnable(#name) { \
    TestSystem::instance().addInit(this); \
} \
    void run(); \
} name##_init_instance; \
    void name##_init::run()


#define PERFTEST(name) \
struct name##_test: Runnable { \
    name##_test(): Runnable(#name) { \
    TestSystem::instance().addTest(this); \
} \
    void run(); \
} name##_test_instance; \
    void name##_test::run()

#define SUBTEST TestSystem::instance().startNewSubtest()

#define CPU_ON \
    while (!TestSystem::instance().cpu_stop()) { \
    TestSystem::instance().cpuOn()
#define CPU_OFF \
    TestSystem::instance().cpuOff(); \
    } TestSystem::instance().cpuComplete()

#define GPU_ON \
    while (!TestSystem::instance().stop()) { \
    TestSystem::instance().gpuOn()
#define GPU_OFF \
    ocl::finish(); \
    TestSystem::instance().gpuOff(); \
    } TestSystem::instance().gpuComplete()

#define GPU_FULL_ON \
    while (!TestSystem::instance().stop()) { \
    TestSystem::instance().gpufullOn()
#define GPU_FULL_OFF \
    TestSystem::instance().gpufullOff(); \
    } TestSystem::instance().gpufullComplete()

#define WARMUP_ON \
    while (!TestSystem::instance().warmupStop()) {
#define WARMUP_OFF \
        ocl::finish(); \
    } TestSystem::instance().warmupComplete()

#endif
