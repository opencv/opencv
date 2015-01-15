#ifndef OPENCV_CUDA_SAMPLE_PERFORMANCE_H_
#define OPENCV_CUDA_SAMPLE_PERFORMANCE_H_

#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>
#include <string>
#include <opencv2/core/utility.hpp>

#define TAB "    "

class Runnable
{
public:
    explicit Runnable(const std::string& nameStr): name_(nameStr) {}
    virtual ~Runnable() {}

    const std::string& name() const { return name_; }

    virtual void run() = 0;

private:
    std::string name_;
};


class TestSystem
{
public:
    static TestSystem& instance()
    {
        static TestSystem me;
        return me;
    }

    void setWorkingDir(const std::string& val) { working_dir_ = val; }
    const std::string& workingDir() const { return working_dir_; }

    void setTestFilter(const std::string& val) { test_filter_ = val; }
    const std::string& testFilter() const { return test_filter_; }

    void setNumIters(int num_iters) { num_iters_ = num_iters; }

    void addInit(Runnable* init) { inits_.push_back(init); }
    void addTest(Runnable* test) { tests_.push_back(test); }
    void run();

    // It's public because OpenCV callback uses it
    void printError(const std::string& msg);

    std::stringstream& startNewSubtest()
    {
        finishCurrentSubtest();
        return cur_subtest_description_;
    }

    bool stop() const { return cur_iter_idx_ >= num_iters_; }

    void cpuOn() { cpu_started_ = cv::getTickCount(); }
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

    void gpuOn() { gpu_started_ = cv::getTickCount(); }
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

    bool isListMode() const { return is_list_mode_; }
    void setListMode(bool value) { is_list_mode_ = value; }

private:
    TestSystem():
            cur_subtest_is_empty_(true), cpu_elapsed_(0),
            gpu_elapsed_(0), speedup_total_(0.0),
            num_subtests_called_(0), is_list_mode_(false),
            num_iters_(10), cur_iter_idx_(0)
    {
        cpu_times_.reserve(num_iters_);
        gpu_times_.reserve(num_iters_);
    }

    void finishCurrentSubtest();
    void resetCurrentSubtest()
    {
        cpu_elapsed_ = 0;
        gpu_elapsed_ = 0;
        cur_subtest_description_.str("");
        cur_subtest_is_empty_ = true;
        cur_iter_idx_ = 0;
        cpu_times_.clear();
        gpu_times_.clear();
    }

    double meanTime(const std::vector<int64> &samples);

    void printHeading();
    void printSummary();
    void printMetrics(double cpu_time, double gpu_time, double speedup);

    std::string working_dir_;
    std::string test_filter_;

    std::vector<Runnable*> inits_;
    std::vector<Runnable*> tests_;

    std::stringstream cur_subtest_description_;
    bool cur_subtest_is_empty_;

    int64 cpu_started_;
    int64 gpu_started_;
    double cpu_elapsed_;
    double gpu_elapsed_;

    double speedup_total_;
    int num_subtests_called_;

    bool is_list_mode_;

    int num_iters_;
    int cur_iter_idx_;
    std::vector<int64> cpu_times_;
    std::vector<int64> gpu_times_;
};


#define GLOBAL_INIT(name) \
    struct name##_init: Runnable { \
        name##_init(): Runnable(#name) { \
            TestSystem::instance().addInit(this); \
        } \
        void run(); \
    } name##_init_instance; \
    void name##_init::run()


#define TEST(name) \
    struct name##_test: Runnable { \
        name##_test(): Runnable(#name) { \
            TestSystem::instance().addTest(this); \
        } \
        void run(); \
    } name##_test_instance; \
    void name##_test::run()

#define SUBTEST TestSystem::instance().startNewSubtest()

#define CPU_ON \
    while (!TestSystem::instance().stop()) { \
        TestSystem::instance().cpuOn()
#define CPU_OFF \
        TestSystem::instance().cpuOff(); \
    } TestSystem::instance().cpuComplete()

#define CUDA_ON \
    while (!TestSystem::instance().stop()) { \
        TestSystem::instance().gpuOn()
#define CUDA_OFF \
        TestSystem::instance().gpuOff(); \
    } TestSystem::instance().gpuComplete()

// Generates a matrix
void gen(cv::Mat& mat, int rows, int cols, int type, cv::Scalar low,
         cv::Scalar high);

// Returns abs path taking into account test system working dir
std::string abspath(const std::string& relpath);

#endif // OPENCV_CUDA_SAMPLE_PERFORMANCE_H_
