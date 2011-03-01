#ifndef OPENCV_GPU_SAMPLE_PERFORMANCE_H_
#define OPENCV_GPU_SAMPLE_PERFORMANCE_H_

#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#define TAB "    "

class Runnable
{
public:
    explicit Runnable(const std::string& name): name_(name) {}
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

    void setWorkingDir(const std::string& val);
    const std::string& workingDir() const { return working_dir_; }

    void setTestFilter(const std::string& val);
    const std::string& testFilter() const { return test_filter_; }

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

    void cpuOn() { cpu_started_ = cv::getTickCount(); }
    void cpuOff() 
    {
        int64 delta = cv::getTickCount() - cpu_started_;
        cpu_elapsed_ += delta;
        cur_subtest_is_empty_ = false;
    }  

    void gpuOn() { gpu_started_ = cv::getTickCount(); }
    void gpuOff() 
    {
        int64 delta = cv::getTickCount() - gpu_started_;
        gpu_elapsed_ += delta;
        cur_subtest_is_empty_ = false;
    }

private:
    TestSystem(): cur_subtest_is_empty_(true), cpu_elapsed_(0),
                  gpu_elapsed_(0), speedup_total_(0.0),
                  num_subtests_called_(0) {}

    void finishCurrentSubtest();
    void resetCurrentSubtest() 
    {
        cpu_elapsed_ = 0;
        gpu_elapsed_ = 0;
        cur_subtest_description_.str("");
        cur_subtest_is_empty_ = true;
    }

    void printHeading();
    void printSummary();
    void printMetrics(double cpu_time, double gpu_time, double speedup);

    std::string working_dir_;
    std::string test_filter_;

    std::vector<Runnable*> inits_;
    std::vector<Runnable*> tests_;

    std::stringstream cur_subtest_description_;
    bool cur_subtest_is_empty_;

    int64 cpu_started_, cpu_elapsed_;
    int64 gpu_started_, gpu_elapsed_;

    double speedup_total_;
    int num_subtests_called_;
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
#define CPU_ON TestSystem::instance().cpuOn()
#define GPU_ON TestSystem::instance().gpuOn()
#define CPU_OFF TestSystem::instance().cpuOff()
#define GPU_OFF TestSystem::instance().gpuOff()

// Generates matrix
void gen(cv::Mat& mat, int rows, int cols, int type, cv::Scalar low, 
         cv::Scalar high);

// Returns abs path taking into account test system working dir
std::string abspath(const std::string& relpath);

#endif // OPENCV_GPU_SAMPLE_PERFORMANCE_H_
