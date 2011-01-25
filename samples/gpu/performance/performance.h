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
    static TestSystem* instance()
    {
        static TestSystem me;
        return &me;
    }

    void addInit(Runnable* init) { inits_.push_back(init); }

    void addTest(Runnable* test) { tests_.push_back(test); }

    void run();

    void cpuOn() { cpu_started_ = cv::getTickCount(); }

    void cpuOff() 
    {
        int64 delta = cv::getTickCount() - cpu_started_;
        cpu_elapsed_ += delta;
        can_flush_ = true;
    }  

    void gpuOn() { gpu_started_ = cv::getTickCount(); }

    void gpuOff() 
    {
        int64 delta = cv::getTickCount() - gpu_started_;
        gpu_elapsed_ += delta;
        can_flush_ = true;
    }

    // Ends current subtest and starts new one
    std::stringstream& subtest()
    {
        flush_subtest_data();
        return description_;
    }

private:
    TestSystem(): can_flush_(false), cpu_elapsed_(0), gpu_elapsed_(0), 
                  speedup_total_(0.0), num_subtests_called_(0) {};

    void flush_subtest_data();

    void reset_subtest_data() 
    {
        cpu_elapsed_ = 0;
        gpu_elapsed_ = 0;
        description_.str("");
        can_flush_ = false;
    }

    std::vector<Runnable*> inits_;
    std::vector<Runnable*> tests_;

    // Current test (subtest) description
    std::stringstream description_;

    bool can_flush_;

    int64 cpu_started_, cpu_elapsed_;
    int64 gpu_started_, gpu_elapsed_;

    double speedup_total_;
    int num_subtests_called_;
};


#define TEST(name) \
    struct name##_test: Runnable \
    { \
        name##_test(): Runnable(#name) { \
            TestSystem::instance()->addTest(this); \
        } \
        void run(); \
    } name##_test_instance; \
    void name##_test::run()


#define INIT(name) \
    struct name##_init: Runnable \
    { \
        name##_init(): Runnable(#name) { \
            TestSystem::instance()->addInit(this); \
        } \
        void run(); \
    } name##_init_instance; \
    void name##_init::run()


#define CPU_ON TestSystem::instance()->cpuOn()
#define GPU_ON TestSystem::instance()->gpuOn()
#define CPU_OFF TestSystem::instance()->cpuOff()
#define GPU_OFF TestSystem::instance()->gpuOff()
#define SUBTEST TestSystem::instance()->subtest()
#define DESCRIPTION TestSystem::instance()->subtest()

void gen(cv::Mat& mat, int rows, int cols, int type, cv::Scalar low, 
         cv::Scalar high);

#endif // OPENCV_GPU_SAMPLE_PERFORMANCE_H_