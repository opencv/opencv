#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#define USE_OPENCL
#ifdef USE_OPENCL
#include "opencv2/ocl/ocl.hpp"
#endif

#define TAB "    "

using namespace std;
using namespace cv;

// This program test most of the functions in ocl module and generate data metrix of x-factor in .csv files
// All images needed in this test are in samples/gpu folder.
// For haar template, please rename it to facedetect.xml

void gen(Mat &mat, int rows, int cols, int type, Scalar low, Scalar high);
string abspath(const string &relpath);
int CV_CDECL cvErrorCallback(int, const char *, const char *, const char *, int, void *);
typedef struct
{
    short x;
    short y;
} COOR;
COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep,
                  cv::Size size, int sp, int sr, int maxIter, float eps, int *tab);
void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi,
                    int sp, int sr, cv::TermCriteria crit);

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

private:
    TestSystem():
        cur_subtest_is_empty_(true), cpu_elapsed_(0),
        gpu_elapsed_(0), gpu_full_elapsed_(0), speedup_total_(0.0),
        num_subtests_called_(0),
        speedup_faster_count_(0), speedup_slower_count_(0), speedup_equal_count_(0),
        speedup_full_faster_count_(0), speedup_full_slower_count_(0), speedup_full_equal_count_(0), is_list_mode_(false),
        num_iters_(10), cpu_num_iters_(2),
        gpu_warmup_iters_(1), cur_iter_idx_(0), cur_warmup_idx_(0),
        record_(0), recordname_("performance"), itname_changed_(true)
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
        cpu_times_.clear();
        gpu_times_.clear();
        gpu_full_times_.clear();
    }

    double meanTime(const std::vector<int64> &samples);

    void printHeading();
    void printSummary();
    void printMetrics(double cpu_time, double gpu_time, double gpu_full_time, double speedup, double fullspeedup);

    void writeHeading();
    void writeSummary();
    void writeMetrics(double cpu_time, double gpu_time, double gpu_full_time,
                      double speedup, double fullspeedup,
                      double gpu_min, double gpu_max, double std_dev);

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
    int cpu_num_iters_;		//there's no need to set cpu running same times with gpu
    int gpu_warmup_iters_;	//gpu warm up times, default is 1
    int cur_iter_idx_;
    int cur_warmup_idx_;	//current gpu warm up times
    std::vector<int64> cpu_times_;
    std::vector<int64> gpu_times_;
    std::vector<int64> gpu_full_times_;

    FILE *record_;
    std::string recordname_;
    std::string itname_;
    bool itname_changed_;
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
    while (!TestSystem::instance().cpu_stop()) { \
        TestSystem::instance().cpuOn()
#define CPU_OFF \
        TestSystem::instance().cpuOff(); \
    } TestSystem::instance().cpuComplete()

#define GPU_ON \
    while (!TestSystem::instance().stop()) { \
        TestSystem::instance().gpuOn()
#define GPU_OFF \
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
    } TestSystem::instance().warmupComplete()

void TestSystem::run()
{
    if (is_list_mode_)
    {
        for (vector<Runnable *>::iterator it = tests_.begin(); it != tests_.end(); ++it)
        {
            cout << (*it)->name() << endl;
        }

        return;
    }

    // Run test initializers
    for (vector<Runnable *>::iterator it = inits_.begin(); it != inits_.end(); ++it)
    {
        if ((*it)->name().find(test_filter_, 0) != string::npos)
        {
            (*it)->run();
        }
    }

    printHeading();
    writeHeading();

    // Run tests
    for (vector<Runnable *>::iterator it = tests_.begin(); it != tests_.end(); ++it)
    {
        try
        {
            if ((*it)->name().find(test_filter_, 0) != string::npos)
            {
                cout << endl << (*it)->name() << ":\n";

                setCurrentTest((*it)->name());
                //fprintf(record_,"%s\n",(*it)->name().c_str());

                (*it)->run();
                finishCurrentSubtest();
            }
        }
        catch (const Exception &)
        {
            // Message is printed via callback
            resetCurrentSubtest();
        }
        catch (const runtime_error &e)
        {
            printError(e.what());
            resetCurrentSubtest();
        }
    }

#ifdef USE_OPENCL
    printSummary();
    writeSummary();
#endif
}


void TestSystem::finishCurrentSubtest()
{
    if (cur_subtest_is_empty_)
        // There is no need to print subtest statistics
    {
        return;
    }

    double cpu_time = cpu_elapsed_ / getTickFrequency() * 1000.0;
    double gpu_time = gpu_elapsed_ / getTickFrequency() * 1000.0;
    double gpu_full_time = gpu_full_elapsed_ / getTickFrequency() * 1000.0;

    double speedup = static_cast<double>(cpu_elapsed_) / std::max(1.0, gpu_elapsed_);
    speedup_total_ += speedup;

    double fullspeedup = static_cast<double>(cpu_elapsed_) / std::max(1.0, gpu_full_elapsed_);
    speedup_full_total_ += fullspeedup;

    if (speedup > top_)
    {
        speedup_faster_count_++;
    }
    else if (speedup < bottom_)
    {
        speedup_slower_count_++;
    }
    else
    {
        speedup_equal_count_++;
    }

    if (fullspeedup > top_)
    {
        speedup_full_faster_count_++;
    }
    else if (fullspeedup < bottom_)
    {
        speedup_full_slower_count_++;
    }
    else
    {
        speedup_full_equal_count_++;
    }

    // compute min, max and
    std::sort(gpu_times_.begin(), gpu_times_.end());
    double gpu_min = gpu_times_.front() / getTickFrequency() * 1000.0;
    double gpu_max = gpu_times_.back() / getTickFrequency() * 1000.0;
    double deviation = 0;

    if (gpu_times_.size() > 1)
    {
        double sum = 0;

        for (size_t i = 0; i < gpu_times_.size(); i++)
        {
            int64 diff = gpu_times_[i] - static_cast<int64>(gpu_elapsed_);
            double diff_time = diff * 1000 / getTickFrequency();
            sum += diff_time * diff_time;
        }

        deviation = std::sqrt(sum / gpu_times_.size());
    }

    printMetrics(cpu_time, gpu_time, gpu_full_time, speedup, fullspeedup);
    writeMetrics(cpu_time, gpu_time, gpu_full_time, speedup, fullspeedup, gpu_min, gpu_max, deviation);

    num_subtests_called_++;
    resetCurrentSubtest();
}


double TestSystem::meanTime(const vector<int64> &samples)
{
    double sum = accumulate(samples.begin(), samples.end(), 0.);
    return sum / samples.size();
}


void TestSystem::printHeading()
{
    cout << endl;
    cout << setiosflags(ios_base::left);
#ifdef USE_OPENCL
    cout << TAB << setw(10) << "CPU, ms" << setw(10) << "GPU, ms"
         << setw(14) << "SPEEDUP" << setw(14) << "GPUTOTAL, ms" << setw(14) << "TOTALSPEEDUP"
         << "DESCRIPTION\n";
#else
    cout << TAB << setw(10) << "CPU, ms\n";
#endif
    cout << resetiosflags(ios_base::left);
}

void TestSystem::writeHeading()
{
    if (!record_)
    {
#ifdef USE_OPENCL
        recordname_ += "_OCL.csv";
#else
        recordname_ += "_CPU.csv";
#endif
        record_ = fopen(recordname_.c_str(), "w");
    }

#ifdef USE_OPENCL
    fprintf(record_, "NAME,DESCRIPTION,CPU (ms),GPU (ms),SPEEDUP,GPUTOTAL (ms),TOTALSPEEDUP,GPU Min (ms),GPU Max (ms), Standard deviation (ms)\n");
#else
    fprintf(record_, "NAME,DESCRIPTION,CPU (ms)\n");
#endif
    fflush(record_);
}

void TestSystem::printSummary()
{
    cout << setiosflags(ios_base::fixed);
    cout << "\naverage GPU speedup: x"
         << setprecision(3) << speedup_total_ / std::max(1, num_subtests_called_)
         << endl;
    cout << "\nGPU exceeded: "
         << setprecision(3) << speedup_faster_count_
         << "\nGPU passed: "
         << setprecision(3) << speedup_equal_count_
         << "\nGPU failed: "
         << setprecision(3) << speedup_slower_count_
         << endl;
    cout << "\nGPU exceeded rate: "
         << setprecision(3) << (float)speedup_faster_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPU passed rate: "
         << setprecision(3) << (float)speedup_equal_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPU failed rate: "
         << setprecision(3) << (float)speedup_slower_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << endl;
    cout << "\naverage GPUTOTAL speedup: x"
         << setprecision(3) << speedup_full_total_ / std::max(1, num_subtests_called_)
         << endl;
    cout << "\nGPUTOTAL exceeded: "
         << setprecision(3) << speedup_full_faster_count_
         << "\nGPUTOTAL passed: "
         << setprecision(3) << speedup_full_equal_count_
         << "\nGPUTOTAL failed: "
         << setprecision(3) << speedup_full_slower_count_
         << endl;
    cout << "\nGPUTOTAL exceeded rate: "
         << setprecision(3) << (float)speedup_full_faster_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPUTOTAL passed rate: "
         << setprecision(3) << (float)speedup_full_equal_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPUTOTAL failed rate: "
         << setprecision(3) << (float)speedup_full_slower_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << endl;
    cout << resetiosflags(ios_base::fixed);
}


void TestSystem::printMetrics(double cpu_time, double gpu_time, double gpu_full_time, double speedup, double fullspeedup)
{
    cout << TAB << setiosflags(ios_base::left);
    stringstream stream;

    stream << cpu_time;
    cout << setw(10) << stream.str();
#ifdef USE_OPENCL
    stream.str("");
    stream << gpu_time;
    cout << setw(10) << stream.str();

    stream.str("");
    stream << "x" << setprecision(3) << speedup;
    cout << setw(14) << stream.str();

    stream.str("");
    stream << gpu_full_time;
    cout << setw(14) << stream.str();

    stream.str("");
    stream << "x" << setprecision(3) << fullspeedup;
    cout << setw(14) << stream.str();
#endif
    cout << cur_subtest_description_.str();
    cout << resetiosflags(ios_base::left) << endl;
}

void TestSystem::writeMetrics(double cpu_time, double gpu_time, double gpu_full_time, double speedup, double fullspeedup, double gpu_min, double gpu_max, double std_dev)
{
    if (!record_)
    {
        recordname_ += ".csv";
        record_ = fopen(recordname_.c_str(), "w");
    }

#ifdef USE_OPENCL
    fprintf(record_, "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", itname_changed_ ? itname_.c_str() : "",
            cur_subtest_description_.str().c_str(),
            cpu_time, gpu_time, speedup, gpu_full_time, fullspeedup,
            gpu_min, gpu_max, std_dev);
#else
    fprintf(record_, "%s,%s,%.3f\n",
            itname_changed_ ? itname_.c_str() : "", cur_subtest_description_.str().c_str(), cpu_time);
#endif

    if (itname_changed_)
    {
        itname_changed_ = false;
    }

    fflush(record_);
}

void TestSystem::writeSummary()
{
    if (!record_)
    {
        recordname_ += ".csv";
        record_ = fopen(recordname_.c_str(), "w");
    }

    fprintf(record_, "\nAverage GPU speedup: %.3f\n"
            "exceeded: %d (%.3f%%)\n"
            "passed: %d (%.3f%%)\n"
            "failed: %d (%.3f%%)\n"
            "\nAverage GPUTOTAL speedup: %.3f\n"
            "exceeded: %d (%.3f%%)\n"
            "passed: %d (%.3f%%)\n"
            "failed: %d (%.3f%%)\n",
            speedup_total_ / std::max(1, num_subtests_called_),
            speedup_faster_count_, (float)speedup_faster_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_equal_count_, (float)speedup_equal_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_slower_count_, (float)speedup_slower_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_total_ / std::max(1, num_subtests_called_),
            speedup_full_faster_count_, (float)speedup_full_faster_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_equal_count_, (float)speedup_full_equal_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_slower_count_, (float)speedup_full_slower_count_ / std::max(1, num_subtests_called_) * 100
           );
    fflush(record_);
}

void TestSystem::printError(const std::string &msg)
{
    cout << TAB << "[error: " << msg << "] " << cur_subtest_description_.str() << endl;
}

void gen(Mat &mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


string abspath(const string &relpath)
{
    return TestSystem::instance().workingDir() + relpath;
}


int CV_CDECL cvErrorCallback(int /*status*/, const char * /*func_name*/,
                             const char *err_msg, const char * /*file_name*/,
                             int /*line*/, void * /*userdata*/)
{
    TestSystem::instance().printError(err_msg);
    return 0;
}

/////////// matchTemplate ////////////////////////
//void InitMatchTemplate()
//{
//	Mat src; gen(src, 500, 500, CV_32F, 0, 1);
//	Mat templ; gen(templ, 500, 500, CV_32F, 0, 1);
//#ifdef USE_OPENCL
//	ocl::oclMat d_src(src), d_templ(templ), d_dst;
//	ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
//#endif
//}
TEST(matchTemplate)
{
    //InitMatchTemplate();

    Mat src, templ, dst;
    int templ_size = 5;


    for (int size = 1000; size <= 4000; size *= 2)
    {
        int all_type[] = {CV_32FC1, CV_32FC4};
        std::string type_name[] = {"CV_32FC1", "CV_32FC4"};

        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            for(templ_size = 5; templ_size <=5; templ_size *= 5)
            {
                gen(src, size, size, all_type[j], 0, 1);

                SUBTEST << src.cols << 'x' << src.rows << "; " << type_name[j] << "; templ " << templ_size << 'x' << templ_size << "; CCORR";

                gen(templ, templ_size, templ_size, all_type[j], 0, 1);

                matchTemplate(src, templ, dst, CV_TM_CCORR);

                CPU_ON;
                matchTemplate(src, templ, dst, CV_TM_CCORR);
                CPU_OFF;

#ifdef USE_OPENCL
                ocl::oclMat d_src(src), d_templ, d_dst;

                d_templ.upload(templ);

                WARMUP_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
                WARMUP_OFF;

                GPU_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
                GPU_OFF;

                GPU_FULL_ON;
                d_src.upload(src);
                d_templ.upload(templ);
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
                d_dst.download(dst);
                GPU_FULL_OFF;
#endif
            }
        }

        int all_type_8U[] = {CV_8UC1};
        std::string type_name_8U[] = {"CV_8UC1"};

        for (size_t j = 0; j < sizeof(all_type_8U) / sizeof(int); j++)
        {
            for(templ_size = 5; templ_size < 200; templ_size *= 5)
            {
                SUBTEST << src.cols << 'x' << src.rows << "; " << type_name_8U[j] << "; templ " << templ_size << 'x' << templ_size << "; CCORR_NORMED";

                gen(src, size, size, all_type_8U[j], 0, 255);

                gen(templ, templ_size, templ_size, all_type_8U[j], 0, 255);

                matchTemplate(src, templ, dst, CV_TM_CCORR_NORMED);

                CPU_ON;
                matchTemplate(src, templ, dst, CV_TM_CCORR_NORMED);
                CPU_OFF;

#ifdef USE_OPENCL
                ocl::oclMat d_src(src);
                ocl::oclMat d_templ(templ), d_dst;

                WARMUP_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR_NORMED);
                WARMUP_OFF;

                GPU_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR_NORMED);
                GPU_OFF;

                GPU_FULL_ON;
                d_src.upload(src);
                d_templ.upload(templ);
                ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR_NORMED);
                d_dst.download(dst);
                GPU_FULL_OFF;
#endif
            }
        }
    }
}

///////////// PyrLKOpticalFlow ////////////////////////
TEST(PyrLKOpticalFlow)
{
    std::string images1[] = {"rubberwhale1.png", "aloeL.jpg"};
    std::string images2[] = {"rubberwhale2.png", "aloeR.jpg"};

    for (size_t i = 0; i < sizeof(images1) / sizeof(std::string); i++)
    {
        Mat frame0 = imread(abspath(images1[i]), i == 0 ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (frame0.empty())
        {
            std::string errstr = "can't open " + images1[i];
            throw runtime_error(errstr);
        }

        Mat frame1 = imread(abspath(images2[i]), i == 0 ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (frame1.empty())
        {
            std::string errstr = "can't open " + images2[i];
            throw runtime_error(errstr);
        }

        Mat gray_frame;

        if (i == 0)
        {
            cvtColor(frame0, gray_frame, COLOR_BGR2GRAY);
        }

        for (int points = 1000; points <= 4000; points *= 2)
        {
            if (i == 0)
                SUBTEST << frame0.cols << "x" << frame0.rows << "; color; " << points << " points";
            else
                SUBTEST << frame0.cols << "x" << frame0.rows << "; gray; " << points << " points";
            Mat nextPts_cpu;
            Mat status_cpu;

            vector<Point2f> pts;
            goodFeaturesToTrack(i == 0 ? gray_frame : frame0, pts, points, 0.01, 0.0);

            vector<Point2f> nextPts;
            vector<unsigned char> status;

            vector<float> err;

            calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);

            CPU_ON;
            calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);
            CPU_OFF;

#ifdef USE_OPENCL
            ocl::PyrLKOpticalFlow d_pyrLK;

            ocl::oclMat d_frame0(frame0);
            ocl::oclMat d_frame1(frame1);

            ocl::oclMat d_pts;
            Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void *)&pts[0]);
            d_pts.upload(pts_mat);

            ocl::oclMat d_nextPts;
            ocl::oclMat d_status;
            ocl::oclMat d_err;

            WARMUP_ON;
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);
            WARMUP_OFF;

            GPU_ON;
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);
            GPU_OFF;

            GPU_FULL_ON;
            d_frame0.upload(frame0);
            d_frame1.upload(frame1);
            d_pts.upload(pts_mat);
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status, &d_err);

            if (!d_nextPts.empty())
            {
                d_nextPts.download(nextPts_cpu);
            }

            if (!d_status.empty())
            {
                d_status.download(status_cpu);
            }

            GPU_FULL_OFF;
#endif
        }

    }
}


///////////// pyrDown //////////////////////
TEST(pyrDown)
{
    Mat src, dst;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            pyrDown(src, dst);

            CPU_ON;
            pyrDown(src, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            ocl::oclMat d_src(src);
            ocl::oclMat d_dst;

            WARMUP_ON;
            ocl::pyrDown(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::pyrDown(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::pyrDown(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }
    }
}

///////////// pyrUp ////////////////////////
TEST(pyrUp)
{
    Mat src, dst;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 500; size <= 2000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            pyrUp(src, dst);

            CPU_ON;
            pyrUp(src, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            ocl::oclMat d_src(src);
            ocl::oclMat d_dst;

            WARMUP_ON;
            ocl::pyrUp(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::pyrUp(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::pyrUp(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }
    }
}

///////////// Canny ////////////////////////
TEST(Canny)
{
    Mat img = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);

    if (img.empty())
    {
        throw runtime_error("can't open aloeL.jpg");
    }

    SUBTEST << img.cols << 'x' << img.rows << "; aloeL.jpg" << "; edges" << "; CV_8UC1";

    Mat edges(img.size(), CV_8UC1);

    CPU_ON;
    Canny(img, edges, 50.0, 100.0);
    CPU_OFF;

#ifdef USE_OPENCL
    ocl::oclMat d_img(img);
    ocl::oclMat d_edges;
    ocl::CannyBuf d_buf;

    WARMUP_ON;
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    WARMUP_OFF;

    GPU_ON;
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    GPU_OFF;

    GPU_FULL_ON;
    d_img.upload(img);
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    d_edges.download(edges);
    GPU_FULL_OFF;
#endif
}

///////////// Haar ////////////////////////
#ifdef USE_OPENCL
namespace cv
{
namespace ocl
{

struct getRect
{
    Rect operator()(const CvAvgComp &e) const
    {
        return e.rect;
    }
};

class CascadeClassifier_GPU : public OclCascadeClassifier
{
public:
    void detectMultiScale(oclMat &image,
                          CV_OUT std::vector<cv::Rect>& faces,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size())
    {
        (void)maxSize;
        MemStorage storage(cvCreateMemStorage(0));
        //CvMat img=image;
        CvSeq *objs = oclHaarDetectObjects(image, storage, scaleFactor, minNeighbors, flags, minSize);
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(objs).copyTo(vecAvgComp);
        faces.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
    }

};

}
}
#endif
TEST(Haar)
{
    Mat img = imread(abspath("basketball1.png"), CV_LOAD_IMAGE_GRAYSCALE);

    if (img.empty())
    {
        throw runtime_error("can't open basketball1.png");
    }

    CascadeClassifier faceCascadeCPU;

    if (!faceCascadeCPU.load(abspath("facedetect.xml")))
    {
        throw runtime_error("can't load facedetect.xml");
    }

    vector<Rect> faces;

    SUBTEST << img.cols << "x" << img.rows << "; scale image";
    CPU_ON;
    faceCascadeCPU.detectMultiScale(img, faces,
                                    1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    CPU_OFF;

#ifdef USE_OPENCL
    ocl::CascadeClassifier_GPU faceCascade;

    if (!faceCascade.load(abspath("facedetect.xml")))
    {
        throw runtime_error("can't load facedetect.xml");
    }

    ocl::oclMat d_img(img);

    faces.clear();

    WARMUP_ON;
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    WARMUP_OFF;

    faces.clear();

    GPU_ON;
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    GPU_OFF;

    GPU_FULL_ON;
    d_img.upload(img);
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    GPU_FULL_OFF;
#endif
}

///////////// blend ////////////////////////
template <typename T>
void blendLinearGold(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &weights1, const cv::Mat &weights2, cv::Mat &result_gold)
{
    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float *weights1_row = weights1.ptr<float>(y);
        const float *weights2_row = weights2.ptr<float>(y);
        const T *img1_row = img1.ptr<T>(y);
        const T *img2_row = img2.ptr<T>(y);
        T *result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < img1.cols * cn; ++x)
        {
            float w1 = weights1_row[x / cn];
            float w2 = weights2_row[x / cn];
            result_gold_row[x] = static_cast<T>((img1_row[x] * w1 + img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}
TEST(blend)
{
    Mat src1, src2, weights1, weights2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_weights1, d_weights2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " and CV_32FC1";

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(weights1, size, size, CV_32FC1, 0, 1);
            gen(weights2, size, size, CV_32FC1, 0, 1);

            blendLinearGold<uchar>(src1, src2, weights1, weights2, dst);

            CPU_ON;
            blendLinearGold<uchar>(src1, src2, weights1, weights2, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);
            d_weights1.upload(weights1);
            d_weights2.upload(weights2);

            WARMUP_ON;
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            d_weights1.upload(weights1);
            d_weights2.upload(weights2);
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }
    }
}
///////////// columnSum////////////////////////
TEST(columnSum)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; CV_32FC1";

        gen(src, size, size, CV_32FC1, 0, 256);

        CPU_ON;
        dst.create(src.size(), src.type());

        for (int i = 1; i < src.rows; ++i)
        {
            for (int j = 0; j < src.cols; ++j)
            {
                dst.at<float>(i, j) = src.at<float>(i, j) += src.at<float>(i - 1, j);
            }
        }

        CPU_OFF;

#ifdef USE_OPENCL
        d_src.upload(src);
        WARMUP_ON;
        ocl::columnSum(d_src, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::columnSum(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::columnSum(d_src, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}

///////////// HOG////////////////////////
TEST(HOG)
{
    Mat src = imread(abspath("road.png"), cv::IMREAD_GRAYSCALE);

    if (src.empty())
    {
        throw runtime_error("can't open road.png");
    }


    cv::HOGDescriptor hog;
    hog.setSVMDetector(hog.getDefaultPeopleDetector());
    std::vector<cv::Rect> found_locations;

    SUBTEST << 768 << 'x' << 576 << "; road.png";

    hog.detectMultiScale(src, found_locations);

    CPU_ON;
    hog.detectMultiScale(src, found_locations);
    CPU_OFF;

#ifdef USE_OPENCL
    cv::ocl::HOGDescriptor ocl_hog;
    ocl_hog.setSVMDetector(ocl_hog.getDefaultPeopleDetector());
    ocl::oclMat d_src;
    d_src.upload(src);

    WARMUP_ON;
    ocl_hog.detectMultiScale(d_src, found_locations);
    WARMUP_OFF;

    GPU_ON;
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_OFF;

    GPU_FULL_ON;
    d_src.upload(src);
    ocl_hog.detectMultiScale(d_src, found_locations);
    GPU_FULL_OFF;
#endif
}

///////////// SURF ////////////////////////

TEST(SURF)
{
    Mat keypoints_cpu;
    Mat descriptors_cpu;

    Mat src = imread(abspath("aloeL.jpg"), CV_LOAD_IMAGE_GRAYSCALE);

    if (src.empty())
    {
        throw runtime_error("can't open aloeL.jpg");
    }

    SUBTEST << src.cols << "x" << src.rows << "; aloeL.jpg";
    SURF surf;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    surf(src, Mat(), keypoints, descriptors);

    CPU_ON;
    keypoints.clear();
    surf(src, Mat(), keypoints, descriptors);
    CPU_OFF;

#ifdef USE_OPENCL
    ocl::SURF_OCL d_surf;
    ocl::oclMat d_src(src);
    ocl::oclMat d_keypoints;
    ocl::oclMat d_descriptors;

    WARMUP_ON;
    d_surf(d_src, ocl::oclMat(), d_keypoints, d_descriptors);
    WARMUP_OFF;

    GPU_ON;
    d_surf(d_src, ocl::oclMat(), d_keypoints, d_descriptors);
    GPU_OFF;

    GPU_FULL_ON;
    d_src.upload(src);
    d_surf(d_src, ocl::oclMat(), d_keypoints, d_descriptors);

    if (!d_keypoints.empty())
    {
        d_keypoints.download(keypoints_cpu);
    }

    if (!d_descriptors.empty())
    {
        d_descriptors.download(descriptors_cpu);
    }

    GPU_FULL_OFF;
#endif
}
//////////////////// BruteForceMatch /////////////////
TEST(BruteForceMatcher)
{
    Mat trainIdx_cpu;
    Mat distance_cpu;
    Mat allDist_cpu;
    Mat nMatches_cpu;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        // Init CPU matcher
        int desc_len = 64;

        BFMatcher matcher(NORM_L2);

        Mat query;
        gen(query, size, desc_len, CV_32F, 0, 1);

        Mat train;
        gen(train, size, desc_len, CV_32F, 0, 1);
        // Output
        vector< vector<DMatch> > matches(2);
#ifdef USE_OPENCL
        // Init GPU matcher
        ocl::BruteForceMatcher_OCL_base d_matcher(ocl::BruteForceMatcher_OCL_base::L2Dist);

        ocl::oclMat d_query(query);
        ocl::oclMat d_train(train);

        ocl::oclMat d_trainIdx, d_distance, d_allDist, d_nMatches;
#endif
        SUBTEST << size << "; match";

        matcher.match(query, train, matches[0]);

        CPU_ON;
        matcher.match(query, train, matches[0]);
        CPU_OFF;

#ifdef USE_OPENCL
        WARMUP_ON;
        d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.match(d_query, d_train, matches[0]);
        GPU_FULL_OFF;
#endif

        SUBTEST << size << "; knnMatch";

        matcher.knnMatch(query, train, matches, 2);

        CPU_ON;
        matcher.knnMatch(query, train, matches, 2);
        CPU_OFF;

#ifdef USE_OPENCL
        WARMUP_ON;
        d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.knnMatch(d_query, d_train, matches, 2);
        GPU_FULL_OFF;
#endif
        SUBTEST << size << "; radiusMatch";

        float max_distance = 2.0f;

        matcher.radiusMatch(query, train, matches, max_distance);

        CPU_ON;
        matcher.radiusMatch(query, train, matches, max_distance);
        CPU_OFF;

#ifdef USE_OPENCL
        d_trainIdx.release();

        WARMUP_ON;
        d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.radiusMatch(d_query, d_train, matches, max_distance);
        GPU_FULL_OFF;
#endif
    }
}
///////////// Lut ////////////////////////
TEST(lut)
{
    Mat src, lut, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_lut, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC3};
    std::string type_name[] = {"CV_8UC1", "CV_8UC3"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);
            gen(lut, 1, 256, CV_8UC1, 0, 1);
            gen(dst, size, size, all_type[j], 0, 256);

            LUT(src, lut, dst);

            CPU_ON;
            LUT(src, lut, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src.upload(src);
            d_lut.upload(lut);

            WARMUP_ON;
            ocl::LUT(d_src, d_lut, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::LUT(d_src, d_lut, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_lut.upload(lut);
            ocl::LUT(d_src, d_lut, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Exp ////////////////////////
TEST(Exp)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; CV_32FC1";

        gen(src, size, size, CV_32FC1, 0, 256);
        gen(dst, size, size, CV_32FC1, 0, 256);

        exp(src, dst);

        CPU_ON;
        exp(src, dst);
        CPU_OFF;
#ifdef USE_OPENCL
        d_src.upload(src);

        WARMUP_ON;
        ocl::exp(d_src, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::exp(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::exp(d_src, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}

///////////// LOG ////////////////////////
TEST(Log)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; 32F";

        gen(src, size, size, CV_32F, 1, 10);

        log(src, dst);

        CPU_ON;
        log(src, dst);
        CPU_OFF;
#ifdef USE_OPENCL
        d_src.upload(src);

        WARMUP_ON;
        ocl::log(d_src, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::log(d_src, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::log(d_src, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}

///////////// Add ////////////////////////

TEST(Add)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 1);
            gen(src2, size, size, all_type[j], 0, 1);

            add(src1, src2, dst);

            CPU_ON;
            add(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::add(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::add(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::add(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Mul ////////////////////////
TEST(Mul)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            multiply(src1, src2, dst);

            CPU_ON;
            multiply(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::multiply(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::multiply(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::multiply(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Div ////////////////////////
TEST(Div)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            divide(src1, src2, dst);

            CPU_ON;
            divide(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::divide(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::divide(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::divide(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Absdiff ////////////////////////
TEST(Absdiff)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            absdiff(src1, src2, dst);

            CPU_ON;
            absdiff(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::absdiff(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::absdiff(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::absdiff(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// CartToPolar ////////////////////////
TEST(CartToPolar)
{
    Mat src1, src2, dst, dst1;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst, d_dst1;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            gen(dst1, size, size, all_type[j], 0, 256);


            cartToPolar(src1, src2, dst, dst1, 1);

            CPU_ON;
            cartToPolar(src1, src2, dst, dst1, 1);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            WARMUP_OFF;

            GPU_ON;
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::cartToPolar(d_src1, d_src2, d_dst, d_dst1, 1);
            d_dst.download(dst);
            d_dst1.download(dst1);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// PolarToCart ////////////////////////
TEST(PolarToCart)
{
    Mat src1, src2, dst, dst1;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst, d_dst1;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            gen(dst1, size, size, all_type[j], 0, 256);


            polarToCart(src1, src2, dst, dst1, 1);

            CPU_ON;
            polarToCart(src1, src2, dst, dst1, 1);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            WARMUP_OFF;

            GPU_ON;
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::polarToCart(d_src1, d_src2, d_dst, d_dst1, 1);
            d_dst.download(dst);
            d_dst1.download(dst1);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Magnitude ////////////////////////
TEST(magnitude)
{
    Mat x, y, mag;
#ifdef USE_OPENCL
    ocl::oclMat d_x, d_y, d_mag;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(x, size, size, all_type[j], 0, 1);
            gen(y, size, size, all_type[j], 0, 1);

            magnitude(x, y, mag);

            CPU_ON;
            magnitude(x, y, mag);
            CPU_OFF;
#ifdef USE_OPENCL
            d_x.upload(x);
            d_y.upload(y);

            WARMUP_ON;
            ocl::magnitude(d_x, d_y, d_mag);
            WARMUP_OFF;

            GPU_ON;
            ocl::magnitude(d_x, d_y, d_mag);
            GPU_OFF;

            GPU_FULL_ON;
            d_x.upload(x);
            d_y.upload(y);
            ocl::magnitude(d_x, d_y, d_mag);
            d_mag.download(mag);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Transpose ////////////////////////
TEST(Transpose)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            transpose(src, dst);

            CPU_ON;
            transpose(src, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::transpose(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::transpose(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::transpose(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Flip ////////////////////////
TEST(Flip)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; FLIP_BOTH";

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);

            flip(src, dst, 0);

            CPU_ON;
            flip(src, dst, 0);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::flip(d_src, d_dst, 0);
            WARMUP_OFF;

            GPU_ON;
            ocl::flip(d_src, d_dst, 0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::flip(d_src, d_dst, 0);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// minMax ////////////////////////
TEST(minMax)
{
    Mat src;
#ifdef USE_OPENCL
    ocl::oclMat d_src;
#endif
    double min_val, max_val;
    Point min_loc, max_loc;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);

            CPU_ON;
            minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::minMax(d_src, &min_val, &max_val);
            WARMUP_OFF;

            GPU_ON;
            ocl::minMax(d_src, &min_val, &max_val);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::minMax(d_src, &min_val, &max_val);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// minMaxLoc ////////////////////////
TEST(minMaxLoc)
{
    Mat src;
#ifdef USE_OPENCL
    ocl::oclMat d_src;
#endif
    double min_val, max_val;
    Point min_loc, max_loc;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 1);

            CPU_ON;
            minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
            WARMUP_OFF;

            GPU_ON;
            ocl::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::minMaxLoc(d_src, &min_val, &max_val, &min_loc, &max_loc);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Sum ////////////////////////
TEST(Sum)
{
    Mat src;
    Scalar cpures, gpures;
#ifdef USE_OPENCL
    ocl::oclMat d_src;
#endif
    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            cpures = sum(src);

            CPU_ON;
            cpures = sum(src);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            gpures = ocl::sum(d_src);
            WARMUP_OFF;

            GPU_ON;
            gpures = ocl::sum(d_src);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            gpures = ocl::sum(d_src);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// countNonZero ////////////////////////
TEST(countNonZero)
{
    Mat src;
#ifdef USE_OPENCL
    ocl::oclMat d_src;
#endif
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            countNonZero(src);

            CPU_ON;
            countNonZero(src);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::countNonZero(d_src);
            WARMUP_OFF;

            GPU_ON;
            ocl::countNonZero(d_src);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::countNonZero(d_src);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Phase ////////////////////////
TEST(Phase)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            phase(src1, src2, dst, 1);

            CPU_ON;
            phase(src1, src2, dst, 1);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::phase(d_src1, d_src2, d_dst, 1);
            WARMUP_OFF;

            GPU_ON;
            ocl::phase(d_src1, d_src2, d_dst, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::phase(d_src1, d_src2, d_dst, 1);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// bitwise_and////////////////////////
TEST(bitwise_and)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_and(src1, src2, dst);

            CPU_ON;
            bitwise_and(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::bitwise_and(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// bitwise_or////////////////////////
TEST(bitwise_or)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_or(src1, src2, dst);

            CPU_ON;
            bitwise_or(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::bitwise_or(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::bitwise_or(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::bitwise_or(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// bitwise_xor////////////////////////
TEST(bitwise_xor)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_xor(src1, src2, dst);

            CPU_ON;
            bitwise_xor(src1, src2, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::bitwise_xor(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::bitwise_xor(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::bitwise_xor(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// bitwise_not////////////////////////
TEST(bitwise_not)
{
    Mat src1, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32SC1};
    std::string type_name[] = {"CV_8UC1", "CV_32SC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            bitwise_not(src1, dst);

            CPU_ON;
            bitwise_not(src1, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);

            WARMUP_ON;
            ocl::bitwise_not(d_src1, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::bitwise_not(d_src1, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::bitwise_not(d_src1, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// compare////////////////////////
TEST(compare)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int CMP_EQ = 0;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            compare(src1, src2, dst, CMP_EQ);

            CPU_ON;
            compare(src1, src2, dst, CMP_EQ);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            WARMUP_OFF;

            GPU_ON;
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::compare(d_src1, d_src2, d_dst, CMP_EQ);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// pow ////////////////////////
TEST(pow)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 100);
            gen(dst, size, size, all_type[j], 0, 100);

            pow(src, -2.0, dst);

            CPU_ON;
            pow(src, -2.0, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);
            d_dst.upload(dst);

            WARMUP_ON;
            ocl::pow(d_src, -2.0, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::pow(d_src, -2.0, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::pow(d_src, -2.0, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// MagnitudeSqr////////////////////////
TEST(MagnitudeSqr)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    int all_type[] = {CV_32FC1};
    std::string type_name[] = {"CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t t = 0; t < sizeof(all_type) / sizeof(int); t++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[t];

            gen(src1, size, size, all_type[t], 0, 256);
            gen(src2, size, size, all_type[t], 0, 256);
            gen(dst, size, size, all_type[t], 0, 256);


            for (int i = 0; i < src1.rows; ++i)

                for (int j = 0; j < src1.cols; ++j)
                {
                    float val1 = src1.at<float>(i, j);
                    float val2 = src2.at<float>(i, j);

                    ((float *)(dst.data))[i * dst.step / 4 + j] = val1 * val1 + val2 * val2;

                }

            CPU_ON;

            for (int i = 0; i < src1.rows; ++i)
                for (int j = 0; j < src1.cols; ++j)
                {
                    float val1 = src1.at<float>(i, j);
                    float val2 = src2.at<float>(i, j);

                    ((float *)(dst.data))[i * dst.step / 4 + j] = val1 * val1 + val2 * val2;

                }

            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::magnitudeSqr(d_src1, d_src2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::magnitudeSqr(d_src1, d_src2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::magnitudeSqr(d_src1, d_src2, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// AddWeighted////////////////////////
TEST(AddWeighted)
{
    Mat src1, src2, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_dst;
#endif
    double alpha = 2.0, beta = 1.0, gama = 3.0;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            addWeighted(src1, alpha, src2, beta, gama, dst);

            CPU_ON;
            addWeighted(src1, alpha, src2, beta, gama, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);
            d_src2.upload(src2);

            WARMUP_ON;
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            ocl::addWeighted(d_src1, alpha, d_src2, beta, gama, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Blur////////////////////////
TEST(Blur)
{
    Mat src1, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_dst;
#endif
    Size ksize = Size(3, 3);
    int bordertype = BORDER_CONSTANT;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            blur(src1, dst, ksize, Point(-1, -1), bordertype);

            CPU_ON;
            blur(src1, dst, ksize, Point(-1, -1), bordertype);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);

            WARMUP_ON;
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            WARMUP_OFF;

            GPU_ON;
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::blur(d_src1, d_dst, ksize, Point(-1, -1), bordertype);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Laplacian////////////////////////
TEST(Laplacian)
{
    Mat src1, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_dst;
#endif
    int ksize = 3;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src1, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);


            Laplacian(src1, dst, -1, ksize, 1);

            CPU_ON;
            Laplacian(src1, dst, -1, ksize, 1);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src1.upload(src1);

            WARMUP_ON;
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            WARMUP_OFF;

            GPU_ON;
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            ocl::Laplacian(d_src1, d_dst, -1, ksize, 1);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Erode ////////////////////
TEST(Erode)
{
    Mat src, dst, ker;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4", "CV_32FC1", "CV_32FC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], Scalar::all(0), Scalar::all(256));
            ker = getStructuringElement(MORPH_RECT, Size(3, 3));

            erode(src, dst, ker);

            CPU_ON;
            erode(src, dst, ker);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::erode(d_src, d_dst, ker);
            WARMUP_OFF;

            GPU_ON;
            ocl::erode(d_src, d_dst, ker);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::erode(d_src, d_dst, ker);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Sobel ////////////////////////
TEST(Sobel)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int dx = 1;
    int dy = 1;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            Sobel(src, dst, -1, dx, dy);

            CPU_ON;
            Sobel(src, dst, -1, dx, dy);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            WARMUP_OFF;

            GPU_ON;
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::Sobel(d_src, d_dst, -1, dx, dy);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Scharr ////////////////////////
TEST(Scharr)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int dx = 1;
    int dy = 0;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            Scharr(src, dst, -1, dx, dy);

            CPU_ON;
            Scharr(src, dst, -1, dx, dy);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            WARMUP_OFF;

            GPU_ON;
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::Scharr(d_src, d_dst, -1, dx, dy);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// GaussianBlur ////////////////////////
TEST(GaussianBlur)
{
    Mat src, dst;
    int all_type[] = {CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4", "CV_32FC1", "CV_32FC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            GaussianBlur(src, dst, Size(9, 9), 0);

            CPU_ON;
            GaussianBlur(src, dst, Size(9, 9), 0);
            CPU_OFF;
#ifdef USE_OPENCL
            ocl::oclMat d_src(src);
            ocl::oclMat d_dst(src.size(), src.type());
            ocl::oclMat d_buf;

            WARMUP_ON;
            ocl::GaussianBlur(d_src, d_dst, Size(9, 9), 0);
            WARMUP_OFF;

            GPU_ON;
            ocl::GaussianBlur(d_src, d_dst, Size(9, 9), 0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::GaussianBlur(d_src, d_dst, Size(9, 9), 0);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// equalizeHist ////////////////////////
TEST(equalizeHist)
{
    Mat src, dst;
    int all_type[] = {CV_8UC1};
    std::string type_name[] = {"CV_8UC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            equalizeHist(src, dst);

            CPU_ON;
            equalizeHist(src, dst);
            CPU_OFF;
#ifdef USE_OPENCL
            ocl::oclMat d_src(src);
            ocl::oclMat d_dst;
            ocl::oclMat d_hist;
            ocl::oclMat d_buf;

            WARMUP_ON;
            ocl::equalizeHist(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::equalizeHist(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::equalizeHist(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
/////////// CopyMakeBorder //////////////////////
TEST(CopyMakeBorder)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_dst;
#endif
    int bordertype = BORDER_CONSTANT;
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;


            gen(src, size, size, all_type[j], 0, 256);

            copyMakeBorder(src, dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));

            CPU_ON;
            copyMakeBorder(src, dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            CPU_OFF;
#ifdef USE_OPENCL
            ocl::oclMat d_src(src);

            WARMUP_ON;
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            WARMUP_OFF;

            GPU_ON;
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::copyMakeBorder(d_src, d_dst, 7, 5, 5, 7, bordertype, cv::Scalar(1.0));
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// cornerMinEigenVal ////////////////////////
TEST(cornerMinEigenVal)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_dst;
#endif
    int blockSize = 7, apertureSize = 1 + 2 * (rand() % 4);
    int borderType = BORDER_REFLECT;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;


            gen(src, size, size, all_type[j], 0, 256);

            cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);

            CPU_ON;
            cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);
            CPU_OFF;
#ifdef USE_OPENCL
            ocl::oclMat d_src(src);

            WARMUP_ON;
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            WARMUP_OFF;

            GPU_ON;
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::cornerMinEigenVal(d_src, d_dst, blockSize, apertureSize, borderType);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// cornerHarris ////////////////////////
TEST(cornerHarris)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; BORDER_REFLECT";

            gen(src, size, size, all_type[j], 0, 1);

            cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT);

            CPU_ON;
            cornerHarris(src, dst, 5, 7, 0.1, BORDER_REFLECT);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            WARMUP_OFF;

            GPU_ON;
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::cornerHarris(d_src, d_dst, 5, 7, 0.1, BORDER_REFLECT);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }


    }
}
///////////// integral ////////////////////////
TEST(integral)
{
    Mat src, sum;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_sum, d_buf;
#endif
    int all_type[] = {CV_8UC1};
    std::string type_name[] = {"CV_8UC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j]  ;

            gen(src, size, size, all_type[j], 0, 256);

            integral(src, sum);

            CPU_ON;
            integral(src, sum);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::integral(d_src, d_sum);
            WARMUP_OFF;

            GPU_ON;
            ocl::integral(d_src, d_sum);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::integral(d_src, d_sum);
            d_sum.download(sum);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// WarpAffine ////////////////////////
TEST(WarpAffine)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    static const double coeffs[2][3] =
    {
        {cos(3.14 / 6), -sin(3.14 / 6), 100.0},
        {sin(3.14 / 6), cos(3.14 / 6), -100.0}
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);
    int interpolation = INTER_NEAREST;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};


    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            Size size1 = Size(size, size);

            warpAffine(src, dst, M, size1, interpolation);

            CPU_ON;
            warpAffine(src, dst, M, size1, interpolation);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            WARMUP_OFF;

            GPU_ON;
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::warpAffine(d_src, d_dst, M, size1, interpolation);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// WarpPerspective ////////////////////////
TEST(WarpPerspective)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    static const double coeffs[3][3] =
    {
        {cos(3.14 / 6), -sin(3.14 / 6), 100.0},
        {sin(3.14 / 6), cos(3.14 / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);
    int interpolation = INTER_NEAREST;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            gen(dst, size, size, all_type[j], 0, 256);
            Size size1 = Size(size, size);

            warpPerspective(src, dst, M, size1, interpolation);

            CPU_ON;
            warpPerspective(src, dst, M, size1, interpolation);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            WARMUP_OFF;

            GPU_ON;
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::warpPerspective(d_src, d_dst, M, size1, interpolation);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// resize ////////////////////////
TEST(resize)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; up";

            gen(src, size, size, all_type[j], 0, 256);

            resize(src, dst, Size(), 2.0, 2.0);

            CPU_ON;
            resize(src, dst, Size(), 2.0, 2.0);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            WARMUP_OFF;

            GPU_ON;
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::resize(d_src, d_dst, Size(), 2.0, 2.0);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; down";

            gen(src, size, size, all_type[j], 0, 256);

            resize(src, dst, Size(), 0.5, 0.5);

            CPU_ON;
            resize(src, dst, Size(), 0.5, 0.5);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            WARMUP_OFF;

            GPU_ON;
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::resize(d_src, d_dst, Size(), 0.5, 0.5);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// threshold////////////////////////
TEST(threshold)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; 8UC1; THRESH_BINARY";

        gen(src, size, size, CV_8U, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_BINARY);
        CPU_OFF;
#ifdef USE_OPENCL
        d_src.upload(src);

        WARMUP_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        WARMUP_OFF;

        GPU_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_BINARY);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; 32FC1; THRESH_TRUNC [NPP]";

        gen(src, size, size, CV_32FC1, 0, 100);

        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);

        CPU_ON;
        threshold(src, dst, 50.0, 0.0, THRESH_TRUNC);
        CPU_OFF;
#ifdef USE_OPENCL
        d_src.upload(src);

        WARMUP_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        WARMUP_OFF;

        GPU_ON;
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::threshold(d_src, d_dst, 50.0, 0.0, THRESH_TRUNC);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}
///////////// meanShiftFiltering////////////////////////
TEST(meanShiftFiltering)
{
    int sp = 10, sr = 10;

    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; 8UC3 vs 8UC4";

        gen(src, size, size, CV_8UC3, Scalar::all(0), Scalar::all(256));

        pyrMeanShiftFiltering(src, dst, sp, sr);

        CPU_ON;
        pyrMeanShiftFiltering(src, dst, sp, sr);
        CPU_OFF;
#ifdef USE_OPENCL
        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));

        d_src.upload(src);

        WARMUP_ON;
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr);
        WARMUP_OFF;

        GPU_ON;
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::meanShiftFiltering(d_src, d_dst, sp, sr);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}
///////////// meanShiftProc////////////////////////
COOR do_meanShift(int x0, int y0, uchar *sptr, uchar *dptr, int sstep, cv::Size size, int sp, int sr, int maxIter, float eps, int *tab)
{

    int isr2 = sr * sr;
    int c0, c1, c2, c3;
    int iter;
    uchar *ptr = NULL;
    uchar *pstart = NULL;
    int revx = 0, revy = 0;
    c0 = sptr[0];
    c1 = sptr[1];
    c2 = sptr[2];
    c3 = sptr[3];

    // iterate meanshift procedure
    for (iter = 0; iter < maxIter; iter++)
    {
        int count = 0;
        int s0 = 0, s1 = 0, s2 = 0, sx = 0, sy = 0;

        //mean shift: process pixels in window (p-sigmaSp)x(p+sigmaSp)
        int minx = x0 - sp;
        int miny = y0 - sp;
        int maxx = x0 + sp;
        int maxy = y0 + sp;

        //deal with the image boundary
        if (minx < 0)
        {
            minx = 0;
        }

        if (miny < 0)
        {
            miny = 0;
        }

        if (maxx >= size.width)
        {
            maxx = size.width - 1;
        }

        if (maxy >= size.height)
        {
            maxy = size.height - 1;
        }

        if (iter == 0)
        {
            pstart = sptr;
        }
        else
        {
            pstart = pstart + revy * sstep + (revx << 2); //point to the new position
        }

        ptr = pstart;
        ptr = ptr + (miny - y0) * sstep + ((minx - x0) << 2); //point to the start in the row

        for (int y = miny; y <= maxy; y++, ptr += sstep - ((maxx - minx + 1) << 2))
        {
            int rowCount = 0;
            int x = minx;
#if CV_ENABLE_UNROLLED

            for (; x + 4 <= maxx; x += 4, ptr += 16)
            {
                int t0, t1, t2;
                t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];

                if (tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x;
                    rowCount++;
                }

                t0 = ptr[4], t1 = ptr[5], t2 = ptr[6];

                if (tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 1;
                    rowCount++;
                }

                t0 = ptr[8], t1 = ptr[9], t2 = ptr[10];

                if (tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 2;
                    rowCount++;
                }

                t0 = ptr[12], t1 = ptr[13], t2 = ptr[14];

                if (tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x + 3;
                    rowCount++;
                }
            }

#endif

            for (; x <= maxx; x++, ptr += 4)
            {
                int t0 = ptr[0], t1 = ptr[1], t2 = ptr[2];

                if (tab[t0 - c0 + 255] + tab[t1 - c1 + 255] + tab[t2 - c2 + 255] <= isr2)
                {
                    s0 += t0;
                    s1 += t1;
                    s2 += t2;
                    sx += x;
                    rowCount++;
                }
            }

            if (rowCount == 0)
            {
                continue;
            }

            count += rowCount;
            sy += y * rowCount;
        }

        if (count == 0)
        {
            break;
        }

        int x1 = sx / count;
        int y1 = sy / count;
        s0 = s0 / count;
        s1 = s1 / count;
        s2 = s2 / count;

        bool stopFlag = (x0 == x1 && y0 == y1) || (abs(x1 - x0) + abs(y1 - y0) +
                        tab[s0 - c0 + 255] + tab[s1 - c1 + 255] + tab[s2 - c2 + 255] <= eps);

        //revise the pointer corresponding to the new (y0,x0)
        revx = x1 - x0;
        revy = y1 - y0;

        x0 = x1;
        y0 = y1;
        c0 = s0;
        c1 = s1;
        c2 = s2;

        if (stopFlag)
        {
            break;
        }
    } //for iter

    dptr[0] = (uchar)c0;
    dptr[1] = (uchar)c1;
    dptr[2] = (uchar)c2;
    dptr[3] = (uchar)c3;

    COOR coor;
    coor.x = static_cast<short>(x0);
    coor.y = static_cast<short>(y0);
    return coor;
}

void meanShiftProc_(const Mat &src_roi, Mat &dst_roi, Mat &dstCoor_roi, int sp, int sr, cv::TermCriteria crit)
{

    if (src_roi.empty())
    {
        CV_Error(CV_StsBadArg, "The input image is empty");
    }

    if (src_roi.depth() != CV_8U || src_roi.channels() != 4)
    {
        CV_Error(CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported");
    }

    CV_Assert((src_roi.cols == dst_roi.cols) && (src_roi.rows == dst_roi.rows) &&
              (src_roi.cols == dstCoor_roi.cols) && (src_roi.rows == dstCoor_roi.rows));
    CV_Assert(!(dstCoor_roi.step & 0x3));

    if (!(crit.type & cv::TermCriteria::MAX_ITER))
    {
        crit.maxCount = 5;
    }

    int maxIter = std::min(std::max(crit.maxCount, 1), 100);
    float eps;

    if (!(crit.type & cv::TermCriteria::EPS))
    {
        eps = 1.f;
    }

    eps = (float)std::max(crit.epsilon, 0.0);

    int tab[512];

    for (int i = 0; i < 512; i++)
    {
        tab[i] = (i - 255) * (i - 255);
    }

    uchar *sptr = src_roi.data;
    uchar *dptr = dst_roi.data;
    short *dCoorptr = (short *)dstCoor_roi.data;
    int sstep = (int)src_roi.step;
    int dstep = (int)dst_roi.step;
    int dCoorstep = (int)dstCoor_roi.step >> 1;
    cv::Size size = src_roi.size();

    for (int i = 0; i < size.height; i++, sptr += sstep - (size.width << 2),
            dptr += dstep - (size.width << 2), dCoorptr += dCoorstep - (size.width << 1))
    {
        for (int j = 0; j < size.width; j++, sptr += 4, dptr += 4, dCoorptr += 2)
        {
            *((COOR *)dCoorptr) = do_meanShift(j, i, sptr, dptr, sstep, size, sp, sr, maxIter, eps, tab);
        }
    }

}
TEST(meanShiftProc)
{
    Mat src, dst, dstCoor_roi;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst, d_dstCoor_roi;
#endif
    TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS, 5, 1);

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; 8UC4 and CV_16SC2 ";

        gen(src, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));
        gen(dst, size, size, CV_8UC4, Scalar::all(0), Scalar::all(256));
        gen(dstCoor_roi, size, size, CV_16SC2, Scalar::all(0), Scalar::all(256));

        meanShiftProc_(src, dst, dstCoor_roi, 5, 6, crit);

        CPU_ON;
        meanShiftProc_(src, dst, dstCoor_roi, 5, 6, crit);
        CPU_OFF;
#ifdef USE_OPENCL
        d_src.upload(src);

        WARMUP_ON;
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor_roi, 5, 6, crit);
        WARMUP_OFF;

        GPU_ON;
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor_roi, 5, 6, crit);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::meanShiftProc(d_src, d_dst, d_dstCoor_roi, 5, 6, crit);
        d_dst.download(dst);
        d_dstCoor_roi.download(dstCoor_roi);
        GPU_FULL_OFF;
#endif
    }
}
///////////// ConvertTo////////////////////////
TEST(ConvertTo)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " to 32FC1";

            gen(src, size, size, all_type[j], 0, 256);
            //gen(dst, size, size, all_type[j], 0, 256);

            //d_dst.upload(dst);

            src.convertTo(dst, CV_32FC1);

            CPU_ON;
            src.convertTo(dst, CV_32FC1);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            d_src.convertTo(d_dst, CV_32FC1);
            WARMUP_OFF;

            GPU_ON;
            d_src.convertTo(d_dst, CV_32FC1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.convertTo(d_dst, CV_32FC1);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// copyTo////////////////////////
TEST(copyTo)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            //gen(dst, size, size, all_type[j], 0, 256);

            //d_dst.upload(dst);

            src.copyTo(dst);

            CPU_ON;
            src.copyTo(dst);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            d_src.copyTo(d_dst);
            WARMUP_OFF;

            GPU_ON;
            d_src.copyTo(d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.copyTo(d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// setTo////////////////////////
TEST(setTo)
{
    Mat src, dst;
    Scalar val(1, 2, 3, 4);
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            src.setTo(val);

            CPU_ON;
            src.setTo(val);
            CPU_OFF;
#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            d_src.setTo(val);
            WARMUP_OFF;

            GPU_ON;
            d_src.setTo(val);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.setTo(val);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// Merge////////////////////////
TEST(Merge)
{
    Mat dst;
#ifdef USE_OPENCL
    ocl::oclMat d_dst;
#endif
    int channels = 4;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;
            Size size1 = Size(size, size);
            std::vector<Mat> src(channels);

            for (int i = 0; i < channels; ++i)
            {
                src[i] = Mat(size1, all_type[j], cv::Scalar::all(i));
            }

            merge(src, dst);

            CPU_ON;
            merge(src, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            std::vector<ocl::oclMat> d_src(channels);

            for (int i = 0; i < channels; ++i)
            {
                d_src[i] = ocl::oclMat(size1, all_type[j], cv::Scalar::all(i));
            }

            WARMUP_ON;
            ocl::merge(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::merge(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;

            for (int i = 0; i < channels; ++i)
            {
                d_src[i] = ocl::oclMat(size1, CV_8U, cv::Scalar::all(i));
            }

            ocl::merge(d_src, d_dst);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// Split////////////////////////
TEST(Split)
{
    //int channels = 4;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];
            Size size1 = Size(size, size);

            Mat src(size1, CV_MAKE_TYPE(all_type[j], 4), cv::Scalar(1, 2, 3, 4));

            std::vector<cv::Mat> dst;

            split(src, dst);

            CPU_ON;
            split(src, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            ocl::oclMat d_src(size1, CV_MAKE_TYPE(all_type[j], 4), cv::Scalar(1, 2, 3, 4));
            std::vector<cv::ocl::oclMat> d_dst;

            WARMUP_ON;
            ocl::split(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::split(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::split(d_src, d_dst);
            GPU_FULL_OFF;
#endif
        }

    }
}


///////////// norm////////////////////////
TEST(norm)
{
    Mat src, buf;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_buf;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size << "; CV_8UC1; NORM_INF";

        gen(src, size, size, CV_8UC1, Scalar::all(0), Scalar::all(1));
        gen(buf, size, size, CV_8UC1, Scalar::all(0), Scalar::all(1));

        norm(src, NORM_INF);

        CPU_ON;
        norm(src, NORM_INF);
        CPU_OFF;

#ifdef USE_OPENCL
        d_src.upload(src);
        d_buf.upload(buf);

        WARMUP_ON;
        ocl::norm(d_src, d_buf, NORM_INF);
        WARMUP_OFF;

        GPU_ON;
        ocl::norm(d_src, d_buf, NORM_INF);
        GPU_OFF;

        GPU_FULL_ON;
        d_src.upload(src);
        ocl::norm(d_src, d_buf, NORM_INF);
        GPU_FULL_OFF;
#endif
    }
}
///////////// remap////////////////////////
TEST(remap)
{
    Mat src, dst, xmap, ymap;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst, d_xmap, d_ymap;
#endif
    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    int interpolation = INTER_LINEAR;
    int borderMode = BORDER_CONSTANT;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t t = 0; t < sizeof(all_type) / sizeof(int); t++)
        {
            SUBTEST << size << 'x' << size << "; src " << type_name[t] << "; map CV_32FC1";

            gen(src, size, size, all_type[t], 0, 256);

            xmap.create(size, size, CV_32FC1);
            dst.create(size, size, CV_32FC1);
            ymap.create(size, size, CV_32FC1);

            for (int i = 0; i < size; ++i)
            {
                float *xmap_row = xmap.ptr<float>(i);
                float *ymap_row = ymap.ptr<float>(i);

                for (int j = 0; j < size; ++j)
                {
                    xmap_row[j] = (j - size * 0.5f) * 0.75f + size * 0.5f;
                    ymap_row[j] = (i - size * 0.5f) * 0.75f + size * 0.5f;
                }
            }


            remap(src, dst, xmap, ymap, interpolation, borderMode);

            CPU_ON;
            remap(src, dst, xmap, ymap, interpolation, borderMode);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src.upload(src);
            d_dst.upload(dst);
            d_xmap.upload(xmap);
            d_ymap.upload(ymap);

            WARMUP_ON;
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            WARMUP_OFF;

            GPU_ON;
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}
///////////// cvtColor////////////////////////
TEST(cvtColor)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif
    int all_type[] = {CV_8UC4};
    std::string type_name[] = {"CV_8UC4"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            gen(src, size, size, all_type[j], 0, 256);
            SUBTEST << size << "x" << size << "; " << type_name[j] << " ; CV_RGBA2GRAY";

            cvtColor(src, dst, CV_RGBA2GRAY, 4);

            CPU_ON;
            cvtColor(src, dst, CV_RGBA2GRAY, 4);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::cvtColor(d_src, d_dst, CV_RGBA2GRAY, 4);
            WARMUP_OFF;

            GPU_ON;
            ocl::cvtColor(d_src, d_dst, CV_RGBA2GRAY, 4);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::cvtColor(d_src, d_dst, CV_RGBA2GRAY, 4);
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }


    }


}
///////////// filter2D////////////////////////
TEST(filter2D)
{
    Mat src;

    for (int size = 1000; size <= 4000; size *= 2)
    {
        int all_type[] = {CV_8UC1, CV_8UC4};
        std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            gen(src, size, size, all_type[j], 0, 256);

            for (int ksize = 3; ksize <= 15; ksize = 2*ksize+1)
            {
                SUBTEST << "ksize = " << ksize << "; " << size << 'x' << size << "; " << type_name[j] ;

                Mat kernel;
                gen(kernel, ksize, ksize, CV_32FC1, 0.0, 1.0);

                Mat dst;
                cv::filter2D(src, dst, -1, kernel);

                CPU_ON;
                cv::filter2D(src, dst, -1, kernel);
                CPU_OFF;
#ifdef USE_OPENCL
                ocl::oclMat d_src(src);
                ocl::oclMat d_dst;

                WARMUP_ON;
                ocl::filter2D(d_src, d_dst, -1, kernel);
                WARMUP_OFF;

                GPU_ON;
                ocl::filter2D(d_src, d_dst, -1, kernel);
                GPU_OFF;

                GPU_FULL_ON;
                d_src.upload(src);
                ocl::filter2D(d_src, d_dst, -1, kernel);
                d_dst.download(dst);
                GPU_FULL_OFF;
#endif
            }

        }


    }
}


///////////// dft ////////////////////////
TEST(dft)
{
    Mat src, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src, d_dst;
#endif

    int all_type[] = {CV_32FC1, CV_32FC2};
    std::string type_name[] = {"CV_32FC1", "CV_32FC2"};

    for (int size = 1000; size <= 4000; size *= 2)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " ; complex-to-complex";

            gen(src, size, size, all_type[j], Scalar::all(0), Scalar::all(1));

            dft(src, dst);

            CPU_ON;
            dft(src, dst);
            CPU_OFF;

#ifdef USE_OPENCL
            d_src.upload(src);

            WARMUP_ON;
            ocl::dft(d_src, d_dst, Size(size, size));
            WARMUP_OFF;

            GPU_ON;
            ocl::dft(d_src, d_dst, Size(size, size));
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::dft(d_src, d_dst, Size(size, size));
            d_dst.download(dst);
            GPU_FULL_OFF;
#endif
        }

    }
}

///////////// gemm ////////////////////////
TEST(gemm)
{
    Mat src1, src2, src3, dst;
#ifdef USE_OPENCL
    ocl::oclMat d_src1, d_src2, d_src3, d_dst;
#endif

    for (int size = 1000; size <= 4000; size *= 2)
    {
        SUBTEST << size << 'x' << size;

        gen(src1, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src2, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src3, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));

        gemm(src1, src2, 1.0, src3, 1.0, dst);

        CPU_ON;
        gemm(src1, src2, 1.0, src3, 1.0, dst);
        CPU_OFF;

#ifdef USE_OPENCL
        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);

        WARMUP_ON;
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        d_dst.download(dst);
        GPU_FULL_OFF;
#endif
    }
}

int main(int argc, const char *argv[])
{
#ifdef USE_OPENCL
    vector<ocl::Info> oclinfo;
    int num_devices = getDevice(oclinfo);

    if (num_devices < 1)
    {
        cerr << "no device found\n";
        return -1;
    }

    int devidx = 0;

    for (size_t i = 0; i < oclinfo.size(); i++)
    {
        for (size_t j = 0; j < oclinfo[i].DeviceName.size(); j++)
        {
            printf("device %d: %s\n", devidx++, oclinfo[i].DeviceName[j].c_str());
        }
    }

#endif
    redirectError(cvErrorCallback);

    const char *keys =
        "{ h | help    | false | print help message }"
        "{ f | filter  |       | filter for test }"
        "{ w | workdir |       | set working directory }"
        "{ l | list    | false | show all tests }"
        "{ d | device  | 0     | device id }"
        "{ i | iters   | 10    | iteration count }"
        "{ m | warmup  | 1     | gpu warm up iteration count}"
        "{ t | xtop    | 1.1	  | xfactor top boundary}"
        "{ b | xbottom | 0.9	  | xfactor bottom boundary}"
        "{ v | verify  | false | only run gpu once to verify if problems occur}";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

#ifdef USE_OPENCL
    int device = cmd.get<int>("device");

    if (device < 0 || device >= num_devices)
    {
        cerr << "Invalid device ID" << endl;
        return -1;
    }

    if (cmd.get<bool>("verify"))
    {
        TestSystem::instance().setNumIters(1);
        TestSystem::instance().setGPUWarmupIters(0);
        TestSystem::instance().setCPUIters(0);
    }

    devidx = 0;

    for (size_t i = 0; i < oclinfo.size(); i++)
    {
        for (size_t j = 0; j < oclinfo[i].DeviceName.size(); j++, devidx++)
        {
            if (device == devidx)
            {
                ocl::setDevice(oclinfo[i], j);
                TestSystem::instance().setRecordName(oclinfo[i].DeviceName[j]);
                printf("\nuse %d: %s\n", devidx, oclinfo[i].DeviceName[j].c_str());
                goto END_DEV;
            }
        }
    }

END_DEV:

#endif
    string filter = cmd.get<string>("filter");
    string workdir = cmd.get<string>("workdir");
    bool list = cmd.get<bool>("list");
    int iters = cmd.get<int>("iters");
    int wu_iters = cmd.get<int>("warmup");
    double x_top = cmd.get<double>("xtop");
    double x_bottom = cmd.get<double>("xbottom");

    TestSystem::instance().setTopThreshold(x_top);
    TestSystem::instance().setBottomThreshold(x_bottom);

    if (!filter.empty())
    {
        TestSystem::instance().setTestFilter(filter);
    }

    if (!workdir.empty())
    {
        if (workdir[workdir.size() - 1] != '/' && workdir[workdir.size() - 1] != '\\')
        {
            workdir += '/';
        }

        TestSystem::instance().setWorkingDir(workdir);
    }

    if (list)
    {
        TestSystem::instance().setListMode(true);
    }

    TestSystem::instance().setNumIters(iters);
    TestSystem::instance().setGPUWarmupIters(wu_iters);

    TestSystem::instance().run();

    return 0;
}
