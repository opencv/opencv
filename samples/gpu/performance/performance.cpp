#include <iomanip>
#include "performance.h"

using namespace std;
using namespace cv;

void TestSystem::run()
{
    // Run initializers
    vector<Runnable*>::iterator it = inits_.begin();
    for (; it != inits_.end(); ++it)
    {
        (*it)->run();
    }

    printHeading();

    // Run tests
    it = tests_.begin();
    for (; it != tests_.end(); ++it)
    {
        cout << endl << (*it)->name() << ":\n";
        try
        {
            (*it)->run();
            flushSubtestData();
        }
        catch (const cv::Exception&)
        {
            resetSubtestData();
        }
    }

    printSummary();
}


void TestSystem::flushSubtestData()
{
    if (!can_flush_)
        return;

    int cpu_time = static_cast<int>(cpu_elapsed_ / getTickFrequency() * 1000.0);
    int gpu_time = static_cast<int>(gpu_elapsed_ / getTickFrequency() * 1000.0);

    double speedup = static_cast<double>(cpu_time) / std::max(1, gpu_time);
    speedup_total_ += speedup;

    printItem(cpu_time, gpu_time, speedup);
    
    num_subtests_called_++;
    resetSubtestData();
}


void TestSystem::printHeading()
{
    cout << setiosflags(ios_base::left);
    cout << TAB << setw(10) << "CPU, ms" << setw(10) << "GPU, ms" 
        << setw(10) << "SPEEDUP" 
        << "DESCRIPTION\n";
    cout << resetiosflags(ios_base::left);
}


void TestSystem::printSummary()
{
    cout << setiosflags(ios_base::fixed);
    cout << "\naverage GPU speedup: x" 
        << setprecision(3) << speedup_total_ / num_subtests_called_ 
        << endl;
    cout << resetiosflags(ios_base::fixed);
}


void TestSystem::printItem(double cpu_time, double gpu_time, double speedup)
{
    cout << TAB << setiosflags(ios_base::left);
    stringstream stream;

    stream << cpu_time;
    cout << setw(10) << stream.str();

    stream.str("");
    stream << gpu_time;
    cout << setw(10) << stream.str();

    stream.str("");
    stream << "x" << setprecision(3) << speedup;
    cout << setw(10) << stream.str();

    cout << description_.str();
    cout << resetiosflags(ios_base::left) << endl;
}


void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


int CV_CDECL cvErrorCallback(int /*status*/, const char* /*func_name*/, 
                             const char* /*err_msg*/, const char* /*file_name*/,
                             int /*line*/, void* /*userdata*/)
{
    return 0;
}


int main()
{
    redirectError(cvErrorCallback);
    TestSystem::instance()->run();
    return 0;
}