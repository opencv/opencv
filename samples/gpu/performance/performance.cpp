#include <iomanip>
#include <stdexcept>
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
        catch (const Exception&)
        {
            // Message is printed via callback
            resetSubtestData();
        }
        catch (const runtime_error& e)
        {
            printError(e.what());
            resetSubtestData();
        }
    }

    printSummary();
}


void TestSystem::setWorkingDir(const string& val)
{
    working_dir_ = val;
}


void TestSystem::flushSubtestData()
{
    if (!can_flush_)
        return;

    int cpu_time = static_cast<int>(cpu_elapsed_ / getTickFrequency() * 1000.0);
    int gpu_time = static_cast<int>(gpu_elapsed_ / getTickFrequency() * 1000.0);

    double speedup = static_cast<double>(cpu_elapsed_) / gpu_elapsed_;
    speedup_total_ += speedup;

    printItem(cpu_time, gpu_time, speedup);
    
    num_subtests_called_++;
    resetSubtestData();
}


void TestSystem::printHeading()
{
    cout << setiosflags(ios_base::left);
    cout << TAB << setw(10) << "CPU, ms" << setw(10) << "GPU, ms" 
        << setw(14) << "SPEEDUP" 
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
    cout << setw(14) << stream.str();

    cout << description_.str();
    cout << resetiosflags(ios_base::left) << endl;
}


void TestSystem::printError(const std::string& msg)
{
    cout << TAB << "[error: " << msg << "] " << description_.str() << endl;
}


void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


string abspath(const string& relpath)
{
    return TestSystem::instance()->workingDir() + relpath;
}


int CV_CDECL cvErrorCallback(int /*status*/, const char* /*func_name*/, 
                             const char* err_msg, const char* /*file_name*/,
                             int /*line*/, void* /*userdata*/)
{
    TestSystem::instance()->printError(err_msg);
    return 0;
}


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Usage: performance_gpu <working_dir_with_slash>\n\n";
    }
    else
    {
        TestSystem::instance()->setWorkingDir(argv[1]);
    }

    redirectError(cvErrorCallback);
    TestSystem::instance()->run();

    return 0;
}