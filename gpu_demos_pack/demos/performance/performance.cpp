#include <iomanip>
#include <stdexcept>
#include <string>
#include "performance.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void TestSystem::run()
{
    if (is_list_mode_)
    {
        for (vector<Runnable*>::iterator it = tests_.begin(); it != tests_.end(); ++it)
            cout << (*it)->name() << endl;

        return;
    }

    // Run test initializers    
    for (vector<Runnable*>::iterator it = inits_.begin(); it != inits_.end(); ++it)
    {
        if ((*it)->name().find(test_filter_, 0) != string::npos)
            (*it)->run();
    }

    printHeading();

    // Run tests
    for (vector<Runnable*>::iterator it = tests_.begin(); it != tests_.end(); ++it)
    {
        try
        {
            if ((*it)->name().find(test_filter_, 0) != string::npos)
            {
                cout << endl << (*it)->name() << ":\n";
                (*it)->run();
                finishCurrentSubtest();
            }
        }
        catch (const Exception&)
        {
            // Message is printed via callback
            resetCurrentSubtest();
        }
        catch (const runtime_error& e)
        {
            printError(e.what());
            resetCurrentSubtest();
        }
    }

    printSummary();
}


void TestSystem::finishCurrentSubtest()
{
    if (cur_subtest_is_empty_)
        // There is no need to print subtest statistics
        return;

    double cpu_time = cpu_elapsed_ / getTickFrequency() * 1000.0;
    double gpu_time = gpu_elapsed_ / getTickFrequency() * 1000.0;

    double speedup = static_cast<double>(cpu_elapsed_) / std::max((int64)1, gpu_elapsed_);
    speedup_total_ += speedup;

    printMetrics(cpu_time, gpu_time, speedup);
    
    num_subtests_called_++;
    resetCurrentSubtest();
}


double TestSystem::meanTime(const vector<int64> &samples)
{
    double sum = accumulate(samples.begin(), samples.end(), 0.);
    if (samples.size() > 1)
        return (sum - samples[0]) / (samples.size() - 1);
    return sum;
}


void TestSystem::printHeading()
{
    cout << endl;
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
        << setprecision(3) << speedup_total_ / std::max(1, num_subtests_called_) 
        << endl;
    cout << resetiosflags(ios_base::fixed);
}


void TestSystem::printMetrics(double cpu_time, double gpu_time, double speedup)
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

    cout << cur_subtest_description_.str();
    cout << resetiosflags(ios_base::left) << endl;
}


void TestSystem::printError(const std::string& msg)
{
    cout << TAB << "[error: " << msg << "] " << cur_subtest_description_.str() << endl;
}


void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


string abspath(const string& relpath)
{
    return TestSystem::instance().workingDir() + relpath;
}


int CV_CDECL cvErrorCallback(int /*status*/, const char* /*func_name*/, 
                             const char* err_msg, const char* /*file_name*/,
                             int /*line*/, void* /*userdata*/)
{
    TestSystem::instance().printError(err_msg);
    return 0;
}


int main(int argc, const char* argv[])
{
    redirectError(cvErrorCallback);

    const char* keys =
       "{ h | help    | false | print help message }"
       "{ f | filter  |       | filter for test }"
       "{ l | list    | false | show all tests }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: demo_performance [options]" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    string filter = cmd.get<string>("filter");
    bool list = cmd.get<bool>("list");

    if (!filter.empty())
        TestSystem::instance().setTestFilter(filter);

    if (list)
        TestSystem::instance().setListMode(true);

    TestSystem::instance().setWorkingDir("data/");
    TestSystem::instance().setNumIters(2);
    TestSystem::instance().run();

    cout << "\nPress ENTER to exit...\n";
    cin.get();

    return 0;
}
