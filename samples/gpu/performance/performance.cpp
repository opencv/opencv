#include <iomanip>
#include "performance.h"

using namespace std;
using namespace cv;

void Test::gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);

    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


void TestSystem::run()
{
    cout << setiosflags(ios_base::left);
    cout << "    " << setw(10) << "CPU, ms" << setw(10) << "GPU, ms" 
        << setw(10) << "SPEEDUP" << "DESCRIPTION\n";
    cout << resetiosflags(ios_base::left);

    vector<Test*>::iterator it = tests_.begin();
    for (; it != tests_.end(); ++it)
    {
        Test* test = *it;

        cout << endl << test->name() << ":\n";
        test->run();

        flush();
    }

    cout << setiosflags(ios_base::fixed | ios_base::left);
    cout << "\nAverage GPU Speedup: x" << setprecision(3) 
        << speedup_total_ / num_subtests_called_ << endl;
    cout << resetiosflags(ios_base::fixed | ios_base::left);
}


void TestSystem::flush()
{
    if (!can_flush_)
        return;

    int cpu_time = static_cast<int>(cpu_elapsed_ / getTickFrequency() * 1000.0);
    int gpu_time = static_cast<int>(gpu_elapsed_ / getTickFrequency() * 1000.0);
    cpu_elapsed_ = 0;
    gpu_elapsed_ = 0;

    double speedup = static_cast<double>(cpu_time) / std::max(1, gpu_time);
    speedup_total_ += speedup;

    cout << "    " << setiosflags(ios_base::fixed | ios_base::left);

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
    description_.str("");

    cout << resetiosflags(ios_base::fixed | ios_base::left) << endl;
    
    can_flush_ = false;
    num_subtests_called_++;
}


int main()
{
    TestSystem::instance()->run();
    return 0;
}