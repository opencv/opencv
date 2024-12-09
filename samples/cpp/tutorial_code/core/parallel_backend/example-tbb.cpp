#include "opencv2/core.hpp"
#include <iostream>

#include <chrono>
#include <thread>

//! [tbb_include]
#include "opencv2/core/parallel/backend/parallel_for.tbb.hpp"
//! [tbb_include]

namespace cv { // private.hpp
CV_EXPORTS const char* currentParallelFramework();
}

static
std::string currentParallelFrameworkSafe()
{
    const char* framework = cv::currentParallelFramework();
    if (framework)
        return framework;
    return std::string();
}

using namespace cv;
int main()
{
    std::cout << "OpenCV builtin parallel framework: '" << currentParallelFrameworkSafe() << "' (nthreads=" << getNumThreads() << ")" << std::endl;

    //! [tbb_backend]
    cv::parallel::setParallelForBackend(std::make_shared<cv::parallel::tbb::ParallelForBackend>());
    //! [tbb_backend]

    std::cout << "New parallel backend: '" << currentParallelFrameworkSafe() << "'" << "' (nthreads=" << getNumThreads() << ")" << std::endl;

    parallel_for_(Range(0, 20), [&](const Range range)
    {
        std::ostringstream out;
        out << "Thread " << getThreadNum() << "(opencv=" << utils::getThreadID() << "): range " << range.start << "-" << range.end << std::endl;
        std::cout << out.str() << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });
}
