#include "perf_precomp.hpp"
#include "opencv2/ts/gpu_perf.hpp"

static const char * impls[] = {
#ifdef HAVE_CUDA
    "cuda",
#endif
#ifdef HAVE_OPENCV_OCL
    "ocl",
#endif
    "plain"
};

#ifdef HAVE_OPENCL
#define DUMP_PROPERTY_XML(propertyName, propertyValue) \
    do { \
        std::stringstream ssName, ssValue;\
        ssName << propertyName;\
        ssValue << propertyValue; \
        ::testing::Test::RecordProperty(ssName.str(), ssValue.str()); \
    } while (false)

#define DUMP_MESSAGE_STDOUT(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while (false)

#include "opencv2/ocl/private/opencl_dumpinfo.hpp"
#endif

int main(int argc, char **argv)
{
    ::perf::TestBase::setPerformanceStrategy(::perf::PERF_STRATEGY_SIMPLE);
#if defined(HAVE_CUDA) && defined(HAVE_OPENCL)
    CV_PERF_TEST_MAIN_INTERNALS(nonfree, impls, perf::printCudaInfo(), dumpOpenCLDevice());
#elif defined(HAVE_CUDA)
    CV_PERF_TEST_MAIN_INTERNALS(nonfree, impls, perf::printCudaInfo());
#elif defined(HAVE_OPENCL)
    CV_PERF_TEST_MAIN_INTERNALS(nonfree, impls, dumpOpenCLDevice());
#else
    CV_PERF_TEST_MAIN_INTERNALS(nonfree, impls)
#endif
}
