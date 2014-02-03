#include "perf_precomp.hpp"
#include "opencv2/ts/gpu_perf.hpp"

static const char * impls[] = {
#ifdef HAVE_CUDA
    "cuda",
#endif
#ifdef HAVE_OPENCL
    "ocl",
#endif
    "plain"
};

CV_PERF_TEST_MAIN_WITH_IMPLS(nonfree, impls, perf::printCudaInfo())
