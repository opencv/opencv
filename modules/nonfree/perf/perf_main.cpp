#include "perf_precomp.hpp"
#include "opencv2/ts/cuda_perf.hpp"

static const char * impls[] = {
#ifdef HAVE_CUDA
    "cuda",
#endif
    "plain"
};

CV_PERF_TEST_MAIN_WITH_IMPLS(nonfree, impls, perf::printCudaInfo())
