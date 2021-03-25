#include "perf_precomp.hpp"
#include "opencv2/ts/cuda_perf.hpp"

static const char * impls[] = {
#ifdef HAVE_CUDA
    "cuda",
#endif
    "plain"
};

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

CV_PERF_TEST_MAIN_WITH_IMPLS(photo, impls, perf::printCudaInfo())
