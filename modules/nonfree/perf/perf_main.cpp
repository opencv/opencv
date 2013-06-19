#include "perf_precomp.hpp"
#include "opencv2/ts/gpu_perf.hpp"

CV_PERF_TEST_MAIN_WITH_IMPLS(nonfree, (
#ifdef HAVE_CUDA
                                       "cuda",
#endif
                                       "plain"), perf::printCudaInfo())
