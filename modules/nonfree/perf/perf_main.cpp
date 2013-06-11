#include "perf_precomp.hpp"
#include "opencv2/ts/gpu_perf.hpp"

CV_PERF_TEST_MAIN_WITH_IMPLS(nonfree, ("cuda", "plain"), perf::printCudaInfo())
