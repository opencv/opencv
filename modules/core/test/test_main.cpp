#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif


#include "test_precomp.hpp"

#ifndef HAVE_CUDA

CV_TEST_MAIN("cv")

#else

#include "opencv2/ts/cuda_test.hpp"

CV_CUDA_TEST_MAIN("cv")

#endif
