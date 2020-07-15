#include "perf_precomp.hpp"
#ifdef _MSC_VER
# if _MSC_VER >= 1700
#  pragma warning(disable:4447) // Disable warning 'main' signature found without threading model
# endif
#endif

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

CV_PERF_TEST_MAIN(core)
