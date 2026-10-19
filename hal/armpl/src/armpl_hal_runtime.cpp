#ifdef HAVE_ARMPL

#include "armpl_hal_runtime.hpp"

#if defined(ARMPL_HAL_CBLAS_LIB) || defined(ARMPL_HAL_OMP_LIB)
#include "opencv2/core/utility.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/plugin_loader.private.hpp"
#endif

void *armpl_hal_get_function(const char *name)
{
#ifdef ARMPL_HAL_CBLAS_LIB
    static cv::plugin::impl::DynamicLib lib(cv::plugin::impl::toFileSystemPath(ARMPL_HAL_CBLAS_LIB));
    return lib.getSymbol(name);
#else
    (void)name;
    return 0;
#endif
}

#ifdef ARMPL_HAL_OMP_LIB
static void *armpl_omp_function(const char *name)
{
    static cv::plugin::impl::DynamicLib lib(cv::plugin::impl::toFileSystemPath(ARMPL_HAL_OMP_LIB));
    return lib.getSymbol(name);
}

static void armpl_omp_set_threads(int n)
{
    static void (*set_threads)(int) = (void (*)(int))armpl_omp_function("omp_set_num_threads");
    if (set_threads)
        set_threads(n);
}
#endif

ArmplSingleThread::ArmplSingleThread(double work) : nthreads(0)
{
#ifdef ARMPL_HAL_OMP_LIB
    static int (*get_threads)() = (int (*)())armpl_omp_function("omp_get_max_threads");
    if (work < (1 << 20) && get_threads && (nthreads = get_threads()) > 1)
        armpl_omp_set_threads(1);
#else
    (void)work;
#endif
}

ArmplSingleThread::~ArmplSingleThread()
{
#ifdef ARMPL_HAL_OMP_LIB
    if (nthreads > 1)
        armpl_omp_set_threads(nthreads);
#endif
}

#endif
