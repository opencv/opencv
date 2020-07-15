// Helper file to include dispatched functions declaration:
//
// Usage:
//     #define CV_CPU_SIMD_FILENAME "<filename>.simd.hpp"
//     #define CV_CPU_DISPATCH_MODE AVX2
//     #include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"
//     #define CV_CPU_DISPATCH_MODE SSE2
//     #include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#ifndef CV_DISABLE_OPTIMIZATION
#ifdef _MSC_VER
#pragma warning(disable: 4702) // unreachable code
#endif
#endif

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#endif

#undef CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#undef CV_CPU_OPTIMIZATION_NAMESPACE_END

#define CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN namespace __CV_CAT(opt_, CV_CPU_DISPATCH_MODE) {
#define CV_CPU_OPTIMIZATION_NAMESPACE_END }

#include CV_CPU_SIMD_FILENAME

#undef CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#undef CV_CPU_OPTIMIZATION_NAMESPACE_END
#undef CV_CPU_DISPATCH_MODE
