#include "test_precomp.hpp"
#include "test_intrin_utils.hpp"

namespace cvtest { namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void test_hal_intrin_float16x4()
{
    TheTest<v_float16x4>()
        .test_loadstore_fp16()
        .test_float_cvt_fp16()
        ;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
