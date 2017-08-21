#include "test_precomp.hpp"
#include "test_intrin_utils.hpp"

namespace cvtest { namespace hal {
TEST(hal_intrin, float16x4) {
    TheTest<v_float16x4>()
        .test_loadstore_fp16()
        .test_float_cvt_fp16()
        ;
}
}}
