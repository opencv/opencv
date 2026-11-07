#include "opencv2/core.hpp"
#include "opencv2/core/simd_intrinsics.hpp"

using namespace cv;

int main(int /*argc*/, char** /*argv*/)
{
    printf("==================  macro dump  ===================\n");
#ifdef CV_SIMD
    printf("CV_SIMD is defined: " CVAUX_STR(CV_SIMD) "\n");
#ifdef CV_SIMD_WIDTH
    printf("CV_SIMD_WIDTH is defined: " CVAUX_STR(CV_SIMD_WIDTH) "\n");
#endif
#ifdef CV_SIMD128
    printf("CV_SIMD128 is defined: " CVAUX_STR(CV_SIMD128) "\n");
#endif
#ifdef CV_SIMD256
    printf("CV_SIMD256 is defined: " CVAUX_STR(CV_SIMD256) "\n");
#endif
#ifdef CV_SIMD512
    printf("CV_SIMD512 is defined: " CVAUX_STR(CV_SIMD512) "\n");
#endif
#ifdef CV_SIMD_64F
    printf("CV_SIMD_64F is defined: " CVAUX_STR(CV_SIMD_64F) "\n");
#endif
#ifdef CV_SIMD_FP16
    printf("CV_SIMD_FP16 is defined: " CVAUX_STR(CV_SIMD_FP16) "\n");
#endif
#else
    printf("CV_SIMD is NOT defined\n");
#endif

#ifdef CV_SIMD
    printf("=================  sizeof checks  =================\n");
    printf("sizeof(v_uint8) = %d\n", (int)sizeof(v_uint8));
    printf("sizeof(v_int32) = %d\n", (int)sizeof(v_int32));
    printf("sizeof(v_float32) = %d\n", (int)sizeof(v_float32));

    printf("==================  arithm check  =================\n");
    v_uint8 a = vx_setall_u8(10);
    v_uint8 c = v_add(a, vx_setall_u8(45));
    printf("v_get0(vx_setall_u8(10) + vx_setall_u8(45)) => %d\n", (int)v_get0(c));
#else
    printf("\nSIMD intrinsics are not available. Check compilation target and passed build options.\n");
#endif

    printf("=====================  done  ======================\n");
    return 0;
}
