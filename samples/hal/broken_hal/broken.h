#ifndef _BROKEN_H_INCLUDED_
#define _BROKEN_H_INCLUDED_

#include "opencv2/core/hal/interface.h"

#if defined(__cplusplus)
extern "C"
{
#endif

int broken_add8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_add8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int broken_add16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int broken_add16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int broken_add32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int broken_add32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int broken_add64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int broken_sub8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_sub8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int broken_sub16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int broken_sub16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int broken_sub32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int broken_sub32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int broken_sub64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int broken_max8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_max8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int broken_max16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int broken_max16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int broken_max32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int broken_max32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int broken_max64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int broken_min8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_min8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int broken_min16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int broken_min16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int broken_min32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int broken_min32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int broken_min64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int broken_absdiff8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_absdiff8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int broken_absdiff16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int broken_absdiff16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int broken_absdiff32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int broken_absdiff32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int broken_absdiff64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int broken_and8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_or8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_xor8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int broken_not8u(const uchar* src1, size_t sz1, uchar* dst, size_t sz, int w, int h);

#undef cv_hal_add8u
#define cv_hal_add8u broken_add8u
#undef cv_hal_add8s
#define cv_hal_add8s broken_add8s
#undef cv_hal_add16u
#define cv_hal_add16u broken_add16u
#undef cv_hal_add16s
#define cv_hal_add16s broken_add16s
#undef cv_hal_add32s
#define cv_hal_add32s broken_add32s
#undef cv_hal_add32f
#define cv_hal_add32f broken_add32f
#undef cv_hal_add64f
#define cv_hal_add64f broken_add64f
#undef cv_hal_sub8u
#define cv_hal_sub8u broken_sub8u
#undef cv_hal_sub8s
#define cv_hal_sub8s broken_sub8s
#undef cv_hal_sub16u
#define cv_hal_sub16u broken_sub16u
#undef cv_hal_sub16s
#define cv_hal_sub16s broken_sub16s
#undef cv_hal_sub32s
#define cv_hal_sub32s broken_sub32s
#undef cv_hal_sub32f
#define cv_hal_sub32f broken_sub32f
#undef cv_hal_sub64f
#define cv_hal_sub64f broken_sub64f
#undef cv_hal_max8u
#define cv_hal_max8u broken_max8u
#undef cv_hal_max8s
#define cv_hal_max8s broken_max8s
#undef cv_hal_max16u
#define cv_hal_max16u broken_max16u
#undef cv_hal_max16s
#define cv_hal_max16s broken_max16s
#undef cv_hal_max32s
#define cv_hal_max32s broken_max32s
#undef cv_hal_max32f
#define cv_hal_max32f broken_max32f
#undef cv_hal_max64f
#define cv_hal_max64f broken_max64f
#undef cv_hal_min8u
#define cv_hal_min8u broken_min8u
#undef cv_hal_min8s
#define cv_hal_min8s broken_min8s
#undef cv_hal_min16u
#define cv_hal_min16u broken_min16u
#undef cv_hal_min16s
#define cv_hal_min16s broken_min16s
#undef cv_hal_min32s
#define cv_hal_min32s broken_min32s
#undef cv_hal_min32f
#define cv_hal_min32f broken_min32f
#undef cv_hal_min64f
#define cv_hal_min64f broken_min64f
#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u broken_absdiff8u
#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s broken_absdiff8s
#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u broken_absdiff16u
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s broken_absdiff16s
#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s broken_absdiff32s
#undef cv_hal_absdiff32f
#define cv_hal_absdiff32f broken_absdiff32f
#undef cv_hal_absdiff64f
#define cv_hal_absdiff64f broken_absdiff64f
#undef cv_hal_and8u
#define cv_hal_and8u broken_and8u
#undef cv_hal_or8u
#define cv_hal_or8u broken_or8u
#undef cv_hal_xor8u
#define cv_hal_xor8u broken_xor8u
#undef cv_hal_not8u
#define cv_hal_not8u broken_not8u

int broken_cmp8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp16s(const short* src1, size_t sz1, const short* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp32s(const int* src1, size_t sz1, const int* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp32f(const float* src1, size_t sz1, const float* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int broken_cmp64f(const double* src1, size_t sz1, const double* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);

#undef cv_hal_cmp8u
#define cv_hal_cmp8u broken_cmp8u
#undef cv_hal_cmp8s
#define cv_hal_cmp8s broken_cmp8s
#undef cv_hal_cmp16u
#define cv_hal_cmp16u broken_cmp16u
#undef cv_hal_cmp16s
#define cv_hal_cmp16s broken_cmp16s
#undef cv_hal_cmp32s
#define cv_hal_cmp32s broken_cmp32s
#undef cv_hal_cmp32f
#define cv_hal_cmp32f broken_cmp32f
#undef cv_hal_cmp64f
#define cv_hal_cmp64f broken_cmp64f

int broken_mul8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int broken_mul8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int broken_mul16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int broken_mul16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int broken_mul32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int broken_mul32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int broken_mul64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);
int broken_div8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int broken_div8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int broken_div16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int broken_div16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int broken_div32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int broken_div32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int broken_div64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);
int broken_recip8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int broken_recip8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int broken_recip16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int broken_recip16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int broken_recip32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int broken_recip32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int broken_recip64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);

#undef cv_hal_mul8u
#define cv_hal_mul8u broken_mul8u
#undef cv_hal_mul8s
#define cv_hal_mul8s broken_mul8s
#undef cv_hal_mul16u
#define cv_hal_mul16u broken_mul16u
#undef cv_hal_mul16s
#define cv_hal_mul16s broken_mul16s
#undef cv_hal_mul32s
#define cv_hal_mul32s broken_mul32s
#undef cv_hal_mul32f
#define cv_hal_mul32f broken_mul32f
#undef cv_hal_mul64f
#define cv_hal_mul64f broken_mul64f
#undef cv_hal_div8u
#define cv_hal_div8u broken_div8u
#undef cv_hal_div8s
#define cv_hal_div8s broken_div8s
#undef cv_hal_div16u
#define cv_hal_div16u broken_div16u
#undef cv_hal_div16s
#define cv_hal_div16s broken_div16s
#undef cv_hal_div32s
#define cv_hal_div32s broken_div32s
#undef cv_hal_div32f
#define cv_hal_div32f broken_div32f
#undef cv_hal_div64f
#define cv_hal_div64f broken_div64f
#undef cv_hal_recip8u
#define cv_hal_recip8u broken_recip8u
#undef cv_hal_recip8s
#define cv_hal_recip8s broken_recip8s
#undef cv_hal_recip16u
#define cv_hal_recip16u broken_recip16u
#undef cv_hal_recip16s
#define cv_hal_recip16s broken_recip16s
#undef cv_hal_recip32s
#define cv_hal_recip32s broken_recip32s
#undef cv_hal_recip32f
#define cv_hal_recip32f broken_recip32f
#undef cv_hal_recip64f
#define cv_hal_recip64f broken_recip64f

int broken_addWeighted8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, const double* scales);
int broken_addWeighted64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, const double* scales);

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u broken_addWeighted8u
#undef cv_hal_addWeighted8s
#define cv_hal_addWeighted8s broken_addWeighted8s
#undef cv_hal_addWeighted16u
#define cv_hal_addWeighted16u broken_addWeighted16u
#undef cv_hal_addWeighted16s
#define cv_hal_addWeighted16s broken_addWeighted16s
#undef cv_hal_addWeighted32s
#define cv_hal_addWeighted32s broken_addWeighted32s
#undef cv_hal_addWeighted32f
#define cv_hal_addWeighted32f broken_addWeighted32f
#undef cv_hal_addWeighted64f
#define cv_hal_addWeighted64f broken_addWeighted64f

#if defined(__cplusplus)
}
#endif

#endif
