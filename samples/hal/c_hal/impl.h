#ifndef _wrong_H_INCLUDED_
#define _wrong_H_INCLUDED_

#include "opencv2/core/hal/interface.h"

#if defined(__cplusplus)
extern "C"
{
#endif

int wrong_add8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_add8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int wrong_add16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int wrong_add16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int wrong_add32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int wrong_add32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int wrong_add64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int wrong_sub8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_sub8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int wrong_sub16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int wrong_sub16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int wrong_sub32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int wrong_sub32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int wrong_sub64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int wrong_max8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_max8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int wrong_max16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int wrong_max16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int wrong_max32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int wrong_max32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int wrong_max64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int wrong_min8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_min8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int wrong_min16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int wrong_min16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int wrong_min32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int wrong_min32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int wrong_min64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int wrong_absdiff8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_absdiff8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h);
int wrong_absdiff16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h);
int wrong_absdiff16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h);
int wrong_absdiff32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h);
int wrong_absdiff32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h);
int wrong_absdiff64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h);
int wrong_and8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_or8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_xor8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h);
int wrong_not8u(const uchar* src1, size_t sz1, uchar* dst, size_t sz, int w, int h);

#undef cv_hal_add8u
#define cv_hal_add8u wrong_add8u
#undef cv_hal_add8s
#define cv_hal_add8s wrong_add8s
#undef cv_hal_add16u
#define cv_hal_add16u wrong_add16u
#undef cv_hal_add16s
#define cv_hal_add16s wrong_add16s
#undef cv_hal_add32s
#define cv_hal_add32s wrong_add32s
#undef cv_hal_add32f
#define cv_hal_add32f wrong_add32f
#undef cv_hal_add64f
#define cv_hal_add64f wrong_add64f
#undef cv_hal_sub8u
#define cv_hal_sub8u wrong_sub8u
#undef cv_hal_sub8s
#define cv_hal_sub8s wrong_sub8s
#undef cv_hal_sub16u
#define cv_hal_sub16u wrong_sub16u
#undef cv_hal_sub16s
#define cv_hal_sub16s wrong_sub16s
#undef cv_hal_sub32s
#define cv_hal_sub32s wrong_sub32s
#undef cv_hal_sub32f
#define cv_hal_sub32f wrong_sub32f
#undef cv_hal_sub64f
#define cv_hal_sub64f wrong_sub64f
#undef cv_hal_max8u
#define cv_hal_max8u wrong_max8u
#undef cv_hal_max8s
#define cv_hal_max8s wrong_max8s
#undef cv_hal_max16u
#define cv_hal_max16u wrong_max16u
#undef cv_hal_max16s
#define cv_hal_max16s wrong_max16s
#undef cv_hal_max32s
#define cv_hal_max32s wrong_max32s
#undef cv_hal_max32f
#define cv_hal_max32f wrong_max32f
#undef cv_hal_max64f
#define cv_hal_max64f wrong_max64f
#undef cv_hal_min8u
#define cv_hal_min8u wrong_min8u
#undef cv_hal_min8s
#define cv_hal_min8s wrong_min8s
#undef cv_hal_min16u
#define cv_hal_min16u wrong_min16u
#undef cv_hal_min16s
#define cv_hal_min16s wrong_min16s
#undef cv_hal_min32s
#define cv_hal_min32s wrong_min32s
#undef cv_hal_min32f
#define cv_hal_min32f wrong_min32f
#undef cv_hal_min64f
#define cv_hal_min64f wrong_min64f
#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u wrong_absdiff8u
#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s wrong_absdiff8s
#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u wrong_absdiff16u
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s wrong_absdiff16s
#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s wrong_absdiff32s
#undef cv_hal_absdiff32f
#define cv_hal_absdiff32f wrong_absdiff32f
#undef cv_hal_absdiff64f
#define cv_hal_absdiff64f wrong_absdiff64f
#undef cv_hal_and8u
#define cv_hal_and8u wrong_and8u
#undef cv_hal_or8u
#define cv_hal_or8u wrong_or8u
#undef cv_hal_xor8u
#define cv_hal_xor8u wrong_xor8u
#undef cv_hal_not8u
#define cv_hal_not8u wrong_not8u

int wrong_cmp8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp16s(const short* src1, size_t sz1, const short* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp32s(const int* src1, size_t sz1, const int* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp32f(const float* src1, size_t sz1, const float* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);
int wrong_cmp64f(const double* src1, size_t sz1, const double* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, int op);

#undef cv_hal_cmp8u
#define cv_hal_cmp8u wrong_cmp8u
#undef cv_hal_cmp8s
#define cv_hal_cmp8s wrong_cmp8s
#undef cv_hal_cmp16u
#define cv_hal_cmp16u wrong_cmp16u
#undef cv_hal_cmp16s
#define cv_hal_cmp16s wrong_cmp16s
#undef cv_hal_cmp32s
#define cv_hal_cmp32s wrong_cmp32s
#undef cv_hal_cmp32f
#define cv_hal_cmp32f wrong_cmp32f
#undef cv_hal_cmp64f
#define cv_hal_cmp64f wrong_cmp64f

int wrong_mul8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int wrong_mul8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int wrong_mul16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int wrong_mul16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int wrong_mul32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int wrong_mul32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int wrong_mul64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);
int wrong_div8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int wrong_div8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int wrong_div16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int wrong_div16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int wrong_div32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int wrong_div32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int wrong_div64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);
int wrong_recip8u(const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, double scale);
int wrong_recip8s(const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, double scale);
int wrong_recip16u(const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, double scale);
int wrong_recip16s(const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, double scale);
int wrong_recip32s(const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, double scale);
int wrong_recip32f(const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, double scale);
int wrong_recip64f(const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, double scale);

#undef cv_hal_mul8u
#define cv_hal_mul8u wrong_mul8u
#undef cv_hal_mul8s
#define cv_hal_mul8s wrong_mul8s
#undef cv_hal_mul16u
#define cv_hal_mul16u wrong_mul16u
#undef cv_hal_mul16s
#define cv_hal_mul16s wrong_mul16s
#undef cv_hal_mul32s
#define cv_hal_mul32s wrong_mul32s
#undef cv_hal_mul32f
#define cv_hal_mul32f wrong_mul32f
#undef cv_hal_mul64f
#define cv_hal_mul64f wrong_mul64f
#undef cv_hal_div8u
#define cv_hal_div8u wrong_div8u
#undef cv_hal_div8s
#define cv_hal_div8s wrong_div8s
#undef cv_hal_div16u
#define cv_hal_div16u wrong_div16u
#undef cv_hal_div16s
#define cv_hal_div16s wrong_div16s
#undef cv_hal_div32s
#define cv_hal_div32s wrong_div32s
#undef cv_hal_div32f
#define cv_hal_div32f wrong_div32f
#undef cv_hal_div64f
#define cv_hal_div64f wrong_div64f
#undef cv_hal_recip8u
#define cv_hal_recip8u wrong_recip8u
#undef cv_hal_recip8s
#define cv_hal_recip8s wrong_recip8s
#undef cv_hal_recip16u
#define cv_hal_recip16u wrong_recip16u
#undef cv_hal_recip16s
#define cv_hal_recip16s wrong_recip16s
#undef cv_hal_recip32s
#define cv_hal_recip32s wrong_recip32s
#undef cv_hal_recip32f
#define cv_hal_recip32f wrong_recip32f
#undef cv_hal_recip64f
#define cv_hal_recip64f wrong_recip64f

int wrong_addWeighted8u(const uchar* src1, size_t sz1, const uchar* src2, size_t sz2, uchar* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted8s(const schar* src1, size_t sz1, const schar* src2, size_t sz2, schar* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted16u(const ushort* src1, size_t sz1, const ushort* src2, size_t sz2, ushort* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted16s(const short* src1, size_t sz1, const short* src2, size_t sz2, short* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted32s(const int* src1, size_t sz1, const int* src2, size_t sz2, int* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted32f(const float* src1, size_t sz1, const float* src2, size_t sz2, float* dst, size_t sz, int w, int h, const double* scales);
int wrong_addWeighted64f(const double* src1, size_t sz1, const double* src2, size_t sz2, double* dst, size_t sz, int w, int h, const double* scales);

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u wrong_addWeighted8u
#undef cv_hal_addWeighted8s
#define cv_hal_addWeighted8s wrong_addWeighted8s
#undef cv_hal_addWeighted16u
#define cv_hal_addWeighted16u wrong_addWeighted16u
#undef cv_hal_addWeighted16s
#define cv_hal_addWeighted16s wrong_addWeighted16s
#undef cv_hal_addWeighted32s
#define cv_hal_addWeighted32s wrong_addWeighted32s
#undef cv_hal_addWeighted32f
#define cv_hal_addWeighted32f wrong_addWeighted32f
#undef cv_hal_addWeighted64f
#define cv_hal_addWeighted64f wrong_addWeighted64f

#if defined(__cplusplus)
}
#endif

#endif
