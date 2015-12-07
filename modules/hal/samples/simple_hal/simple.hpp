#ifndef _SIMPLE_HPP_INCLUDED_
#define _SIMPLE_HPP_INCLUDED_

#include "opencv2/hal/interface.hpp"

int slow_and8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
int slow_or8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
int slow_xor8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
int slow_not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);

#undef hal_and8u
#define hal_and8u slow_and8u
#undef hal_or8u
#define hal_or8u slow_or8u
#undef hal_xor8u
#define hal_xor8u slow_xor8u
#undef hal_not8u
#define hal_not8u slow_not8u

#endif
