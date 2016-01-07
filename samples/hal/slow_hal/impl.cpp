#include "impl.hpp"

int slow_and8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    for(; height--; src1 = src1 + step1, src2 = src2 + step2, dst = dst + step)
        for(int x = 0 ; x < width; x++ )
            dst[x] = src1[x] & src2[x];
    return CV_HAL_ERROR_OK;
}

int slow_or8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    for(; height--; src1 = src1 + step1, src2 = src2 + step2, dst = dst + step)
        for(int x = 0 ; x < width; x++ )
            dst[x] = src1[x] | src2[x];
    return CV_HAL_ERROR_OK;
}

int slow_xor8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    for(; height--; src1 = src1 + step1, src2 = src2 + step2, dst = dst + step)
        for(int x = 0 ; x < width; x++ )
            dst[x] = src1[x] ^ src2[x];
    return CV_HAL_ERROR_OK;
}

int slow_not8u(const uchar* src1, size_t step1, uchar* dst, size_t step, int width, int height)
{
    for(; height--; src1 = src1 + step1, dst = dst + step)
        for(int x = 0 ; x < width; x++ )
            dst[x] = ~src1[x];
    return CV_HAL_ERROR_OK;
}
