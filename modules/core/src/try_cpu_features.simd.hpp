// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <inttypes.h>
#include <signal.h>
#include <stdlib.h>
#include <setjmp.h>

#if defined __ARM_NEON
#include "arm_neon.h"
#endif

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

bool haveCpuFeatureNeon_();
bool haveCpuFeatureFp16_();
bool haveCpuFeatureDotProd_();
bool haveCpuFeatureFp16SIMD_();
bool haveCpuFeatureBf16SIMD_();

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if defined __ARM_NEON || \
    defined __ARM_FP16_FORMAT_IEEE || \
    defined __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || \
    defined __ARM_FEATURE_BF16_VECTOR_ARITHMETIC || \
    defined __ARM_FEATURE_DOTPROD
static bool tryCPUFeature(void (*signal_handler)(int), int (*try_feature)(void), jmp_buf* jmp)
{
    volatile bool flag = true;
    struct sigaction oldact;
    sigaction(SIGILL, NULL, &oldact);
    signal(SIGILL, signal_handler);
    if (setjmp(*jmp) == 0) {
        flag = try_feature() != 0;
    } else {
        flag = false;
    }
    sigaction(SIGILL, &oldact, NULL);
    return flag;
}
#endif

#ifdef __ARM_NEON
static jmp_buf haveNeonCatch;

static void noNeonHandler(int) {
    longjmp(haveNeonCatch, 1);
}

static int tryNeon() {
    uint16_t xbuf[] = {0, 1, 2, 3, 4, 5, 6, 7, (uint16_t)rand()};
    
    uint16x8_t x = vld1q_u16(xbuf);
    uint16x8_t y = vaddq_u16(x, x);
    uint16_t ybuf[8];
    vst1q_u16(ybuf, y);
    return (int)(ybuf[0] + ybuf[1] + ybuf[2] + ybuf[3] +
                 ybuf[4] + ybuf[5] + ybuf[6] + ybuf[7]);
}

bool haveCpuFeatureNeon_()
{
    return tryCPUFeature(noNeonHandler, tryNeon, &haveNeonCatch);
}
#else
bool haveCpuFeatureNeon_()
{
    return false;
}
#endif

#ifdef __ARM_FP16_FORMAT_IEEE
static jmp_buf haveFp16Catch;

static void noFp16Handler(int) {
    longjmp(haveFp16Catch, 1);
}

static int tryFp16() {
    int16_t xbuf[] = {0, 1, 2, 3, 4, 5, 6, 7, (int16_t)rand()};
    int16x8_t x = vld1q_s16(xbuf);
    float16x8_t xf = vcvtq_f16_s16(x);
    float32x4_t y0 = vcvt_f32_f16(vget_low_f16(xf));
    float32x4_t y1 = vcvt_f32_f16(vget_high_f16(xf));
    y0 = vaddq_f32(y0, y1);
    float ybuf[4];
    vst1q_f32(ybuf, y0);
    return (int)(ybuf[0] + ybuf[1] + ybuf[2] + ybuf[3]);
}

bool haveCpuFeatureFp16_()
{
    return tryCPUFeature(noFp16Handler, tryFp16, &haveFp16Catch);
}
#else
bool haveCpuFeatureFp16_()
{
    return false;
}
#endif

#ifdef __ARM_FEATURE_DOTPROD
static jmp_buf haveDotProdCatch;

static void noDotProdHandler(int) {
    longjmp(haveDotProdCatch, 1);
}

static int tryDotProd() {
    uint8_t xbuf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, (uint8_t)(rand()%16)};
    
    uint8x16_t x = vld1q_u8(xbuf);
    uint32x4_t y = vdotq_u32(vdupq_n_u32(0u), x, x);
    uint32_t ybuf[4];
    vst1q_u32(ybuf, y);
    return (int)(ybuf[0] + ybuf[1] + ybuf[2] + ybuf[3]);
}

bool haveCpuFeatureDotProd_()
{
    return tryCPUFeature(noDotProdHandler, tryDotProd, &haveDotProdCatch);
}
#else
bool haveCpuFeatureDotProd_()
{
    return false;
}
#endif

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static jmp_buf haveFp16SIMDCatch;

static void noFp16SIMDHandler(int) {
    longjmp(haveFp16SIMDCatch, 1);
}

static int tryFp16SIMD() {
    float abuf[] = {(float)rand(), (float)rand(), (float)rand(), (float)rand()};
    float16x4_t x_ = vcvt_f16_f32(vld1q_f32(abuf));
    float16x8_t x = vcombine_f16(x_, x_);
    x = vfmaq_laneq_f16(x, x, x, 7);
    vst1q_f32(abuf, vcvt_f32_f16(vget_low_f16(x)));
    return (int)(abuf[0] + abuf[1] + abuf[2] + abuf[3]);
}

bool haveCpuFeatureFp16SIMD_()
{
    return tryCPUFeature(noFp16SIMDHandler, tryFp16SIMD, &haveFp16SIMDCatch);
}
#else
bool haveCpuFeatureFp16SIMD_()
{
    return false;
}
#endif

#if defined __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
static jmp_buf haveBf16SIMDCatch;

static void noBf16SIMDHandler(int) {
    longjmp(haveBf16SIMDCatch, 1);
}

static int tryBf16SIMD() {
    float abuf[] = {(float)(rand()%100), (float)(rand()%100), (float)(rand()%100), (float)(rand()%100)};
    bfloat16x4_t x_ = vcvt_bf16_f32(vld1q_f32(abuf));
    bfloat16x8_t x = vcombine_bf16(x_, x_);
    float32x4_t y = vbfdotq_f32(vdupq_n_f32(0.f), x, x);
    vst1q_f32(abuf, y);
    return (int)(abuf[0] + abuf[1] + abuf[2] + abuf[3]);
}

bool haveCpuFeatureBf16SIMD_()
{
    return tryCPUFeature(noBf16SIMDHandler, tryBf16SIMD, &haveBf16SIMDCatch);
}
#else
bool haveCpuFeatureBf16SIMD_()
{
    return false;
}
#endif

#endif

CV_CPU_OPTIMIZATION_NAMESPACE_END
}
