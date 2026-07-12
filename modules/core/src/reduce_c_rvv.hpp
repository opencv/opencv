// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace reduce_c_rvv
{

// Optimized ReduceC support in this backend:
//
// | Input -> output type/channel       | SUM | AVG | MIN | MAX | SUM2 |
// |------------------------------------|:---:|:---:|:---:|:---:|:----:|
// | 8UC1/C3/C4 -> 8UC1/C3/C4          |  -  |  -  |  x  |  x  |  -   |
// | 8UC1/C3/C4 -> 32SC1/C3/C4         |  x  |  x  |  -  |  -  |  x   |
// | 8UC1/C3/C4 -> 32FC1/C3/C4         |  x  |  x  |  -  |  -  |  x   |
// | 8UC1/C3/C4 -> 64FC1/C3/C4         |  -  |  -  |  -  |  -  |  x   |
// | 16UC1/C3/C4 -> 16UC1/C3/C4        |  -  |  -  |  x  |  x  |  -   |
// | 16UC1 -> 32FC1                     |  x  |  x  |  -  |  -  |  x   |
// | 16UC3/C4 -> 32FC3/C4              |  x  |  x  |  -  |  -  |  -   |
// | 16UC1 -> 64FC1                     |  -  |  -  |  -  |  -  |  x*  |
// | 16UC3/C4 -> 64FC3/C4              |  -  |  -  |  -  |  -  |  -   |
// | 16SC1/C3/C4 -> 16SC1/C3/C4        |  -  |  -  |  x  |  x  |  -   |
// | 16SC1 -> 32FC1                     |  x  |  x  |  -  |  -  |  x   |
// | 16SC3/C4 -> 32FC3/C4              |  x  |  x  |  -  |  -  |  -   |
// | 16SC1 -> 64FC1                     |  -  |  -  |  -  |  -  |  x*  |
// | 16SC3/C4 -> 64FC3/C4              |  -  |  -  |  -  |  -  |  -   |
// | 32FC1/C3/C4 -> 32FC1/C3/C4        |  x  |  x  |  x  |  x  |  x   |
// | 32FC1 -> 64FC1                     | x*  | x*  |  -  |  -  |  x*  |
// | 32FC3/C4 -> 64FC3/C4              | x*  | x*  |  -  |  -  |  -   |
// | 64FC1 -> 64FC1                     | x*  | x*  | x*  | x*  |  x*  |
// | 64FC3/C4 -> 64FC3/C4              | x*  | x*  |  -  |  -  |  -   |
//
// 'x' in SUM/AVG denotes the existing shared universal-intrinsics kernel; 'x'
// in MIN/MAX/SUM2 denotes a native RVV kernel. '*' requires
// CV_SIMD_SCALABLE_64F. For legal MIN/MAX/SUM2 combinations marked '-', and for
// other channel counts, dispatch uses the shared generic fallback.

template<bool isMax>
static void minMax8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            uchar result = isMax ? 0 : UCHAR_MAX;
            int x = 0;
            const int vlmax = __riscv_vsetvlmax_e8m8();
            vuint8m8_t acc = __riscv_vmv_v_x_u8m8(result, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m8(cols - x);
                vuint8m8_t v = __riscv_vle8_v_u8m8(src + x, vl);
                acc = isMax ? __riscv_vmaxu_tu(acc, acc, v, vl)
                            : __riscv_vminu_tu(acc, acc, v, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(result, __riscv_vsetvlmax_e8m1());
            vuint8m1_t reduced = isMax ? __riscv_vredmaxu(acc, seed, vlmax)
                                       : __riscv_vredminu(acc, seed, vlmax);
            result = (uchar)__riscv_vmv_x(reduced);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dst[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            const ushort initial = isMax ? 0 : USHRT_MAX;
            const int vlmax = __riscv_vsetvlmax_e16m8();
            vuint16m8_t acc = __riscv_vmv_v_x_u16m8(initial, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m8(cols - x);
                vuint16m8_t v = __riscv_vle16_v_u16m8(src + x, vl);
                acc = isMax ? __riscv_vmaxu_tu(acc, acc, v, vl)
                            : __riscv_vminu_tu(acc, acc, v, vl);
                x += vl;
            }
            vuint16m1_t seed = __riscv_vmv_s_x_u16m1(initial, __riscv_vsetvlmax_e16m1());
            vuint16m1_t reduced = isMax ? __riscv_vredmaxu(acc, seed, vlmax)
                                        : __riscv_vredminu(acc, seed, vlmax);
            dstmat.ptr<ushort>(y)[0] = (ushort)__riscv_vmv_x(reduced);
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            const short initial = isMax ? SHRT_MIN : SHRT_MAX;
            const int vlmax = __riscv_vsetvlmax_e16m8();
            vint16m8_t acc = __riscv_vmv_v_x_i16m8(initial, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m8(cols - x);
                vint16m8_t v = __riscv_vle16_v_i16m8(src + x, vl);
                acc = isMax ? __riscv_vmax_tu(acc, acc, v, vl)
                            : __riscv_vmin_tu(acc, acc, v, vl);
                x += vl;
            }
            vint16m1_t seed = __riscv_vmv_s_x_i16m1(initial, __riscv_vsetvlmax_e16m1());
            vint16m1_t reduced = isMax ? __riscv_vredmax(acc, seed, vlmax)
                                       : __riscv_vredmin(acc, seed, vlmax);
            dstmat.ptr<short>(y)[0] = (short)__riscv_vmv_x(reduced);
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const ushort initial = isMax ? 0 : USHRT_MAX;
    const int vlmax = __riscv_vsetvlmax_e16m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            ushort* dst = dstmat.ptr<ushort>(y);
            vuint16m2_t acc0 = __riscv_vmv_v_x_u16m2(initial, vlmax);
            vuint16m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vuint16m2x4_t v = __riscv_vlseg4e16_v_u16m2x4(src + x * 4, vl);
                vuint16m2_t v0 = __riscv_vget_v_u16m2x4_u16m2(v, 0);
                vuint16m2_t v1 = __riscv_vget_v_u16m2x4_u16m2(v, 1);
                vuint16m2_t v2 = __riscv_vget_v_u16m2x4_u16m2(v, 2);
                vuint16m2_t v3 = __riscv_vget_v_u16m2x4_u16m2(v, 3);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl)
                             : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl)
                             : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl)
                             : __riscv_vminu_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vmaxu_tu(acc3, acc3, v3, vl)
                             : __riscv_vminu_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vuint16m1_t seed = __riscv_vmv_s_x_u16m1(initial, __riscv_vsetvlmax_e16m1());
            dst[0] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax) : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax) : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax) : __riscv_vredminu(acc2, seed, vlmax));
            dst[3] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc3, seed, vlmax) : __riscv_vredminu(acc3, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const short initial = isMax ? SHRT_MIN : SHRT_MAX;
    const int vlmax = __riscv_vsetvlmax_e16m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            short* dst = dstmat.ptr<short>(y);
            vint16m2_t acc0 = __riscv_vmv_v_x_i16m2(initial, vlmax);
            vint16m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vint16m2x4_t v = __riscv_vlseg4e16_v_i16m2x4(src + x * 4, vl);
                vint16m2_t v0 = __riscv_vget_v_i16m2x4_i16m2(v, 0);
                vint16m2_t v1 = __riscv_vget_v_i16m2x4_i16m2(v, 1);
                vint16m2_t v2 = __riscv_vget_v_i16m2x4_i16m2(v, 2);
                vint16m2_t v3 = __riscv_vget_v_i16m2x4_i16m2(v, 3);
                acc0 = isMax ? __riscv_vmax_tu(acc0, acc0, v0, vl)
                             : __riscv_vmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmax_tu(acc1, acc1, v1, vl)
                             : __riscv_vmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmax_tu(acc2, acc2, v2, vl)
                             : __riscv_vmin_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vmax_tu(acc3, acc3, v3, vl)
                             : __riscv_vmin_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vint16m1_t seed = __riscv_vmv_s_x_i16m1(initial, __riscv_vsetvlmax_e16m1());
            dst[0] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc0, seed, vlmax) : __riscv_vredmin(acc0, seed, vlmax));
            dst[1] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc1, seed, vlmax) : __riscv_vredmin(acc1, seed, vlmax));
            dst[2] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc2, seed, vlmax) : __riscv_vredmin(acc2, seed, vlmax));
            dst[3] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc3, seed, vlmax) : __riscv_vredmin(acc3, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const ushort initial = isMax ? 0 : USHRT_MAX;
    const int vlmax = __riscv_vsetvlmax_e16m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            ushort* dst = dstmat.ptr<ushort>(y);
            vuint16m2_t acc0 = __riscv_vmv_v_x_u16m2(initial, vlmax);
            vuint16m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vuint16m2x3_t v = __riscv_vlseg3e16_v_u16m2x3(src + x * 3, vl);
                vuint16m2_t v0 = __riscv_vget_v_u16m2x3_u16m2(v, 0);
                vuint16m2_t v1 = __riscv_vget_v_u16m2x3_u16m2(v, 1);
                vuint16m2_t v2 = __riscv_vget_v_u16m2x3_u16m2(v, 2);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl) : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl) : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl) : __riscv_vminu_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vuint16m1_t seed = __riscv_vmv_s_x_u16m1(initial, __riscv_vsetvlmax_e16m1());
            dst[0] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax) : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax) : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (ushort)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax) : __riscv_vredminu(acc2, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax16sC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const short initial = isMax ? SHRT_MIN : SHRT_MAX;
    const int vlmax = __riscv_vsetvlmax_e16m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            short* dst = dstmat.ptr<short>(y);
            vint16m2_t acc0 = __riscv_vmv_v_x_i16m2(initial, vlmax);
            vint16m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vint16m2x3_t v = __riscv_vlseg3e16_v_i16m2x3(src + x * 3, vl);
                vint16m2_t v0 = __riscv_vget_v_i16m2x3_i16m2(v, 0);
                vint16m2_t v1 = __riscv_vget_v_i16m2x3_i16m2(v, 1);
                vint16m2_t v2 = __riscv_vget_v_i16m2x3_i16m2(v, 2);
                acc0 = isMax ? __riscv_vmax_tu(acc0, acc0, v0, vl) : __riscv_vmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmax_tu(acc1, acc1, v1, vl) : __riscv_vmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmax_tu(acc2, acc2, v2, vl) : __riscv_vmin_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vint16m1_t seed = __riscv_vmv_s_x_i16m1(initial, __riscv_vsetvlmax_e16m1());
            dst[0] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc0, seed, vlmax) : __riscv_vredmin(acc0, seed, vlmax));
            dst[1] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc1, seed, vlmax) : __riscv_vredmin(acc1, seed, vlmax));
            dst[2] = (short)__riscv_vmv_x(isMax ? __riscv_vredmax(acc2, seed, vlmax) : __riscv_vredmin(acc2, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const int vlmax = __riscv_vsetvlmax_e8m1();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            vuint8m1_t acc0 = __riscv_vmv_v_x_u8m1(initial, vlmax);
            vuint8m1_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x3_t v = __riscv_vlseg3e8_v_u8m1x3(src + x * 3, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x3_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x3_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x3_u8m1(v, 2);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl)
                             : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl)
                             : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl)
                             : __riscv_vminu_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(initial, vlmax);
            dst[0] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax)
                                                 : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax)
                                                 : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax)
                                                 : __riscv_vredminu(acc2, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const uchar initial = isMax ? 0 : UCHAR_MAX;
    const int vlmax = __riscv_vsetvlmax_e8m1();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            uchar* dst = dstmat.ptr<uchar>(y);
            vuint8m1_t acc0 = __riscv_vmv_v_x_u8m1(initial, vlmax);
            vuint8m1_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x4_t v = __riscv_vlseg4e8_v_u8m1x4(src + x * 4, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x4_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x4_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x4_u8m1(v, 2);
                vuint8m1_t v3 = __riscv_vget_v_u8m1x4_u8m1(v, 3);
                acc0 = isMax ? __riscv_vmaxu_tu(acc0, acc0, v0, vl)
                             : __riscv_vminu_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vmaxu_tu(acc1, acc1, v1, vl)
                             : __riscv_vminu_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vmaxu_tu(acc2, acc2, v2, vl)
                             : __riscv_vminu_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vmaxu_tu(acc3, acc3, v3, vl)
                             : __riscv_vminu_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vuint8m1_t seed = __riscv_vmv_s_x_u8m1(initial, vlmax);
            dst[0] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc0, seed, vlmax)
                                                 : __riscv_vredminu(acc0, seed, vlmax));
            dst[1] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc1, seed, vlmax)
                                                 : __riscv_vredminu(acc1, seed, vlmax));
            dst[2] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc2, seed, vlmax)
                                                 : __riscv_vredminu(acc2, seed, vlmax));
            dst[3] = (uchar)__riscv_vmv_x(isMax ? __riscv_vredmaxu(acc3, seed, vlmax)
                                                 : __riscv_vredminu(acc3, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = src[0];
            int x = 0;
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(result, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m8(cols - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(src + x, vl);
                acc = isMax ? __riscv_vfmax_tu(acc, acc, v, vl)
                            : __riscv_vfmin_tu(acc, acc, v, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(result, __riscv_vsetvlmax_e32m1());
            vfloat32m1_t reduced = isMax ? __riscv_vfredmax(acc, seed, vlmax)
                                         : __riscv_vfredmin(acc, seed, vlmax);
            result = __riscv_vfmv_f(reduced);
            for (; x < cols; x++)
                result = reduceScalarMinMax<isMax>(result, src[x]);
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(initial, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x3_t v = __riscv_vlseg3e32_v_f32m2x3(src + x * 3, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x3_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x3_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x3_f32m2(v, 2);
                acc0 = isMax ? __riscv_vfmax_tu(acc0, acc0, v0, vl)
                             : __riscv_vfmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vfmax_tu(acc1, acc1, v1, vl)
                             : __riscv_vfmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vfmax_tu(acc2, acc2, v2, vl)
                             : __riscv_vfmin_tu(acc2, acc2, v2, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(initial, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc0, seed, vlmax)
                                          : __riscv_vfredmin(acc0, seed, vlmax));
            dst[1] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc1, seed, vlmax)
                                          : __riscv_vfredmin(acc1, seed, vlmax));
            dst[2] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc2, seed, vlmax)
                                          : __riscv_vfredmin(acc2, seed, vlmax));
        }
    });
    v_cleanup();
}

template<bool isMax>
static void minMax32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const float initial = isMax ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max();
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(initial, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x4_t v = __riscv_vlseg4e32_v_f32m2x4(src + x * 4, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x4_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x4_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x4_f32m2(v, 2);
                vfloat32m2_t v3 = __riscv_vget_v_f32m2x4_f32m2(v, 3);
                acc0 = isMax ? __riscv_vfmax_tu(acc0, acc0, v0, vl)
                             : __riscv_vfmin_tu(acc0, acc0, v0, vl);
                acc1 = isMax ? __riscv_vfmax_tu(acc1, acc1, v1, vl)
                             : __riscv_vfmin_tu(acc1, acc1, v1, vl);
                acc2 = isMax ? __riscv_vfmax_tu(acc2, acc2, v2, vl)
                             : __riscv_vfmin_tu(acc2, acc2, v2, vl);
                acc3 = isMax ? __riscv_vfmax_tu(acc3, acc3, v3, vl)
                             : __riscv_vfmin_tu(acc3, acc3, v3, vl);
                x += vl;
            }
            vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(initial, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc0, seed, vlmax)
                                          : __riscv_vfredmin(acc0, seed, vlmax));
            dst[1] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc1, seed, vlmax)
                                          : __riscv_vfredmin(acc1, seed, vlmax));
            dst[2] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc2, seed, vlmax)
                                          : __riscv_vfredmin(acc2, seed, vlmax));
            dst[3] = __riscv_vfmv_f(isMax ? __riscv_vfredmax(acc3, seed, vlmax)
                                          : __riscv_vfredmin(acc3, seed, vlmax));
        }
    });
    v_cleanup();
}

#if CV_SIMD_SCALABLE_64F
template<bool isMax>
static void minMax64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const double initial = isMax ? std::numeric_limits<double>::lowest()
                                 : std::numeric_limits<double>::max();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const double* src = srcmat.ptr<double>(y);
            const int vlmax = __riscv_vsetvlmax_e64m8();
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(initial, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e64m8(cols - x);
                vfloat64m8_t v = __riscv_vle64_v_f64m8(src + x, vl);
                acc = isMax ? __riscv_vfmax_tu(acc, acc, v, vl)
                            : __riscv_vfmin_tu(acc, acc, v, vl);
                x += vl;
            }
            vfloat64m1_t seed = __riscv_vfmv_s_f_f64m1(initial, __riscv_vsetvlmax_e64m1());
            vfloat64m1_t reduced = isMax ? __riscv_vfredmax(acc, seed, vlmax)
                                         : __riscv_vfredmin(acc, seed, vlmax);
            dstmat.ptr<double>(y)[0] = __riscv_vfmv_f(reduced);
        }
    });
    v_cleanup();
}
#endif

template<typename DT>
static void sum2_8uC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;

    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            uint32_t result = 0;
            int x = 0;
            vuint32m1_t acc = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m4(cols - x);
                vuint8m4_t v = __riscv_vle8_v_u8m4(src + x, vl);
                acc = __riscv_vwredsumu(__riscv_vwmulu(v, v, vl), acc, vl);
                x += vl;
            }
            result = (uint32_t)__riscv_vmv_x(acc);
            for (; x < cols; x++)
                result += (uint32_t)src[x] * src[x];
            dst[0] = (DT)(int32_t)result;
        }
    });
    v_cleanup();
}

static void sum2_16u32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m4(cols - x);
                vuint16m4_t v = __riscv_vle16_v_u16m4(src + x, vl);
                vfloat32m8_t vf = __riscv_vfwcvt_f_xu_v_f32m8(v, vl);
                acc = __riscv_vfmacc_tu(acc, vf, vf, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dstmat.ptr<float>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}

static void sum2_16s32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m4(cols - x);
                vint16m4_t v = __riscv_vle16_v_i16m4(src + x, vl);
                vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v, vl);
                acc = __riscv_vfmacc_tu(acc, vf, vf, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dstmat.ptr<float>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}

#if CV_SIMD_SCALABLE_64F
static void sum2_16u64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            const int vlmax = __riscv_vsetvlmax_e64m8();
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vuint16m2_t v = __riscv_vle16_v_u16m2(src + x, vl);
                vfloat32m4_t vf = __riscv_vfwcvt_f_xu_v_f32m4(v, vl);
                vfloat64m8_t vd = __riscv_vfwcvt_f_f_v_f64m8(vf, vl);
                acc = __riscv_vfmacc_tu(acc, vd, vd, vl);
                x += vl;
            }
            vfloat64m1_t zero = __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1());
            dstmat.ptr<double>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}

static void sum2_16s64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            const int vlmax = __riscv_vsetvlmax_e64m8();
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e16m2(cols - x);
                vint16m2_t v = __riscv_vle16_v_i16m2(src + x, vl);
                vfloat32m4_t vf = __riscv_vfwcvt_f_x_v_f32m4(v, vl);
                vfloat64m8_t vd = __riscv_vfwcvt_f_f_v_f64m8(vf, vl);
                acc = __riscv_vfmacc_tu(acc, vd, vd, vl);
                x += vl;
            }
            vfloat64m1_t zero = __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1());
            dstmat.ptr<double>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}

static void sum2_32f64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            const int vlmax = __riscv_vsetvlmax_e64m8();
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m4(cols - x);
                vfloat32m4_t v = __riscv_vle32_v_f32m4(src + x, vl);
                vfloat64m8_t vd = __riscv_vfwcvt_f_f_v_f64m8(v, vl);
                acc = __riscv_vfmacc_tu(acc, vd, vd, vl);
                x += vl;
            }
            vfloat64m1_t zero = __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1());
            dstmat.ptr<double>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}

static void sum2_64f64fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const double* src = srcmat.ptr<double>(y);
            const int vlmax = __riscv_vsetvlmax_e64m8();
            vfloat64m8_t acc = __riscv_vfmv_v_f_f64m8(0, vlmax);
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e64m8(cols - x);
                vfloat64m8_t v = __riscv_vle64_v_f64m8(src + x, vl);
                acc = __riscv_vfmacc_tu(acc, v, v, vl);
                x += vl;
            }
            vfloat64m1_t zero = __riscv_vfmv_s_f_f64m1(0, __riscv_vsetvlmax_e64m1());
            dstmat.ptr<double>(y)[0] = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
        }
    });
    v_cleanup();
}
#endif

template<typename DT>
static void sum2_8uC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            vuint32m1_t acc0 = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            vuint32m1_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x3_t v = __riscv_vlseg3e8_v_u8m1x3(src + x * 3, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x3_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x3_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x3_u8m1(v, 2);
                acc0 = __riscv_vwredsumu(__riscv_vwmulu(v0, v0, vl), acc0, vl);
                acc1 = __riscv_vwredsumu(__riscv_vwmulu(v1, v1, vl), acc1, vl);
                acc2 = __riscv_vwredsumu(__riscv_vwmulu(v2, v2, vl), acc2, vl);
                x += vl;
            }
            dst[0] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc0);
            dst[1] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc1);
            dst[2] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc2);
        }
    });
    v_cleanup();
}

template<typename DT>
static void sum2_8uC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            DT* dst = dstmat.ptr<DT>(y);
            vuint32m1_t acc0 = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
            vuint32m1_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e8m1(cols - x);
                vuint8m1x4_t v = __riscv_vlseg4e8_v_u8m1x4(src + x * 4, vl);
                vuint8m1_t v0 = __riscv_vget_v_u8m1x4_u8m1(v, 0);
                vuint8m1_t v1 = __riscv_vget_v_u8m1x4_u8m1(v, 1);
                vuint8m1_t v2 = __riscv_vget_v_u8m1x4_u8m1(v, 2);
                vuint8m1_t v3 = __riscv_vget_v_u8m1x4_u8m1(v, 3);
                acc0 = __riscv_vwredsumu(__riscv_vwmulu(v0, v0, vl), acc0, vl);
                acc1 = __riscv_vwredsumu(__riscv_vwmulu(v1, v1, vl), acc1, vl);
                acc2 = __riscv_vwredsumu(__riscv_vwmulu(v2, v2, vl), acc2, vl);
                acc3 = __riscv_vwredsumu(__riscv_vwmulu(v3, v3, vl), acc3, vl);
                x += vl;
            }
            dst[0] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc0);
            dst[1] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc1);
            dst[2] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc2);
            dst[3] = (DT)(int32_t)(uint32_t)__riscv_vmv_x(acc3);
        }
    });
    v_cleanup();
}

static void sum2_32fC1(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float result = 0;
            int x = 0;
            const int vlmax = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0, vlmax);
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m8(cols - x);
                vfloat32m8_t v = __riscv_vle32_v_f32m8(src + x, vl);
                acc = __riscv_vfmacc_tu(acc, v, v, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            result = __riscv_vfmv_f(__riscv_vfredusum(acc, zero, vlmax));
            for (; x < cols; x++)
                result += src[x] * src[x];
            dstmat.ptr<float>(y)[0] = result;
        }
    });
    v_cleanup();
}

static void sum2_32fC3(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x3_t v = __riscv_vlseg3e32_v_f32m2x3(src + x * 3, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x3_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x3_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x3_f32m2(v, 2);
                acc0 = __riscv_vfmacc_tu(acc0, v0, v0, vl);
                acc1 = __riscv_vfmacc_tu(acc1, v1, v1, vl);
                acc2 = __riscv_vfmacc_tu(acc2, v2, v2, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(__riscv_vfredusum(acc0, zero, vlmax));
            dst[1] = __riscv_vfmv_f(__riscv_vfredusum(acc1, zero, vlmax));
            dst[2] = __riscv_vfmv_f(__riscv_vfredusum(acc2, zero, vlmax));
        }
    });
    v_cleanup();
}

static void sum2_32fC4(const Mat& srcmat, Mat& dstmat)
{
    const int cols = srcmat.cols;
    const int vlmax = __riscv_vsetvlmax_e32m2();
    parallel_for_(Range(0, srcmat.rows), [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);
            vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0, vlmax);
            vfloat32m2_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
            int x = 0;
            for (; x < cols; )
            {
                const int vl = __riscv_vsetvl_e32m2(cols - x);
                vfloat32m2x4_t v = __riscv_vlseg4e32_v_f32m2x4(src + x * 4, vl);
                vfloat32m2_t v0 = __riscv_vget_v_f32m2x4_f32m2(v, 0);
                vfloat32m2_t v1 = __riscv_vget_v_f32m2x4_f32m2(v, 1);
                vfloat32m2_t v2 = __riscv_vget_v_f32m2x4_f32m2(v, 2);
                vfloat32m2_t v3 = __riscv_vget_v_f32m2x4_f32m2(v, 3);
                acc0 = __riscv_vfmacc_tu(acc0, v0, v0, vl);
                acc1 = __riscv_vfmacc_tu(acc1, v1, v1, vl);
                acc2 = __riscv_vfmacc_tu(acc2, v2, v2, vl);
                acc3 = __riscv_vfmacc_tu(acc3, v3, v3, vl);
                x += vl;
            }
            vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0, __riscv_vsetvlmax_e32m1());
            dst[0] = __riscv_vfmv_f(__riscv_vfredusum(acc0, zero, vlmax));
            dst[1] = __riscv_vfmv_f(__riscv_vfredusum(acc1, zero, vlmax));
            dst[2] = __riscv_vfmv_f(__riscv_vfredusum(acc2, zero, vlmax));
            dst[3] = __riscv_vfmv_f(__riscv_vfredusum(acc3, zero, vlmax));
        }
    });
    v_cleanup();
}

} // namespace reduce_c_rvv
