// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

namespace reduce_c_rvv
{

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
