// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "ndsrvp_hal.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/hal/interface.h"

namespace cv {

namespace ndsrvp {

class WarpAffineInvoker : public ParallelLoopBody {
public:
    WarpAffineInvoker(const Mat& _src, Mat& _dst, int _interpolation, int _borderType,
        const Scalar& _borderValue, int* _adelta, int* _bdelta, const double* _M)
        : ParallelLoopBody()
        , src(_src)
        , dst(_dst)
        , interpolation(_interpolation)
        , borderType(_borderType)
        , borderValue(_borderValue)
        , adelta(_adelta)
        , bdelta(_bdelta)
        , M(_M)
    {
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        const int BLOCK_SZ = 64;
        AutoBuffer<short, 0> __XY(BLOCK_SZ * BLOCK_SZ * 2), __A(BLOCK_SZ * BLOCK_SZ);
        short *XY = __XY.data(), *A = __A.data();
        const int AB_BITS = MAX(10, (int)INTER_BITS);
        const int AB_SCALE = 1 << AB_BITS;
        int round_delta = interpolation == CV_HAL_INTER_NEAREST ? AB_SCALE / 2 : AB_SCALE / INTER_TAB_SIZE / 2, x, y, x1, y1;

        int bh0 = std::min(BLOCK_SZ / 2, dst.rows);
        int bw0 = std::min(BLOCK_SZ * BLOCK_SZ / bh0, dst.cols);
        bh0 = std::min(BLOCK_SZ * BLOCK_SZ / bw0, dst.rows);

        for (y = range.start; y < range.end; y += bh0) {
            for (x = 0; x < dst.cols; x += bw0) {
                int bw = std::min(bw0, dst.cols - x);
                int bh = std::min(bh0, range.end - y);

                Mat _XY(bh, bw, CV_16SC2, XY);
                Mat dpart(dst, Rect(x, y, bw, bh));

                for (y1 = 0; y1 < bh; y1++) {
                    short* xy = XY + y1 * bw * 2;
                    int X0 = saturate_cast<int>((M[1] * (y + y1) + M[2]) * AB_SCALE) + round_delta;
                    int Y0 = saturate_cast<int>((M[4] * (y + y1) + M[5]) * AB_SCALE) + round_delta;

                    if (interpolation == CV_HAL_INTER_NEAREST) {
                        x1 = 0;

                        for (; x1 < bw; x1 += 2) {
                            int32x2_t vX = { X0 + adelta[x + x1], X0 + adelta[x + x1 + 1] };
                            int32x2_t vY = { Y0 + bdelta[x + x1], Y0 + bdelta[x + x1 + 1] };

                            vX = __nds__v_sclip32(__nds__v_sra32(vX, AB_BITS), 15);
                            vY = __nds__v_sclip32(__nds__v_sra32(vY, AB_BITS), 15);

                            *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vY, (unsigned long)vX);
                        }

                        for (; x1 < bw; x1++) {
                            int X = (X0 + adelta[x + x1]) >> AB_BITS;
                            int Y = (Y0 + bdelta[x + x1]) >> AB_BITS;
                            xy[x1 * 2] = saturate_cast<short>(X);
                            xy[x1 * 2 + 1] = saturate_cast<short>(Y);
                        }
                    } else {
                        short* alpha = A + y1 * bw;
                        x1 = 0;

                        const int INTER_MASK = INTER_TAB_SIZE - 1;
                        const uint32x2_t vmask = { INTER_MASK, INTER_MASK };
                        for (; x1 < bw; x1 += 2) {
                            int32x2_t vX = { X0 + adelta[x + x1], X0 + adelta[x + x1 + 1] };
                            int32x2_t vY = { Y0 + bdelta[x + x1], Y0 + bdelta[x + x1 + 1] };
                            vX = __nds__v_sra32(vX, (AB_BITS - INTER_BITS));
                            vY = __nds__v_sra32(vY, (AB_BITS - INTER_BITS));

                            int32x2_t vx = __nds__v_sclip32(__nds__v_sra32(vX, INTER_BITS), 15);
                            int32x2_t vy = __nds__v_sclip32(__nds__v_sra32(vY, INTER_BITS), 15);

                            *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vy, (unsigned long)vx);

                            uint32x2_t valpha = __nds__v_uadd32(__nds__v_sll32((uint32x2_t)(vY & vmask), INTER_BITS), (uint32x2_t)(vX & vmask));
                            *(int16x2_t*)(alpha + x1) = (int16x2_t) { (short)(valpha[0]), (short)(valpha[1]) };
                        }

                        for (; x1 < bw; x1++) {
                            int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                            int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                            xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
                            xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
                            alpha[x1] = (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (X & (INTER_TAB_SIZE - 1)));
                        }
                    }
                }

                if (interpolation == CV_HAL_INTER_NEAREST)
                    remap(src, dpart, _XY, Mat(), interpolation, borderType, borderValue);
                else {
                    Mat _matA(bh, bw, CV_16U, A);
                    remap(src, dpart, _XY, _matA, interpolation, borderType, borderValue);
                }
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int interpolation, borderType;
    Scalar borderValue;
    int *adelta, *bdelta;
    const double* M;
};

int warpAffine(int src_type,
    const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
    const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(Size(dst_width, dst_height), src_type, dst_data, dst_step);

    int x;
    AutoBuffer<int> _abdelta(dst.cols * 2);
    int *adelta = &_abdelta[0], *bdelta = adelta + dst.cols;
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    const int AB_SCALE = 1 << AB_BITS;

    for (x = 0; x < dst.cols; x++) {
        adelta[x] = saturate_cast<int>(M[0] * x * AB_SCALE);
        bdelta[x] = saturate_cast<int>(M[3] * x * AB_SCALE);
    }

    Range range(0, dst.rows);
    WarpAffineInvoker invoker(src, dst, interpolation, borderType,
        Scalar(borderValue[0], borderValue[1], borderValue[2], borderValue[3]),
        adelta, bdelta, M);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
