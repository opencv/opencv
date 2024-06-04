// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "ndsrvp_hal.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/hal/interface.h"

namespace cv {

namespace ndsrvp {

class WarpPerspectiveInvoker : public ParallelLoopBody {
public:
    WarpPerspectiveInvoker(const Mat& _src, Mat& _dst, const double* _M, int _interpolation,
        int _borderType, const Scalar& _borderValue)
        : ParallelLoopBody()
        , src(_src)
        , dst(_dst)
        , M(_M)
        , interpolation(_interpolation)
        , borderType(_borderType)
        , borderValue(_borderValue)
    {
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        const int BLOCK_SZ = 32;
        short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];
        int x, y, y1, width = dst.cols, height = dst.rows;

        int bh0 = std::min(BLOCK_SZ / 2, height);
        int bw0 = std::min(BLOCK_SZ * BLOCK_SZ / bh0, width);
        bh0 = std::min(BLOCK_SZ * BLOCK_SZ / bw0, height);

        for (y = range.start; y < range.end; y += bh0) {
            for (x = 0; x < width; x += bw0) {
                int bw = std::min(bw0, width - x);
                int bh = std::min(bh0, range.end - y); // height

                Mat _XY(bh, bw, CV_16SC2, XY);
                Mat dpart(dst, Rect(x, y, bw, bh));

                for (y1 = 0; y1 < bh; y1++) {
                    short* xy = XY + y1 * bw * 2;
                    double X0 = M[0] * x + M[1] * (y + y1) + M[2];
                    double Y0 = M[3] * x + M[4] * (y + y1) + M[5];
                    double W0 = M[6] * x + M[7] * (y + y1) + M[8];

                    if (interpolation == CV_HAL_INTER_NEAREST) {
                        int x1 = 0;

                        for (; x1 < bw; x1 += 2) {
                            double W1 = W0 + M[6] * x1, W2 = W1 + M[6];
                            W1 = W1 ? 1. / W1 : 0;
                            W2 = W2 ? 1. / W2 : 0;
                            double fX1 = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1) * W1));
                            double fX2 = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * (x1 + 1)) * W2));
                            double fY1 = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1) * W1));
                            double fY2 = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * (x1 + 1)) * W2));

                            int32x2_t vX = {saturate_cast<int>(fX1), saturate_cast<int>(fX2)};
                            int32x2_t vY = {saturate_cast<int>(fY1), saturate_cast<int>(fY2)};

                            vX = __nds__v_sclip32(vX, 15);
                            vY = __nds__v_sclip32(vY, 15);

                            *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vY, (unsigned long)vX);
                        }

                        for (; x1 < bw; x1++) {
                            double W = W0 + M[6] * x1;
                            W = W ? 1. / W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
                            int X = saturate_cast<int>(fX);
                            int Y = saturate_cast<int>(fY);

                            xy[x1 * 2] = saturate_cast<short>(X);
                            xy[x1 * 2 + 1] = saturate_cast<short>(Y);
                        }
                    } else {
                        short* alpha = A + y1 * bw;
                        int x1 = 0;

                        const int INTER_MASK = INTER_TAB_SIZE - 1;
                        const uint32x2_t vmask = { INTER_MASK, INTER_MASK };
                        for (; x1 < bw; x1 += 2) {
                            double W1 = W0 + M[6] * x1, W2 = W1 + M[6];
                            W1 = W1 ? INTER_TAB_SIZE / W1 : 0;
                            W2 = W2 ? INTER_TAB_SIZE / W2 : 0;
                            double fX1 = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1) * W1));
                            double fX2 = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * (x1 + 1)) * W2));
                            double fY1 = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1) * W1));
                            double fY2 = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * (x1 + 1)) * W2));

                            int32x2_t vX = {saturate_cast<int>(fX1), saturate_cast<int>(fX2)};
                            int32x2_t vY = {saturate_cast<int>(fY1), saturate_cast<int>(fY2)};

                            int32x2_t vx = __nds__v_sclip32(__nds__v_sra32(vX, INTER_BITS), 15);
                            int32x2_t vy = __nds__v_sclip32(__nds__v_sra32(vY, INTER_BITS), 15);

                            *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vy, (unsigned long)vx);

                            uint32x2_t valpha = __nds__v_uadd32(__nds__v_sll32((uint32x2_t)(vY & vmask), INTER_BITS), (uint32x2_t)(vX & vmask));
                            *(int16x2_t*)(alpha + x1) = (int16x2_t) { (short)(valpha[0]), (short)(valpha[1]) };
                        }

                        for (; x1 < bw; x1++) {
                            double W = W0 + M[6] * x1;
                            W = W ? INTER_TAB_SIZE / W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
                            int X = saturate_cast<int>(fX);
                            int Y = saturate_cast<int>(fY);

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
    const double* M;
    int interpolation, borderType;
    Scalar borderValue;
};

int warpPerspective(int src_type,
    const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
    const double M[9], int interpolation, int borderType, const double borderValue[4])
{
    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(Size(dst_width, dst_height), src_type, dst_data, dst_step);

    Range range(0, dst.rows);
    WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType, Scalar(borderValue[0], borderValue[1], borderValue[2], borderValue[3]));
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
