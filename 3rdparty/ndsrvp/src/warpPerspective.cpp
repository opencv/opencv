// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

int warpPerspectiveBlocklineNN(const double *M, short* xy, double X0, double Y0, double W0, int bw)
{
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

    return CV_HAL_ERROR_OK;
}

int warpPerspectiveBlockline(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw)
{
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
        alpha[x1] = (short)((Y & INTER_MASK) * INTER_TAB_SIZE + (X & INTER_MASK));
    }

    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
