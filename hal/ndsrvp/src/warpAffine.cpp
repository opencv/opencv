// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ndsrvp_hal.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

int warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw)
{
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;

    for (; x1 < bw; x1 += 2) {
        int32x2_t vX = { X0 + adelta[x1], X0 + adelta[x1 + 1] };
        int32x2_t vY = { Y0 + bdelta[x1], Y0 + bdelta[x1 + 1] };

        vX = __nds__v_sclip32(__nds__v_sra32(vX, AB_BITS), 15);
        vY = __nds__v_sclip32(__nds__v_sra32(vY, AB_BITS), 15);

        *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vY, (unsigned long)vX);
    }

    for (; x1 < bw; x1++) {
        int X = X0 + adelta[x1];
        int Y = Y0 + bdelta[x1];
        xy[x1 * 2] = saturate_cast<short>(X);
        xy[x1 * 2 + 1] = saturate_cast<short>(Y);
    }

    return CV_HAL_ERROR_OK;
}

int warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw)
{
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;

    const int INTER_MASK = INTER_TAB_SIZE - 1;
    const uint32x2_t vmask = { INTER_MASK, INTER_MASK };
    for (; x1 < bw; x1 += 2) {
        int32x2_t vX = { X0 + adelta[x1], X0 + adelta[x1 + 1] };
        int32x2_t vY = { Y0 + bdelta[x1], Y0 + bdelta[x1 + 1] };
        vX = __nds__v_sra32(vX, (AB_BITS - INTER_BITS));
        vY = __nds__v_sra32(vY, (AB_BITS - INTER_BITS));

        int32x2_t vx = __nds__v_sclip32(__nds__v_sra32(vX, INTER_BITS), 15);
        int32x2_t vy = __nds__v_sclip32(__nds__v_sra32(vY, INTER_BITS), 15);

        *(uint16x4_t*)(xy + x1 * 2) = (uint16x4_t)__nds__pkbb16((unsigned long)vy, (unsigned long)vx);

        uint32x2_t valpha = __nds__v_uadd32(__nds__v_sll32((uint32x2_t)(vY & vmask), INTER_BITS), (uint32x2_t)(vX & vmask));
        *(int16x2_t*)(alpha + x1) = (int16x2_t) { (short)(valpha[0]), (short)(valpha[1]) };
    }

    for (; x1 < bw; x1++) {
        int X = X0 + adelta[x1];
        int Y = Y0 + bdelta[x1];
        xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
        xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
        alpha[x1] = (short)((Y & INTER_MASK) * INTER_TAB_SIZE + (X & INTER_MASK));
    }

    return CV_HAL_ERROR_OK;
}

} // namespace ndsrvp

} // namespace cv
