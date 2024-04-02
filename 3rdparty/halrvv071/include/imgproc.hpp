#ifndef _RVV071_HPP_INCLUDED_
#define _RVV071_HPP_INCLUDED_

#include "opencv2/core/hal/interface.h"

int cvt_hal_BGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue);

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cvt_hal_BGRtoBGR

#endif