#ifndef OPENCV_FASTCV_HAL_HPP_INCLUDED
#define OPENCV_FASTCV_HAL_HPP_INCLUDED

#include "opencv2/core/hal/interface.h"

int fastcv_hal_add_8u(const uchar *a, size_t astep, const uchar *b, size_t bstep, uchar *c, size_t cstep, int w, int h);

#undef cv_hal_add8u
#define cv_hal_add8u fastcv_hal_add_8u

int fastcv_hal_setto_mask(uchar *dst_data, int dst_step, int dst_cols, int dst_rows,
                             const uchar* mask_data, int mask_step,
                             uchar *value_data, int value_size);

//! @cond IGNORED
#undef cv_hal_setto_mask
#define cv_hal_setto_mask fastcv_hal_setto_mask
//! @endcond

#endif
