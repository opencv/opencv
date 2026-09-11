/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//M*/

#include "precomp.hpp"
#include "hal_replacement.hpp"

#include "resize.hpp"
#include "resize.simd.hpp"
#include "resize.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL based on CMakeLists.txt

namespace cv {

namespace hal {

void resize(int src_type,
            const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
            double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((dst_width > 0 && dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
    if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
    {
        inv_scale_x = static_cast<double>(dst_width) / src_width;
        inv_scale_y = static_cast<double>(dst_height) / src_height;
    }

    CALL_HAL(resize, cv_hal_resize, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);

    int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
    Size dsize = Size(saturate_cast<int>(src_width*inv_scale_x),
                        saturate_cast<int>(src_height*inv_scale_y));
    CV_Assert( !dsize.empty() );
    CV_UNUSED(depth);
    CV_UNUSED(cn);

    if (interpolation == INTER_LINEAR_EXACT)
    {
        if (resizeLinearExact(src_type, src_data, src_step, src_width, src_height,
                              dst_data, dst_step, dst_width, dst_height,
                              inv_scale_x, inv_scale_y, &interpolation))
            return;
    }

    CV_CPU_DISPATCH(resize_cpu, (src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace hal

} // namespace cv
