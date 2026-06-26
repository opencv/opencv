// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

#include "moments.simd.hpp"
#include "moments.simd_declarations.hpp"

namespace cv {

MomentsInTileFunc getMomentsInTileFunc(int depth);

MomentsInTileFunc getMomentsInTileFunc(int depth)
{
    CV_CPU_DISPATCH(getMomentsInTileFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace cv
