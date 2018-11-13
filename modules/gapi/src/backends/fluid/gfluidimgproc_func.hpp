// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#pragma once

#if !defined(GAPI_STANDALONE)

namespace cv {
namespace gapi {
namespace fluid {

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

template<typename DST, typename SRC>
void run_sobel_impl(DST out[], const SRC *in[], int width, int chan,
                    const float kx[], const float ky[], int border,
                    float scale, float delta, float *buf[],
                    int y, int y0);

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
