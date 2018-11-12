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

// Sobel 3x3: vertical pass
template<bool noscale, typename DST>
void run_sobel3x3_vert(DST out[], int length, const float ky[],
         float scale, float delta, const int r[], float *buf[]);

}  // namespace fluid
}  // namespace gapi
}  // namespace cv

#endif // !defined(GAPI_STANDALONE)
