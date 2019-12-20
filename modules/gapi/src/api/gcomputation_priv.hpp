// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPUTATION_PRIV_HPP
#define OPENCV_GAPI_GCOMPUTATION_PRIV_HPP

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/gcall.hpp"

#include "opencv2/gapi/util/variant.hpp"

namespace cv {

class GComputation::Priv
{
public:
    GCompiled   m_lastCompiled;
    GMetaArgs   m_lastMetas; // TODO: make GCompiled remember its metas?
    GProtoArgs  m_ins;
    GProtoArgs  m_outs;
};

}

#endif // OPENCV_GAPI_GCOMPUTATION_PRIV_HPP
