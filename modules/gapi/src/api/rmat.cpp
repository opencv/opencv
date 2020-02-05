// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/rmat.hpp>

using View = cv::RMat::View;

// There is an issue with default generated operator=(View&&) on Mac:
// it doesn't nullify m_cb of a moved object
View& View::operator=(View&& v) {
    m_desc = v.m_desc;
    m_data = v.m_data;
    m_step = v.m_step;
    m_cb   = v.m_cb;
    v.m_desc = {};
    v.m_data = nullptr;
    v.m_step = 0u;
    v.m_cb   = nullptr;
    return *this;
}
