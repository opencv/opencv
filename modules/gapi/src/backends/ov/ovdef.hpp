// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation
//

#ifndef OPENCV_GAPI_OV_DEF_HPP
#define OPENCV_GAPI_OV_DEF_HPP

// FIXME: Should this test be made at the CMake level?
#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000
#  include <openvino/openvino.hpp>
#  define HAVE_OPENVINO_2_0
#endif // HAVE_INF_ENGINE

#endif // OPENCV_GAPI_OV_DEF_HPP
