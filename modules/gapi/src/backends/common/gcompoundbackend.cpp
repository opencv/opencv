// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gcompoundkernel.hpp> // compound::backend()

#include "api/gbackend_priv.hpp"
#include "compiler/gislandmodel.hpp" // GIslandExecutable

cv::gapi::GBackend cv::gapi::compound::backend()
{
    // A pointer to dummy Priv is used to uniquely identify backends
    static cv::gapi::GBackend this_backend(std::make_shared<cv::gapi::GBackend::Priv>());
    return this_backend;
}
