// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GCALL_PRIV_HPP
#define OPENCV_GCALL_PRIV_HPP

#include <vector>
#include <unordered_map>

#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gcall.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gnode.hpp"

namespace cv {

class GCall::Priv
{
public:
    std::vector<GArg> m_args;
    const GKernel     m_k;

    // FIXME: Document that there's no recursion here.
    // TODO: Rename to "constructionNode" or smt to reflect its lifetime
    GNode             m_node;

    explicit Priv(const GKernel &k);
};

}

#endif // OPENCV_GCALL_PRIV_HPP
