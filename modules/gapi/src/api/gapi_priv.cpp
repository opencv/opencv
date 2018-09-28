// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <ade/util/assert.hpp>

#include "api/gapi_priv.hpp"
#include "api/gnode_priv.hpp"

cv::GOrigin::GOrigin(GShape s,
                    const cv::GNode& n,
                    std::size_t p,
                    const cv::gimpl::HostCtor c)
    : shape(s), node(n), port(p), ctor(c)
{
}

cv::GOrigin::GOrigin(GShape s, cv::gimpl::ConstVal v)
    : shape(s), node(cv::GNode::Const()), value(v), port(INVALID_PORT)
{
}

bool cv::detail::GOriginCmp::operator() (const cv::GOrigin &lhs,
                                         const cv::GOrigin &rhs) const
{
    const GNode::Priv* lhs_p = &lhs.node.priv();
    const GNode::Priv* rhs_p = &rhs.node.priv();
    if (lhs_p == rhs_p)
    {
        if (lhs.port == rhs.port)
        {
            // A data Origin is uniquely identified by {node/port} pair.
            // The situation when there're two Origins with same {node/port}s
            // but with different shapes (data formats) is illegal!
            GAPI_Assert(lhs.shape == rhs.shape);
        }
        return lhs.port < rhs.port;
    }
    else return lhs_p < rhs_p;
}
