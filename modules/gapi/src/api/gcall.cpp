// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <cassert>
#include "opencv2/gapi/gcall.hpp"
#include "api/gcall_priv.hpp"

// GCall private implementation ////////////////////////////////////////////////
cv::GCall::Priv::Priv(const cv::GKernel &k)
    : m_k(k)
{
}

// GCall public implementation /////////////////////////////////////////////////

cv::GCall::GCall(const cv::GKernel &k)
    : m_priv(new Priv(k))
{
    // Here we have a reference to GNode,
    // and GNode has a reference to us. Cycle! Now see destructor.
    m_priv->m_node = GNode::Call(*this);
}

cv::GCall::~GCall()
{
    // FIXME: current behavior of the destructor can cause troubles in a threaded environment. GCall
    // is not supposed to be accessed for modification within multiple threads. There should be a
    // way to ensure somehow that no problem occurs in future. For now, this is a reminder that
    // GCall is not supposed to be copied inside a code block that is executed in parallel.

    // When a GCall object is destroyed (and GCall::Priv is likely still alive,
    // as there might be other references), reset m_node to break cycle.
    m_priv->m_node = GNode();
}

void cv::GCall::setArgs(std::vector<GArg> &&args)
{
    // FIXME: Check if argument number is matching kernel prototype
    m_priv->m_args = std::move(args);
}

cv::GMat cv::GCall::yield(int output)
{
    return cv::GMat(m_priv->m_node, output);
}

cv::GScalar cv::GCall::yieldScalar(int output)
{
    return cv::GScalar(m_priv->m_node, output);
}

cv::detail::GArrayU cv::GCall::yieldArray(int output)
{
    return cv::detail::GArrayU(m_priv->m_node, output);
}

cv::GCall::Priv& cv::GCall::priv()
{
    return *m_priv;
}

const cv::GCall::Priv& cv::GCall::priv() const
{
    return *m_priv;
}
