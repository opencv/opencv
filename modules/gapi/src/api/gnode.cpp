// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <cassert>

#include "api/gnode.hpp"
#include "api/gnode_priv.hpp"

// GNode private implementation
cv::GNode::Priv::Priv()
    : m_shape(NodeShape::EMPTY)
{
}

cv::GNode::Priv::Priv(GCall c)
    : m_shape(NodeShape::CALL), m_spec(c)
{
}

cv::GNode::Priv::Priv(ParamTag)
    : m_shape(NodeShape::PARAM)
{
}

cv::GNode::Priv::Priv(ConstTag)
    : m_shape(NodeShape::CONST_BOUNDED)
{
}

// GNode public implementation
cv::GNode::GNode()
    : m_priv(new Priv())
{
}

cv::GNode::GNode(const GCall &c)
    : m_priv(new Priv(c))
{
}

cv::GNode::GNode(ParamTag)
    : m_priv(new Priv(Priv::ParamTag()))
{
}

cv::GNode::GNode(ConstTag)
    : m_priv(new Priv(Priv::ConstTag()))
{
}

cv::GNode cv::GNode::Call(const GCall &c)
{
    return GNode(c);
}

cv::GNode cv::GNode::Param()
{
    return GNode(ParamTag());
}

cv::GNode cv::GNode::Const()
{
    return GNode(ConstTag());
}

cv::GNode::Priv& cv::GNode::priv()
{
    return *m_priv;
}

const cv::GNode::Priv& cv::GNode::priv() const
{
    return *m_priv;
}

const cv::GNode::NodeShape& cv::GNode::shape() const
{
    return m_priv->m_shape;
}

const cv::GCall& cv::GNode::call()  const
{
    return util::get<GCall>(m_priv->m_spec);
}
