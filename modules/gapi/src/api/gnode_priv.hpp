// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GNODE_PRIV_HPP
#define OPENCV_GNODE_PRIV_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "opencv2/gapi/util/variant.hpp"

#include "opencv2/gapi/gcall.hpp"
#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gnode.hpp"

namespace cv {

enum class GNode::NodeShape: unsigned int
{
    EMPTY,
    CALL,
    PARAM,
    CONST_BOUNDED
};

class GNode::Priv
{
public:
    // TODO: replace with optional?
    typedef util::variant<util::monostate, GCall> NodeSpec;
    const NodeShape m_shape;
    const NodeSpec  m_spec;
    std::string     m_island; // user-modifiable attribute
    struct ParamTag {};
    struct ConstTag {};

    Priv();                    // Empty (invalid) constructor
    explicit Priv(GCall c);    // Call conctrustor
    explicit Priv(ParamTag u); // Param constructor
    explicit Priv(ConstTag u); // Param constructor
};

}

#endif // OPENCV_GNODE_PRIV_HPP
