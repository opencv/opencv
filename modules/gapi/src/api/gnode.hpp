// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GNODE_HPP
#define OPENCV_GAPI_GNODE_HPP

#include <memory> // std::shared_ptr

namespace cv {

class GCall;

// TODO Move "internal" namespace
// TODO Expose details?

// This class won't be public

// data GNode = Call Operation [GNode]
//            | Const <T>
//            | Param <GMat|GParam>

class GNode
{
public:
    class Priv;

    // Constructors
    GNode();                               // Empty (invalid) constructor
    static GNode Call (const GCall &c);    // Call constructor
    static GNode Param();                  // Param constructor
    static GNode Const();

    // Internal use only
    Priv& priv();
    const Priv& priv() const;
    enum class NodeShape: unsigned int;

    const NodeShape& shape() const;
    const GCall&     call()  const;

protected:
    struct ParamTag {};
    struct ConstTag {};

    explicit GNode(const GCall &c);
    explicit GNode(ParamTag unused);
    explicit GNode(ConstTag unused);

    std::shared_ptr<Priv> m_priv;
};

}

#endif // OPENCV_GAPI_GNODE_HPP
