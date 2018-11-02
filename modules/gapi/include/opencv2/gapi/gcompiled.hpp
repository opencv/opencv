// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILED_HPP
#define OPENCV_GAPI_GCOMPILED_HPP

#include <vector>

#include "opencv2/gapi/opencv_includes.hpp"
#include "opencv2/gapi/own/assert.hpp"
#include "opencv2/gapi/garg.hpp"

namespace cv {

// This class represents a compiled computation.
// In theory (and ideally), it can be used w/o the rest of APIs.
// In theory (and ideally), it can be serialized/deserialized.
// It can enable scenarious like deployment to an autonomous devince, FuSa, etc.
//
// Currently GCompiled assumes all GMats you used to pass data to G-API
// are valid and not destroyed while you use a GCompiled object.
//
// FIXME: In future, there should be a way to name I/O objects and specify it
// to GCompiled externally (for example, when it is loaded on the target system).

class GAPI_EXPORTS GCompiled
{
public:
    class GAPI_EXPORTS Priv;

    GCompiled();

    void operator() (GRunArgs &&ins, GRunArgsP &&outs);          // Generic arg-to-arg
#if !defined(GAPI_STANDALONE)
    void operator() (cv::Mat in, cv::Mat &out);                  // Unary overload
    void operator() (cv::Mat in, cv::Scalar &out);               // Unary overload (scalar)
    void operator() (cv::Mat in1, cv::Mat in2, cv::Mat &out);    // Binary overload
    void operator() (cv::Mat in1, cv::Mat in2, cv::Scalar &out); // Binary overload (scalar)
    void operator() (const std::vector<cv::Mat> &ins,            // Compatibility overload
                     const std::vector<cv::Mat> &outs);
#endif  // !defined(GAPI_STANDALONE)
    Priv& priv();

    explicit operator bool () const; // Check if GCompiled is runnable or empty

    const GMetaArgs& metas() const; // Meta passed to compile()
    const GMetaArgs& outMetas() const; // Inferred output metadata

    bool canReshape() const; // is reshape mechanism supported by GCompiled
    void reshape(const GMetaArgs& inMetas, const GCompileArgs& args); // run reshape procedure

protected:
    std::shared_ptr<Priv> m_priv;
};

}

#endif // OPENCV_GAPI_GCOMPILED_HPP
