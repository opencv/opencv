// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILED_PRIV_HPP
#define OPENCV_GAPI_GCOMPILED_PRIV_HPP

#include <memory> // unique_ptr

#include "opencv2/gapi/util/optional.hpp"
#include "compiler/gmodel.hpp"
#include "executor/gexecutor.hpp"

// NB: BTW, GCompiled is the only "public API" class which
// private part (implementation) is hosted in the "compiler/" module.
//
// This file is here just to keep ADE hidden from the top-level APIs.
//
// As the thing becomes more complex, appropriate API and implementation
// part will be placed to api/ and compiler/ modules respectively.

namespace cv {

namespace gimpl
{
    struct GRuntimeArgs;
};

// FIXME: GAPI_EXPORTS is here only due to tests and Windows linker issues
class GAPI_EXPORTS GCompiled::Priv
{
    // NB: For now, a GCompiled keeps the original ade::Graph alive.
    // If we want to go autonomous, we might to do something with this.
    GMetaArgs  m_metas;    // passed by user
    GMetaArgs  m_outMetas; // inferred by compiler
    std::unique_ptr<cv::gimpl::GExecutor> m_exec;

    void checkArgs(const cv::gimpl::GRuntimeArgs &args) const;

public:
    void setup(const GMetaArgs &metaArgs,
               const GMetaArgs &outMetas,
               std::unique_ptr<cv::gimpl::GExecutor> &&pE);
    bool isEmpty() const;

    bool canReshape() const;
    void reshape(const GMetaArgs& inMetas, const GCompileArgs &args);
    void prepareForNewStream();

    void run(cv::gimpl::GRuntimeArgs &&args);
    const GMetaArgs& metas() const;
    const GMetaArgs& outMetas() const;

    const cv::gimpl::GModel::Graph& model() const;
};

}

#endif // OPENCV_GAPI_GCOMPILED_PRIV_HPP
