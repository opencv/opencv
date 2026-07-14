// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPILER_HPP
#define OPENCV_GAPI_GCOMPILER_HPP


#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/gcomputation.hpp>

#include <ade/execution_engine/execution_engine.hpp>

namespace cv { namespace gimpl {

// FIXME: exported for internal tests only!
class GAPI_EXPORTS GCompiler
{
    const GComputation&      m_c;
    const GMetaArgs          m_metas;
    GCompileArgs             m_args;
    ade::ExecutionEngine     m_e;

    cv::GKernelPackage       m_all_kernels;
    cv::gapi::GNetPackage    m_all_networks;

    // Patterns built from transformations
    std::vector<std::unique_ptr<ade::Graph>> m_all_patterns;


    void validateInputMeta();
    void validateOutProtoArgs();

public:
    // Metas may be empty in case when graph compiling for streaming
    // In this case graph get metas from first frame
    explicit GCompiler(const GComputation &c,
                             GMetaArgs    &&metas,
                             GCompileArgs &&args);

    // The method which does everything...
    GCompiled compile();

    // This too.
    GStreamingCompiled compileStreaming();

    // But those are actually composed of this:
    using GPtr = std::unique_ptr<ade::Graph>;
    GPtr        generateGraph();               // Unroll GComputation into a GModel
    void        runPasses(ade::Graph &g);      // Apply all G-API passes on a GModel
    void        compileIslands(ade::Graph &g); // Instantiate GIslandExecutables in GIslandModel
    static void compileIslands(ade::Graph &g, const cv::GCompileArgs &args);
    GCompiled   produceCompiled(GPtr &&pg);    // Produce GCompiled from processed GModel
    GStreamingCompiled  produceStreamingCompiled(GPtr &&pg); // Produce GStreamingCompiled from processed GMbodel
    static void runMetaPasses(ade::Graph &g, const cv::GMetaArgs &metas);

    static GPtr makeGraph(const cv::GComputation::Priv &);
};

}}

#endif // OPENCV_GAPI_GCOMPILER_HPP
