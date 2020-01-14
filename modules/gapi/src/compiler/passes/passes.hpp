// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_COMPILER_PASSES_HPP
#define OPENCV_GAPI_COMPILER_PASSES_HPP

#include <ostream>
#include <ade/passes/pass_base.hpp>

#include "opencv2/gapi/garg.hpp"
#include "opencv2/gapi/gcommon.hpp"

// Forward declarations - external
namespace ade {
    class Graph;

    namespace passes {
        struct PassContext;
    }
}

namespace cv {

// Forward declarations - internal
namespace gapi {
    class GKernelPackage;
    struct GNetPackage;
}  // namespace gapi

namespace gimpl { namespace passes {

void dumpDot(const ade::Graph &g, std::ostream& os);
void dumpDot(ade::passes::PassContext &ctx, std::ostream& os);
void dumpDotStdout(ade::passes::PassContext &ctx);
void dumpGraph(ade::passes::PassContext     &ctx, const std::string& dump_path);
void dumpDotToFile(ade::passes::PassContext &ctx, const std::string& dump_path);

void initIslands(ade::passes::PassContext &ctx);
void checkIslands(ade::passes::PassContext &ctx);
void checkIslandsContent(ade::passes::PassContext &ctx);

void initMeta(ade::passes::PassContext &ctx, const GMetaArgs &metas);
void inferMeta(ade::passes::PassContext &ctx, bool meta_is_initialized);
void storeResultingMeta(ade::passes::PassContext &ctx);

void expandKernels(ade::passes::PassContext &ctx,
                   const gapi::GKernelPackage& kernels);

void bindNetParams(ade::passes::PassContext   &ctx,
                   const gapi::GNetPackage    &networks);

void resolveKernels(ade::passes::PassContext   &ctx,
                    const gapi::GKernelPackage &kernels);

void fuseIslands(ade::passes::PassContext &ctx);
void syncIslandTags(ade::passes::PassContext &ctx);
void topoSortIslands(ade::passes::PassContext &ctx);

void applyTransformations(ade::passes::PassContext &ctx,
                          const gapi::GKernelPackage &pkg,
                          const std::vector<std::unique_ptr<ade::Graph>> &preGeneratedPatterns);

void addStreaming(ade::passes::PassContext &ctx);

}} // namespace gimpl::passes

} // namespace cv

#endif // OPENCV_GAPI_COMPILER_PASSES_HPP
