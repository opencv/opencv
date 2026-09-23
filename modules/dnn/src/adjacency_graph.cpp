// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

typedef std::map<std::type_index, FusionOps> FusionRegistry;

// std::map, not unordered_map: fusionOpsFor hands out a pointer to a mapped value,
// and only a node-based container keeps that valid across later registrations.
FusionRegistry& registry()
{
    static FusionRegistry reg;
    return reg;
}

Mutex& registryMutex()
{
    static Mutex m;
    return m;
}

} // namespace

void registerFusionOps(const std::type_info& layerType, const FusionOps& ops)
{
    AutoLock lock(registryMutex());
    registry()[std::type_index(layerType)] = ops;
}

const FusionOps* fusionOpsFor(const Layer* layer)
{
    if (!layer)
        return nullptr;

    AutoLock lock(registryMutex());
    const FusionRegistry& reg = registry();
    FusionRegistry::const_iterator it = reg.find(std::type_index(typeid(*layer)));
    return it == reg.end() ? nullptr : &it->second;
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
