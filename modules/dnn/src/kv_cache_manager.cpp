// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "kv_cache_manager.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void setKVCacheManager(Net::Impl* netimpl)
{
    CV_Assert(netimpl != nullptr);
    CV_Assert(netimpl->mainGraph);

    const std::vector<Arg>& inputs = netimpl->mainGraph->inputs();
    CV_Assert(!inputs.empty());
    Arg inputEmbeddingArg = inputs[0];
    MatShape shape = netimpl->argData(inputEmbeddingArg).shape;

    CV_Assert(shape.size() == 3);
    KVCacheManager manager;
    manager.netimpl = netimpl;
    manager.dim = shape[2];
    manager.blockSize = 325;
    manager.initialized = true;
    netimpl->kvCacheManager = manager;
}

void KVCacheManager::init(int batchSize_)
{
    clear();

    batchSize = batchSize_;
    curPageOffset = 0;

    // Allocate cache buffers for each attention layer
    for (const auto& layerPair : netimpl->layers)
    {
        const LayerData& layer = layerPair.second;
        if (layer.type == "AttentionONNXAI")
        {
            std::string layerId = layer.name;
            Mat kCache(Size(dim, blockSize * batchSize), CV_32F, Scalar(0));
            Mat vCache(Size(dim, blockSize * batchSize), CV_32F, Scalar(0));
            kData[layerId] = kCache;
            vData[layerId] = vCache;
        }
    }
}


CV__DNN_INLINE_NS_END
}}
