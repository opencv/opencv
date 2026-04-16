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

void KVCacheManager::init()
{
    if (isInitialised)
        return;

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

void KVCache::grow(const Mat& newData) {
    CV_Assert(newData.dims == 4 || newData.dims == 3);

    if (nHeads == -1) {
        CV_Assert(newData.dims == 4); // to derive shape from data, we need 4D
        headDim = newData.size[3];
        nHeads = newData.size[1];
        batchSize = newData.size[0];
    } else {
        if (headDim == -1) {
            if (newData.dims == 4)
                headDim = newData.size[3];
            else{
                CV_Assert(newData.dims == 3);
                CV_Assert(newData.size[2] % nHeads == 0);
                headDim = newData.size[2] / nHeads;
            }
        } else {
            if (newData.dims == 4)
                CV_Assert(newData.size[3] == headDim);
            else
                CV_Assert(newData.size[2] == headDim * nHeads);
        }
    }

    int T = newData.dims == 4 ? newData.size[2] : newData.size[1];

    if (T > 1){
        // prefetch
        if(!pages.empty())
            CV_Error(
                cv::Error::StsNotImplemented,
                "storing multiple tokens to a non-empty cache is not supported yet. Either clear the cache (to reenter the prefetch phase) or provide tokens one-by-one"
            );

        // add pages
        int totalPages = (T + pageSize - 1) / pageSize;
        for (int i = 0; i < totalPages - 1; i++) {
            pages.push_back(
                Mat(
                    {batchSize, nHeads, fastGemmPackBSize(headDim, pageSize, opt)},
                    CV_32F, Scalar(0)
                );
            );
        }
        growPrefetch(newData, T);
    } else{
        // generate
        growGenerate(newData);
    }
}

void KVCache::growPrefetch(const Mat& newData, int T){
    int totalPages = (T + pageSize - 1) / pageSize;
    int ps = isKCache ? fastGemmPackBSize(headDim, pageSize, opt) : fastGemmPackBSize(pageSize, headDim, opt);

    bool is3Dlayout = newData.dims == 3;

    auto fn = [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            int page = i / (batchSize * nHeads);
            int b = (i - page * batchSize * nHeads) / nHeads;
            int h = i % nHeads;

            // source
            size_t step_source = b * nHeads * T * headDim ;
            if(is3Dlayout)
                step_source = page * pageSize * nHeads * headDim +
                                h * headDim;
            else
                step_source = h * headDim * T;
            const auto*source = newData.ptr<float>() + step_source;

            // dst
            size_t step_dst = b * nHeads * ps +
                              h * ps;
            auto*dst = pages[page].ptr<float>() + step_dst;

            fastGemmPackB(
                isKCache,
                fastGemmNR(opt), headDim,
                source, is3Dlayout ? headDim * nHeads : headDim
                dst,
                opt
            );
        }
    };
    int total = totalPages * batchSize * nHeads;
    parallel_for_(Range(0, total), fn);
    nTokens += T;
}

virtual void KCache::growGenerate(const Mat& newData){
    bool is3Dlayout = newData.dims == 3;
    int cur_page = (nTokens + 1) / pageSize;
    int _kp = (nTokens + 1) % pageSize;
    const int batch_size = newData.size[0];
    const int ps = fastGemmPackBSize(headDim, pageSize, opt);

    auto* page = pages[cur_page].ptr<float>();
    const auto* data = newData.ptr<float>();

    for (int b = 0; b < batch_size; b++){
        for (int h = 0; h < nHeads; h++){
            for(int j = 0; j < headDim; j++) {
                int step =
                    b * nHeads * headDim +
                    h * headDim;
                page[
                    ps * (b * nHeads * ps + h) +
                    _kp + j * pageSize
                ] = data + step;
            }
        }
    }
}

virtual void VCache::growGenerate(const Mat& newData){
    const size_t Nr = fastGemmNR(opt);
    size_t width = (headDim + Nr - 1) / Nr * Nr;
    const size_t t0 = ((nTokens + 1) % pageSize);
    const size_t step_packed = pageSize * Nr;
    auto* page = pages[cur_page].ptr<float>();
    const auto* data = newData.ptr<float>();

    for (int b = 0; b < batch_size; b++){
        for (int h = 0; h < nHeads; h++){
            for (int j=0; j < headDim; j+=fasGemmNR(opt)){
                int step =
                    b * nHeads * headDim +
                    h * headDim +
                    j;
                size_t copy_size = std::min(Nr, headDim - j);
                std::memcpy(
                    page + width * t0 + j,
                    data + step,
                    copy_size * sizeof(float)
                );
            }
        }
    }
}

CV__DNN_INLINE_NS_END
}}
