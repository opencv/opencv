// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "kv_cache_manager.hpp"
#include "net_impl.hpp"

#include <memory>
#include "layers/cpu_kernels/fast_gemm.hpp"
namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void setKVCacheManager(Ptr<Net::Impl> netimpl)
{
    CV_Assert(netimpl != nullptr);

    CV_Assert(!netimpl->layers.empty());

    auto manager = KVCacheManager();
    manager.netimpl = netimpl;
    manager.opt.init();

    for (const auto& layer : netimpl->mainGraph->prog()) {
        if (layer->type != "AttentionOnnxAi")
            continue;

        int kvNumHeads = layer.dynamicCast<AttentionOnnxAiLayer>()->kv_num_heads;

        if (kvNumHeads > 0)
        {
            manager.kData.emplace(layer->name, KCache(manager.opt, kvNumHeads));
            manager.vData.emplace(layer->name, VCache(manager.opt, kvNumHeads));
        }
        else
        {
            manager.kData.emplace(layer->name, KCache(manager.opt));
            manager.vData.emplace(layer->name, VCache(manager.opt));
        }
    }

    manager.isInitialized = true;
    netimpl->useKVCache = true;
    netimpl->kvCacheManager = std::move(manager);
}

void KVCacheManager::init()
{
    // Construction of per-layer caches happens in setKVCacheManager.
    // This method is retained as a callable hook for deferred re-init if needed.
    CV_Assert(isInitialized);
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
        if (batchSize == -1)
            batchSize = newData.size[0];
    }

    int T = newData.dims == 4 ? newData.size[2] : newData.size[1];

    if (T > 1 || pages.empty()) {
        // prefetch
        if(!pages.empty())
            CV_Error(
                cv::Error::StsNotImplemented,
                "storing multiple tokens to a non-empty cache is not supported yet. Either clear the cache (to reenter the prefetch phase) or provide tokens one-by-one"
            );

        // add pages
        int totalPages = (T + pageSize - 1) / pageSize;
        for (int i = 0; i < totalPages; i++) {
            int page_size = isKCache ?
                (int)fastGemmPackBSize(pageSize, headDim, opt):
                (int)fastGemmPackBSize(headDim, pageSize, opt);

            pages.push_back(
                Mat({batchSize, nHeads, page_size}, CV_32F, Scalar(0))
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
    int ps = isKCache ? (int)fastGemmPackBSize(pageSize, headDim, opt)
                      : (int)fastGemmPackBSize(headDim, pageSize, opt);

    bool is3Dlayout = newData.dims == 3;

    auto fn = [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            int page = i / (batchSize * nHeads);
            int b = (i - page * batchSize * nHeads) / nHeads;
            int h = i % nHeads;

            // source
            size_t step_source = b * nHeads * T * headDim;
            if(is3Dlayout)
                step_source += page * pageSize * nHeads * headDim +
                              h * headDim;
            else
                step_source += h * headDim * T + page * pageSize * headDim;
            const auto* source = newData.ptr<float>() + step_source;

            // dst
            size_t step_dst = b * nHeads * ps +
                              h * ps;
            auto* dst = pages[page].ptr<float>() + step_dst;

            const int N = (isKCache ? pageSize : headDim);
            const int K = (isKCache ? headDim : pageSize);

            fastGemmPackB(
                isKCache,
                N, K,
                source, is3Dlayout ? headDim * nHeads : headDim,
                dst,
                opt
            );
        }
    };
    int total = totalPages * batchSize * nHeads;
    parallel_for_(Range(0, total), fn);
    nTokens += T;
}

void KCache::growGenerate(const Mat& newData){
    int cur_page = nTokens / pageSize;
    const int Ps = fastGemmPackBSize(pageSize, headDim, opt);
    int t0 = nTokens % pageSize;
    const int batch_size = newData.size[0];

    if (cur_page >= (int)pages.size()) {
        pages.push_back(Mat({batchSize, nHeads, Ps}, CV_32F, Scalar(0)));
    }

    auto* page = pages[cur_page].ptr<float>();
    const auto* data = newData.ptr<float>();

    for (int b = 0; b < batch_size; b++){
        for (int h = 0; h < nHeads; h++){
            for(int j = 0; j < headDim; j++) {
                int step =
                    b * nHeads * headDim +
                    h * headDim +
                    j;
                page[
                    b * nHeads * Ps +
                    h * Ps +
                    t0 + pageSize * j
                ] = *(data + step);
            }
        }
    }

    nTokens += 1;
}

void VCache::growGenerate(const Mat& newData){
    const int batch_size = newData.size[0];
    const int Nr = fastGemmNR(opt);
    const int Ps = fastGemmPackBSize(headDim, pageSize, opt);
    const int t0 = nTokens  % pageSize;
    const int step_packed = pageSize * Nr;
    int cur_page = nTokens  / pageSize;

    if (cur_page >= (int)pages.size()) {
        pages.push_back(Mat({batchSize, nHeads, Ps}, CV_32F, Scalar(0)));
    }

    auto* page = pages[cur_page].ptr<float>();
    const auto* data = newData.ptr<float>();

    for (int b = 0; b < batch_size; b++){
        for (int h = 0; h < nHeads; h++){
            for (int j=0; j <= headDim / Nr; j+=1){
                int step =
                    b * nHeads * headDim +
                    h * headDim +
                    j * Nr;

                size_t copy_size = std::min(Nr, headDim - j * Nr);

                auto* cur_page =
                    page + b * nHeads * Ps +
                    h * Ps +
                    t0 * Nr +
                    j * step_packed;

                std::memcpy(
                    cur_page,
                    data + step,
                    copy_size * sizeof(float)
                );
            }
        }
    }

    nTokens += 1;
}


CV__DNN_INLINE_NS_END
}}
