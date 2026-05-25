// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "kv_cache_manager.hpp"
#include "net_impl.hpp"

#include <memory>
#include <opencv2/core/utils/tls.hpp>
#include "layers/cpu_kernels/fast_gemm.hpp"
namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN



void initKVDataRecursively(const Ptr<Graph>& graph, std::map<std::string, KCache>& kData, std::map<std::string, VCache>& vData, FastGemmOpt& opt) {
    for (const auto& layer : graph->prog()) {

        for (const auto& subgraph : layer->subgraphs() ? *layer->subgraphs() : std::vector<Ptr<Graph>>()) {
            initKVDataRecursively(subgraph, kData, vData, opt);
        }

        if (layer->type == "AttentionOnnxAi") {
            int kvNumHeads = layer.dynamicCast<AttentionOnnxAiLayer>()->kv_num_heads;

            if (kvNumHeads > 0)
            {
                kData.emplace(layer->name, KCache(opt, kvNumHeads));
                vData.emplace(layer->name, VCache(opt, kvNumHeads));
            }
            else
            {
                kData.emplace(layer->name, KCache(opt));
                vData.emplace(layer->name, VCache(opt));
            }
        }
    }
}

void setKVCacheManager(Ptr<Net::Impl> netimpl)
{
    CV_Assert(netimpl != nullptr);

    CV_Assert(!netimpl->layers.empty());

    auto manager = KVCacheManager();
    manager.netimpl = netimpl;
    manager.opt.init();
    initKVDataRecursively(netimpl->mainGraph, manager.kData, manager.vData, manager.opt);
    manager.buildRoutes();

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

void KVCacheManager::buildRoutes()
{
    presentToPastRoutes.clear();
    hasRoutes = false;

    if (!netimpl || !netimpl->mainGraph)
        return;

    const std::vector<Arg>& gr_outputs = netimpl->mainGraph->outputs();
    for (const Arg& out_arg : gr_outputs)
    {
        const ArgData& out_adata = netimpl->args.at(out_arg.idx);
        const std::string& out_name = out_adata.name;

        if (out_name.compare(0, 8, "present.") != 0)
            continue;

        std::string past_name = "past_key_values." + out_name.substr(8);
        auto it = netimpl->argnames.find(past_name);
        if (it == netimpl->argnames.end())
            continue;

        Arg past_arg((int)it->second);
        if (netimpl->args.at(past_arg.idx).kind != DNN_ARG_INPUT)
            continue;

        presentToPastRoutes.emplace_back(out_arg.idx, past_arg.idx);
    }

    hasRoutes = !presentToPastRoutes.empty();
    if (hasRoutes)
        initPastTensors();
}

void KVCacheManager::initPastTensors()
{
    for (const auto& route : presentToPastRoutes)
    {
        const ArgData& past_adata = netimpl->args.at(route.second);
        const MatShape& decl_shape = past_adata.shape;
        if (decl_shape.dims <= 0)
            continue;

        // Replace symbolic dims (stored as <=0): batch (dim 0) -> 1, all others -> 0 for empty-sequence state.
        std::vector<int> shape_vec(decl_shape.dims);
        for (int d = 0; d < decl_shape.dims; d++)
            shape_vec[d] = (decl_shape[d] > 0) ? decl_shape[d] : (d == 0 ? 1 : 0);

        int dtype = past_adata.type;
        if (dtype < 0)
            dtype = CV_32F;

        Mat& past_t = netimpl->__tensors__.at(route.second);
        past_t = Mat(shape_vec, dtype, Scalar(0));
        netimpl->finalizeLayers = true;
    }
}

void KVCacheManager::applyRoutes()
{
    for (const auto& route : presentToPastRoutes)
    {
        const Mat& present_t = netimpl->argTensor(Arg(route.first));
        Mat& past_t = netimpl->__tensors__.at(route.second);
        if (present_t.empty())
            continue;
        if (past_t.shape() != present_t.shape() || past_t.type() != present_t.type())
            netimpl->finalizeLayers = true;
        present_t.copyTo(past_t);
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
        growPrefill(newData, T);
    } else{
        // generate
        growGenerate(newData);
    }

}

void KVCache::growPrefill(const Mat& newData, int T){
    int totalPages = (T + pageSize - 1) / pageSize;
    int ps = isKCache ? (int)fastGemmPackBSize(pageSize, headDim, opt)
                      : (int)fastGemmPackBSize(headDim, pageSize, opt);

    bool is3Dlayout = newData.dims == 3;

    int total = totalPages * batchSize * nHeads;

    cv::TLSData<std::vector<float>> tls_temp_buf;

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

            int chunk_T = std::min(pageSize, T - page * pageSize);
            const float* actual_source = source;

            int lds = is3Dlayout ? headDim * nHeads : headDim;

            if (chunk_T < pageSize) {
                std::vector<float>& temp_buf = *tls_temp_buf.get();
                temp_buf.assign(pageSize * headDim, 0.0f);
                for (int i = 0; i < chunk_T; i++) {
                    std::memcpy(temp_buf.data() + i * headDim, source + i * lds, headDim * sizeof(float));
                }
                actual_source = temp_buf.data();
            }

            // dst
            size_t step_dst = b * nHeads * ps +
                              h * ps;
            auto* dst = pages[page].ptr<float>() + step_dst;

            const int N = headDim;
            const int K = pageSize;

            fastGemmPackB(
                isKCache,
                N, K,
                actual_source, chunk_T < pageSize ? headDim : lds,
                dst,
                opt
            );
        }
    };
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

    const int nstripes = batch_size * nHeads;
    parallel_for_(Range(0, nstripes), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            int b = i / nHeads, h = i % nHeads;
            const float* src = data + (b * nHeads + h) * headDim;
            float* dst = page + b * nHeads * Ps + h * Ps + t0;
            for (int j = 0; j < headDim; j++)
                dst[j * pageSize] = src[j];
        }
    });

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

    const int nstripes = batch_size * nHeads;
    parallel_for_(Range(0, nstripes), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            int b = i / nHeads, h = i % nHeads;
            for (int j = 0; j <= (headDim - 1) / Nr; j++) {
                int step = b * nHeads * headDim + h * headDim + j * Nr;
                int copy_size = std::min(Nr, headDim - j * Nr);

                auto* cur_page_ptr = page + b * nHeads * Ps + h * Ps + t0 * Nr + j * step_packed;
                const float* src_ptr = data + step;
                std::memcpy(cur_page_ptr, src_ptr, copy_size * sizeof(float));
                if (copy_size < Nr) {
                    float replication_val = src_ptr[0];
                    for (int k = copy_size; k < Nr; k++) {
                        cur_page_ptr[k] = replication_val;
                    }
                }
            }
        }
    });

    nTokens += 1;
}

CV__DNN_INLINE_NS_END
}}
