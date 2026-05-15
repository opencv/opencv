// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_KV_CACHE_MANAGER_HPP__
#define __OPENCV_DNN_KV_CACHE_MANAGER_HPP__

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <map>
#include <string>
#include <vector>

#include "layers/cpu_kernels/fast_gemm.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class KVCache
{
    public:
        virtual ~KVCache() = default;
        KVCache(FastGemmOpt opt, int nHeads) : nHeads(nHeads),  headDim(-1), offset(0), opt(opt) {}
        KVCache(FastGemmOpt opt) : nHeads(-1), headDim(-1),offset(0), opt(opt) {}
        void grow(const Mat& newData);
        void clear() {
            pages.clear();
            nTokens = 0;
        }
        const std::vector<Mat>& getPages() const { return pages; }
        int getPageSize() const { return pageSize; }
        int getNumTokens() const { return nTokens; }
    protected:
        void growPrefill(const Mat& newData, int T);

        virtual void growGenerate(const Mat& newData) = 0;
        std::vector<Mat> pages;

        int nTokens = 0;
        int pageSize = -1;
        int nHeads;
        int headDim;
        int batchSize = -1;
        int offset;
        bool isKCache = false;
        FastGemmOpt opt;
};

class VCache : public KVCache
{
    public:
        VCache(FastGemmOpt opt) : KVCache(opt) {
            isKCache = false;
            pageSize = fastGemmKC(opt);
        }
        VCache(FastGemmOpt opt, int nHeads) : KVCache(opt, nHeads) {
            isKCache = false;
            pageSize = fastGemmKC(opt);
        }
    protected:
        void growGenerate(const Mat& newData) CV_OVERRIDE;
};

class KCache : public KVCache
{
    public:
        KCache(FastGemmOpt opt) : KVCache(opt) {
            isKCache = true;
            pageSize = fastGemmNR(opt);
        }
        KCache(FastGemmOpt opt, int nHeads) : KVCache(opt, nHeads) {
            isKCache = true;
            pageSize = fastGemmNR(opt);
        }
    protected:
        void growGenerate(const Mat& newData) CV_OVERRIDE;
};


struct KVCacheManager
{
    Net::Impl* netimpl = nullptr;
    std::map<std::string, KCache> kData;
    std::map<std::string, VCache> vData;
    FastGemmOpt opt;
    bool isInitialized = false;

    // present.* output arg idx -> past_key_values.* input arg idx
    std::vector<std::pair<int, int>> presentToPastRoutes;
    bool hasRoutes = false;

    void init();
    void buildRoutes();
    void applyRoutes();
    void initPastTensors();
};

void setKVCacheManager(Ptr<Net::Impl> netimpl);


CV__DNN_INLINE_NS_END
}}

#endif
