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
        KVCache(FastGemmOpt opt, int nHeads) : nHeads(nHeads), offset(0), opt(opt), headDim(-1) {}
        KVCache(FastGemmOpt opt) : nHeads(-1), isShapeDerived(false), offset(0), opt(opt), headDim(-1) {}
        void grow(const Mat& newData);
        void clear() {
            pages.clear();
            nTokens = 0;
        }
        const std::vector<Mat>& getPages() const { return pages; }
    protected:
        void growPrefetch(const Mat& newData, int T);
        virtual void growGenerate(const Mat& newData);

        std::vector<Mat> pages;

        int nTokens = 0;
        int pageSize = 325;
        int nHeads;
        int headDim;
        int batchSize = -1;
        int offset;
        bool isShapeDerived = true;
        bool isKCache = false;
        FastGemmOpt opt;
};

class VCache : public KVCache
{
    public:
        VCache(FastGemmOpt opt) : KVCache(opt) {
            isKCache = false;
        }
        VCache(FastGemmOpt opt, int nHeads) : KVCache(opt, nHeads) {
            isKCache = false;
        }
    protected:
        void growGenerate(const Mat& newData) CV_OVERRIDE;
};

class KCache : public KVCache
{
    public:
        KCache(FastGemmOpt opt) : KVCache(opt) {
            isKCache = true;
        }
        KCache(FastGemmOpt opt, int nHeads) : KVCache(opt, nHeads) {
            isKCache = true;
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

    void init();
};

void setKVCacheManager(Net::Impl* netimpl);


CV__DNN_INLINE_NS_END
}}

#endif
