// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_KV_CACHE_MANAGER_HPP__
#define __OPENCV_DNN_KV_CACHE_MANAGER_HPP__

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp> // For Net::Impl forward declaration via Net

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

struct KVCacheManager
{
    Net::Impl* netimpl;

    int dim;
    int blockSize;
    int batchSize;
    int curPageOffset;
    bool initialized = false;

    std::map<std::string, Mat> kData;
    std::map<std::string, Mat> vData;


    KVCacheManager() :  netimpl(nullptr), dim(0), blockSize(0), batchSize(0), curPageOffset(0), initialized(false) {}

    void clear()
    {
        kData.clear();
        vData.clear();
        curPageOffset = 0;
        initialized = false;
    }

    void init(int batchSize_);

    void updateCache(std::string layerId, const Mat& k, const Mat& v)
    {

        CV_Assert(k.size[0] == batchSize);
        CV_Assert(v.size[0] == batchSize);
        CV_Assert(k.size[2] == dim);
        CV_Assert(v.size[2] == dim);

        Mat& kCache = kData[layerId];
        Mat& vCache = vData[layerId];

        k.rowRange(0, k.rows).copyTo(kCache.rowRange(curPageOffset, curPageOffset + k.rows));
        v.rowRange(0, v.rows).copyTo(vCache.rowRange(curPageOffset, curPageOffset + v.rows));
        curPageOffset += k.rows;
    }

    const Mat& getKeyCache(std::string layerId)
    {
        return kData[layerId];
    }

    const Mat& getValueCache(std::string layerId)
    {
        return vData[layerId];
    }
};

void setKVCacheManager(Net::Impl* netimpl);


CV__DNN_INLINE_NS_END
}}

#endif