// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_BINARZERMGR_HPP__
#define __OPENCV_BINARZERMGR_HPP__

#include "zxing/zxing.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/binarizer.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/common/hybrid_binarizer.hpp"
#include "zxing/common/fast_window_binarizer.hpp"
#include "zxing/common/simple_adaptive_binarizer.hpp"

namespace cv {
class BinarizerMgr
{
    enum BINARIZER
    {
        Hybrid = 0,
        FastWindow = 1,
        SimpleAdaptive = 2,
        GlobalHistogram = 3,
        OTSU = 4,
        Niblack = 5,
        Adaptive = 6,
        HistogramBackground = 7
    };
    
public:
    BinarizerMgr();
    ~BinarizerMgr();
    
    zxing::Ref<zxing::Binarizer> Binarize(zxing::Ref<zxing::LuminanceSource> source);
    
    void switchBinarizer();
    
    int getCurBinarizer();
    
    void setNextOnceBinarizer(int iBinarizerIndex);
    
private:
    int m_iNowRotateIndex;
    int m_iNextOnceBinarizer;
    std::vector<BINARIZER> m_vecRotateBinarizer;
};
}  // namesapce cv
#endif // __OPENCV_BINARZERMGR_HPP__