// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "binarzermgr.hpp"

#include "qbarsource.hpp"

using namespace zxing;

namespace cv {
BinarizerMgr::BinarizerMgr() :m_iNowRotateIndex(0), m_iNextOnceBinarizer(-1)
{
    m_vecRotateBinarizer.push_back(Hybrid);
    m_vecRotateBinarizer.push_back(FastWindow);
    m_vecRotateBinarizer.push_back(SimpleAdaptive);
}

BinarizerMgr::~BinarizerMgr()
{
}

zxing::Ref<Binarizer> BinarizerMgr::Binarize(zxing::Ref<LuminanceSource> source)
{
    BINARIZER binarizerIdx = m_vecRotateBinarizer[m_iNowRotateIndex];
    if (m_iNextOnceBinarizer >= 0)
    {
        binarizerIdx = (BINARIZER)m_iNextOnceBinarizer;
    }
    
    zxing::Ref<Binarizer> binarizer;
    
    switch (binarizerIdx)
    {
        case Hybrid:
            binarizer = new HybridBinarizer(source);
            break;
        case FastWindow:
            binarizer = new FastWindowBinarizer(source);
            break;
        case SimpleAdaptive:
            binarizer = new SimpleAdaptiveBinarizer(source);
            break;
        default:
            binarizer = new HybridBinarizer(source);
            break;
    }
    
    return binarizer;
}

void BinarizerMgr::switchBinarizer()
{
    m_iNowRotateIndex = (m_iNowRotateIndex+1) % m_vecRotateBinarizer.size();
}

int BinarizerMgr::getCurBinarizer()
{
    if (m_iNextOnceBinarizer != -1) return m_iNextOnceBinarizer;
    return m_vecRotateBinarizer[m_iNowRotateIndex];
}

void BinarizerMgr::setNextOnceBinarizer(int iBinarizerIndex)
{
    m_iNextOnceBinarizer = iBinarizerIndex;
}
}  // namesapce cv