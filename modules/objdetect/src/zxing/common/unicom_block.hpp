// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_UNICOM_BLOCK_HPP__
#define __ZXING_COMMON_UNICOM_BLOCK_HPP__
#include "counted.hpp"
#include "bit_matrix.hpp"
#include <vector>
#include <cstring>

namespace zxing
{
class UnicomBlock : public Counted
{
public:
    UnicomBlock(int iMaxHeight, int iMaxWidth);
    ~UnicomBlock();
    
    void init();
    void reset(Ref<BitMatrix> poImage);
    
    unsigned short getUnicomBlockIndex(int y, int x);
    
    int getUnicomBlockSize(int y, int x);
    
    int getMinPoint(int y, int x, int &iMinY, int &iMinX);
    int getMaxPoint(int y, int x, int &iMaxY, int &iMaxX);
    
private:
    void bfs(int y, int x);
    
    int m_iHeight;
    int m_iWidth;
    
    unsigned short m_iNowIdx;
    bool m_bInit;
    std::vector<unsigned short> m_vcIndex;
    std::vector<unsigned short> m_vcCount;
    std::vector<int> m_vcMinPnt;
    std::vector<int> m_vcMaxPnt;
    std::vector<int> m_vcQueue;
    static short SEARCH_POS[4][2];
    
    Ref<BitMatrix> m_poImage;
};
}  // namespace zxing
#endif // __ZXING_COMMON_UNICOM_BLOCK_HPP__
