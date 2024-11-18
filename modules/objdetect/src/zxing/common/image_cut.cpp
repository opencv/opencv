// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "image_cut.hpp"

#include <cstdio>

namespace zxing
{

ImageCut::ImageCut()
{
    
}

ImageCut::~ImageCut()
{
    
}

int ImageCut::cut(uint8_t * poImageData, int iWidth, int iHeight, int iTopLeftX, int iTopLeftY, int iBottomRightX, int iBottomRightY, ImageCutResult & result)
{
    if (iTopLeftX < 0 || iTopLeftX > iBottomRightX || iBottomRightX >= iWidth) return -1;
    if (iTopLeftY < 0 || iTopLeftY > iBottomRightY || iBottomRightY >= iHeight) return -1;
    int iNewWidth = iBottomRightX - iTopLeftX + 1;
    int iNewHeight = iBottomRightY - iTopLeftY + 1;
    
    result.arrImage = new Array<uint8_t>(iNewWidth * iNewHeight);
    result.iHeight = iNewHeight;
    result.iWidth = iNewWidth;
    
    int idx = 0;
    for (int y = 0; y < iHeight; ++y)
    {
        if (y < iTopLeftY || y > iBottomRightY) continue;
        for (int x = 0; x < iWidth; ++x)
        {
            if (x < iTopLeftX || x > iBottomRightX) continue;
            result.arrImage[idx++] = poImageData[y * iWidth + x];
        }
    }
    return 0;
}

int ImageCut::cut( Ref<ByteMatrix> matrix, float fRatio, ImageCutResult & result)
{
    int iWidth = matrix->getWidth();
    int iHeight = matrix->getHeight();
    
    int iMinX = iWidth * (1 - fRatio) / 2;
    int iMinY = iHeight * (1 - fRatio) / 2;
    int iMaxX = iWidth * (1 + fRatio) / 2 -1;
    int iMaxY = iHeight * (1 + fRatio) / 2 -1;
    
    if (iMinY < 0 || iMinY > iMaxX || iMaxX >= iWidth) return -1;
    if (iMinX < 0 || iMinX > iMaxX || iMaxX >= iWidth) return -1;
    int iNewHeight = iMaxY - iMinY + 1;
    int iNewWidth = iMaxX - iMinX + 1;
    
    result.arrImage = new Array<uint8_t>(iNewWidth * iNewHeight);
    result.iWidth = iNewWidth;
    result.iHeight = iNewHeight;
    
    int idx = 0;
    for (int y = 0; y < iNewHeight; ++y)
    {
        for (int x = 0; x < iNewWidth; ++x)
        {
            result.arrImage[idx++] = matrix->get(x+iMinX, y+iMinY);
        }
    }
    return 0;
}

}  // namespace zxing
