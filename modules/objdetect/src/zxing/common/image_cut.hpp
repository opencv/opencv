// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_IMAGE_CUT_HPP__
#define __ZXING_COMMON_IMAGE_CUT_HPP__

#include <stdint.h>
#include <vector>
#include "counted.hpp"
#include "byte_matrix.hpp"

namespace zxing
{

typedef struct _ImageCutResult
{
    ArrayRef<uint8_t> arrImage;
    int iWidth;
    int iHeight;
}ImageCutResult;

class ImageCut
{
public:
    ImageCut();
    ~ImageCut();
    
    static int cut(uint8_t * poImageData, int iWidth, int iHeight, int iTopLeftX, int iTopLeftY, int iBottomRightX, int iBottomRightY, ImageCutResult & result);
    static int cut( Ref<ByteMatrix> matrix , float fRatio, ImageCutResult & result);
};

}  // namespace zxing
#endif // __ZXING_COMMON_IMAGE_CUT_HPP__