// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

//
//  dmtximage.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXIMAGE_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXIMAGE_HPP__

#include <stdio.h>

namespace dmtx {

class DmtxImage {
public:
    DmtxImage() {}
    ~DmtxImage();
    
    int dmtxImageCreate(unsigned char *pxl_, int width_, int height_);
    int dmtxImageGetProp(int prop);
    int dmtxImageGetByteOffset(int x, int y);
    unsigned int dmtxImageGetPixelValue(int x, int y, /*@out@*/ int *value);
    unsigned int dmtxImageContainsInt(int margin, int x, int y);
    
private:
    int             width;
    int             height;
    int             bitsPerPixel;
    int             bytesPerPixel;
    int             rowPadBytes;
    int             rowSizeBytes;
    int             imageFlip;
    
    unsigned char  *pxl;
};

}  // namespace dmtx
#endif  // __ZXING_DATAMATRIX_LIBDMTX_DMTXIMAGE_HPP__ 
