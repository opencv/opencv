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
//  dmtxplacemod.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXPLACEMOD_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXPLACEMOD_HPP__

#include <stdio.h>

namespace dmtx {
int modulePlacementEcc200(unsigned char *modules, unsigned char *codewords, int sizeIdx, int moduleOnColor);
static void patternShapeStandard(unsigned char *modules, int mappingRows, int mappingCols, int row, int col, unsigned char *codeword, int moduleOnColor);
static void patternShapeSpecial1(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void patternShapeSpecial2(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void patternShapeSpecial3(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void patternShapeSpecial4(unsigned char *modules, int mappingRows, int mappingCols, unsigned char *codeword, int moduleOnColor);
static void placeModule(unsigned char *modules, int mappingRows, int mappingCols, int row, int col,
      unsigned char *codeword, int mask, int moduleOnColor);
}  // namespace dmtx
#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXPLACEMOD_HPP__
