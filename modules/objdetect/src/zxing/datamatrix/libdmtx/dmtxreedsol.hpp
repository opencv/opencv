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
//  dmtxreedsol.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXREEDSOL_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXREEDSOL_HPP__

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

unsigned int rsDecode(unsigned char *code, int sizeIdx, int fix);

static DmtxBoolean rsComputeSyndromes(DmtxByteList *syn, const DmtxByteList *rec, int blockErrorWords);
static DmtxBoolean rsFindErrorLocatorPoly(DmtxByteList *elp, const DmtxByteList *syn, int errorWordCount, int maxCorrectable);
static DmtxBoolean rsFindErrorLocations(DmtxByteList *loc, const DmtxByteList *elp);
static unsigned int rsRepairErrors(DmtxByteList *rec, const DmtxByteList *loc, const DmtxByteList *elp, const DmtxByteList *syn);

}  // namespace dmtx

#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXREEDSOL_HPP__
