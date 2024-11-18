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
//  dmtxbytelist.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXBYTELIST_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXBYTELIST_HPP__

#include <stdio.h>
#include "common.hpp"


namespace dmtx {

DmtxByteList dmtxByteListBuild(DmtxByte *storage, int capacity);
unsigned int dmtxByteListInit(DmtxByteList *list, int length, DmtxByte value);
// void dmtxByteListClear(DmtxByteList *list);
// unsigned int dmtxByteListHasCapacity(DmtxByteList *list);
unsigned int dmtxByteListCopy(DmtxByteList *dst, const DmtxByteList *src);
unsigned int dmtxByteListPush(DmtxByteList *list, DmtxByte value);
DmtxByte dmtxByteListPop(DmtxByteList *list, unsigned int *passFail);

}  // namespace dmtx

#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXBYTELIST_HPP__
