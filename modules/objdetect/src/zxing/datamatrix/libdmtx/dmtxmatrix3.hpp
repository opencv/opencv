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
//  dmtxmatrix3.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXMATRIX3_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXMATRIX3_HPP__

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

void dmtxMatrix3Translate(/*@out@*/ DmtxMatrix3 m, double tx, double ty);
void dmtxMatrix3Rotate(/*@out@*/ DmtxMatrix3 m, double angle);
void dmtxMatrix3Scale(/*@out@*/ DmtxMatrix3 m, double sx, double sy);
void dmtxMatrix3Shear(/*@out@*/ DmtxMatrix3 m, double shx, double shy);
unsigned int dmtxMatrix3LineSkewTop(DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewTopInv(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewSide(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
unsigned int dmtxMatrix3LineSkewSideInv(/*@out@*/ DmtxMatrix3 m, double b0, double b1, double sz);
void dmtxMatrix3Multiply(/*@out@*/ DmtxMatrix3 mOut, DmtxMatrix3 m0, DmtxMatrix3 m1);
void dmtxMatrix3MultiplyBy(DmtxMatrix3 m0, DmtxMatrix3 m1);
int dmtxMatrix3VMultiply(/*@out@*/ DmtxVector2 *vOut, DmtxVector2 *vIn, DmtxMatrix3 m);
int dmtxMatrix3VMultiplyBy(DmtxVector2 *v, DmtxMatrix3 m);

}  // namespace dmtx


#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXMATRIX3_HPP__
