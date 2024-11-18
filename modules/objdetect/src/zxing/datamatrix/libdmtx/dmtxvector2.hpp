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
//  dmtxvector2.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXVECTOR2_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXVECTOR2_HPP__

#include <stdio.h>
#include "common.hpp"

namespace dmtx {

DmtxVector2* dmtxVector2Sub(DmtxVector2 *vOut,const DmtxVector2 *v1,const DmtxVector2 *v2);

double dmtxVector2Cross(const DmtxVector2 *v1, const DmtxVector2 *v2);

double dmtxVector2Norm(DmtxVector2 *v);

double dmtxVector2Dot(const DmtxVector2 *v1,const DmtxVector2 *v2);

double dmtxVector2Mag(const DmtxVector2 *v);

unsigned int dmtxRay2Intersect(DmtxVector2 *point, const DmtxRay2 *p0, const DmtxRay2 *p1);

unsigned int dmtxPointAlongRay2(DmtxVector2 *point, const DmtxRay2 *r, double t);

}  // namespace dmtx

#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXVECTOR2_HPP__
