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
//  BMP.hpp
//  QQView
//
//  Created by Tencent Research on 9/30/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#ifndef __ZXING_COMMON_UTIL_BMP_HPP__
#define __ZXING_COMMON_UTIL_BMP_HPP__

bool saveBMP(const char* BMPfname, int nWidth, int nHeight, unsigned char* buffer);

bool loadBMP(const char* BMPfname, int &nWidth, int &nHeight, unsigned char* buffer);

#endif // __ZXING_COMMON_UTIL_BMP_HPP__