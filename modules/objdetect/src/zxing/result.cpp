// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Result.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 13/05/2008.
 *  Copyright 2008 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "result.hpp"

using zxing::Result;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::String;
using zxing::ResultPoint;

// VC++
using zxing::BarcodeFormat;

Result::Result(Ref<String> text,
               ArrayRef<char> rawBytes,
               ArrayRef< Ref<ResultPoint> > resultPoints,
               BarcodeFormat format) :
text_(text), rawBytes_(rawBytes), resultPoints_(resultPoints), format_(format) {
    charset_ = "UTF-8";
    
    qrcodeVersion_ = -1;
    pyramidLv_ = -1;
    binaryMethod_ = -1;
    ecLevel_ = '0';
#ifdef CALC_CODE_AREA_SCORE
    codeAreaScore_ = 0;
#endif
}

Result::Result(Ref<String> text,
               ArrayRef<char> rawBytes,
               ArrayRef< Ref<ResultPoint> > resultPoints,
               BarcodeFormat format,
               std::string charset) :
text_(text), rawBytes_(rawBytes), resultPoints_(resultPoints), format_(format), charset_(charset) {
    
    qrcodeVersion_ = -1;
    pyramidLv_ = -1;
    binaryMethod_ = -1;
    ecLevel_ = '0';
#ifdef CALC_CODE_AREA_SCORE
    codeAreaScore_ = 0;
#endif
}

Result::Result(Ref<String> text,
               ArrayRef<char> rawBytes,
               ArrayRef< Ref<ResultPoint> > resultPoints,
               BarcodeFormat format,
               std::string charset,
               int QRCodeVersion, std::string ecLevel, std::string charsetMode) :
text_(text), rawBytes_(rawBytes), resultPoints_(resultPoints), format_(format), charset_(charset),
qrcodeVersion_(QRCodeVersion), ecLevel_(ecLevel), charsetMode_(charsetMode) {
    pyramidLv_ = -1;
    binaryMethod_ = -1;
#ifdef CALC_CODE_AREA_SCORE
    codeAreaScore_ = 0;
#endif
}

Result::~Result() {
}

Ref<String> Result::getText() {
    return text_;
}

ArrayRef<char> Result::getRawBytes() {
    return rawBytes_;
}

ArrayRef< Ref<ResultPoint> > const& Result::getResultPoints() const {
    return resultPoints_;
}

ArrayRef< Ref<ResultPoint> >& Result::getResultPoints() {
    return resultPoints_;
}

void Result::setResultPoints(int idx, float x, float y){
    if (idx < 0 || idx >= resultPoints_->size()) return;
    resultPoints_[idx]->setX(x);
    resultPoints_[idx]->setY(y);
}

void Result::enlargeResultPoints(int scale){
    for (int i = 0; i < resultPoints_->size(); i++){
        resultPoints_[i]=Ref<ResultPoint>(new ResultPoint(resultPoints_[i]->getX()*static_cast<float>(scale), resultPoints_[i]->getY()*static_cast<float>(scale)));
    }
    return;
}
zxing::BarcodeFormat Result::getBarcodeFormat() const {
    return format_;
}

std::string Result::getCharset() const {
    return charset_;
}

std::string zxing::Result::getChartsetMode() const
{
    return charsetMode_;
}
