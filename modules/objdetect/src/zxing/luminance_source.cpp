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
 *  LuminanceSource.cpp
 *  zxing
 *
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

#include <sstream>
#include "luminance_source.hpp"
#include "inverted_luminance_source.hpp"
#include "common/illegal_argument_exception.hpp"

using zxing::Ref;
using zxing::LuminanceSource;

LuminanceSource::LuminanceSource(int width_, int height_) :width(width_), height(height_), tvInter(-1) {}

LuminanceSource::~LuminanceSource() {}

bool LuminanceSource::isCropSupported() const {
    return false;
}

Ref<LuminanceSource> LuminanceSource::crop(int, int, int, int) const {
    return Ref<LuminanceSource>();
}

bool LuminanceSource::isRotateSupported() const {
    return false;
}

Ref<LuminanceSource> LuminanceSource::rotateCounterClockwise() const {
    return Ref<LuminanceSource>();
}

LuminanceSource::operator std::string() const {
    ArrayRef<char> row;
    std::ostringstream oss;
    zxing::ErrorHandler  err_handler;
    for (int y = 0; y < getHeight(); y++) {
        err_handler.reset();
        row = getRow(y, row, err_handler);
        if (err_handler.errCode()) continue;
        for (int x = 0; x < getWidth(); x++) {
            int luminance = row[x] & 0xFF;
            char c;
            if (luminance < 0x40)
            {
                c = '#';
            }
            else if (luminance < 0x80)
            {
                c = '+';
            }
            else if (luminance < 0xC0)
            {
                c = '.';
            }
            else
            {
                c = ' ';
            }
            oss << c;
        }
        oss << '\n';
    }
    return oss.str();
}

Ref<LuminanceSource> LuminanceSource::invert() const
{
    // N.B.: this only works because we use counted objects with the
    // count in the object. This is _not_ how things like shared_ptr
    // work. They do not allow you to make a smart pointer from a native
    // pointer more than once. If we ever switch to (something like)
    // shared_ptr's, the luminace source is going to have keep a weak
    // pointer to itself from which it can create a strong pointer as
    // needed. And, FWIW, that has nasty semantics in the face of
    // exceptions during construction.
    
    return Ref<LuminanceSource>
    (new InvertedLuminanceSource(Ref<LuminanceSource>(const_cast<LuminanceSource*>(this))));
}

void LuminanceSource::denoseLuminanceSource(int inter){
    tvInter = inter;
}
