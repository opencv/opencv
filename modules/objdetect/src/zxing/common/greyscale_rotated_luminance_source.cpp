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
 *  GreyscaleRotatedLuminanceSource.cpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
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


#include "greyscale_rotated_luminance_source.hpp"
#include "illegal_argument_exception.hpp"
#include "byte_matrix.hpp"

using zxing::ArrayRef;
using zxing::GreyscaleRotatedLuminanceSource;
using zxing::ByteMatrix;
using zxing::Ref;
using zxing::ErrorHandler;

// Note that dataWidth and dataHeight are not reversed, as we need to
// be able to traverse the greyData correctly, which does not get
// rotated.
GreyscaleRotatedLuminanceSource::
GreyscaleRotatedLuminanceSource(ArrayRef<char> greyData,
                                int dataWidth, int dataHeight,
                                int left, int top,
                                int width, int height,
                                ErrorHandler & err_handler)
: Super(width, height),
greyData_(greyData),
dataWidth_(dataWidth),
left_(left), top_(top) {
    // Intentionally comparing to the opposite dimension since we're rotated.
    if (left + width > dataHeight || top + height > dataWidth)
    {
        err_handler = IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
    }
}

// The API asks for rows, but we're rotated, so we return columns.
ArrayRef<char>
GreyscaleRotatedLuminanceSource::getRow(int y, ArrayRef<char> row, ErrorHandler & err_handler) const {
    if (y < 0 || y >= getHeight())
    {
        err_handler = IllegalArgumentErrorHandler("Requested row is outside the image.");
        return ArrayRef<char>();
    }
    if (!row || row->size() < getWidth())
    {
        row = ArrayRef<char>(getWidth());
    }
    int offset = (left_ * dataWidth_) + (dataWidth_ - 1 - (y + top_));

    for (int x = 0; x < getWidth(); x++) {
        row[x] = greyData_[offset];
        offset += dataWidth_;
    }
    return row;
}

ArrayRef<char> GreyscaleRotatedLuminanceSource::getMatrix() const {
    ArrayRef<char> result(getWidth() * getHeight());
    for (int y = 0; y < getHeight(); y++) {
        char* row = &result[y * getWidth()];
        int offset = (left_ * dataWidth_) + (dataWidth_ - 1 - (y + top_));
        for (int x = 0; x < getWidth(); x++) {
            row[x] = greyData_[offset];
            offset += dataWidth_;
        }
    }
    return result;
}

Ref<ByteMatrix> GreyscaleRotatedLuminanceSource::getByteMatrix() const{
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
