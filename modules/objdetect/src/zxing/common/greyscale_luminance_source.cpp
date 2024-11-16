// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  GreyscaleLuminanceSource.cpp
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

#include "greyscale_luminance_source.hpp"
#include "greyscale_rotated_luminance_source.hpp"
#include "illegal_argument_exception.hpp"
#include "byte_matrix.hpp"

using zxing::Ref;
using zxing::ArrayRef;
using zxing::LuminanceSource;
using zxing::GreyscaleLuminanceSource;
using zxing::ByteMatrix;
using zxing::ErrorHandler;

GreyscaleLuminanceSource::
GreyscaleLuminanceSource(ArrayRef<char> greyData,
                         int dataWidth, int dataHeight,
                         int left, int top,
                         int width, int height,
                         ErrorHandler & err_handler)
: Super(width, height),
greyData_(greyData),
dataWidth_(dataWidth), dataHeight_(dataHeight),
left_(left), top_(top)
{
    if (left + width > dataWidth || top + height > dataHeight || top < 0 || left < 0)
    {
        err_handler = IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
    }
}

ArrayRef<char> GreyscaleLuminanceSource::getRow(int y, ArrayRef<char> row, ErrorHandler & err_handler) const
{
    if (y < 0 || y >= this->getHeight())
    {
        err_handler = IllegalArgumentErrorHandler("Requested row is outside the image.");
        return ArrayRef<char>();
    }
    int width = getWidth();
    if (!row || row->size() < width)
    {
        ArrayRef<char> temp(width);
        row = temp;
    }
    int offset = (y + top_) * dataWidth_ + left_;
    memcpy(&row[0], &greyData_[offset], width);
    return row;
}

ArrayRef<char> GreyscaleLuminanceSource::getMatrix() const
{
    int size = getWidth() * getHeight();
    ArrayRef<char> result(size);
    if (left_ == 0 && top_ == 0 && dataWidth_ == getWidth() && dataHeight_ == getHeight())
    {
        memcpy(&result[0], &greyData_[0], size);
    }
    else
    {
        for (int row = 0; row < getHeight(); row++) {
            memcpy(&result[row * getWidth()], &greyData_[(top_ + row) * dataWidth_ + left_], getWidth());
        }
    }
    return result;
}

Ref<LuminanceSource> GreyscaleLuminanceSource::rotateCounterClockwise(ErrorHandler & err_handler) const
{
    // Intentionally flip the left, top, width, and height arguments as
    // needed. dataWidth and dataHeight are always kept unrotated.
    Ref<LuminanceSource> result(
                                 new GreyscaleRotatedLuminanceSource(greyData_,
                                                                     dataWidth_, dataHeight_,
                                                                     top_, left_, getHeight(), getWidth(), err_handler));
    if (err_handler.ErrCode()) return Ref<LuminanceSource>();
    return result;
}

Ref<ByteMatrix> GreyscaleLuminanceSource::getByteMatrix() const
{
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
