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
 *  Copyright 2010 ZXing authors. All rights reserved.
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

#include "bit_matrix.hpp"
#include "illegal_argument_exception.hpp"

#include <climits>
#include <iostream>
#include <sstream>
#include <string>

using std::ostream;
using std::ostringstream;

using zxing::BitMatrix;
using zxing::BitArray;
using zxing::ArrayRef;
using zxing::Ref;
using zxing::ErrorHandler;

void BitMatrix::init(int width_, int height_, ErrorHandler & err_handler)
{
    if (width_ < 1 || height_ < 1)
    {
        err_handler = IllegalArgumentErrorHandler("Both dimensions must be greater than 0");
        return;
    }
    
    this->width = width_;
    this->height = height_;
    this->rowBitsSize = width_;
    bits = ArrayRef<unsigned char>(width_ * height_);
    rowOffsets = ArrayRef<int>(height_);
    
    rowOffsets[0] = 0;
    for (int i = 1; i < height_; i++)
    {
        rowOffsets[i] = rowOffsets[i - 1] + width_;
    }
    
    isInitRowCounters = false;
    isInitColsCounters = false;
}

void BitMatrix::init(int width_, int height_, bool* bitsPtr, ErrorHandler & err_handler)
{
    init(width_, height_, err_handler);
    if (err_handler.errCode())   return;
    memcpy(bits->data(), bitsPtr, width_ * height_ * sizeof(bool));
}

void BitMatrix::initRowCounters()
{
    if (isInitRowCounters == true)
    {
        return;
    }
    
    row_counters = std::vector<COUNTER_TYPE>(width * height, 0);
    row_counters_offset = std::vector<COUNTER_TYPE>(width * height, 0);
    row_point_offset = std::vector<COUNTER_TYPE>(width * height, 0);
    row_counter_offset_end = std::vector<COUNTER_TYPE>(height, 0);
    
    row_counters_recorded = std::vector<bool>(height, false);
    
    isInitRowCounters = true;
}

void BitMatrix::initColsCounters()
{
    if (isInitColsCounters == true)
    {
        return;
    }
    
    cols_counters = std::vector<COUNTER_TYPE>(width * height, 0);
    cols_counters_offset = std::vector<COUNTER_TYPE>(width * height, 0);
    cols_point_offset = std::vector<COUNTER_TYPE>(width * height, 0);
    cols_counter_offset_end = std::vector<COUNTER_TYPE>(width, 0);
    
    cols_counters_recorded = std::vector<bool>(width, false);
    
    isInitColsCounters = true;
}

BitMatrix::BitMatrix(int dimension, ErrorHandler & err_handler) {
    init(dimension, dimension, err_handler);
}

BitMatrix::BitMatrix(int width, int height, ErrorHandler &err_handler) {
    init(width, height, err_handler);
}

BitMatrix::BitMatrix(int width, int height, bool* bitsPtr, ErrorHandler & err_handler)
{
    init(width, height, bitsPtr, err_handler);
}

// Copy bitMatrix
void BitMatrix::copyOf(Ref<BitMatrix> bits_, ErrorHandler & err_handler)
{
    int width_ = bits_->getWidth();
    int height_ = bits_->getHeight();
    init(width_, height_, err_handler);
    
    for (int y = 0; y < height_; y++)
    {
        bool* rowPtr = bits_->getRowBoolPtr(y);
        setRowBool(y, rowPtr);
    }
}

bool BitMatrix::copyOf2(Ref<BitMatrix> bits_)
{
    if (bits_->getWidth() != this->getWidth() || bits_->getHeight() != this->getHeight())
    {
        return false;
    }
    memcpy(this->getPtr(), bits_->getPtr(), width * height * sizeof(bool));
    return true;
}

void BitMatrix::xxor(Ref<BitMatrix> bits_)
{
    if (width != bits_->getWidth() || height != bits_->getHeight())
    {
        return;
    }
    
    for (int y = 0; y < height && y < bits_->getHeight(); ++y)
    {
        bool* rowPtrA = bits_->getRowBoolPtr(y);
        bool* rowPtrB = getRowBoolPtr(y);
        
        for (int x = 0; x < width && x < bits_->getWidth(); ++x)
        {
            rowPtrB[x] = rowPtrB[x] ^ rowPtrA[x];
        }
        setRowBool(y, rowPtrB);
    }
}

BitMatrix::~BitMatrix()
{
}

void BitMatrix::flip(int x, int y)
{
    bits[rowOffsets[y]+x] = !(static_cast<bool>(bits[rowOffsets[y]+x]));
}

void BitMatrix::flipAll()
{
    bool* matrixBits = reinterpret_cast<bool*>(bits->data());
    for (int i = 0; i < bits->size(); i++)
    {
        matrixBits[i] = !matrixBits[i];
    }
}

void BitMatrix::flipRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler)
{
    if (top < 0 || left < 0)
    {
        err_handler = IllegalArgumentErrorHandler("Left and top must be nonnegative");
        return;
    }
    if (height_ < 1 || width_ < 1)
    {
        err_handler = IllegalArgumentErrorHandler("Height and width must be at least 1");
        return;
    }
    int right = left + width_;
    int bottom = top + height_;
    if (bottom > this->height || right > this->width)
    {
        err_handler = IllegalArgumentErrorHandler("The region must fit inside the matrix");
        return;
    }
    
    for (int y = top; y < bottom; y++) {
        for (int x = left; x < right; x++) {
            bits[rowOffsets[y] + x] ^= 1;
        }
    }
}

void BitMatrix::randomFlipRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler)
{
    if (top < 0 || left < 0)
    {
        err_handler = IllegalArgumentErrorHandler("Left and top must be nonnegative");
        return;
    }
    if (height_ < 1 || width_ < 1)
    {
        err_handler = IllegalArgumentErrorHandler("Height and width must be at least 1");
        return;
    }
    int right = left + width_;
    int bottom = top + height_;
    if (bottom > this->height || right > this->width)
    {
        err_handler = IllegalArgumentErrorHandler("The region must fit inside the matrix");
        return;
    }
    
    for (int y = top; y < bottom; y++)
    {
        for (int x = left; x < right; x++)
        {
            if ((x + y) % 2)
            {
                bits[rowOffsets[y] + x] = true;
            }
            else
            {
                bits[rowOffsets[y] + x] = false;
            }
        }
    }
}


void BitMatrix::setRegion(int left, int top, int width_, int height_, ErrorHandler & err_handler)
{
    if (top < 0 || left < 0)
    {
        err_handler = IllegalArgumentErrorHandler("Left and top must be nonnegative");
        return;
    }
    if (height_ < 1 || width_ < 1)
    {
        err_handler = IllegalArgumentErrorHandler("Height and width must be at least 1");
        return;
    }
    int right = left + width_;
    int bottom = top + height_;
    if (bottom > this->height || right > this->width)
    {
        err_handler = IllegalArgumentErrorHandler("The region must fit inside the matrix");
        return;
    }
    
    for (int y = top; y < bottom; y++) {
        for (int x = left; x < right; x++) {
            bits[rowOffsets[y] + x] = true;
        }
    }
}

void BitMatrix::fillHoles()
{
}

Ref<BitArray> BitMatrix::getRow(int y, Ref<BitArray> row)
{
    if (row.empty() || row->getSize() < width)
    {
        row = new BitArray(width);
    }
    
    // row->
    unsigned char* src = bits.data() + rowOffsets[y];
    row->setOneRow(src, width);
    
    return row;
}

ArrayRef<int> BitMatrix::getTopLeftOnBit() const
{
    int bitsOffset = 0;
    while (bitsOffset < bits->size() && bits[bitsOffset] != 0)
    {
        bitsOffset++;
    }
    if (bitsOffset == bits->size())
    {
        return ArrayRef<int>();
    }
    int y = bitsOffset / width;
    int x = bitsOffset % width;
    ArrayRef<int> res(2);
    res[0] = x;
    res[1] = y;
    return res;
}

ArrayRef<int> BitMatrix::getTopLeftOnBitNew() const
{
    int min_i_j = INT_MAX;
    int min_i = 0, min_j = 0;
    for (int i = 0; i < height/4; i++)
    {
        for (int j = 0; j < width/4; j++)
        {
            if (get(j, i))
            {
                if (i + j < min_i_j)
                {
                    min_i_j = i + j;
                    min_i = i;
                    min_j = j;
                }
            }
        }
    }
    
    ArrayRef<int> res(2);
    res[0] = min_j;
    res[1] = min_i;
    return res;
}

ArrayRef<int> BitMatrix::getTopRightOnBitNew() const
{
    int min_i_j = INT_MAX;
    int min_i = 0, min_j = width - 1;
    for (int i = 0; i < height / 4; i++)
    {
        for (int j = width - 1; j > width / 4; j--)
        {
            if (get(j, i))
            {
                int b = width - j - 1;
                if (i + b < min_i_j)
                {
                    min_i_j = i + b;
                    min_i = i;
                    min_j = j;
                }
            }
        }
    }
    ArrayRef<int> res(2);
    res[0] = min_j;
    res[1] = min_i;
    return res;
}

ArrayRef<int> BitMatrix::getBottomLeftOnBitNew() const
{
    int min_i_j = INT_MAX;
    int min_i = height - 1, min_j = 0;
    for (int i = height - 1; i > height / 4; i--)
    {
        int b = height - i - 1;
        for (int j = 0; j < width / 4; j++)
        {
            if (get(j, i))
            {
                if (j + b < min_i_j)
                {
                    min_i_j = j + b;
                    min_i = i;
                    min_j = j;
                }
            }
        }
    }
    ArrayRef<int> res(2);
    res[0] = min_j;
    res[1] = min_i;
    return res;
}

ArrayRef<int> BitMatrix::getPointOnRight(int x, int y, int x_max, bool flag)
{
    x_max = (std::min)(width - 1, x_max);
    int i = x;
    bool found = false;
    while (i < (x_max - 2)) {
        if (!get(i, y) && !get(i+1, y) && !get(x+2, y))
        {
            found = true;
            break;
        }
        i++;
    }
    int j = y;
    if (flag)
    {
        while (j < (y - 10))
        {
            if (!get(i, j) || j == 0)
            {
                break;
            }
            j--;
        }
    }
    else
    {
        while (j > (y + 10)) {
            if (!get(i, j) || j == (height - 1))
            {
                break;
            }
            j++;
        }
    }
    ArrayRef<int> res(2);
    if (found)
    {
        res[0] = (std::max)(i - 1, 0);
        res[1] = (std::max)(j - 1, 0);
    }
    else
    {
        res[0] = -1;
        res[1] = -1;
    }
    return res;
}

ArrayRef<int> BitMatrix::getPointOnLeft(int x, int y, int x_min, bool flag)
{
    (void)x;
    (void)y;
    (void)x_min;
    (void)flag;

    ArrayRef<int> res(2);
    
    return res;
}
ArrayRef<int> BitMatrix::getPointOnTop(int x, int y, int x_min, bool flag)
{
    (void)x;
    (void)y;
    (void)x_min;
    (void)flag;

    ArrayRef<int> res(2);
    return res;
}
ArrayRef<int> BitMatrix::getPointOnBottom(int x, int y, int x_max, bool flag)
{
    (void)x;
    (void)y;
    (void)x_max;
    (void)flag;

    ArrayRef<int> res(2);
    return res;
}

ArrayRef<int> BitMatrix::getBottomRightOnBitNew() const
{
    int min_i_j = INT_MAX;
    int min_i = height - 1, min_j = width - 1;
    for (int i = height - 1; i > height / 4; i--)
    {
        int a = height - i - 1;
        for (int j = width - 1; j > width / 4; j--)
        {
            if (get(j, i))
            {
                int b = width - j - 1;
                if (a + b < min_i_j)
                {
                    min_i_j = a + b;
                    min_i = i;
                    min_j = j;
                }
            }
        }
    }
    ArrayRef<int> res(2);
    res[0] = min_j;
    res[1] = min_i;
    return res;
}

ArrayRef<int> BitMatrix::getBottomRightOnBit() const
{
    int bitsOffset = bits->size() - 1;
    while (bitsOffset >= 0 && bits[bitsOffset] != 0)
    {
        bitsOffset--;
    }
    if (bitsOffset < 0)
    {
        return ArrayRef<int>();
    }
    
    int y = bitsOffset / width;
    int x = bitsOffset % width;
    ArrayRef<int> res(2);
    res[0] = x;
    res[1] = y;
    return res;
}

void BitMatrix::getRowBool(int y, bool* getrow)
{
    int offset = rowOffsets[y];
    unsigned char* src = bits.data() + offset;
    memcpy(getrow, src, rowBitsSize*sizeof(bool));
}


void BitMatrix::setRowBool(int y, bool* row)
{
    int offset = rowOffsets[y];
    unsigned char* dst = bits.data() + offset;
    memcpy(dst, row, rowBitsSize*sizeof(bool));
    
    return;
}

bool* BitMatrix::getRowBoolPtr(int y)
{
    int offset = y * rowBitsSize;
    unsigned char* src = bits.data() + offset;
    return reinterpret_cast<bool*>(src);
}


void BitMatrix::clear()
{
    int size = bits->size();
    
    unsigned char* dst = bits->data();
    memset(dst, 0, size*sizeof(unsigned char));
}

int BitMatrix::getWidth() const
{
    return width;
}

int BitMatrix::getHeight() const
{
    return height;
}

COUNTER_TYPE* BitMatrix::getRowPointInRecords(int y)
{
    if (!row_point_offset[y])
    {
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_point_offset[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

COUNTER_TYPE* BitMatrix::getRowRecords(int y)
{
    if (!row_counters_recorded[y]){
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_counters[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

COUNTER_TYPE* BitMatrix::getRowRecordsOffset(int y)
{
    if (!row_counters_recorded[y])
    {
        setRowRecords(y);
    }
    int offset = y * width;
    COUNTER_TYPE* counters = &row_counters_offset[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

bool BitMatrix::getRowFirstIsWhite(int y)
{
    bool is_white = !get(0, y);
    return is_white;
}

bool BitMatrix::getRowLastIsWhite(int y)
{
    bool last_is_white = !get(width - 1, y);
    return last_is_white;
}

COUNTER_TYPE BitMatrix::getRowCounterOffsetEnd(int y)
{
    if (!row_counters_recorded[y]){
        setRowRecords(y);
    }
    return row_counter_offset_end[y];
}

void BitMatrix::setRowRecords(int y)
{
    COUNTER_TYPE* cur_row_counters = &row_counters[0] + y * width;
    COUNTER_TYPE* cur_row_counters_offset = &row_counters_offset[0] + y * width;
    COUNTER_TYPE* cur_row_point_in_counters = &row_point_offset[0] + y * width;
    int end = width;
    
    bool* rowBit=getRowBoolPtr(y);
    bool isWhite = !rowBit[0];
    int counterPosition = 0;
    int i = 0;
    cur_row_counters_offset[0] = 0;
    while (i < end)
    {
        if (rowBit[i] ^ isWhite)
        {  // that is, exactly one is true
            cur_row_counters[counterPosition]++;
        }
        else
        {
            counterPosition++;
            if (counterPosition == end)
            {
                break;
            }
            else
            {
                cur_row_counters[counterPosition] = 1;
                isWhite = !isWhite;
                cur_row_counters_offset[counterPosition] = i;
            }
        }
        cur_row_point_in_counters[i] = counterPosition;
        i++;
    }
    
    // use the last row__onedReaderData->counter_size to record _onedReaderData->counter_size
    row_counter_offset_end[y] = counterPosition < end ? (counterPosition + 1):end;
    
    row_counters_recorded[y] = true;
}

COUNTER_TYPE* BitMatrix::getColsPointInRecords(int x)
{
    if (!cols_point_offset[x])
    {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_point_offset[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

COUNTER_TYPE* BitMatrix::getColsRecords(int x)
{
    if (!cols_counters_recorded[x])
    {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_counters[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

COUNTER_TYPE* BitMatrix::getColsRecordsOffset(int x)
{
    if (!cols_counters_recorded[x])
    {
        setColsRecords(x);
    }
    int offset = x * height;
    COUNTER_TYPE* counters = &cols_counters_offset[0] + offset;
    return  reinterpret_cast<COUNTER_TYPE*>(counters);
}

COUNTER_TYPE BitMatrix::getColsCounterOffsetEnd(int x)
{
    if (!cols_counters_recorded[x])
    {
        setColsRecords(x);
    }
    return cols_counter_offset_end[x];
}

void BitMatrix::setColsRecords(int x)
{
    
    COUNTER_TYPE* cur_cols_counters = &cols_counters[0] + x*height;
    COUNTER_TYPE* cur_cols_counters_offset = &cols_counters_offset[0] + x*height;
    COUNTER_TYPE* cur_cols_point_in_counters = &cols_point_offset[0] + x*height;
    int end = height;
    
    bool* rowBit = getRowBoolPtr(0);
    bool isWhite = !rowBit[0];
    int counterPosition = 0;
    int i = 0;
    cur_cols_counters_offset[0] = 0;
    while (i < end)
    {
        if (rowBit[i] ^ isWhite)
        {  // that is, exactly one is true
            cur_cols_counters[counterPosition]++;
        }
        else
        {
            counterPosition++;
            if (counterPosition == end)
            {
                break;
            }
            else
            {
                cur_cols_counters[counterPosition] = 1;
                isWhite = !isWhite;
                cur_cols_counters_offset[counterPosition] = i;
            }
        }
        cur_cols_point_in_counters[i] = counterPosition;
        i++;
        rowBit += width;
    }
    
    // use the last row__onedReaderData->counter_size to record _onedReaderData->counter_size
    cols_counter_offset_end[x] = counterPosition < end ? (counterPosition + 1) : end;
    
    cols_counters_recorded[x] = true;
};
