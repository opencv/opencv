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
 *  Binarizer.cpp
 *  zxing
 *
 *  Created by Ralf Kistner on 16/10/2009.
 *  Copyright 2008 ZXing authors All rights reserved.
 *  Modified by Lukasz Warchol on 02/02/2010.
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

#include "binarizer.hpp"
#include <iostream>

namespace zxing {

Binarizer::Binarizer(Ref<LuminanceSource> source) : source_(source) {
    dataWidth = source->getWidth();
    dataHeight = source->getHeight();
    
    width = dataWidth;
    height = dataHeight;
    
#ifdef SUPPORT_ROTATE
    _bitCached = new uint8_t[dataWidth * dataHeight];
    memset(_bitCached, (uint8_t)0, (dataWidth * dataHeight * sizeof(uint8_t)));
    
    // By default cache mode is 0
    _cacheMode = 0;
    
    _binarizeCached = 0;
#endif
    
    matrix_ = NULL;
    matrix0_ = NULL;
    matrixInverted_ = NULL;
    
#ifdef SUPPORT_ROTATE
    matrix45_ = NULL;
    matrix90_ = NULL;
#endif
    
    histogramBinarized=false;
    usingHistogram=false;
}

Binarizer::~Binarizer() {
#ifdef SUPPORT_ROTATE
    delete [] _bitCached;
#endif
}

Ref<LuminanceSource> Binarizer::getLuminanceSource() const {
    return source_;
}

int Binarizer::getWidth() const {
    return width;
}

int Binarizer::getHeight() const {
    return height;
}

#ifdef SUPPORT_ROTATE
void Binarizer::setBinCache(int x, int y)
{
    int offset = y*dataWidth + x;
    _bitCached[offset] = 1;
}

int Binarizer::getBinCache(int x, int y)
{
    int offset = y*dataWidth + x;
    return static_cast<int>(_bitCached[offset]);
}

int Binarizer::rotateCounterClockwise()
{
    if (_binarizeCached == 0)
    {
        return -1;
    }
    
    _cacheMode = 1;
    
    if (matrix90_ != NULL)
    {
        return 0;
    }
    
    // Swap width and height when rotate 90 degrees
    width = dataHeight;
    height = dataWidth;
    
    Ref<BitMatrix> newMatrix (new BitMatrix(width, height));
    
    for (int y = 0; y < dataHeight; y++)
    {
        int newX = y;
        
        for (int x=0; x<dataWidth; x++)
        {
            int newY = dataWidth - x - 1;
            
            int bitInt = getBinCache(x, y);
            if (bitInt == 1)
            {
                newMatrix->set(newX, newY);
            }
        }
    }
    
    matrix90_ = newMatrix;
    
    matrix_ = matrix90_;
}

int Binarizer::rotateCounterClockwise45()
{
    return 0;
}
#else
int Binarizer::rotateCounterClockwise()
{
    return 0;
}

int Binarizer::rotateCounterClockwise45()
{
    return 0;
}
#endif

Ref<BitMatrix> Binarizer::getInvertedMatrix(ErrorHandler & err_handler)
{
    if (!matrix_)
    {
        return Ref<BitMatrix>();
    }
    
    if (matrixInverted_ == NULL)
    {
        matrixInverted_ = new BitMatrix(matrix_->getWidth(), matrix_->getHeight(), err_handler);
        matrixInverted_->copyOf(matrix_, err_handler);
        matrixInverted_->flipAll();
    }
    
    return matrixInverted_;
}

// Return different black matrix according to cacheMode
Ref<BitMatrix> Binarizer::getBlackMatrix(ErrorHandler & err_handler)
{
    (void) err_handler;
#ifdef SUPPORT_ROTATE
    if (_cacheMode == 0)
    {
        matrix_ = matrix0_;
    }
    else if (_cacheMode == 1)
    {
        matrix_ = matrix90_;
    }
    else if (_cacheMode == 2)
    {
        matrix_ = matrix45_;
    }
    // By default return matrix0
    else
    {
        matrix_ = matrix0_;
    }
#else
    matrix_ = matrix0_;
#endif
    
    
    return matrix_;
}

Ref<BitArray> Binarizer::getBlackRow(int y, Ref<BitArray> row, ErrorHandler & err_handler)
{
    if (!matrix_)
    {
        matrix_ = getBlackMatrix(err_handler);
        if (err_handler.errCode())   return Ref<BitArray>();
    }
    
    matrix_->getRow(y, row);
    return row;
}

#ifdef SUPPORT_ROTATE
void Binarizer::startBinCache()
{
    _binarizeCached = 0;
}
void Binarizer::endBinCache()
{
    _binarizeCached = 1;
}
#endif

ArrayRef<BINARIZER_BLOCK> Binarizer::getBlockArray(int size)
{
    ArrayRef<BINARIZER_BLOCK> blocks(new Array<BINARIZER_BLOCK>(size));
    
    for (int i = 0; i < blocks->size(); i++)
    {
        blocks[i].sum = 0;
        blocks[i].min = 0xFF;
        blocks[i].max = 0;
    }
    
    return blocks;
}
}  // namespace zxing
