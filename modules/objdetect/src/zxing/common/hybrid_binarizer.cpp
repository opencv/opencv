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
 *  HybridBinarizer.cpp
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

#include "hybrid_binarizer.hpp"
#include "illegal_argument_exception.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <stdint.h>

using namespace zxing;

// This class uses 5*5 blocks to compute local luminance, where each block is 8*8 pixels
// So this is the smallest dimension in each axis we can accept.
namespace {
const int BLOCK_SIZE_POWER = 3;
const int BLOCK_SIZE = 1 << BLOCK_SIZE_POWER;  // ...0100...00
const int BLOCK_SIZE_MASK = BLOCK_SIZE - 1;   // ...0011...11
const int MINIMUM_DIMENSION = BLOCK_SIZE * 5;  // bLOCK_SIZE * 5;
}  // namespace

HybridBinarizer::HybridBinarizer(Ref<LuminanceSource> source) :
GlobalHistogramBinarizer(source) {
    int width = source->getWidth();
    int height = source->getHeight();
    
    int subWidth = width >> BLOCK_SIZE_POWER;
    if ((width & BLOCK_SIZE_MASK) != 0)
    {
        subWidth++;
    }
    int subHeight = height >> BLOCK_SIZE_POWER;
    if ((height & BLOCK_SIZE_MASK) != 0)
    {
        subHeight++;
    }
    
    grayByte_ = source->getByteMatrix();
    
    blocks_ = getBlockArray(subWidth * subHeight);
    
    width_ = width;
    height_ = height;
    subWidth_ = subWidth;
    subHeight_ = subHeight;
    
    initBlocks();
    initBlockIntegral();
}

HybridBinarizer::~HybridBinarizer() {
}

Ref<Binarizer>
HybridBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer> (new GlobalHistogramBinarizer(source));
}

/* init integral
 */
int HybridBinarizer::initBlockIntegral()
{
    int width = subWidth_ + 1;
    int height = subHeight_ + 1;
    
    blockIntegral_ = new Array<int>(width * height);
    
    int* integral = blockIntegral_->data();
    
    // first row only
    int rs = 0;
    
    for (int j = 0; j < width; j++)
    {
        integral[j] = 0;
    }
    for (int i = 0; i < height; i++)
    {
        integral[i * width] = 0;
    }
    for (int j = 0; j < subWidth_; j++)
    {
        rs += blocks_[j].threshold;
        integral[width + j + 1] = rs;
    }
    
    // remaining cells are sum above and to the left
    int offset = width;
    int offsetBlock = 0;
    
    for (int i = 1; i < subHeight_; ++i)
    {
        offsetBlock = i * subWidth_;
        
        rs = 0;
        
        offset += width;
        
        for (int j = 0; j < subWidth_; ++j)
        {
            rs += blocks_[offsetBlock + j].threshold;
            integral[offset + j + 1] = rs + integral[offset - width + j + 1];
        }
    }
    
    return 1;
}

/**
 * Calculates the final BitMatrix once for all requests. This could be called once from the
 * constructor instead, but there are some advantages to doing it lazily, such as making
 * profiling easier, and not doing heavy lifting when callers don't expect it.
 */
Ref<BitMatrix> HybridBinarizer::getBlackMatrix(ErrorHandler &err_handler)
{
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_)
    {
        binarizeByBlock(0, err_handler);
        if (err_handler.errCode())   return Ref<BitMatrix>();
    }
    
    // First call binarize image in child class to get matrix0_ and binCache
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackMatrix(err_handler);
}

/**
 * Calculate black row from BitMatrix
 * If BitMatrix has been calculated then just get the row
 * If BitMatrix has not been calculated then call getBlackMatrix first
 * @Author : valiantliu
 */
Ref<BitArray> HybridBinarizer::getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler) 
{
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_)
    {
        binarizeByBlock(0, err_handler);
        if (err_handler.errCode())   return Ref<BitArray>();
    }
    
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

namespace {
inline int cap(int value, int min, int max) {
    return value < min ? min : value > max ? max : value;
}
}  // namespace

// For each block in the image, calculates the average black point using a 5*5 grid
// of the blocks around it. Also handles the corner cases (fractional blocks are computed based
// on the last pixels in the row/column which are also used in the previous block.)

#define THRES_BLOCKSIZE 2

#ifdef USE_LEVEL_BINARIZER
// No use of level now
ArrayRef<int> HybridBinarizer::getBlackPoints(int level)
{
    (void)level;

    int blackWidth, blackHeight;
    
    blackWidth = subWidth_;
    blackHeight = subHeight_;
    
    ArrayRef<int> blackPoints(blackWidth*blackHeight);
    
    int* blackArray = blackPoints->data();
    
    int offset = 0;
    for (int i = 0; i < blackHeight; i++)
    {
        offset = i*blackWidth;
        for (int j = 0; j<blackWidth; j++)
        {
            blackArray[offset+j] = blocks_[offset+j].threshold;
        }
    }
    
    return blackPoints;
}

// Original code 20140606
void HybridBinarizer::calculateThresholdForBlock(Ref<ByteMatrix>& luminances,
                                                 int subWidth,
                                                 int subHeight,
                                                 int width,
                                                 int height,
                                                 int SIZE_POWER,
                                                 // arrayRef<int> &blackPoints,
                                                 Ref<BitMatrix> const& matrix,
                                                 ErrorHandler & err_handler)
{
    int block_size = 1 << SIZE_POWER;
    
    int maxYOffset = height - block_size;
    int maxXOffset = width - block_size;
    
    int* blockIntegral = blockIntegral_->data();
    
    int blockArea = ((2*THRES_BLOCKSIZE+1)*(2*THRES_BLOCKSIZE+1));
    
    for (int y = 0; y < subHeight; y++) {
        int yoffset = y << SIZE_POWER;
        if (yoffset > maxYOffset)
        {
            yoffset = maxYOffset;
        }
        for (int x = 0; x < subWidth; x++) {
            int xoffset = x << SIZE_POWER;
            if (xoffset > maxXOffset)
            {
                xoffset = maxXOffset;
            }
            int left = cap(x, THRES_BLOCKSIZE, subWidth - THRES_BLOCKSIZE - 1);
            int top = cap(y, THRES_BLOCKSIZE, subHeight - THRES_BLOCKSIZE - 1);
            
            int sum = 0;
            
            int offset1 = (top - THRES_BLOCKSIZE)*(subWidth+1) + left - THRES_BLOCKSIZE;
            int offset2 = (top + THRES_BLOCKSIZE + 1)*(subWidth+1) + left - THRES_BLOCKSIZE;
            
            int blocksize = THRES_BLOCKSIZE*2 + 1;
            
            sum = blockIntegral[offset1] - blockIntegral[offset1 + blocksize] - blockIntegral[offset2] + blockIntegral[offset2 + blocksize];
            
            int average = sum / blockArea;
            thresholdBlock(luminances, xoffset, yoffset, average, width, matrix, err_handler);
            if (err_handler.errCode())   return;
        }
    }
}
#else
void HybridBinarizer::calculateThresholdForBlock(Ref<ByteMatrix>& luminances,
                                                 int subWidth,
                                                 int subHeight,
                                                 int width,
                                                 int height,
                                                 ArrayRef<int> &blackPoints,
                                                 Ref<BitMatrix> const& matrix,
                                                 ErrorHandler & err_handler) {
    int maxYOffset = height - BLOCK_SIZE;
    int maxXOffset = width - BLOCK_SIZE;
    
#ifndef USE_GOOGLE_CODE
    
    int subSumWidth = subWidth + 5;
    int subSumHeight = subHeight + 5;
    int subSumSize = subSumWidth * subSumHeight;
    
    int* secondSumRow=&subSumPoints[subSumWidth];
    for (int i = 0; i < subSumWidth; i++) {
        // first row
        subSumPoints[i] = 0;
        // second row
        if (i == 0)
        {
            subSumColumn[i] = 0;
            secondSumRow[i] = 0;
        }
        else
        {
            int tmp_i = cap(i, 3, subWidth+2)-3;
            secondSumRow[i] = secondSumRow[i-1]+blackPoints[tmp_i];
            subSumColumn[i] = blackPoints[tmp_i];
        }
    }
    
    for (int i = 2; i < subSumHeight; i++){
        int* tmpSumRow = &subSumPoints[i*subSumWidth];
        int tmp_i = cap(i, 3, subHeight+2)-3;
        int* tmpPointRow = &blackPoints[tmp_i*subWidth];
        tmpSumRow[0] = 0;
        for (int j = 1; j < subSumWidth; j++){
            int tmp_j = cap(j, 3, subWidth + 2) - 3;
            subSumColumn[j] = subSumColumn[j] + tmpPointRow[tmp_j];
            tmpSumRow[j] = tmpSumRow[j-1] + subSumColumn[j];
        }
    }
    
#ifdef USE_SET_INT
    int setIntCircle = BITS_PER_WORD/BITS_PER_BYTE;
    int* averages = new int[setIntCircle];
#endif
    
    for (int y = 0; y < subHeight; y++){
        int yoffset = y << BLOCK_SIZE_POWER;
        if (yoffset > maxYOffset) yoffset = maxYOffset;
        
        int* tmpTopSumRow = &subSumPoints[y*subSumWidth];
        int* tmpDownSumRow = &subSumPoints[(y+5)*subSumWidth];
        unsigned char* pLuminaceTemp=luminances->getByteRow(yoffset);
        for (int x = 0; x < subWidth; x++) {
#ifndef USE_SET_INT
            int xoffset = x << BLOCK_SIZE_POWER;
            if (xoffset > maxXOffset) xoffset = maxXOffset;
#endif
            int sum=tmpDownSumRow[x+5]-tmpDownSumRow[x]
            -tmpTopSumRow[x+5]+tmpTopSumRow[x];
            int average = sum / 25;
#ifndef USE_SET_INT
            thresholdBlock(luminances, xoffset, yoffset, average, width, matrix, err_handler);
            if (err_handler.errCode())  return;
#else
            // handle 4 blacks one time
            int k = x % setIntCircle;
            averages[k] = average;
            if (k == (setIntCircle-1)){
                int tmp_x = x-setIntCircle+1;
                int xoffset = tmp_x<< BLOCK_SIZE_POWER;
                thresholdFourBlocks(luminances, xoffset, yoffset, averages, width, matrix);
            }
#endif
        }
        
    }
#ifdef USE_SET_INT
    delete [] averages;
#endif
    
#else
    ////////  The original google codes ////////
    
    int blockArea = ((2*THRES_BLOCKSIZE+1)*(2*THRES_BLOCKSIZE+1));
    
    for (int y = 0; y < subHeight; y++) {
        int yoffset = y << BLOCK_SIZE_POWER;
        if (yoffset > maxYOffset)
        {
            yoffset = maxYOffset;
        }
        for (int x = 0; x < subWidth; x++) {
            int xoffset = x << BLOCK_SIZE_POWER;
            if (xoffset > maxXOffset)
            {
                xoffset = maxXOffset;
            }
            int left = cap(x, THRES_BLOCKSIZE, subWidth - THRES_BLOCKSIZE - 1);
            int top = cap(y, THRES_BLOCKSIZE, subHeight - THRES_BLOCKSIZE - 1);
            int sum = 0;
            for (int z = -THRES_BLOCKSIZE; z <= THRES_BLOCKSIZE; z++) {
                int *blackRow = &blackPoints[(top + z) * subWidth];
                
                for (int k=-THRES_BLOCKSIZE; k<=THRES_BLOCKSIZE; k++)
                {
                    sum += blackRow[left+k];
                }
                
            }
            int average = sum / blockArea;
            thresholdBlock(luminances, xoffset, yoffset, average, width, matrix, err_handler);
            if (err_handler.errCode())  return;
        }
    }
#endif
    
}

#endif

#ifdef USE_SET_INT
void HybridBinarizer::thresholdFourBlocks(Ref<ByteMatrix>& luminances,
                                          int xoffset,
                                          int yoffset,
                                          int* thresholds,
                                          int stride,
                                          Ref<BitMatrix> const& matrix){
    int setIntCircle = BITS_PER_WORD / BITS_PER_BYTE;
    for (int y = 0; y < BLOCK_SIZE; y++){
        unsigned char* pTemp = luminances->getByteRow(yoffset + y);
        pTemp = pTemp + xoffset;
        unsigned int valueInt = 0;
        int bitPosition = 0;
        for (int k = 0; k < setIntCircle; k++){
            for (int x = 0; x < BLOCK_SIZE; x++){
                int pixel = *pTemp++;
                if (pixel <= thresholds[k])
                {
                    valueInt |= (unsigned int)1 << bitPosition;
                }
                bitPosition++;
            }
        }
        matrix->setIntOneTime(xoffset, yoffset+y, valueInt);
    }
    return;
}
#endif

// Applies a single threshold to a block of pixels
void HybridBinarizer::thresholdBlock(Ref<ByteMatrix>& luminances,
                                     int xoffset,
                                     int yoffset,
                                     int threshold,
                                     int stride,
                                     Ref<BitMatrix> const& matrix,
                                     ErrorHandler & err_handler) {
    (void)stride;
    
    int rowBitsSize = matrix->getRowBitsSize();
    int rowSize = width;
    
    int rowBitStep = rowBitsSize - BLOCK_SIZE;
    int rowStep = rowSize - BLOCK_SIZE;
    
    unsigned char* pTemp = luminances->getByteRow(yoffset, err_handler);
    if (err_handler.errCode())   return;
    bool* bpTemp = matrix->getRowBoolPtr(yoffset);
    
    pTemp += xoffset;
    bpTemp += xoffset;
    
    for (int y = 0; y < BLOCK_SIZE; y++)
    {
        for (int x = 0; x < BLOCK_SIZE; x++)
        {
            // comparison needs to be <= so that black == 0 pixels are black even if the threshold is 0.
            *bpTemp++ = (*pTemp++ <= threshold) ? true : false;
        }
        
        pTemp += rowBitStep;
        bpTemp += rowStep;
    }
}


void HybridBinarizer::thresholdIrregularBlock(Ref<ByteMatrix>& luminances,
                                              int xoffset,
                                              int yoffset,
                                              int blockWidth,
                                              int blockHeight,
                                              int threshold,
                                              int stride,
                                              Ref<BitMatrix> const& matrix,
                                              ErrorHandler & err_handler){
    (void)stride;

    for (int y = 0; y < blockHeight; y++) {
        unsigned char* pTemp = luminances->getByteRow(yoffset+y, err_handler);
        if (err_handler.errCode())   return;
        pTemp = pTemp + xoffset;
        for (int x = 0; x < blockWidth; x++){
            // comparison needs to be <= so that black == 0 pixels are black even if the threshold is 0.
            int pixel = *pTemp++;
            if (pixel <= threshold)
            {
                matrix->set(xoffset + x, yoffset + y);
            }
        }
    }
}

namespace {

#ifndef USE_GOOGLE_CODE
int getBlackPointFromNeighbors(ArrayRef<int> &blackPoints, int subWidth, int x, int y) {
    int neihbors=0;
    int* pTemp=&blackPoints[(y-1)*subWidth+x];
    neihbors+=*pTemp;
    --pTemp;
    neihbors+=*pTemp;
    pTemp=pTemp+subWidth;
    neihbors+=*pTemp;
    return neihbors>>2;
}
#else

#ifdef USE_LEVEL_BINARIZER
////////  The original google codes ////////
inline int getBlackPointFromNeighbors(ArrayRef<BINARIZER_BLOCK> block, int subWidth, int x, int y) {
    return (block[(y-1)*subWidth+x].threshold +
            2*block[y*subWidth+x-1].threshold +
            block[(y-1)*subWidth+x-1].threshold) >> 2;
}
#else
////////  The original google codes ////////
inline int getBlackPointFromNeighbors(ArrayRef<int> blackPoints, int subWidth, int x, int y) {
    return (blackPoints[(y-1)*subWidth+x] +
            2*blackPoints[y*subWidth+x-1] +
            blackPoints[(y-1)*subWidth+x-1]) >> 2;
}
#endif

#endif
}  // namespace

#define MIN_DYNAMIC_RANGE 24

#ifdef USE_LEVEL_BINARIZER
// Calculates a single black point for each block of pixels and saves it away. 
int HybridBinarizer::initBlocks() 
{
    Ref<ByteMatrix>& luminances = grayByte_;
    int subWidth = subWidth_;
    int subHeight = subHeight_;
    int width = width_;
    int height = height_;
    
    unsigned char* bytes = luminances->bytes;
    
    const int minDynamicRange = 24;
    
    for (int y = 0; y < subHeight; y++){
        int yoffset = y << BLOCK_SIZE_POWER;
        int maxYOffset = height - BLOCK_SIZE;
        if (yoffset > maxYOffset) yoffset = maxYOffset;
        for (int x = 0; x < subWidth; x++){
            int xoffset = x << BLOCK_SIZE_POWER;
            int maxXOffset = width - BLOCK_SIZE;
            if (xoffset > maxXOffset) xoffset = maxXOffset;
            int sum = 0;
            int min = 0xFF;
            int max = 0;
            for (int yy = 0, offset = yoffset * width + xoffset;
                 yy < BLOCK_SIZE;
                 yy++, offset += width) {
                for (int xx = 0; xx < BLOCK_SIZE; xx++) {
                    int pixel = bytes[offset + xx];
                    sum += pixel;
                    
                    // still looking for good contrast
                    if (pixel < min)
                    {
                        min = pixel;
                    }
                    if (pixel > max)
                    {
                        max = pixel;
                    }
                }
                
                // short-circuit min/max tests once dynamic range is met
                if (max - min > minDynamicRange)
                {
                    // finish the rest of the rows quickly
                    for (yy++, offset += width; yy < BLOCK_SIZE; yy++, offset += width) {
                        for (int xx = 0; xx < BLOCK_SIZE; xx += 2) {
                            sum += bytes[offset + xx];
                            sum += bytes[offset + xx + 1];
                        }
                    }
                }
            }
            
            blocks_[y * subWidth + x].min = min;
            blocks_[y * subWidth + x].max = max;
            blocks_[y * subWidth + x].sum = sum;
            blocks_[y * subWidth + x].threshold = getBlockThreshold(x, y, subWidth, sum, min, max, minDynamicRange, BLOCK_SIZE_POWER);
        }
    }
    
    return 1;
}

int HybridBinarizer::getBlockThreshold(int x, int y, int subWidth, int sum, int min, int max, int minDynamicRange, int SIZE_POWER)
{
    // See
    // http:// groups.google.com/group/zxing/browse_thread/thread/d06efa2c35a7ddc0
    
    // The default estimate is the average of the values in the block.
    int average = sum >> (SIZE_POWER * 2);
    if (max - min <= minDynamicRange)
    {
        // If variation within the block is low, assume this is a block withe only light or only dark pixels.
        // In that case we do not want to use the average, as it would divide this low contrast area into black
        // and white pixels, essentially creating data out of noise.
        // The default assumption is that the block is light/background. Since no estimate for the level of dark
        // pixels exists locally, use half the min for the block.
        average = min >> 1;
        if (y > 0 && x > 0)
        {
            // Correct the "white background" assumption for blocks that have neighbors by comparing the pixels in
            // this block to the previously calculated black points. This is based on the fact that dark barcode symbology
            // is always surrounded by some amout of light background for which reasonable black point estimates were
            // made. The bp estimated at the boundaries is used for the interior.
            int bp = getBlackPointFromNeighbors(blocks_, subWidth, x, y);
            // The (min<bp) is arbitrary but works better than other heuristics that were tried.
            if (min < bp)
            {
                average = bp;
            }
        }
    }
    
    return average;
}

#else
// Calculates a single black point for each block of pixels and saves it away. 
ArrayRef<int> HybridBinarizer::calculateBlackPoints(Ref<ByteMatrix>& luminances,
                                                    int subWidth,
                                                    int subHeight,
                                                    int width,
                                                    int height) {
    const int minDynamicRange = 24;
    int maxYOffset = height - BLOCK_SIZE;
    int maxXOffset = width - BLOCK_SIZE;
    
#ifndef  USE_GOOGLE_CODE
    
    ArrayRef<int> blackPoints(subHeight * subWidth);
    ArrayRef<BlacKPointsInfo> blackPointsInfo=new Array<BlacKPointsInfo>(subHeight*subWidth);
    for (int i = 0; i < subWidth*subHeight; i++){
        blackPointsInfo[i].sum = 0;
        blackPointsInfo[i].min = 0xFF;
        blackPointsInfo[i].max = 0;
    }
    unsigned char* pLuminanceTemp = luminances->bytes;
    
    
    for (int i = 0; i < height; i++){
        int i_black = i / 8;
        BlacKPointsInfo* pInfoTemp=&blackPointsInfo[i_black * subWidth];
        for (int j = 0; j < width; j++){
            int j_black = j / 8;
            int pixel = *pLuminanceTemp++;
#ifdef USE_MAX_MIN
            if (pixel<pInfoTemp[j_black].min) pInfoTemp[j_black].min = pixel;
            if (pixel>pInfoTemp[j_black].max) pInfoTemp[j_black].max = pixel;
#endif
            pInfoTemp[j_black].sum += pixel;
        }
    }
    
    for (int y = 0, k = 0; y < subHeight; y++){
        for (int x = 0; x < subWidth; x++, k++){
            int sum = blackPointsInfo[k].sum;
            int average = sum >> (BLOCK_SIZE_POWER*2);
#ifdef USE_MAX_MIN
            int max = blackPointsInfo[k].max;
            int min = blackPointsInfo[k].min;
            if (max - min <= minDynamicRange){
                average = min >> 1;
                if (x > 0 && y > 0){
                    int bp = getBlackPointFromNeighbors(blackPoints, subWidth, x, y);
                    if (min < bp) 	average = bp;
                }
            }
#endif
            blackPoints[k]=average;
        }
    }
    
#else
    
    ////////  The original google codes ////////
    ArrayRef<int> blackPoints(subWidth*subHeight);
    for (int y = 0; y < subHeight; y++) {
        int yoffset = y<<BLOCK_SIZE_POWER;
        int maxYOffset = height - BLOCK_SIZE;
        if (yoffset > maxYOffset) yoffset = maxYOffset;
        for (int x = 0; x < subWidth; x++){
            int xoffset = x<<BLOCK_SIZE_POWER;
            int maxXOffset = width -  BLOCK_SIZE;
            if (xoffset > maxXOffset) xoffset = maxXOffset;
            int sum = 0;
            int min = 0xFF;
            int max = 0;
            for (int yy = 0, offset = yoffset * width + xoffset;
                 yy < BLOCK_SIZE;
                 yy++, offset += width) {
                for (int xx = 0; xx < BLOCK_SIZE; xx++) {
                    int pixel = luminances->bytes[offset + xx] & 0xFF;
                    sum += pixel;
                    // still looking for good contrast
                    if (pixel < min)
                    {
                        min = pixel;
                    }
                    if (pixel > max)
                    {
                        max = pixel;
                    }
                }
                
                // short-circuit min/max tests once dynamic range is met
                if (max - min > minDynamicRange)
                {
                    // finish the rest of the rows quickly
                    for (yy++, offset += width; yy < BLOCK_SIZE; yy++, offset += width) {
                        for (int xx = 0; xx < BLOCK_SIZE; xx += 2) {
                            sum += luminances->bytes[offset + xx] & 0xFF;
                            sum += luminances->bytes[offset + xx + 1] & 0xFF;
                        }
                    }
                }
            }
            // See
            // http:// groups.google.com/group/zxing/browse_thread/thread/d06efa2c35a7ddc0
            
            // The default estimate is the average of the values in the block.
            int average = sum >> (BLOCK_SIZE_POWER * 2);
            if (max - min <= minDynamicRange)
            {
                // If variation within the block is low, assume this is a block withe only light or only dark pixels.
                // In that case we do not want to use the average, as it would divide this low contrast area into black
                // and white pixels, essentially creating data out of noise.
                // The default assumption is that the block is light/background. Since no estimate for the level of dark
                // pixels exists locally, use half the min for the block.
                average = min >> 1;
                if (y > 0 && x > 0)
                {
                    // Correct the "white background" assumption for blocks that have neighbors by comparing the pixels in
                    // this block to the previously calculated black points. This is based on the fact that dark barcode symbology
                    // is always surrounded by some amout of light background for which reasonable black point estimates were
                    // made. The bp estimated at the boundaries is used for the interior.
                    int bp = getBlackPointFromNeighbors(blackPoints, subWidth, x, y);
                    // The (min<bp) is arbitrary but works better than other heuristics that were tried.
                    if (min < bp)
                    {
                        average = bp;
                    }
                }
            }
            blackPoints[y * subWidth + x] = average;
        }
    }
#endif
    
    return blackPoints;
}
#endif

int HybridBinarizer::binarizeByBlock(int blockLevel, ErrorHandler & err_handler)
{
    (void)blockLevel;
    
    if (width >= MINIMUM_DIMENSION && height >= MINIMUM_DIMENSION)
    {
        Ref<BitMatrix> newMatrix (new BitMatrix(width, height, err_handler));
        if (err_handler.errCode())   return -1;
        
        calculateThresholdForBlock(grayByte_, subWidth_, subHeight_, width, height, BLOCK_SIZE_POWER, newMatrix, err_handler);
        if (err_handler.errCode())   return -1;
        
        matrix0_ = newMatrix;
    }
    else
    {
        // If the image is too small, fall back to the global histogram approach.
        matrix0_ = GlobalHistogramBinarizer::getBlackMatrix(err_handler);
        if (err_handler.errCode())   return 1;
    }
    
    return 1;
}
