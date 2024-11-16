// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
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

#include "../zxing.hpp"
#include "../barcode_format.hpp"
#include "one_dreader.hpp"
#include "../reader_exception.hpp"
#include "one_dresult_point.hpp"
#include "../not_found_exception.hpp"
#include <math.h>
#include <limits.h>

#include<iostream>

using std::vector;
using zxing::Ref;
using zxing::Result;
using zxing::NotFoundException;
using zxing::oned::OneDReader;

// VC++
using zxing::BarcodeFormat;

// VC++
using zxing::BinaryBitmap;
using zxing::BitArray;
using zxing::DecodeHints;
using zxing::BitMatrix;
using zxing::ErrorHandler;

OneDReader::OneDReader() {
}


Ref<Result> OneDReader::decode(Ref<BinaryBitmap> image, 
                               DecodeHints hints) {
    ErrorHandler err_handler;
    Ref<Result> rst = doDecode(image, hints, err_handler);
    if (err_handler.ErrCode() == 0)
        return rst;
    else
    {
        err_handler.Reset();
        bool tryHarder = hints.getTryHarder();
        if (tryHarder && image->isRotateSupported())
        {
            Ref<BinaryBitmap> rotatedImage(image->rotateCounterClockwise());
            Ref<Result> result = doDecode(rotatedImage, hints, err_handler);
            if (err_handler.ErrCode())
                return Ref<Result>();
            // Doesn't have java metadata stuff
            ArrayRef< Ref<ResultPoint> >& points(result->getResultPoints());
            if (points && !points->empty()) {
                int height = rotatedImage->getHeight();
                for (int i = 0; i < points->size(); i++) {
                    points[i].reset(new OneDResultPoint(height - points[i]->getY() - 1, points[i]->getX()));
                }
            }
            
            return result;
        }
        else
        {
            return Ref<Result>();
        }
    }
}

#include <typeinfo>

Ref<Result> OneDReader::doDecode(Ref<BinaryBitmap> image, DecodeHints hints, ErrorHandler & err_handler) {
    int width = image->getWidth();
    int height = image->getHeight();
    Ref<BitArray> row(new BitArray(width));
    
    int middle = height >> 1;
    bool tryHarder = hints.getTryHarder();
    int rowStep = (std::max)(1, height >> (tryHarder ? 8 : 5));
    
    int maxLines;
    if (tryHarder)
    {
        maxLines = height;  // Look at the whole image, not just the center
    }
    else
    {
        maxLines = 15;  // 15 rows spaced 1/32 apart is roughly the middle half of the image
    }
    
#ifdef USE_ERROR_CORRECTION
    sPreTexts.clear();
    bChecked = false;
#endif
    
    Ref<BitMatrix> matrix = image->getBlackMatrix(err_handler);
    if (err_handler.ErrCode())   return Ref<Result>();
    
    // If we need to use getRowRecords or getRowCounterOffsetEnd, we should call initRowCounters first : Valiantliu
    matrix->initRowCounters();
    
    //  -- Attempt
    int lastAttempt = -1;
    _lastReaderIdx = -1;
    bNeedCheck = false;
    
    for (int x = 0; x < maxLines; x++) {
        
        // Scanning from the middle out. Determine which row we're looking at next:
        int rowStepsAboveOrBelow = (x + 1) >> 1;
        bool isAbove = (x & 0x01) == 0;  // i.e. is x even?
        int rowNumber = middle + rowStep * (isAbove ? rowStepsAboveOrBelow : -rowStepsAboveOrBelow);
        if (rowNumber < 0 || rowNumber >= height) {
            // Oops, if we run off the top or bottom, stop
            break;
        }
        
        // Estimate black point for this row and load it:
        row = image->getBlackRow(rowNumber, row, err_handler);
        if (err_handler.ErrCode()){
            err_handler.Reset();
            continue;
        }
        
        recordAllPattern(matrix, rowNumber, _onedReaderData);
        
        // To speed up get next sets function
        row->initAllNextSets();
        
        // While we have the image data in a BitArray, it's fairly cheap to reverse it in place to
        // handle decoding upside down barcodes.
        int maxAttempt = 2;
        int attempts[2] = {0, 1};
      
        if (lastAttempt >=0)
        {
            maxAttempt = 1;
            
            attempts[0] = lastAttempt;
        }
        
        for (int tA = 0; tA < maxAttempt; tA++) {
            int attempt = attempts[tA];
            if (attempt == 1) {
                row->reverse();  // reverse the row and continue
                
                reverseAllPattern(_onedReaderData);
                
                // To speed up get next sets function
                row->initAllNextSets();
            }
            
            // Java hints stuff missing
            
            Ref<Result> result = decodeRow(rowNumber, row);
            
            if (result == NULL) {
                continue;
            }
            
#ifdef USE_ERROR_CORRECTION
            std::string sCurrBarcodeText = result->getText()->getText();
            
            if (sCurrBarcodeText.size() < 14)
                // Check result only for UPC-A / UPC-E / ITF / EAN-8, others usually right
            {
                bChecked = checkResultRight(sPreTexts, sCurrBarcodeText);
                
                if ((bChecked == false) && (sCurrBarcodeText.size() > 0))
                {
                    sPreTexts.push_back(sCurrBarcodeText);
                    
                    lastAttempt = attempt;
                    
                    bNeedCheck = true;
                }
                
            }
            // For other barcode format, no need to check
            else
            {
                bChecked = true;
            }
            
            if (bChecked == true)
            {
                return result;
            }
#else
            return result;
#endif
        }
    }
    
    err_handler = NotFoundErrorHandler("NotFoundErrorHandler");
    return Ref<Result>();
}


#ifdef USE_ERROR_CORRECTION
bool OneDReader::checkResultRight(std::vector<std::string> prevTexts, std::string currText)
{
    // Check if it has a pre result
    if ((prevTexts.size() > 0) && (currText.size() > 0))
    {
        for (size_t i = 0; i < prevTexts.size(); i++)
        {
            if (currText == prevTexts.at(i))
            {
                return true;
            }
        }
    }
    
    return false;
    
}
#endif

int OneDReader::patternMatchVariance(vector<int>& counters,
                                     vector<int> const& pattern,
                                     int maxIndividualVariance) {
    return patternMatchVariance(counters, &pattern[0], maxIndividualVariance);
}

int OneDReader::patternMatchVariance(vector<int>& counters,
                                     int const pattern[],
                                     int maxIndividualVariance) {
    int numCounters = counters.size();
    unsigned int total = 0;
    unsigned int patternLength = 0;
    for (int i = 0; i < numCounters; i++) {
        total += counters[i];
        patternLength += pattern[i];
    }
    if (total < patternLength || patternLength == 0)
    {
        // If we don't even have one pixel per unit of bar width, assume this is too small
        // to reliably match, so fail:
        return INT_MAX;
    }
    // We're going to fake floating-point math in integers. We just need to use more bits.
    // Scale up patternLength so that intermediate values below like scaledCounter will have
    // more "significant digits"
    int unitBarWidth = (total << INTEGER_MATH_SHIFT) / patternLength;
    maxIndividualVariance = (maxIndividualVariance * unitBarWidth) >> INTEGER_MATH_SHIFT;
    
    int totalVariance = 0;
    for (int x = 0; x < numCounters; x++) {
        int counter = counters[x] << INTEGER_MATH_SHIFT;
        int scaledPattern = pattern[x] * unitBarWidth;
        int variance = counter > scaledPattern ? counter - scaledPattern : scaledPattern - counter;
        if (variance > maxIndividualVariance) {
            return INT_MAX;
        }
        totalVariance += variance;
    }
    return totalVariance / total;
}


// Records the size of successive runs of white and pixels in a row, starting at a given point
// The values are recorded in the given array, and the number of runs records is equal to the size
// of the array. If the row starts on a white pixel at the given start point, then the first count 
// recorded is the run of white pixels starting from that point; likewise it is the count of a run
// of black pixels if the row begin on a black pixels at that point.
bool OneDReader::recordPattern(Ref<BitArray> row,
                               int start,
                               vector<int>& counters,
                               ONED_READER_DATA* onedReaderData) {
    if (static_cast<int>(onedReaderData->all_counters.size()) != row->getSize())
        recordAllPattern(row, onedReaderData);
    
    int numCounters = counters.size();
    int counterPosition = 0;
    for (int i = 0; i < numCounters; i++) {
        counters[i] = 0;
    }
    int end = row->getSize();
    if (start >= end) {
        return false;
    }
    
    int pixel_i = 0;
    int all_counters_i = 0;
    
    // Fix for memory leak by Valiantliu
    while (pixel_i < start && all_counters_i < onedReaderData->counter_size-1){
        all_counters_i++;
        pixel_i = onedReaderData->all_counters_offsets[all_counters_i];
    }
    
    if (pixel_i == start){
        for (counterPosition = 0; counterPosition < numCounters; counterPosition++){
            if ((all_counters_i+counterPosition)>=onedReaderData->counter_size)
                break;
            counters[counterPosition] = onedReaderData->all_counters[all_counters_i+counterPosition];
        }
    }
    else
    {
        --all_counters_i;
        counters[0]=onedReaderData->all_counters[all_counters_i]-(pixel_i-start);
        for (counterPosition=1; counterPosition < numCounters; counterPosition++){
            if ((all_counters_i+counterPosition)>=onedReaderData->counter_size)
                break;
            counters[counterPosition] = onedReaderData->all_counters[all_counters_i+counterPosition];
        }
    }
    
    if (counters[0]==0)
    {
        return false;
    }
    
    // If we read fully the last section of pixels and filled up our counters -- or filled
    // the last counter but ran off the side of the image, OK. Otherwise, a problem.
    if (!(counterPosition == numCounters ||
          (counterPosition == (numCounters - 1) && counterPosition+all_counters_i == onedReaderData->counter_size-1)
         ))
    {
        return false;
    }
    
    return true;
}


void OneDReader::recordAllPattern(Ref<BitMatrix> matrix, int row_num, ONED_READER_DATA* onedReaderData){
    onedReaderData->counter_size = matrix->getWidth();
    if (static_cast<int>(onedReaderData->all_counters.size()) != onedReaderData->counter_size)
        onedReaderData->all_counters.resize(onedReaderData->counter_size, 0);
    if (static_cast<int>(onedReaderData->all_counters_offsets.size()) != onedReaderData->counter_size)
        onedReaderData->all_counters_offsets.resize(onedReaderData->counter_size, 0);
    
    COUNTER_TYPE* recorded_counters = matrix->getRowRecords(row_num);
    COUNTER_TYPE* recorded_counter_offsets = matrix->getRowRecordsOffset(row_num);
    
#ifdef COUNTER_TYPE
    for (int i = 0; i < onedReaderData->counter_size; i++){
        onedReaderData->all_counters[i] = static_cast<int>(recorded_counters[i]);
        onedReaderData->all_counters_offsets[i] = static_cast<int>(recorded_counter_offsets[i]);
    }
#else
    memcpy(&onedReaderData->all_counters[0], recorded_counters, onedReaderData->counter_size*sizeof(int));
    memcpy(&onedReaderData->all_counters_offsets[0], recorded_counter_offsets, onedReaderData->counter_size*sizeof(int));
#endif
    
    onedReaderData->counter_size = matrix->getRowCounterOffsetEnd(row_num);
    onedReaderData->first_is_white = matrix->getRowFirstIsWhite(row_num);
    
    return;
}

void OneDReader::recordAllPattern(Ref<BitArray> row, ONED_READER_DATA* onedReaderData){
    onedReaderData->counter_size = row->getSize();
    if (static_cast<int>(onedReaderData->all_counters.size()) != onedReaderData->counter_size)
        onedReaderData->all_counters.resize(onedReaderData->counter_size, 0);
    if (static_cast<int>(onedReaderData->all_counters_offsets.size()) != onedReaderData->counter_size)
        onedReaderData->all_counters_offsets.resize(onedReaderData->counter_size, 0);
    
    // Modified by Valiantliu : Speed up
    memset(&onedReaderData->all_counters[0], 0, onedReaderData->counter_size*sizeof(int));
    memset(&onedReaderData->all_counters_offsets[0], 0, onedReaderData->counter_size*sizeof(int));
    
    int end = row->getSize();
    onedReaderData->first_is_white = !row->get(0);
    bool isWhite = onedReaderData->first_is_white;
    int counterPosition = 0;
    int i = 0;
    onedReaderData->all_counters_offsets[0] = 0;
    
    bool* rowBit = row->getRowBoolPtr();
    
    while (i < end) {
        if (rowBit[i] ^ isWhite) {  // that is, exactly one is true
            onedReaderData->all_counters[counterPosition]++;
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
                onedReaderData->all_counters[counterPosition] = 1;
                isWhite = !isWhite;
                onedReaderData->all_counters_offsets[counterPosition]= i;
            }
        }
        
        i++;
    }
    
    onedReaderData->counter_size = counterPosition < end? (counterPosition+1):end;
    
    // If we read fully the last section of pixels and filled up our counters -- or filled
    // the last counter but ran off the side of the image, OK. Otherwise, a problem.
    if (!(counterPosition == end || (counterPosition == end - 1 && i == end))) {
        return;
    }
}

void OneDReader::reverseAllPattern(ONED_READER_DATA* onedReaderData){
    // reverse vector<int> _onedReaderData->all_counters_offsets
    int rowsize = onedReaderData->all_counters.size();
    vector<int> all_counters_offset_tmp(onedReaderData->counter_size, 0);
    for (int i = 0; i < onedReaderData->counter_size; i++){
        all_counters_offset_tmp[i]=onedReaderData->all_counters_offsets[i];
    }
    for (int i=1; i < onedReaderData->counter_size; i++){
        onedReaderData->all_counters_offsets[i]=rowsize-all_counters_offset_tmp[onedReaderData->counter_size-i];
    }
    
    // reverse vector<int> _onedReaderData->all_counters
    for (int i = 0; i < onedReaderData->counter_size/2; i++){
        int tmp=onedReaderData->all_counters[i];
        int reverse_i = onedReaderData->counter_size-i-1;
        onedReaderData->all_counters[i]=onedReaderData->all_counters[reverse_i];
        onedReaderData->all_counters[reverse_i] = tmp;
    }
    
    // reverse the bool value: _onedReaderData->first_is_white (if it is necessary)
    if (onedReaderData->counter_size%2==0)
        onedReaderData->first_is_white = !onedReaderData->first_is_white;
}

OneDReader::~OneDReader() {
}
