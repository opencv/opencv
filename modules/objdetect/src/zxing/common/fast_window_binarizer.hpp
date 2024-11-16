// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __FASTWINDOWBINARIZER_H__
#define __FASTWINDOWBINARIZER_H__
/*
 *  FastWindowBinarizer.hpp
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

#include <vector>
#include "../binarizer.hpp"
#include "global_histogram_binarizer.hpp"
#include "bit_array.hpp"
#include "bit_matrix.hpp"
#include "../error_handler.hpp"

namespace zxing {

class FastWindowBinarizer : public GlobalHistogramBinarizer {
private:
    Ref<BitMatrix> matrix_;
    Ref<BitArray> cached_row_;
    
    int* _luminancesInt;
    int* _blockTotals;
    int* _totals;
    int* _rowTotals;
    
    unsigned int* _internal;
    
public:
    FastWindowBinarizer(Ref<LuminanceSource> source);
    virtual ~FastWindowBinarizer();
    
    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler & err_handler);
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler);
    
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source);

private:
    void calcBlockTotals(int* luminancesInt, int* output, int width, int height, int aw, int ah);
    void cumulative(int* data, int* output, int width, int height);
    int binarizeImage0(ErrorHandler & err_handler);
    void fastIntegral(const unsigned char* inputMatrix, unsigned int* outputMatrix, int width, int height);
    int binarizeImage1(ErrorHandler &err_handler);
    void fastWindow(const unsigned char* src, unsigned char*dst, int width, int height, ErrorHandler & err_handler);
    void fastWin2(unsigned char * src, unsigned char *dst, int width, int height);
};

}  // namespace zxing


#endif
