// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __SIMPLEADAPTIVEBINARIZER_H__
#define __SIMPLEADAPTIVEBINARIZER_H__
/*
 *  SimpleAdaptiveBinarizer.hpp
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

#include "../binarizer.hpp"
#include "bit_array.hpp"
#include "bit_matrix.hpp"
#include "array.hpp"
#include "global_histogram_binarizer.hpp"

#ifndef USE_GOOGLE_CODE
#define USE_GOOGLE_CODE 0
#endif

namespace zxing {

class SimpleAdaptiveBinarizer : public GlobalHistogramBinarizer {
private:
    ArrayRef<char> luminances;
   
public:
    SimpleAdaptiveBinarizer(Ref<LuminanceSource> source);
    virtual ~SimpleAdaptiveBinarizer();
    
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler);
    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler &err_handler);
    
    Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source);
private:
    int binarizeImage0(ErrorHandler &err_handler);
    int qrBinarize(const unsigned char *_img, unsigned char* _dst, int _width, int _height);
    
    bool filtered;
};

}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_COMMON_SIMPLEADAPTIVEBINARIZER_H_
