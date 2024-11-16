// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __CODA_BAR_READER_H__
#define __CODA_BAR_READER_H__
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

#include "one_dreader.hpp"
#include "../common/bit_array.hpp"
#include "../result.hpp"
#include "../error_handler.hpp"

namespace zxing {
namespace oned {

class CodaBarReader : public OneDReader {
private:
    static const int MAX_ACCEPTABLE;
    static const int PADDING;
    
    // Keep some instance variables to avoid reallocations
    std::string decodeRowResult;
    std::vector<int> counters;
    int counterLength;
    
public:
    CodaBarReader();
    
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
    
    void validatePattern(int start, ErrorHandler & err_handler);
    
    static bool arrayContains(char const array[], char key);
    
private:
    void setCounters(Ref<BitArray> row);
    void counterAppend(int e);
    int findStartPattern();
    
    int toNarrowWidePattern(int position);
};

}  // namespace oned
}  // namespace zxing

#endif
