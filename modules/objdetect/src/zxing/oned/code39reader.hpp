// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __CODE_39_READER_H__
#define __CODE_39_READER_H__
/*
 *  Code39Reader.hpp
 *  ZXing
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

#include "one_dreader.hpp"
#include "../common/bit_array.hpp"
#include "../result.hpp"
#include "../error_handler.hpp"

namespace zxing {
namespace oned {

/**
 * <p>Decodes Code 39 barcodes. This does not support "Full ASCII Code 39" yet.</p>
 * Ported form Java (author Sean Owen)
 * @author Lukasz Warchol
 */
class Code39Reader : public OneDReader {
private:
    bool usingCheckDigit;
    bool extendedMode;
    std::string decodeRowResult;
    std::vector<int> counters;
    
    void init(bool usingCheckDigit = false, bool extendedMode = false);
    
    static std::vector<int> findAsteriskPattern(Ref<BitArray> row,
                                                std::vector<int>& counters, ONED_READER_DATA* onedReaderData);
    static int toNarrowWidePattern(std::vector<int>& counters);
    static char patternToChar(int pattern, ErrorHandler & err_handler);
    static Ref<String> decodeExtended(std::string encoded, ErrorHandler & err_handler);
    
    void append(char* s, char c);
    
public:
    Code39Reader();
    Code39Reader(bool usingCheckDigit_);
    Code39Reader(bool usingCheckDigit_, bool extendedMode_);
    
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
};

}  // namespace oned
}  // namespace zxing

#endif
