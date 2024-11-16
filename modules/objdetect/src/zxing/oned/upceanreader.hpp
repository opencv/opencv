// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __UPC_EAN_READER_H__
#define __UPC_EAN_READER_H__

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

#define USE_PRE_ROWOFFSET 1

class UPCEANReader : public OneDReader {
public:
    static bool checkStandardUPCEANChecksum(Ref<String> const& s);
    static int getStandardUPCEANChecksum(Ref<String> const& s);
    
#ifdef USE_PRE_BESTMATCH
    static int getCounterOffset(std::vector<int> counters);
#endif
    
private:
    std::string decodeRowStringBuffer;
    
    static const int MAX_AVG_VARIANCE;
    static const int MAX_INDIVIDUAL_VARIANCE;
    
#ifdef USE_PRE_BESTMATCH
    
    static int initbestMatchDigit(Ref<BitArray> row, ONED_READER_DATA* onedReaderData);
    
#else
#endif
    
    static Range findStartGuardPattern(Ref<BitArray> row,
                                       ONED_READER_DATA* onedReaderData,
                                       ErrorHandler &err_handler);
    
    virtual Range decodeEnd(Ref<BitArray> row, int endStart);
    
    static Range findGuardPattern(Ref<BitArray> row,
                                  int rowOffset,
                                  bool whiteFirst,
                                  std::vector<int> const& pattern,
                                  std::vector<int>& counters,
                                  ONED_READER_DATA* onedReaderData);
    
    
protected:
    static const std::vector<int> START_END_PATTERN;
    static const std::vector<int> MIDDLE_PATTERN;
    
    static const std::vector<int const*> L_PATTERNS;
    static const std::vector<int const*> G_PATTERNS;
    static const std::vector<int const*> L_AND_G_PATTERNS;
    
    static Range findGuardPattern(Ref<BitArray> row,
                                  int rowOffset,
                                  bool whiteFirst,
                                  std::vector<int> const& pattern,
                                  ONED_READER_DATA* onedReaderData);
    
public:
    UPCEANReader();
    
    virtual int decodeMiddle(Ref<BitArray> row,
                             Range const& startRange,
                             std::string& resultString) = 0;
    
    virtual Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
    virtual Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row, Range const& range);
    
#ifdef USE_PRE_BESTMATCH
    static DigitResult decodeDigit(Ref<BitArray> row,
                                   std::vector<int>& counters,
                                   int rowOffset,
                                   std::vector<int const*> const& patterns,
                                   ONED_READER_DATA* onedReaderData);
    
#else
    static int decodeDigit(Ref<BitArray> row,
                           std::vector<int>& counters,
                           int rowOffset,
                           std::vector<int const*> const& patterns);
#endif
    
    virtual bool checkChecksum(Ref<String> const& s);
    virtual int getChecksum(Ref<String> const& s);
    
    virtual BarcodeFormat getBarcodeFormat() = 0;
    virtual ~UPCEANReader();
    
    friend class MultiFormatUPCEANReader;
};

}  // namespace oned
}  // namespace zxing

#endif
