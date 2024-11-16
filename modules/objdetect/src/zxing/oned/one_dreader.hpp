// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ONED_READER_H__
#define __ONED_READER_H__

/*
 *  OneDReader.hpp
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

#include "../reader.hpp"
#include "../error_handler.hpp"

// Use error correction for oned barcode
// Added by Skylook
#define USE_PRE_BESTMATCH 1
#define USE_ERROR_CORRECTION 1

namespace zxing {
namespace oned {

class OneDReader : public Reader {
private:
    Ref<Result> doDecode(Ref<BinaryBitmap> image, DecodeHints hints, ErrorHandler & err_handler);
    
protected:
    static const int INTEGER_MATH_SHIFT = 8;
    
    struct Range {
    private:
        int data[2];
        bool valid;

    public:
        Range() {
            data[0] = 0;
            data[1] = 0;
            valid = true;
        }
        Range(bool valid_){
            data[0] = 0;
            data[1] = 1;
            valid = valid_;
        }
        Range(int zero, int one) {
            data[0] = zero;
            data[1] = one;
            valid = true;
        }
        bool isValid(){
            return valid;
        }
        int& operator [] (int index) {
            return data[index];
        }
        int const& operator [] (int index) const {
            return data[index];
        }
    };
    
    static int patternMatchVariance(std::vector<int>& counters,
                                    std::vector<int> const& pattern,
                                    int maxIndividualVariance);
    static int patternMatchVariance(std::vector<int>& counters,
                                    int const pattern[],
                                    int maxIndividualVariance);
    
protected:
    static const int PATTERN_MATCH_RESULT_SCALE_FACTOR = 1 << INTEGER_MATH_SHIFT;
    
public:
    //  -- Oned Reader Data : Start
#ifdef USE_PRE_BESTMATCH
    struct DigitResultCache
    {
        int bestMatch[2];
        int counterOffset;
    };
    struct DigitResult
    {
        int bestMatch;
        int counterOffset;
    };
#endif
    
    struct ONED_READER_DATA
    {
        std::vector<int> all_counters;
        std::vector<int> all_counters_offsets;
        bool first_is_white;
        int counter_size;
        
        bool ean13_checked;
        int ean13_lg_pattern_found;
        int ean13_decode_middle_middle_offset;
        int ean13_decode_middle_final_offset;
        std::string ean13_decode_middle_middle_string;
        std::string ean13_decode_middle_final_string;
        
        std::vector<DigitResultCache> digitResultCache;
    };
    
    ONED_READER_DATA* _onedReaderData;
    
    virtual void setData(ONED_READER_DATA* onedReaderData)
    {
        _onedReaderData = onedReaderData;
    };
    //  -- Oned Reader Data : End
    
    OneDReader();
    using Reader::decode;
    virtual Ref<Result> decode(Ref<BinaryBitmap> image,
                               DecodeHints hints);
    
    // Implementations must not throw any exceptions. If a barcode is not found on this row,
    // a empty ref should be returned e.g. return Ref<Result>();
    virtual Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row) = 0;
    // virtual Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row, int readerIdx) = 0;
    
    static bool recordPattern(Ref<BitArray> row,
                              int start,
                              std::vector<int>& counters,
                              ONED_READER_DATA* onedReaderData);
    
#ifdef USE_ERROR_CORRECTION
    bool checkResultRight(std::vector<std::string> prevText, std::string currText);
#endif
    
    static void recordAllPattern(Ref<BitArray> row, ONED_READER_DATA* onedReaderData);
    static void reverseAllPattern(ONED_READER_DATA* onedReaderData);
    static void recordAllPattern(Ref<BitMatrix> matrix, int row_num, ONED_READER_DATA* onedReaderData);
    
#ifdef USE_ERROR_CORRECTION
    bool bChecked;
    Ref<Result> preResult;
    std::vector<std::string> sPreTexts;
    
    int _lastReaderIdx;
    bool bNeedCheck;
#endif
    
    virtual ~OneDReader(); 
};

}  // namespace oned
}  // namespace zxing

#endif
