#ifndef __MULTI_FORMAT_ONED_READER_H__
#define __MULTI_FORMAT_ONED_READER_H__
/*
 *  MultiFormatOneDReader.hpp
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
#include "../error_handler.hpp"

namespace zxing {
namespace oned {
class MultiFormatOneDReader : public OneDReader {
private:
    std::vector<Ref<OneDReader> > readers;
    
    //  -- Add Reader Data
    ONED_READER_DATA* readerData;
    
public:
    MultiFormatOneDReader(DecodeHints hints);
    std::string name() { return "oned"; }
    
    ~MultiFormatOneDReader();
    
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row, int readerIdx);
};
}  // namespace oned
}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_ONED_MULTIFORMATONEDREADER_H_ 
