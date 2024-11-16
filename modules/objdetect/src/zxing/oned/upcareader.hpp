// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __UPCA_READER_H__
#define __UPCA_READER_H__
/*
 *  UPCAReader.hpp
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

#include "ean13reader.hpp"
#include "../decode_hints.hpp"
#include "../error_handler.hpp"

namespace zxing {
namespace oned {

class UPCAReader : public UPCEANReader {
private:
    EAN13Reader ean13Reader;
    static Ref<Result> maybeReturnResult(Ref<Result> result);
    
public:
    UPCAReader();
    
    int decodeMiddle(Ref<BitArray> row, Range const& startRange, std::string& resultString);
    
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row, Range const& startGuardRange);
    Ref<Result> decode(Ref<BinaryBitmap> image, DecodeHints hints);
    
    BarcodeFormat getBarcodeFormat();
};

}  // namespace oned
}  // namespace zxing

#endif
