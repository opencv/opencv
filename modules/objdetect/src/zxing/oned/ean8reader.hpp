// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __EAN_8_READER_H__
#define __EAN_8_READER_H__

/*
 *  EAN8Reader.hpp
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

#include "upceanreader.hpp"
#include "../result.hpp"

namespace zxing {
namespace oned {

class EAN8Reader : public UPCEANReader {
private:
    std::vector<int> decodeMiddleCounters;
    
public:
    EAN8Reader();
    
    int decodeMiddle(Ref<BitArray> row,
                     Range const& startRange,
                     std::string& resultString);
    
    BarcodeFormat getBarcodeFormat();
};

}  // namespace oned
}  // namespace zxing

#endif
