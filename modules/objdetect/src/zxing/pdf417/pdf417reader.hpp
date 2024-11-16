// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __PDF417_READER_H__
#define __PDF417_READER_H__

/*
 *  PDF417Reader.hpp
 *  zxing
 *
 *  Copyright 2010,2012 ZXing authors All rights reserved.
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
#include "decoder/decoder.hpp"
#include "../decode_hints.hpp"
#include "../error_handler.hpp"

namespace zxing {
namespace pdf417 {

class PDF417Reader : public Reader {
private:
    decoder::Decoder decoder;
    
    static Ref<BitMatrix> extractPureBits(Ref<BitMatrix> image, ErrorHandler & err_handler);
    static int moduleSize(ArrayRef<int> leftTopBlack, Ref<BitMatrix> image, ErrorHandler & err_handler);
    static int findPatternStart(int x, int y, Ref<BitMatrix> image, ErrorHandler & err_handler);
    static int findPatternEnd(int x, int y, Ref<BitMatrix> image, ErrorHandler &err_handler);
    
public:
    std::string name() { return "pdf417"; }
    using Reader::decode;
    Ref<Result> decode(Ref<BinaryBitmap> image, DecodeHints hints);
    
    void reset();
};

}  // namespace pdf417
}  // namespace zxing

#endif  // __PDF417_READER_H__
