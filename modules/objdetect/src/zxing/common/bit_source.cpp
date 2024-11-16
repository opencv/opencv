// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  BitSource.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 09/05/2008.
 *  Copyright 2008 Google UK. All rights reserved.
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

#include "bit_source.hpp"
#include <sstream>
#include "illegal_argument_exception.hpp"

namespace zxing {

int BitSource::readBits(int numBits, ErrorHandler & err_handler) {
    if (numBits < 0 || numBits > 32 || numBits > available())
    {
        std::ostringstream oss;
        oss << numBits;
        err_handler = IllegalArgumentErrorHandler(oss.str().c_str());
        return -1;
    }
    
    int result = 0;
    
    // First, read remainder from current byte
    if (bitOffset_ > 0)
    {
        int bitsLeft = 8 - bitOffset_;
        int toRead = numBits < bitsLeft ? numBits : bitsLeft;
        int bitsToNotRead = bitsLeft - toRead;
        int mask = (0xFF >> (8 - toRead)) << bitsToNotRead;
        result = (bytes_[byteOffset_] & mask) >> bitsToNotRead;
        numBits -= toRead;
        bitOffset_ += toRead;
        if (bitOffset_ == 8)
        {
            bitOffset_ = 0;
            byteOffset_++;
        }
    }
    
    // Next read whole bytes
    if (numBits > 0)
    {
        while (numBits >= 8) {
            result = (result << 8) | (bytes_[byteOffset_] & 0xFF);
            byteOffset_++;
            numBits -= 8;
        }
        
        // Finally read a partial byte
        if (numBits > 0)
        {
            int bitsToNotRead = 8 - numBits;
            int mask = (0xFF >> bitsToNotRead) << bitsToNotRead;
            result = (result << numBits) | ((bytes_[byteOffset_] & mask) >> bitsToNotRead);
            bitOffset_ += numBits;
        }
    }
    
    return result;
}

int BitSource::available() {
    return 8 * (bytes_->size() - byteOffset_) - bitOffset_;
}
}  // namespace zxing

