// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  ErrorCorrectionLevel.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 15/05/2008.
 *  Copyright 2008-2011 ZXing authors All rights reserved.
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

#include "error_correction_level.hpp"

using std::string;
using zxing::ErrorHandler;

namespace zxing {

ErrorCorrectionLevel::ErrorCorrectionLevel(int inOrdinal,
                                           int bits,
                                           char const* name) :
ordinal_(inOrdinal), bits_(bits), name_(name) {}

int ErrorCorrectionLevel::ordinal() const {
    return ordinal_;
}

int ErrorCorrectionLevel::bits() const {
    return bits_;
}

string const& ErrorCorrectionLevel::name() const {
    return name_;
}

ErrorCorrectionLevel::operator string const& () const {
    return name_;
}

ErrorCorrectionLevel& ErrorCorrectionLevel::forBits(int bits, ErrorHandler & err_handler) {
    if (bits < 0 || bits >= N_LEVELS) {
        err_handler = zxing::ReaderErrorHandler("Ellegal error correction level bits");
        return *FOR_BITS[0];
    }
    return *FOR_BITS[bits];
}

ErrorCorrectionLevel ErrorCorrectionLevel::L(0, 0x01, "L");
ErrorCorrectionLevel ErrorCorrectionLevel::M(1, 0x00, "M");
ErrorCorrectionLevel ErrorCorrectionLevel::Q(2, 0x03, "Q");
ErrorCorrectionLevel ErrorCorrectionLevel::H(3, 0x02, "H");
ErrorCorrectionLevel *ErrorCorrectionLevel::FOR_BITS[] = { &M, &L, &H, &Q };
int ErrorCorrectionLevel::N_LEVELS = 4;

}  // namespace zxing
