#ifndef __ERROR_CORRECTION_LEVEL_H__
#define __ERROR_CORRECTION_LEVEL_H__

/*
 *  ErrorCorrectionLevel.hpp
 *  zxing
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

#include "../reader_exception.hpp"
#include "../error_handler.hpp"

namespace zxing {
// namespace qrcode {

class ErrorCorrectionLevel {
private:
    int ordinal_;
    int bits_;
    std::string name_;
    ErrorCorrectionLevel(int inOrdinal, int bits, char const* name);
    static ErrorCorrectionLevel *FOR_BITS[];
    static int N_LEVELS;

public:
    static ErrorCorrectionLevel L;
    static ErrorCorrectionLevel M;
    static ErrorCorrectionLevel Q;
    static ErrorCorrectionLevel H;
    
    int ordinal() const;
    int bits() const;
    std::string const& name() const;
    operator std::string const& () const;
    
    static ErrorCorrectionLevel& forBits(int bits, ErrorHandler & err_handler);
};
//}  // namespace qrcode
}  // namespace zxing

#endif  // __ERROR_CORRECTION_LEVEL_H__
