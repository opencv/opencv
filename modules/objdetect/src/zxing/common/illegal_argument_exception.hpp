#ifndef __ILLEGAL_ARGUMENT_EXCEPTION_H__
#define __ILLEGAL_ARGUMENT_EXCEPTION_H__

/*
 *  IllegalArgumentException.hpp
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

#include "../exception.hpp"

namespace zxing {

class IllegalArgumentException : public Exception {
public:
    IllegalArgumentException();
    IllegalArgumentException(const char *msg);
    IllegalArgumentException(std::string msg);
    ~IllegalArgumentException() throw();
};

}  // namespace zxing

#endif  // __ILLEGAL_ARGUMENT_EXCEPTION_H__
