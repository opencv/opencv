// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-

#ifndef __NOT_FOUND_EXCEPTION_H__
#define __NOT_FOUND_EXCEPTION_H__

/*
 * Copyright 20011 ZXing authors
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

#include "reader_exception.hpp"

namespace zxing {

class NotFoundException : public ReaderException {
public:
  NotFoundException() throw() {}
  NotFoundException(const char *msg) throw() : ReaderException(msg) {}
  ~NotFoundException() throw() {}
};

}

#endif  // __NOT_FOUND_EXCEPTION_H__
