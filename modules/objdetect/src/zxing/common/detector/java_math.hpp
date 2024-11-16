// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_COMMON_DETECTOR_MATH_H__
#define __ZXING_COMMON_DETECTOR_MATH_H__
/*
 *  Copyright 2012 ZXing authors All rights reserved.
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

#include <cmath>

namespace zxing {
namespace common {
namespace detector {

class Math {
 private:
  Math();
  ~Math();
 public:

  // Java standard Math.round
  static inline int round(float a) {
    return (int)std::floor(a +0.5f);
  }

};

}
}
}

#endif
