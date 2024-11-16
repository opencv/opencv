/*
 *  CornerPoint.cpp
 *  zxing
 *
 *  Created by Luiz Silva on 09/02/2010.
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

#include "corner_point.hpp"


namespace zxing {
namespace datamatrix {

CornerPoint::CornerPoint(float posX, float posY) :
ResultPoint(posX, posY), counter_(0) {
}

int CornerPoint::getCount() const {
    return counter_;
}

void CornerPoint::incrementCount() {
    counter_++;
}

bool CornerPoint::equals(Ref<CornerPoint> other) const {
    return posX_ == other->getX() && posY_ == other->getY();
}

}  // namespace datamatrix
}  // namespace zxing
