// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  DetectorResult.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 14/05/2008.
 *  Copyright 2008 ZXing authors All rights reserved.
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

#include "detector_result.hpp"

namespace zxing {

DetectorResult::DetectorResult(Ref<BitMatrix> bits,
                               ArrayRef< Ref<ResultPoint> > points, int dimension, float modulesize)
: bits_(bits), points_(points), dimension_(dimension), modulesize_(modulesize) {
}

void DetectorResult::SetGray(Ref<ByteMatrix> gray)
{
    gray_ = gray;
}

Ref<BitMatrix> DetectorResult::getBits() {
    return bits_;
}

Ref<ByteMatrix> DetectorResult::getGray() {
    return gray_;
}

ArrayRef< Ref<ResultPoint> > DetectorResult::getPoints() {
    return points_;
}

}  // namespace zxing
