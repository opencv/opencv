// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-

#ifndef __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP__
#define __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP__

/*
 *  AlignmentPattern.hpp
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

#include "../../result_point.hpp"
#include "../../common/bit_matrix.hpp"
#include <cmath>

namespace zxing {
namespace qrcode {

class AlignmentPattern : public ResultPoint {
private:
    float estimatedModuleSize_;

public:
    AlignmentPattern(float posX, float posY, float estimatedModuleSize);
    bool aboutEquals(float moduleSize, float i, float j) const;
    float getModuleSize(){return estimatedModuleSize_;};
    
    Ref<AlignmentPattern> combineEstimate(float i, float j,
                                          float newModuleSize) const;
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __ZXING_QRCODE_DETECTOR_ALIGNMENT_PATTERN_HPP__
