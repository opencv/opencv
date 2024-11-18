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
/*
 *  GenericGFPoly.hpp
 *  zxing
 *
 *  Created by Lukas Stabe on 13/02/2012.
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

#ifndef __ZXING_COMMON_REEDSOLOMON_GENERIC_GFPOLY_HPP_
#define __ZXING_COMMON_REEDSOLOMON_GENERIC_GFPOLY_HPP_

#include <vector>
#include "../array.hpp"
#include "../counted.hpp"
#include "../../error_handler.hpp"

namespace zxing {

class GenericGF;

class GenericGFPoly : public Counted {
private:
    GenericGF &field_;
    ArrayRef<int> coefficients_;
    
public:
    GenericGFPoly(GenericGF &field, ArrayRef<int> coefficients, ErrorHandler & err_handler);
    ArrayRef<int> getCoefficients();
    int getDegree();
    bool isZero();
    int getCoefficient(int degree);
    int evaluateAt(int a);
    Ref<GenericGFPoly> addOrSubtract(Ref<GenericGFPoly> other, ErrorHandler & err_handler);
    Ref<GenericGFPoly> multiply(Ref<GenericGFPoly> other, ErrorHandler & err_handler);
    Ref<GenericGFPoly> multiply(int scalar , ErrorHandler & err_handler);
    Ref<GenericGFPoly> multiplyByMonomial(int degree, int coefficient, ErrorHandler & err_handler);
    std::vector<Ref<GenericGFPoly> > divide(Ref<GenericGFPoly> other, ErrorHandler & err_handler);
};

}  // namespace zxing

#endif  // __ZXING_COMMON_REEDSOLOMON_GENERIC_GFPOLY_HPP_
