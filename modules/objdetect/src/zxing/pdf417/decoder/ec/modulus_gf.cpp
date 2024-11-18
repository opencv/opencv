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
 * Copyright 2012 ZXing authors
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
 *
 * 2012-09-19 HFN translation from Java into C++
 */

#include "modulus_gf.hpp"
#include "modulus_poly.hpp"

using zxing::Ref;
using zxing::pdf417::decoder::ec::ModulusGF;
using zxing::pdf417::decoder::ec::ModulusPoly;
using zxing::IllegalArgumentErrorHandler;
using zxing::ErrorHandler;

/**
 * <p>A field based on powers of a generator integer, modulo some modulus.</p>
 *
 * @author Sean Owen
 * @see com.google.zxing.common.reedsolomon.GenericGF
 */

ModulusGF::ModulusGF(int modulus, int generator, ErrorHandler & err_handler)
: modulus_(modulus) {
    expTable_ = new Array<int>(modulus_);
    logTable_ = new Array<int>(modulus_);
    int x = 1, i;
    for (i = 0; i < modulus_; i++) {
        expTable_[i] = x;
        x = (x * generator) % modulus_;
    }
    for (i = 0; i < modulus_-1; i++) {
        logTable_[expTable_[i]] = i;
    }
    
    ArrayRef<int>aZero(new Array<int>(1)), aOne(new Array<int>(1));
    aZero[0] = 0; aOne[0] = 1;
    zero_ = new ModulusPoly(*this, aZero, err_handler);
    if (err_handler.errCode())   return;
    
    one_ = new ModulusPoly(*this, aOne, err_handler);
    if (err_handler.errCode())   return;
}

Ref<ModulusPoly> ModulusGF::getZero() {
    return zero_;
}

Ref<ModulusPoly> ModulusGF::getOne() {
    return one_;
}

Ref<ModulusPoly> ModulusGF::buildMonomial(int degree, int coefficient, ErrorHandler & err_handler)
{
    if (degree < 0) {
        err_handler = IllegalArgumentErrorHandler("monomial: degree < 0!");
        return Ref<ModulusPoly>();
    }
    if (coefficient == 0) {
        return zero_;
    }
    int nCoefficients = degree + 1;
    ArrayRef<int> coefficients(new Array<int>(nCoefficients));
    coefficients[0] = coefficient;
    Ref<ModulusPoly> result(new ModulusPoly(*this, coefficients, err_handler));
    if (err_handler.errCode())   return Ref<ModulusPoly>();
    
    return result;
}

int ModulusGF::add(int a, int b) {
    return (a + b) % modulus_;
}

int ModulusGF::subtract(int a, int b) {
    return (modulus_ + a - b) % modulus_;
}

int ModulusGF::exp(int a) {
    return expTable_[a];
}

int ModulusGF::log(int a, ErrorHandler & err_handler) {
    if (a == 0) {
        err_handler = IllegalArgumentErrorHandler("log of zero!");
        return -1;
    }
    return logTable_[a];
}

int ModulusGF::inverse(int a, ErrorHandler & err_handler) {
    if (a == 0) {
        err_handler  = IllegalArgumentErrorHandler("inverse of zero!");;
        return -1;
    }
    return expTable_[modulus_ - logTable_[a] - 1];
}

int ModulusGF::multiply(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return expTable_[(logTable_[a] + logTable_[b]) % (modulus_ - 1)];
}

int ModulusGF::getSize() {
    return modulus_;
}
