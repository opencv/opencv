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
 *  GenericGFPoly.cpp
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

#include <iostream>
#include "generic_gfpoly.hpp"
#include "generic_gf.hpp"
#include "../illegal_argument_exception.hpp"

using zxing::GenericGFPoly;
using zxing::ArrayRef;
using zxing::Ref;
using zxing::ErrorHandler;

// VC++
using zxing::GenericGF;

GenericGFPoly::GenericGFPoly(GenericGF &field,
                             ArrayRef<int> coefficients,
                             ErrorHandler & err_handler)
:  field_(field) {
    if (coefficients->size() == 0)
    {
        err_handler = IllegalArgumentErrorHandler("need coefficients");
        return;
    }
    int coefficientsLength = coefficients->size();
    if (coefficientsLength > 1 && coefficients[0] == 0)
    {
        // Leading term must be non-zero for anything except the constant polynomial "0"
        int firstNonZero = 1;
        while (firstNonZero < coefficientsLength && coefficients[firstNonZero] == 0) {
            firstNonZero++;
        }
        if (firstNonZero == coefficientsLength)
        {
            coefficients_ = field.getZero()->getCoefficients();
        }
        else
        {
            coefficients_ = ArrayRef<int>(new Array<int>(coefficientsLength-firstNonZero));
            for (int i = 0; i < coefficients_->size(); i++) {
                coefficients_[i] = coefficients[i + firstNonZero];
            }
        }
    }
    else
    {
        coefficients_ = coefficients;
    }
}

ArrayRef<int> GenericGFPoly::getCoefficients() {
    return coefficients_;
}

int GenericGFPoly::getDegree() {
    return coefficients_->size() - 1;
}

bool GenericGFPoly::isZero() {
    return coefficients_[0] == 0;
}

int GenericGFPoly::getCoefficient(int degree) {
    return coefficients_[coefficients_->size() - 1 - degree];
}

int GenericGFPoly::evaluateAt(int a) {
    if (a == 0)
    {
        // Just return the x^0 coefficient
        return getCoefficient(0);
    }
    
    int size = coefficients_->size();
    if (a == 1)
    {
        // Just the sum of the coefficients
        int result = 0;
        for (int i = 0; i < size; i++) {
            result = GenericGF::addOrSubtract(result, coefficients_[i]);
        }
        return result;
    }
    int result = coefficients_[0];
    for (int i = 1; i < size; i++) {
        result = GenericGF::addOrSubtract(field_.multiply(a, result), coefficients_[i]);
    }
    return result;
}

Ref<GenericGFPoly> GenericGFPoly::addOrSubtract(Ref<zxing::GenericGFPoly> other, ErrorHandler & err_handler) {
    if (!(&field_ == &other->field_))
    {
        err_handler = IllegalArgumentErrorHandler("GenericGFPolys do not have same GenericGF field");
        return Ref<GenericGFPoly>();
    }
    if (isZero())
    {
        return other;
    }
    if (other->isZero())
    {
        return Ref<GenericGFPoly>(this);
    }
    
    ArrayRef<int> smallerCoefficients = coefficients_;
    ArrayRef<int> largerCoefficients = other->getCoefficients();
    if (smallerCoefficients->size() > largerCoefficients->size())
    {
        ArrayRef<int> temp = smallerCoefficients;
        smallerCoefficients = largerCoefficients;
        largerCoefficients = temp;
    }
    
    ArrayRef<int> sumDiff(new Array<int>(largerCoefficients->size()));
    int lengthDiff = largerCoefficients->size() - smallerCoefficients->size();
    // Copy high-order terms only found in higher-degree polynomial's coefficients
    for (int i = 0; i < lengthDiff; i++) {
        sumDiff[i] = largerCoefficients[i];
    }
    
    for (int i = lengthDiff; i < largerCoefficients->size(); i++) {
        sumDiff[i] = GenericGF::addOrSubtract(smallerCoefficients[i-lengthDiff],
                                              largerCoefficients[i]);
    }
    
    Ref<GenericGFPoly> gfpoly(new GenericGFPoly(field_, sumDiff, err_handler));
    if (err_handler.errCode()) return Ref<GenericGFPoly>();
    return gfpoly;
}

Ref<GenericGFPoly> GenericGFPoly::multiply(Ref<zxing::GenericGFPoly> other, ErrorHandler & err_handler) {
    if (!(&field_ == &other->field_))
    {
        err_handler = IllegalArgumentErrorHandler("GenericGFPolys do not have same GenericGF field");
        return Ref<GenericGFPoly>();
    }
    
    if (isZero() || other->isZero())
    {
        return field_.getZero();
    }
    
    ArrayRef<int> aCoefficients = coefficients_;
    int aLength = aCoefficients->size();
    
    ArrayRef<int> bCoefficients = other->getCoefficients();
    int bLength = bCoefficients->size();
    
    ArrayRef<int> product(new Array<int>(aLength + bLength - 1));
    for (int i = 0; i < aLength; i++) {
        int aCoeff = aCoefficients[i];
        for (int j = 0; j < bLength; j++) {
            product[i+j] = GenericGF::addOrSubtract(product[i+j],
                                                    field_.multiply(aCoeff, bCoefficients[j]));
        }
    }
    
    Ref<GenericGFPoly> gfpoly(new GenericGFPoly(field_, product, err_handler));
    if (err_handler.errCode()) return Ref<GenericGFPoly>();
    return gfpoly;
}

Ref<GenericGFPoly> GenericGFPoly::multiply(int scalar, ErrorHandler & err_handler) {
    if (scalar == 0)
    {
        return field_.getZero();
    }
    if (scalar == 1)
    {
        return Ref<GenericGFPoly>(this);
    }
    int size = coefficients_->size();
    ArrayRef<int> product(new Array<int>(size));
    for (int i = 0; i < size; i++) {
        product[i] = field_.multiply(coefficients_[i], scalar);
    }
    
    Ref<GenericGFPoly> gfpoly(new GenericGFPoly(field_, product, err_handler));
    if (err_handler.errCode()) return Ref<GenericGFPoly>();
    return gfpoly;
}

Ref<GenericGFPoly> GenericGFPoly::multiplyByMonomial(int degree, int coefficient, ErrorHandler & err_handler) {
    if (degree < 0) {
        err_handler = IllegalArgumentErrorHandler("degree must not be less then 0");
        return Ref<GenericGFPoly>();
    }
    if (coefficient == 0) {
        return field_.getZero();
    }
    int size = coefficients_->size();
    ArrayRef<int> product(new Array<int>(size+degree));
    for (int i = 0; i < size; i++) {
        product[i] = field_.multiply(coefficients_[i], coefficient);
    }

    Ref<GenericGFPoly> gfpoly(new GenericGFPoly(field_, product, err_handler));
    if (err_handler.errCode()) return Ref<GenericGFPoly>();
    return gfpoly;
}

std::vector<Ref<GenericGFPoly> > GenericGFPoly::divide(Ref<GenericGFPoly> other, ErrorHandler & err_handler) {
    if (!(&field_ == &other->field_))
    {
        err_handler = IllegalArgumentErrorHandler("GenericGFPolys do not have same GenericGF field");
        return std::vector<Ref<GenericGFPoly> >();
    }
    if (other->isZero())
    {
        err_handler = IllegalArgumentErrorHandler("divide by 0");
        return std::vector<Ref<GenericGFPoly> >();
    }
    
    Ref<GenericGFPoly> quotient = field_.getZero();
    Ref<GenericGFPoly> remainder = Ref<GenericGFPoly>(this);
    
    int denominatorLeadingTerm = other->getCoefficient(other->getDegree());
    int inverseDenominatorLeadingTerm = field_.inverse(denominatorLeadingTerm, err_handler);
    if (err_handler.errCode()) return std::vector<Ref<GenericGFPoly> >();
    
    while (remainder->getDegree() >= other->getDegree() && !remainder->isZero()) {
        int degreeDifference = remainder->getDegree() - other->getDegree();
        int scale = field_.multiply(remainder->getCoefficient(remainder->getDegree()),
                                    inverseDenominatorLeadingTerm);
        Ref<GenericGFPoly> term = other->multiplyByMonomial(degreeDifference, scale, err_handler);
        if (err_handler.errCode()) return std::vector<Ref<GenericGFPoly> >();
        Ref<GenericGFPoly> iterationQuotiont = field_.buildMonomial(degreeDifference, scale, err_handler);
        if (err_handler.errCode()) return std::vector<Ref<GenericGFPoly> >();
        quotient = quotient->addOrSubtract(iterationQuotiont, err_handler);
        remainder = remainder->addOrSubtract(term, err_handler);
        if (err_handler.errCode()) return std::vector<Ref<GenericGFPoly> >();
    }
    
    std::vector<Ref<GenericGFPoly> > returnValue(2);
    returnValue[0] = quotient;
    returnValue[1] = remainder;
    return returnValue;
}
