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
 *  Created by Christian Brunschen on 05/05/2008.
 *  Copyright 2008 Google UK. All rights reserved.
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

#include <memory>
#include "reed_solomon_decoder.hpp"
#include "reed_solomon_exception.hpp"
#include "../illegal_argument_exception.hpp"
#include "../../illegal_state_exception.hpp"

using std::vector;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::ReedSolomonDecoder;
using zxing::GenericGFPoly;
using zxing::IllegalStateException;
using zxing::ErrorHandler;

// VC++
using zxing::GenericGF;

ReedSolomonDecoder::ReedSolomonDecoder(Ref<GenericGF> field_) : field(field_) {}

ReedSolomonDecoder::~ReedSolomonDecoder() {
}

void ReedSolomonDecoder::decode(ArrayRef<int> received, int twoS, ErrorHandler & err_handler) {
    Ref<GenericGFPoly> poly(new GenericGFPoly(*field, received, err_handler));
    if (err_handler.errCode())   return;
    ArrayRef<int> syndromeCoefficients(twoS);
    bool noError = true;
    for (int i = 0; i < twoS; i++) {
        int eval = poly->evaluateAt(field->exp(i + field->getGeneratorBase()));
        syndromeCoefficients[syndromeCoefficients->size() - 1 - i] = eval;
        if (eval != 0)
        {
            noError = false;
        }
    }
    if (noError)
    {
        return;
    }
    Ref<GenericGFPoly> syndrome(new GenericGFPoly(*field, syndromeCoefficients, err_handler));
    Ref<GenericGFPoly> monomial = field->buildMonomial(twoS, 1, err_handler);
    if (!monomial || err_handler.errCode())
    {
        err_handler = ErrorHandler("buildMonomial was zero");
        return;
    }
    vector<Ref<GenericGFPoly> > sigmaOmega =
    runEuclideanAlgorithm(monomial, syndrome, twoS, err_handler);
    if (err_handler.errCode()) return;
    
    Ref<GenericGFPoly> sigma = sigmaOmega[0];
    Ref<GenericGFPoly> omega = sigmaOmega[1];
    ArrayRef<int> errorLocations = findErrorLocations(sigma, err_handler);
    if (err_handler.errCode()) return;
    
    ArrayRef<int> errorMagitudes = findErrorMagnitudes(omega, errorLocations, err_handler);
    if (err_handler.errCode()) return;
    for (int i = 0; i < errorLocations->size(); i++) {
        int position = received->size() - 1 - field->log(errorLocations[i], err_handler);
        if (position < 0 || err_handler.errCode()) {
            err_handler = ErrorHandler("Bad error location");
            return;
        }
        received[position] = GenericGF::addOrSubtract(received[position], errorMagitudes[i]);
    }
}

vector<Ref<GenericGFPoly> > ReedSolomonDecoder::runEuclideanAlgorithm(Ref<GenericGFPoly> a,
                                                                      Ref<GenericGFPoly> b,
                                                                      int R, ErrorHandler & err_handler) {
    vector<Ref<GenericGFPoly> > result(2);
    
    // Assume a's degree is >= b's
    if (a->getDegree() < b->getDegree())
    {
        Ref<GenericGFPoly> tmp = a;
        a = b;
        b = tmp;
    }
    
    Ref<GenericGFPoly> rLast(a);
    Ref<GenericGFPoly> r(b);
    Ref<GenericGFPoly> tLast(field->getZero());
    Ref<GenericGFPoly> t(field->getOne());
    
    // Run Euclidean algorithm until r's degree is less than R/2
    while (r->getDegree() >= R / 2) {
        Ref<GenericGFPoly> rLastLast(rLast);
        Ref<GenericGFPoly> tLastLast(tLast);
        rLast = r;
        tLast = t;
        
        // Divide rLastLast by rLast, with quotient q and remainder r
        if (rLast->isZero())
        {
            // Oops, Euclidean algorithm already terminated?
            err_handler = ErrorHandler("r_{i-1} was zero");
            return vector<Ref<GenericGFPoly> >();
        }
        r = rLastLast;
        Ref<GenericGFPoly> q = field->getZero();
        int denominatorLeadingTerm = rLast->getCoefficient(rLast->getDegree());
        int dltInverse = field->inverse(denominatorLeadingTerm, err_handler);
        if (err_handler.errCode())   return vector<Ref<GenericGFPoly> >();
        while (r->getDegree() >= rLast->getDegree() && !r->isZero()) {
            int degreeDiff = r->getDegree() - rLast->getDegree();
            int scale = field->multiply(r->getCoefficient(r->getDegree()), dltInverse);
            q = q->addOrSubtract(field->buildMonomial(degreeDiff, scale, err_handler), err_handler);
            r = r->addOrSubtract(rLast->multiplyByMonomial(degreeDiff, scale, err_handler), err_handler);
            if (err_handler.errCode())   return vector<Ref<GenericGFPoly> >();
        }
        
        Ref<GenericGFPoly> tmp = q->multiply(tLast, err_handler);
        if (err_handler.errCode())   return vector<Ref<GenericGFPoly> >();
        t = tmp->addOrSubtract(tLastLast, err_handler);
        if (err_handler.errCode())   return vector<Ref<GenericGFPoly> >();
        
        if (r->getDegree() >= rLast->getDegree())
        {
            err_handler = ErrorHandler("Division algorithm failed to reduce polynomial?");
            return vector<Ref<GenericGFPoly> >();
        }
    }
    
    int sigmaTildeAtZero = t->getCoefficient(0);
    if (sigmaTildeAtZero == 0)
    {
        err_handler = ErrorHandler("sigmaTilde(0) was zero");
        return vector<Ref<GenericGFPoly> >();
    }
    
    int inverse = field->inverse(sigmaTildeAtZero, err_handler);
    Ref<GenericGFPoly> sigma(t->multiply(inverse, err_handler));
    Ref<GenericGFPoly> omega(r->multiply(inverse, err_handler));
    if (err_handler.errCode())   return vector<Ref<GenericGFPoly> >();
    
    result[0] = sigma;
    result[1] = omega;
    return result;
}

ArrayRef<int> ReedSolomonDecoder::findErrorLocations(Ref<GenericGFPoly> errorLocator, ErrorHandler & err_handler) {
    // This is a direct application of Chien's search
    int numErrors = errorLocator->getDegree();
    if (numErrors == 1)
    {  // shortcut
        ArrayRef<int> result(new Array<int>(1));
        result[0] = errorLocator->getCoefficient(1);
        return result;
    }
    ArrayRef<int> result(new Array<int>(numErrors));
    int e = 0;
    for (int i = 1; i < field->getSize() && e < numErrors; i++) {
        if (errorLocator->evaluateAt(i) == 0)
        {
            result[e] = field->inverse(i, err_handler);
            e++;
        }
    }
    if (e != numErrors || err_handler.errCode())
    {
        err_handler = ErrorHandler("Error locator degree does not match number of root");
        return ArrayRef<int>();
    }
    return result;
}

ArrayRef<int> ReedSolomonDecoder::findErrorMagnitudes(Ref<GenericGFPoly> errorEvaluator, ArrayRef<int> errorLocations, ErrorHandler & err_handler) {
    // This is directly applying Forney's Formula
    int s = errorLocations->size();
    ArrayRef<int> result(new Array<int>(s));
    for (int i = 0; i < s; i++) {
        int xiInverse = field->inverse(errorLocations[i], err_handler);
        int denominator = 1;
        for (int j = 0; j < s; j++) {
            if (i != j)
            {
                int term = field->multiply(errorLocations[j], xiInverse);
                int termPlus1 = (term & 0x1) == 0 ? term | 1 : term & ~1;
                denominator = field->multiply(denominator, termPlus1);
            }
        }
        result[i] = field->multiply(errorEvaluator->evaluateAt(xiInverse), field->inverse(denominator, err_handler));
        if (field->getGeneratorBase() != 0) {
            result[i] = field->multiply(result[i], xiInverse);
        }
    }
    if (err_handler.errCode())   return ArrayRef<int>();
    return result;
}
