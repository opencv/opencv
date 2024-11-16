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

#include "error_correction.hpp"
#include "modulus_poly.hpp"
#include "modulus_gf.hpp"

using std::vector;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::pdf417::decoder::ec::ErrorCorrection;
using zxing::pdf417::decoder::ec::ModulusPoly;
using zxing::pdf417::decoder::ec::ModulusGF;
using zxing::ReedSolomonErrorHandler;
using zxing::ErrorHandler;

/**
 * <p>PDF417 error correction implementation.</p>
 *
 * <p>This <a href="http:// en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction#Example">example</a>
 * is quite useful in understanding the algorithm.</p>
 *
 * @author Sean Owen
 * @see com.google.zxing.common.reedsolomon.ReedSolomonDecoder
 */

ErrorCorrection::ErrorCorrection(): field_(MODULUS_GF_PDF417)
{
}

void ErrorCorrection::decode(ArrayRef<int> received,
                             int numECCodewords,
                             ArrayRef<int> erasures,
                             ErrorHandler & err_handler)
{
    Ref<ModulusPoly> poly (new ModulusPoly(field_, received, err_handler));
    if (err_handler.ErrCode())   return;
    
    ArrayRef<int> S(new Array<int>(numECCodewords));
    bool error = false;
    for (int i = numECCodewords; i > 0; i--) {
        int eval = poly->evaluateAt(field_.exp(i));
        S[numECCodewords - i] = eval;
        if (eval != 0) {
            error = true;
        }
    }
    
    if (error) {
        Ref<ModulusPoly> knownErrors = field_.getOne();
        for (int i = 0; i<erasures->size(); i++) {
            int b = field_.exp(received->size() - 1 - erasures[i]);
            // Add (1 - bx) term:
            ArrayRef<int> one_minus_b_x(new Array<int>(2));
            one_minus_b_x[1]=field_.subtract(0, b);
            one_minus_b_x[0] = 1;
            Ref<ModulusPoly> term (new ModulusPoly(field_, one_minus_b_x, err_handler));
            if (err_handler.ErrCode()) return;
            
            knownErrors = knownErrors->multiply(term, err_handler);
            if (err_handler.ErrCode()) return;
        }
        
        Ref<ModulusPoly> syndrome (new ModulusPoly(field_, S, err_handler));
        if (err_handler.ErrCode()) return;
        
        Ref<ModulusPoly> tmp = field_.buildMonomial(numECCodewords, 1, err_handler);
        if (err_handler.ErrCode() || tmp == NULL) return;
        
        vector<Ref<ModulusPoly> > sigmaOmega (runEuclideanAlgorithm(tmp, syndrome, numECCodewords, err_handler));
        if (err_handler.ErrCode())   return;
        
        Ref<ModulusPoly> sigma = sigmaOmega[0];
        Ref<ModulusPoly> omega = sigmaOmega[1];
        
        ArrayRef<int> errorLocations = findErrorLocations(sigma, err_handler);
        if (err_handler.ErrCode()) return;
        
        ArrayRef<int> errorMagnitudes = findErrorMagnitudes(omega, sigma, errorLocations, err_handler);
        if (err_handler.ErrCode()) return;
        
        for (int i = 0; i < errorLocations->size(); i++) {
            int ret_val = field_.log(errorLocations[i], err_handler);
            if (err_handler.ErrCode())   return;
            
            int position = received->size() - 1 - ret_val;
            if (position < 0) {
                err_handler = ReedSolomonErrorHandler("Bad error location!");
                return;
            }
            received[position] = field_.subtract(received[position], errorMagnitudes[i]);
        }
    }
}

vector<Ref<ModulusPoly> >  ErrorCorrection::runEuclideanAlgorithm(Ref<ModulusPoly> a, Ref<ModulusPoly> b, int R, ErrorHandler & err_handler)
{
    // Assume a's degree is >= b's
    if (a->getDegree() < b->getDegree()) {
        Ref<ModulusPoly> temp = a;
        a = b;
        b = temp;
    }
    
    Ref<ModulusPoly> rLast (a);
    Ref<ModulusPoly> r (b);
    Ref<ModulusPoly> tLast (field_.getZero());
    Ref<ModulusPoly> t (field_.getOne());
    
    // Run Euclidean algorithm until r's degree is less than R/2
    while (r->getDegree() >= R / 2) {
        Ref<ModulusPoly> rLastLast (rLast);
        Ref<ModulusPoly> tLastLast (tLast);
        rLast = r;
        tLast = t;
        
        // Divide rLastLast by rLast, with quotient in q and remainder in r
        if (rLast->isZero()) {
            // Oops, Euclidean algorithm already terminated?
            err_handler = ReedSolomonErrorHandler("Euclidean algorithm already terminated?");
            return vector<Ref<ModulusPoly> >();
        }
        r = rLastLast;
        Ref<ModulusPoly> q (field_.getZero());
        int denominatorLeadingTerm = rLast->getCoefficient(rLast->getDegree());
        int dltInverse = field_.inverse(denominatorLeadingTerm, err_handler);
        if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
        
        while (r->getDegree() >= rLast->getDegree() && !r->isZero()) {
            int degreeDiff = r->getDegree() - rLast->getDegree();
            int scale = field_.multiply(r->getCoefficient(r->getDegree()), dltInverse);
            
            Ref<ModulusPoly> tmp = field_.buildMonomial(degreeDiff, scale, err_handler);
            if (err_handler.ErrCode() || tmp == NULL)  return vector<Ref<ModulusPoly> >();
            
            q = q->add(tmp, err_handler);
            if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
            
            Ref<ModulusPoly> tmp1 = rLast->multiplyByMonomial(degreeDiff, scale, err_handler);
            if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
            
            r = r->subtract(tmp1, err_handler);
            if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
        }
        
        Ref<ModulusPoly> tmp1 = q->multiply(tLast, err_handler);
        if (err_handler.ErrCode() || tmp1 == NULL)  return vector<Ref<ModulusPoly> >();
        
        tmp1 = tmp1->subtract(tLastLast, err_handler);
        if (err_handler.ErrCode() || tmp1 == NULL)  return vector<Ref<ModulusPoly> >();
        
        t = tmp1->negative(err_handler);
        if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
    }
    
    int sigmaTildeAtZero = t->getCoefficient(0);
    if (sigmaTildeAtZero == 0) {
        err_handler = ReedSolomonErrorHandler("sigmaTilde = 0!");
        return vector<Ref<ModulusPoly> >();
    }
    
    int inverse = field_.inverse(sigmaTildeAtZero, err_handler);
    if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
    
    Ref<ModulusPoly> sigma (t->multiply(inverse, err_handler));
    if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
    
    Ref<ModulusPoly> omega (r->multiply(inverse, err_handler));
    if (err_handler.ErrCode())  return vector<Ref<ModulusPoly> >();
    
    vector<Ref<ModulusPoly> > v(2);
    v[0] = sigma;
    v[1] = omega;
    return v;
}

ArrayRef<int> ErrorCorrection::findErrorLocations(Ref<ModulusPoly> errorLocator, ErrorHandler & err_handler)  {
    // This is a direct application of Chien's search
    int numErrors = errorLocator->getDegree();
    ArrayRef<int> result(new Array<int>(numErrors));
    int e = 0;
    for (int i = 1; i < field_.getSize() && e < numErrors; i++) {
        if (errorLocator->evaluateAt(i) == 0) {
            result[e] = field_.inverse(i, err_handler);
            if (err_handler.ErrCode())  return ArrayRef<int>();
            
            e++;
        }
    }
    if (e != numErrors) {
        err_handler = ReedSolomonErrorHandler("Error number inconsistency!");
        return ArrayRef<int>();
    }

    return result;
}

ArrayRef<int> ErrorCorrection::findErrorMagnitudes(Ref<ModulusPoly> errorEvaluator,
                                                   Ref<ModulusPoly> errorLocator,
                                                   ArrayRef<int> errorLocations,
                                                   ErrorHandler & err_handler) {
    int i;
    int errorLocatorDegree = errorLocator->getDegree();
    ArrayRef<int> formalDerivativeCoefficients(new Array<int>(errorLocatorDegree));
    for (i = 1; i <= errorLocatorDegree; i++) {
        formalDerivativeCoefficients[errorLocatorDegree - i] =
        field_.multiply(i, errorLocator->getCoefficient(i));
    }
    Ref<ModulusPoly> formalDerivative (new ModulusPoly(field_, formalDerivativeCoefficients, err_handler));
    if (err_handler.ErrCode())   return ArrayRef<int>();
    
    // This is directly applying Forney's Formula
    int s = errorLocations->size();
    ArrayRef<int> result(new Array<int>(s));
    for (i = 0; i < s; i++) {
        int xiInverse = field_.inverse(errorLocations[i], err_handler);
        if (err_handler.ErrCode())   return ArrayRef<int>();
        
        int numerator = field_.subtract(0, errorEvaluator->evaluateAt(xiInverse));
        int denominator = field_.inverse(formalDerivative->evaluateAt(xiInverse), err_handler);
        if (err_handler.ErrCode())   return ArrayRef<int>();
        
        result[i] = field_.multiply(numerator, denominator);
    }
    return result;
}
