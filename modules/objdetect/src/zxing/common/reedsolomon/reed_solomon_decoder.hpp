#ifndef __REED_SOLOMON_DECODER_H__
#define __REED_SOLOMON_DECODER_H__

/*
 *  ReedSolomonDecoder.hpp
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

#include <memory>
#include <vector>
#include "../counted.hpp"
#include "../array.hpp"
#include "generic_gfpoly.hpp"
#include "generic_gf.hpp"
#include "../../error_handler.hpp"

namespace zxing {
class GenericGFPoly;
class GenericGF;

class ReedSolomonDecoder {
private:
    Ref<GenericGF> field;
public:
    ReedSolomonDecoder(Ref<GenericGF> fld);
    ~ReedSolomonDecoder();
    void decode(ArrayRef<int> received, int twoS, ErrorHandler & err_handler);
    std::vector<Ref<GenericGFPoly> > runEuclideanAlgorithm(Ref<GenericGFPoly> a, Ref<GenericGFPoly> b, int R, ErrorHandler & err_handler);
    
private:
    ArrayRef<int> findErrorLocations(Ref<GenericGFPoly> errorLocator, ErrorHandler & err_handler);
    ArrayRef<int> findErrorMagnitudes(Ref<GenericGFPoly> errorEvaluator, ArrayRef<int> errorLocations, ErrorHandler & err_handler);
};
}  // namespace zxing

#endif  // __REED_SOLOMON_DECODER_H__
