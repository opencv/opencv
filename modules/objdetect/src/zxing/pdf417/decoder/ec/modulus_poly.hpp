#ifndef __MODULUS_GFPOLY_PDF_H__
#define __MODULUS_GFPOLY_PDF_H__

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
 * 2012-09-17 HFN translation from Java into C++
 */

#include "../../../common/counted.hpp"
#include "../../../common/array.hpp"
#include "../../../common/decoder_result.hpp"
#include "../../../common/bit_matrix.hpp"
#include "../../../error_handler.hpp"

namespace zxing {
namespace pdf417 {
namespace decoder {
namespace ec {

class ModulusGF;

/**
 * @author Sean Owen
 * @see com.google.zxing.common.reedsolomon.GenericGFPoly
 */
class ModulusPoly: public Counted {
private:
    ModulusGF &field_;
    ArrayRef<int> coefficients_;
    
public:
    ModulusPoly(ModulusGF& field, ArrayRef<int> coefficients, ErrorHandler & err_handler);
    ~ModulusPoly();
    ArrayRef<int> getCoefficients();
    int getDegree();
    bool isZero();
    int getCoefficient(int degree);
    int evaluateAt(int a);
    Ref<ModulusPoly> add(Ref<ModulusPoly> other, ErrorHandler & err_handler);
    Ref<ModulusPoly> subtract(Ref<ModulusPoly> other, ErrorHandler & err_handler);
    Ref<ModulusPoly> multiply(Ref<ModulusPoly> other, ErrorHandler & err_handler);
    Ref<ModulusPoly> negative(ErrorHandler & err_handler);
    Ref<ModulusPoly> multiply(int scalar, ErrorHandler &err_handler);
    Ref<ModulusPoly> multiplyByMonomial(int degree, int coefficient, ErrorHandler & err_handler);
    std::vector<Ref<ModulusPoly> > divide(Ref<ModulusPoly> other, ErrorHandler & err_handler);
};

}  // namespace ec
}  // namespace decoder
}  // namespace pdf417
}  // namespace zxing

#endif /* __MODULUS_GFPOLY_PDF_H__ */
