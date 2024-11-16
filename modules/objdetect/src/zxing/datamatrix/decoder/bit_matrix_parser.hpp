#ifndef __BIT_MATRIX_PARSER_DM_H__
#define __BIT_MATRIX_PARSER_DM_H__

/*
 *  BitMatrixParser.hpp
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

#include "../../reader_exception.hpp"
#include "../../common/bit_matrix.hpp"
#include "../../common/counted.hpp"
#include "../../common/array.hpp"
#include "../version.hpp"
#include "../../error_handler.hpp"

namespace zxing {
namespace datamatrix {

class BitMatrixParser : public Counted {
private:
    Ref<BitMatrix> bitMatrix_;
    Version * parsedVersion_;
    Ref<BitMatrix> readBitMatrix_;
    
    int copyBit(size_t x, size_t y, int versionBits);
    
public:
    BitMatrixParser(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler);
    Version * readVersion(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler);
    
    ArrayRef<char> readCodewords(ErrorHandler& err_handler);
    bool readModule(int row, int column, int numRows, int numColumns);
    
private:
    int readUtah(int row, int column, int numRows, int numColumns);
    int readCorner1(int numRows, int numColumns);
    int readCorner2(int numRows, int numColumns);
    int readCorner3(int numRows, int numColumns);
    int readCorner4(int numRows, int numColumns);
    Ref<BitMatrix> extractDataRegion(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler);
};

}  // namespace datamatrix
}  // namespace zxing

#endif  // __BIT_MATRIX_PARSER_DM_H__
