#ifndef __DATA_BLOCK_H__
#define __DATA_BLOCK_H__

/*
 *  DataBlock.hpp
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

#include "../../common/counted.hpp"
#include "../../common/array.hpp"
#include "../version.hpp"
#include "../error_correction_level.hpp"
#include "../../error_handler.hpp"

namespace zxing {
namespace qrcode {

class DataBlock : public Counted {
private:
    int numDataCodewords_;
    ArrayRef<char> codewords_;
    
    DataBlock(int numDataCodewords, ArrayRef<char> codewords);
    
public:
    static std::vector<Ref<DataBlock> > getDataBlocks(ArrayRef<char> rawCodewords, Version *version, ErrorCorrectionLevel &ecLevel, ErrorHandler & err_handler);
    
    int getNumDataCodewords();
    ArrayRef<char> getCodewords();
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __DATA_BLOCK_H__
