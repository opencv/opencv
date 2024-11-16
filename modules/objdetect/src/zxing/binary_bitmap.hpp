#ifndef __BINARYBITMAP_H__
#define __BINARYBITMAP_H__

/*
 *  BinaryBitmap.hpp
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

#include "common/counted.hpp"
#include "common/bit_matrix.hpp"
#include "common/bit_array.hpp"
#include "common/unicom_block.hpp"
#include "binarizer.hpp"
#include "error_handler.hpp"

namespace zxing {

class BinaryBitmap : public Counted {
private:
    Ref<Binarizer> binarizer_;
    
public:
    BinaryBitmap(Ref<Binarizer> binarizer);
    virtual ~BinaryBitmap();
    
    Ref<BitArray> getBlackRow(int y, Ref<BitArray> row, ErrorHandler & err_handler);
    Ref<BitMatrix> getBlackMatrix(ErrorHandler & err_handler);
    Ref<BitMatrix> getInvertedMatrix(ErrorHandler & err_handler);
    
    Ref<LuminanceSource> getLuminanceSource() const;
    Ref<UnicomBlock> m_poUnicomBlock;
    
    int getWidth() const;
    int getHeight() const;
    
    bool isRotateSupported() const;
    Ref<BinaryBitmap> rotateCounterClockwise();
    
    bool isCropSupported() const;
    Ref<BinaryBitmap> crop(int left, int top, int width, int height);
    
    bool isHistogramBinarized() const;
    bool ifUseHistogramBinarize()const;
};

}  // namespace zxing

#endif /* BINARYBITMAP_H_ */
