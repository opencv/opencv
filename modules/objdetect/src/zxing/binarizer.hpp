#ifndef BINARIZER_H_
#define BINARIZER_H_

/*
 *  Binarizer.hpp
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

#include "luminance_source.hpp"
#include "common/bit_array.hpp"
#include "common/bit_matrix.hpp"
#include "common/counted.hpp"
#include "error_handler.hpp"

namespace zxing {

struct BINARIZER_BLOCK
{
    int sum;
    int min;
    int max;
    int threshold;
};

class Binarizer : public Counted {
private:
    Ref<LuminanceSource> source_;
    bool histogramBinarized;
    bool usingHistogram;
    
public:
    Binarizer(Ref<LuminanceSource> source);
    virtual ~Binarizer();
    
    // Added for store binarized result
#ifdef SUPPORT_ROTATE
    uint8_t* _bitCached;
#endif
    
    int dataWidth;
    int dataHeight;
    int width;
    int height;
    
#ifdef SUPPORT_ROTATE
    int _binarizeCached;
    
    // For cache mode equals to 0 it is matrix_, 1 is matrix90_, 2 is matrix45_
    // Added by Skylook
    int _cacheMode;
#endif
    
    // Store dynamicalli choice of which matrix is currently used
    Ref<BitMatrix> matrix_;
    
    // Restore 0 degree result
    Ref<BitMatrix> matrix0_;
    
    Ref<BitMatrix> matrixInverted_;
    
#ifdef SUPPORT_ROTATE
    // Restore 45 degree result
    Ref<BitMatrix> matrix45_;
    
    // Restore 90 degree result
    Ref<BitMatrix> matrix90_;
    
    void setBinCache(int x, int y);
    int getBinCache(int x, int y);
    void startBinCache();
    void endBinCache();
    
    bool isRotateSupported() const {
        return true;
    }
#else
    bool isRotateSupported() const {
        return false;
    }
#endif
    
    // rotate counter clockwise 45 & 90 degree from binarized cache
    int rotateCounterClockwise();
    int rotateCounterClockwise45();
    
    virtual Ref<BitMatrix> getBlackMatrix(ErrorHandler & err_handler);
    virtual Ref<BitMatrix> getInvertedMatrix(ErrorHandler & err_handler);
    virtual Ref<BitArray> getBlackRow(int y, Ref<BitArray> row , ErrorHandler & err_handler);
    
    Ref<LuminanceSource> getLuminanceSource() const;
    
    virtual Ref<Binarizer> createBinarizer(Ref<LuminanceSource> source) {
        return Ref<Binarizer> (new Binarizer(source));
    }
    
    int getWidth() const;
    int getHeight() const;
    
    ArrayRef<BINARIZER_BLOCK> getBlockArray(int size);
};

}  // namespace zxing
#endif /* BINARIZER_H_ */
