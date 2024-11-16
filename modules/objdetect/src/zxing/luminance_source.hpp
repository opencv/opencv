// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __LUMINANCESOURCE_H__
#define __LUMINANCESOURCE_H__
/*
 *  LuminanceSource.hpp
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
#include "common/array.hpp"
#include "common/byte_matrix.hpp"
#include <string.h>
#include "error_handler.hpp"

namespace zxing {

class LuminanceSource : public Counted {
private:
    int width;
    int height;
    
public:
    LuminanceSource(int width, int height);
    virtual ~LuminanceSource();
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    void setWidth(int w){width = w;}
    void setHeight(int h){height = h;}
    void filter();
    int tvInter;
    virtual void denoseLuminanceSource(int inter);
    
    // Callers take ownership of the returned memory and must call delete [] on it themselves.
    virtual ArrayRef<char> getRow(int y, ArrayRef<char> row, zxing::ErrorHandler & err_handler) const = 0;
    virtual ArrayRef<char> getMatrix() const = 0;
    virtual Ref<ByteMatrix> getByteMatrix() const = 0;
    
    virtual bool isCropSupported() const;
    virtual Ref<LuminanceSource> crop(int left, int top, int width, int height) const;
    
    virtual bool isRotateSupported() const;
    
    virtual Ref<LuminanceSource> invert() const;
    
    virtual Ref<LuminanceSource> rotateCounterClockwise() const;
    
    operator std::string () const;
};

}  // namespace zxing

#endif /* LUMINANCESOURCE_H_ */
