// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  SimpleAdaptiveBinarizer.cpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors. All rights reserved.
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

#include "simple_adaptive_binarizer.hpp"
#include "../not_found_exception.hpp"
#include "array.hpp"
#include <cstdlib>

using namespace zxing;
// VC++
using zxing::LuminanceSource;

namespace {

const ArrayRef<char> EMPTY(0);
}  // namespace

SimpleAdaptiveBinarizer::SimpleAdaptiveBinarizer(Ref<LuminanceSource> source) 
: GlobalHistogramBinarizer(source), luminances(EMPTY) {
    filtered=false;
}

SimpleAdaptiveBinarizer::~SimpleAdaptiveBinarizer() {}

// Applies simple sharpening to the row data to improve performance of the 1D readers.
Ref<BitArray> SimpleAdaptiveBinarizer::getBlackRow(int y, Ref<BitArray> row, ErrorHandler &err_handler) {
    
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage0(err_handler);
        if (err_handler.ErrCode())   return Ref<BitArray>();
    }
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}

// Does not sharpen the data, as this call is intended to only be used by 2D readers.
Ref<BitMatrix> SimpleAdaptiveBinarizer::getBlackMatrix(ErrorHandler &err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeImage0(err_handler);
        if (err_handler.ErrCode())   return Ref<BitMatrix>();
    }
    
    // First call binarize image in child class to get matrix0_ and binCache
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackMatrix(err_handler);
}

int SimpleAdaptiveBinarizer::binarizeImage0(ErrorHandler &err_handler){
    LuminanceSource& source = *getLuminanceSource();
    int width = source.getWidth();
    int height = source.getHeight();
    Ref<BitMatrix> matrix(new BitMatrix(width, height, err_handler));
    if (err_handler.ErrCode())   return -1;
    
    ArrayRef<char> localLuminances = source.getMatrix();
    
    unsigned char* src = (unsigned char*)localLuminances->data();
    unsigned char* dst = matrix->getPtr();
    
    qrBinarize(src, dst, width, height);
    
    matrix0_ = matrix;
    
    return 0;
}

#define QR_MAXI(_a,_b)      ((_a)-((_a)-(_b)&-((_b)>(_a))))
#define QR_MINI(_a,_b)      ((_a)+((_b)-(_a)&-((_b)<(_a))))

/*A simplified adaptive thresholder.
 This compares the current pixel value to the mean value of a (large) window
 surrounding it.*/
int SimpleAdaptiveBinarizer::qrBinarize(const unsigned char *_img, unsigned char* _dst, int _width, int _height){
    unsigned char *mask = _dst;
    
    if (_width>0&&_height>0){
        unsigned      *col_sums;
        int            logwindw;
        int            logwindh;
        int            windw;
        int            windh;
        int            y0offs;
        int            y1offs;
        unsigned       g;
        int            x;
        int            y;
        
        /*We keep the window size fairly large to ensure it doesn't fit completely
         inside the center of a finder pattern of a version 1 QR code at full
         resolution.*/
        for (logwindw=4; logwindw<8&&(1<<logwindw)<((_width+7)>>3); logwindw++);
        for (logwindh=4; logwindh<8&&(1<<logwindh)<((_height+7)>>3); logwindh++);
        windw=1<<logwindw;
        windh=1<<logwindh;
        
        int logwinds = (logwindw+logwindh);
        
        col_sums=(unsigned *)malloc(_width*sizeof(*col_sums));
        /*Initialize sums down each column.*/
        for (x=0; x<_width; x++){
            g=_img[x];
            col_sums[x]=(g<<(logwindh-1))+g;
        }
        for (y=1; y<(windh>>1); y++){
            y1offs=QR_MINI(y,_height-1)*_width;
            for (x=0; x<_width; x++){
                g=_img[y1offs+x];
                col_sums[x]+=g;
            }
        }
        for (y = 0; y<_height; y++){
            unsigned m;
            int      x0;
            int      x1;
            /*Initialize the sum over the window.*/
            m=(col_sums[0]<<(logwindw-1))+col_sums[0];
            for (x=1; x<(windw>>1); x++){
                x1=QR_MINI(x,_width-1);
                m+=col_sums[x1];
            }
            
            int offset = y*width;
            
            for (x=0; x<_width; x++){
                /*Perform the test against the threshold T = (m/n)-D,
                 where n=windw*windh and D=3.*/
                g=_img[offset+x];
                mask[offset+x]=((g+3)<<(logwinds)<m);
                /*Update the window sum.*/
                if (x+1<_width){
                    x0=QR_MAXI(0, x-(windw>>1));
                    x1=QR_MINI(x+(windw>>1),_width-1);
                    m+=col_sums[x1]-col_sums[x0];
                }
            }
            /*Update the column sums.*/
            if (y+1<_height){
                y0offs=QR_MAXI(0, y-(windh>>1))*_width;
                y1offs=QR_MINI(y+(windh>>1),_height-1)*_width;
                for (x=0; x<_width; x++){
                    col_sums[x]-=_img[y0offs+x];
                    col_sums[x]+=_img[y1offs+x];
                }
            }
        }
        free(col_sums);
    }
    
    return 1;
}

Ref<Binarizer> SimpleAdaptiveBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer> (new SimpleAdaptiveBinarizer(source));
}
