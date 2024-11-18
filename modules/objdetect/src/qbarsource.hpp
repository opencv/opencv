// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

/*
 *  Copyright 2010-2011 ZXing authors
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

#ifndef __OPENCV_QBARSOURCE_HPP__
#define __OPENCV_QBARSOURCE_HPP__

#include "zxing/luminance_source.hpp"
#include "zxing/luminance_source.hpp"
#include "zxing/common/byte_matrix.hpp"
#include "zxing/error_handler.hpp"

namespace cv {
class QBarSource : public zxing::LuminanceSource {
private:
    typedef LuminanceSource Super;
    
    int _comps;
    int _pixelStep;
    
    // Added by Skylook for crop function
    zxing::ArrayRef<char> _matrix;
    
    unsigned char* rgbs;
    unsigned char* luminances;
    int dataWidth;
    int dataHeight;
    int maxDataWidth;
    int maxDataHeight;
    int left;
    int top;
    
    unsigned char convertPixel(unsigned char const* pixel, zxing::ErrorHandler & err_handler) const;
    void makeGrayRow(int y, zxing::ErrorHandler & err_handler);
    void makeGray(zxing::ErrorHandler & err_handler);
    void makeGrayReset(zxing::ErrorHandler & err_handler);
    
    void arrayCopy(unsigned char* src, int inputOffset, char* dst, int outputOffset, int length) const;
    
    // Added by L. Wei
    zxing::ArrayRef<char> downSample(zxing::ArrayRef<char> image,  int& width, int& height, int pixelStep);
    
    ~QBarSource();
    
public:
    QBarSource(unsigned char* pixels, int width, int height, int comps, int pixelStep, zxing::ErrorHandler & err_handler);
    QBarSource(unsigned char* pixels, int width, int height, int left, int top, int cropWidth, int cropHeight, int comps, int pixelStep, bool needReverseHorizontal, zxing::ErrorHandler & err_handler);
    
#ifdef USE_IMAGE_DEBUG
    static zxing::Ref<LuminanceSource> create(std::string const& filename);
#endif
    
    static zxing::Ref<QBarSource> create(unsigned char* pixels, int width, int height, int comps, int pixelStep, zxing::ErrorHandler & err_handler);
    static zxing::Ref<QBarSource> create(unsigned char* pixels, int width, int height, int left, int top, int cropWidth, int cropHeight, int comps, int pixelStep, bool needReverseHorizontal, zxing::ErrorHandler & err_handler);
    
    void reset(unsigned char* pixels, int width, int height, int comps, int pixelStep , zxing::ErrorHandler & err_handler);
    
    zxing::ArrayRef<char> getRow(int y, zxing::ArrayRef<char> row, zxing::ErrorHandler & err_handler) const;
    zxing::ArrayRef<char> getMatrix() const;
    zxing::Ref<zxing::ByteMatrix> getByteMatrix()const;
    // int tvInter;
    virtual void denoseLuminanceSource(int inter);
    void tvDenoising() const;
    
    bool isCropSupported() const;
    using LuminanceSource::crop;
    zxing::Ref<LuminanceSource> crop(int left, int top, int width, int height, zxing::ErrorHandler & err_handler) const;
    
    bool isRotateSupported() const;
    using LuminanceSource::rotateCounterClockwise;
    zxing::Ref<LuminanceSource> rotateCounterClockwise(zxing::ErrorHandler & err_handler) const;
    
    void reverseHorizontal(int width, int height);
    
    int getMaxSize(){
        return maxDataHeight*maxDataWidth;
    }
};
}  // namespace cv
#endif // __OPENCV_QBARSOURCE_HPP__