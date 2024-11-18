// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __ZXING_DATAMATRIX_DETECTOR_DETECTOR_HPP__
#define __ZXING_DATAMATRIX_DETECTOR_DETECTOR_HPP__

/*
 *  Detector.hpp
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

#include "../../common/counted.hpp"
#include "../../common/detector_result.hpp"
#include "../../common/bit_matrix.hpp"
#include "../../common/perspective_transform.hpp"
#include "../../common/detector/white_rectangle_detector.hpp"
#include "../../error_handler.hpp"

namespace zxing {
namespace datamatrix {

class ResultPointsAndTransitions: public Counted {
private:
    Ref<ResultPoint> to_;
    Ref<ResultPoint> from_;
    int transitions_;
    
public:
    ResultPointsAndTransitions();
    ResultPointsAndTransitions(Ref<ResultPoint> from, Ref<ResultPoint> to, int transitions);
    Ref<ResultPoint> getFrom();
    Ref<ResultPoint> getTo();
    int getTransitions();
};

class Detector: public Counted {
private:
    Ref<BitMatrix> image_;
    
protected:
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY,
                              Ref<PerspectiveTransform> transform, ErrorHandler &err_handler);
    
    void insertionSort(std::vector<Ref<ResultPointsAndTransitions> >& vector);
    
    Ref<ResultPoint> correctTopRightRectangular(Ref<ResultPoint> bottomLeft,
                                                Ref<ResultPoint> bottomRight, Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                                int dimensionTop, int dimensionRight);
    Ref<ResultPoint> correctTopRight(Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight,
                                     Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight, int dimension);
    Ref<ResultPoint> correctTopRightV2(std::vector<Ref<ResultPoint>> & points);
    bool isValid(Ref<ResultPoint> p);
    int distance(Ref<ResultPoint> a, Ref<ResultPoint> b);
    Ref<ResultPointsAndTransitions> transitionsBetween(Ref<ResultPoint> from, Ref<ResultPoint> to);
    int transitionsBetweenV2(Ref<ResultPoint> from, Ref<ResultPoint> to);
    Ref<ResultPoint> shiftPoint(Ref<ResultPoint> point, Ref<ResultPoint> to, int div);
    std::vector<Ref<ResultPoint>> shiftToModuleCenter(std::vector<Ref<ResultPoint>> points);
    Ref<ResultPoint> moveAway(Ref<ResultPoint> point, float fromX, float fromY);
    int min(int a, int b) {
        return a > b ? b : a;
    }
    /**
     * Ends up being a bit faster than round(). This merely rounds its
     * argument to the nearest int, where x.5 rounds up.
     */
    int round(float d) {
        return (int) (d + 0.5f);
    }
    
public:
    Ref<BitMatrix> getImage();
    Detector(Ref<BitMatrix> image);
    
    virtual Ref<PerspectiveTransform> createTransform(Ref<ResultPoint> topLeft,
                                                      Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight,
                                                      int dimensionX, int dimensionY);
    virtual Ref<PerspectiveTransform> createTransformV3(Ref<ResultPoint> topLeft,
                                                        Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight,
                                                        int dimensionX, int dimensionY);
    
    Ref<DetectorResult> detectV1(ErrorHandler & err_handler);
    Ref<DetectorResult> detectV2(ErrorHandler & err_handler);
    Ref<DetectorResult> detectV3(ErrorHandler & err_handler);
    
    Ref<DetectorResult> detect(bool use_v2, bool use_v3, ErrorHandler & err_handler);
    
private:
    int compare(Ref<ResultPointsAndTransitions> a, Ref<ResultPointsAndTransitions> b);
};

}  // namespace datamatrix
}  // namespace zxing

#endif  // __ZXING_DATAMATRIX_DETECTOR_DETECTOR_HPP__
