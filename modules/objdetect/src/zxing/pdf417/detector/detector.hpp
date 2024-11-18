// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_PDF417_DETECTOR_DETECTOR_HPP__
#define __ZXING_PDF417_DETECTOR_DETECTOR_HPP__

/*
 *  Detector.hpp
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

#include "../../common/point.hpp"
#include "../../common/detector_result.hpp"
#include "../../not_found_exception.hpp"
#include "../../error_handler.hpp"
#include "../../binary_bitmap.hpp"
#include "../../decode_hints.hpp"

namespace zxing {
namespace pdf417 {
namespace detector {

class Detector {
private:
    static const int INTEGER_MATH_SHIFT = 8;
    static const int PATTERN_MATCH_RESULT_SCALE_FACTOR = 1 << INTEGER_MATH_SHIFT;
    static const int MAX_AVG_VARIANCE;
    static const int MAX_INDIVIDUAL_VARIANCE;
    
    static const int START_PATTERN[];
    static const int START_PATTERN_LENGTH;
    static const int START_PATTERN_REVERSE[];
    static const int START_PATTERN_REVERSE_LENGTH;
    static const int STOP_PATTERN[];
    static const int STOP_PATTERN_LENGTH;
    static const int STOP_PATTERN_REVERSE[];
    static const int STOP_PATTERN_REVERSE_LENGTH;
    
    static const int MIN_BARCODE_HEIGH = 40;
    static const int MAX_PATTERN_DRIFT = 5;
    static const int SKIPPED_ROW_COUNT_MAX = 25;
    static const int INDEXES_START_PATTERN[];
    static const int INDEXES_START_PATTERN_LENGTH;
    static const int INDEXES_STOP_PATTERN[];
    static const int INDEXES_STOP_PATTERN_LENGTH;
    static const int SIMPLE_ROW_STEP = 8;
    static const int DETAIL_ROW_STEP = 2;
    static const int DETAIL_ROW_COUNT = 10;
    
    Ref<BinaryBitmap> image_;
    
    static ArrayRef< Ref<ResultPoint> > findVertices(Ref<BitMatrix> matrix, int rowStep);
    static ArrayRef< Ref<ResultPoint> > findVertices180(Ref<BitMatrix> matrix, int rowStep);
    
    static ArrayRef<int> findGuardPattern(Ref<BitMatrix> matrix,
                                          int column,
                                          int row,
                                          int width,
                                          bool whiteFirst,
                                          const int pattern[],
                                          int patternSize,
                                          ArrayRef<int>& counters);
    static int patternMatchVariance(ArrayRef<int>& counters, const int pattern[],
                                    int maxIndividualVariance);
    
    static zxing::ErrorHandler correctVertices(Ref<BitMatrix> matrix,
                                               ArrayRef< Ref<ResultPoint> >& vertices,
                                               bool upsideDown);
    static void findWideBarTopBottom(Ref<BitMatrix> matrix,
                                     ArrayRef< Ref<ResultPoint> >& vertices,
                                     int offsetVertice,
                                     int startWideBar,
                                     int lenWideBar,
                                     int lenPattern,
                                     int nIncrement);
    
    static ErrorHandler findCrossingPoint(ArrayRef< Ref<ResultPoint> >& vertices,
                                          int idxResult,
                                          int idxLineA1, int idxLineA2,
                                          int idxLineB1, int idxLineB2,
                                          Ref<BitMatrix>& matrix);
    static Point intersection(Line a, Line b);
    static float computeModuleWidth(ArrayRef< Ref<ResultPoint> >& vertices);
    static int computeDimension(Ref<ResultPoint> const& topLeft,
                                Ref<ResultPoint> const& topRight,
                                Ref<ResultPoint> const& bottomLeft,
                                Ref<ResultPoint> const& bottomRight,
                                float moduleWidth);
    int computeYDimension(Ref<ResultPoint> const& topLeft,
                          Ref<ResultPoint> const& topRight,
                          Ref<ResultPoint> const& bottomLeft,
                          Ref<ResultPoint> const& bottomRight,
                          float moduleWidth);
    
    Ref<BitMatrix> sampleLines(ArrayRef< Ref<ResultPoint> > const& vertices, int dimensionY, int dimension, ErrorHandler & err_handler);
    
    ArrayRef< Ref<ResultPoint> > findRowsWithPattern(Ref<BitMatrix> matrix,
                                                     int startRow, int startColumn,
                                                     const int pattern[], const int patternSize);
    
    ArrayRef< Ref<ResultPoint> > findVerticesNew(Ref<BitMatrix> matrix);
    
    void copyToResult(ArrayRef< Ref<ResultPoint> > result, ArrayRef< Ref<ResultPoint> > tmpResult,
                      const int* destinationIndexes, int iLength);

public:
    Detector(Ref<BinaryBitmap> image);
    Ref<BinaryBitmap> getImage();

    zxing::ErrorHandler detect(Ref<DetectorResult> & detect_rst);
    zxing::ErrorHandler detect(DecodeHints const& hints, Ref<DetectorResult> & detect_rst);
};

}  // namespace detector
}  // namespace pdf417
}  // namespace zxing

#endif  // __ZXING_PDF417_DETECTOR_DETECTOR_HPP__
