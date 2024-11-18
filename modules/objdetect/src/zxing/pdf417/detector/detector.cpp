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
/*
 * Copyright 2010 ZXing authors All rights reserved.
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

#include <limits>
#include "detector.hpp"
#include "lines_sampler.hpp"
#include "../../common/grid_sampler.hpp"
#include "../../common/detector/java_math.hpp"
#include "../../common/detector/math_utils.hpp"
#include <iostream>

using std::max;
using std::abs;
using std::numeric_limits;
using zxing::pdf417::detector::Detector;
using zxing::common::detector::Math;
using zxing::common::detector::MathUtils;
using zxing::Ref;
using zxing::ArrayRef;
using zxing::DetectorResult;
using zxing::ResultPoint;
using zxing::Point;
using zxing::BitMatrix;
using zxing::GridSampler;

// VC++

using zxing::BinaryBitmap;
using zxing::DecodeHints;
using zxing::Line;
using zxing::ErrorHandler;

/**
 * <p>Encapsulates logic that can detect a PDF417 Code in an image, even if the
 * PDF417 Code is rotated or skewed, or partially obscured.</p>
 *
 * @author SITA Lab (kevin.osullivan@sita.aero)
 * @author Daniel Switkin (dswitkin@google.com)
 * @author Schweers Informationstechnologie GmbH (hartmut.neubauer@schweers.de)
 * @author creatale GmbH (christoph.schulz@creatale.de)
 */

const int Detector::MAX_AVG_VARIANCE= static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.42f);
const int Detector::MAX_INDIVIDUAL_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.8f);

// B S B S B S B S Bar/Space pattern
// 11111111 0 1 0 1 0 1 000
const int Detector::START_PATTERN[] = {8, 1, 1, 1, 1, 1, 1, 3};
const int Detector::START_PATTERN_LENGTH = sizeof(START_PATTERN) / sizeof(int);

// 11111111 0 1 0 1 0 1 000
const int Detector::START_PATTERN_REVERSE[] = {3, 1, 1, 1, 1, 1, 1, 8};
const int Detector::START_PATTERN_REVERSE_LENGTH = sizeof(START_PATTERN_REVERSE) / sizeof(int);

// 1111111 0 1 000 1 0 1 00 1
const int Detector::STOP_PATTERN[] = {7, 1, 1, 3, 1, 1, 1, 2, 1};
const int Detector::STOP_PATTERN_LENGTH = sizeof(STOP_PATTERN) / sizeof(int);

// B S B S B S B S B Bar/Space pattern
// 1111111 0 1 000 1 0 1 00 1
const int Detector::STOP_PATTERN_REVERSE[] = {1, 2, 1, 1, 1, 3, 1, 1, 7};
const int Detector::STOP_PATTERN_REVERSE_LENGTH = sizeof(STOP_PATTERN_REVERSE) / sizeof(int);

const int Detector::INDEXES_START_PATTERN[] = { 0, 4, 1, 5 };
const int Detector::INDEXES_START_PATTERN_LENGTH = sizeof(INDEXES_START_PATTERN) / sizeof(int);
const int Detector::INDEXES_STOP_PATTERN[] = { 6, 2, 7, 3 };
const int Detector::INDEXES_STOP_PATTERN_LENGTH = sizeof(INDEXES_START_PATTERN) / sizeof(int);

Detector::Detector(Ref<BinaryBitmap> image) : image_(image) {}

ErrorHandler Detector::detect(Ref<DetectorResult> & detect_rst){
    return detect(DecodeHints(), detect_rst);
}

ErrorHandler Detector::detect(DecodeHints const& hints, Ref<DetectorResult> & detect_rst)
{
    (void)hints;
    // Fetch the 1 bit matrix once up front.
    ErrorHandler err_handler;
    Ref<BitMatrix> matrix = image_->getBlackMatrix(err_handler);
    if (err_handler.errCode()) return err_handler;
    
    // Try to find the vertices assuming the image is upright.
    const int rowStep = 8;
    
    ArrayRef< Ref<ResultPoint> > vertices (findVertices(matrix, rowStep));
    if (!vertices) {
        // Maybe the image is rotated 180 degrees?
        vertices = findVertices180(matrix, rowStep);
        if (vertices) {
            err_handler = correctVertices(matrix, vertices, true);
            if (err_handler.errCode()) return err_handler;
        }
    }
    else
    {
        err_handler = correctVertices(matrix, vertices, false);
        if (err_handler.errCode()) return err_handler;
    }
    
    
    if (!vertices) {
        return NotFoundErrorHandler("No vertices found.");
    }
    
    {
        if (vertices[12]->getY() > 0)
            vertices[12] = new ResultPoint(vertices[12]->getX(), vertices[12]->getY() - 1);
        if (vertices[13]->getY() < matrix->getHeight())
            vertices[13] = new ResultPoint(vertices[13]->getX(), vertices[13]->getY() + 1);
    }
    {
        if (vertices[14]->getY() >0)
            vertices[14] = new ResultPoint(vertices[14]->getX(), vertices[14]->getY() - 1);
        if (vertices[15]->getY() < matrix->getHeight())
            vertices[15] = new ResultPoint(vertices[15]->getX(), vertices[15]->getY() + 1);
    }
    
    float moduleWidth = computeModuleWidth(vertices);
    if (moduleWidth < 1.0f) {
        return NotFoundErrorHandler("Bad module width.");
    }
    
    int dimension = computeDimension(vertices[12], vertices[14],
                                     vertices[13], vertices[15], moduleWidth);
    if (dimension < 1) {
        return NotFoundErrorHandler("Bad dimension.");
    }
    
    int yDimension = max(computeYDimension(vertices[12], vertices[14],
                                           vertices[13], vertices[15], moduleWidth), dimension);
    
    // Deskew and sample lines from image.
    Ref<BitMatrix> linesMatrix = sampleLines(vertices, dimension, yDimension, err_handler);
    if (err_handler.errCode()) return err_handler;
    
    ArrayRef< Ref<ResultPoint> > points(4);
    points[0] = vertices[5];
    points[1] = vertices[4];
    points[2] = vertices[6];
    points[3] = vertices[7];
    
    detect_rst = new DetectorResult(linesMatrix, points, dimension);
    return ErrorHandler(0);
}

/**
 * Locate the vertices and the codewords area of a black blob using the Start
 * and Stop patterns as locators.
 *
 * @param matrix the scanned barcode image.
 * @param rowStep the step size for iterating rows (every n-th row).
 * @return an array containing the vertices:
 *           vertices[0] x, y top left barcode
 *           vertices[1] x, y bottom left barcode
 *           vertices[2] x, y top right barcode
 *           vertices[3] x, y bottom right barcode
 *           vertices[4] x, y top left codeword area
 *           vertices[5] x, y bottom left codeword area
 *           vertices[6] x, y top right codeword area
 *           vertices[7] x, y bottom right codeword area
 */
ArrayRef< Ref<ResultPoint> > Detector::findRowsWithPattern(Ref<BitMatrix> matrix,
                                                           int startRow, int startColumn,
                                                           const int pattern[], const int patternSize){
    
    const int height = matrix->getHeight();
    const int width = matrix->getWidth();
    
    int iFastDetectBeginHeight = height / 2 - DETAIL_ROW_COUNT / 2 * DETAIL_ROW_STEP;
    int iFastDetectEndHeight = height / 2 + DETAIL_ROW_COUNT / 2 * DETAIL_ROW_STEP;
    iFastDetectBeginHeight = max(0, iFastDetectBeginHeight);
    iFastDetectEndHeight = (std::min)(iFastDetectEndHeight, height);
    
    ArrayRef< Ref<ResultPoint> > result(4);
    bool found = false;
    ArrayRef<int> counters(new Array<int>(patternSize));
    std::vector<int> vcStartRowList;
    
    for (; startRow < height; startRow += SIMPLE_ROW_STEP)
    {
        ArrayRef<int> loc = findGuardPattern(matrix, startColumn, startRow, width, false, pattern,
                                             patternSize, counters);
        if (loc) {
            while (startRow > 0) {
                ArrayRef<int> previousRowLoc = findGuardPattern(matrix, startColumn, --startRow, width, false, pattern,
                                                                patternSize, counters);
                if (previousRowLoc) {
                    loc = previousRowLoc;
                }
                else
                {
                    startRow++;
                    break;
                }
            }
            result[0] = new ResultPoint(loc[0], startRow);
            result[1] = new ResultPoint(loc[1], startRow);
            found = true;
            break;
        }
    }
    int stopRow = startRow + 1;
    // Last row of the current symbol that contains pattern
    if (found) {
        int skippedRowCount = 0;
        ArrayRef<int> previousRowLoc = new Array<int>(2);
        previousRowLoc[0] = static_cast<int>(result[0]->getX());
        previousRowLoc[1] = static_cast<int>(result[1]->getX());
        for (; stopRow < height; stopRow++) {
            ArrayRef<int> loc = findGuardPattern(matrix, previousRowLoc[0], stopRow, width, false, pattern,
                                                 patternSize, counters);
            // a found pattern is only considered to belong to the same barcode if the start and end positions
            // don't differ too much. Pattern drift should be not bigger than two for consecutive rows. With
            // a higher number of skipped rows drift could be larger. To keep it simple for now, we allow a slightly
            // larger drift and don't check for skipped rows.
            if (loc &&
                abs(previousRowLoc[0] - loc[0]) < MAX_PATTERN_DRIFT &&
                abs(previousRowLoc[1] - loc[1]) < MAX_PATTERN_DRIFT) {
                previousRowLoc = loc;
                skippedRowCount = 0;
            }
            else
            {
                if (skippedRowCount > SKIPPED_ROW_COUNT_MAX) {
                    break;
                }
                else
                {
                    skippedRowCount++;
                }
            }
        }
        stopRow -= skippedRowCount + 1;
        result[2] = new ResultPoint(previousRowLoc[0], stopRow);
        result[3] = new ResultPoint(previousRowLoc[1], stopRow);
    }
    return result;
}

ArrayRef< Ref<ResultPoint> > Detector::findVerticesNew(Ref<BitMatrix> matrix)
{
    int startRow = 0;
    int startColumn = 0;
    
    ArrayRef< Ref<ResultPoint> > result(16);
    copyToResult(result,
                 findRowsWithPattern(matrix, startRow, startColumn,
                                     START_PATTERN, START_PATTERN_LENGTH),
                 INDEXES_START_PATTERN, INDEXES_START_PATTERN_LENGTH);
    
    if (result[4]) {
        startColumn = static_cast<int>(result[4]->getX());
        startRow = static_cast<int>(result[4]->getY());
    }
    copyToResult(result, findRowsWithPattern(matrix, startRow, startColumn,
                                             STOP_PATTERN, STOP_PATTERN_LENGTH),
                 INDEXES_STOP_PATTERN, INDEXES_STOP_PATTERN_LENGTH);
    if (result[0] && result[1] && result[2] && result[3]) return result;
    return ArrayRef< Ref<ResultPoint> >();
}

void Detector::copyToResult(ArrayRef< Ref<ResultPoint> > result, ArrayRef< Ref<ResultPoint> > tmpResult,
                            const int* destinationIndexes, int iLength) {
    for (int i = 0; i < iLength; i++) {
        result[destinationIndexes[i]] = tmpResult[i];
    }
}
ArrayRef< Ref<ResultPoint> > Detector::findVertices(Ref<BitMatrix> matrix, int rowStep)
{
    const int height = matrix->getHeight();
    const int width = matrix->getWidth();
    
    ArrayRef< Ref<ResultPoint> > result(16);
    bool found = false;
    
    ArrayRef<int> counters(new Array<int>(START_PATTERN_LENGTH));
    
    // Top Left
    for (int i = 0; i < height; i += rowStep) {
        ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, START_PATTERN,
                                             START_PATTERN_LENGTH, counters);
        if (loc) {
            result[0] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
            result[4] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
            found = true;
            break;
        }
    }
    // Bottom left
    if (found) {  // Found the Top Left vertex
        found = false;
        for (int i = height - 1; i > 0; i -= rowStep) {
            ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, START_PATTERN,
                                                 START_PATTERN_LENGTH, counters);
            if (loc) {
                result[1] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                result[5] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    
    counters = new Array<int>(STOP_PATTERN_LENGTH);
    
    // Top right
    if (found) {  // Found the Bottom Left vertex
        found = false;
        for (int i = 0; i < height; i += rowStep) {
            ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, STOP_PATTERN,
                                                 STOP_PATTERN_LENGTH, counters);
            if (loc) {
                result[2] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                result[6] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    // Bottom right
    if (found) {  // Found the Top right vertex
        found = false;
        for (int i = height - 1; i > 0; i -= rowStep) {
            ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, STOP_PATTERN,
                                                 STOP_PATTERN_LENGTH, counters);
            if (loc) {
                result[3] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                result[7] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    return found ? result : ArrayRef< Ref<ResultPoint> >();
}

ArrayRef< Ref<ResultPoint> > Detector::findVertices180(Ref<BitMatrix> matrix, int rowStep) {
    const int height = matrix->getHeight();
    const int width = matrix->getWidth();
    // const int halfWidth = width >> 1;
    
    ArrayRef< Ref<ResultPoint> > result(16);
    bool found = false;
    
    ArrayRef<int> counters = new Array<int>(START_PATTERN_REVERSE_LENGTH);
    
    // Top Left
    for (int i = height - 1; i > 0; i -= rowStep) {
        ArrayRef<int> loc =
        findGuardPattern(matrix, 0, i, width, true, START_PATTERN_REVERSE,
                         START_PATTERN_REVERSE_LENGTH, counters);
        if (loc) {
            result[0] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
            result[4] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
            found = true;
            break;
        }
    }
    // Bottom Left
    if (found) {  // Found the Top Left vertex
        found = false;
        for (int i = 0; i < height; i += rowStep) {
            ArrayRef<int> loc =
            findGuardPattern(matrix, 0, i, width, true, START_PATTERN_REVERSE,
                             START_PATTERN_REVERSE_LENGTH, counters);
            if (loc) {
                result[1] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                result[5] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    
    counters = new Array<int>(STOP_PATTERN_REVERSE_LENGTH);
    
    // Top Right
    if (found) {  // Found the Bottom Left vertex
        found = false;
        for (int i = height - 1; i > 0; i -= rowStep) {
            ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, STOP_PATTERN_REVERSE,
                                                 STOP_PATTERN_REVERSE_LENGTH, counters);
            if (loc) {
                result[2] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                result[6] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    // Bottom Right
    if (found) {  // Found the Top Right vertex
        found = false;
        for (int i = 0; i < height; i += rowStep) {
            ArrayRef<int> loc = findGuardPattern(matrix, 0, i, width, false, STOP_PATTERN_REVERSE,
                                                 STOP_PATTERN_REVERSE_LENGTH, counters);
            if (loc) {
                result[3] = new ResultPoint(static_cast<float>(loc[0]), static_cast<float>(i));
                result[7] = new ResultPoint(static_cast<float>(loc[1]), static_cast<float>(i));
                found = true;
                break;
            }
        }
    }
    
    return found ? result : ArrayRef< Ref<ResultPoint> >();
}

/**
 * @param matrix row of black/white values to search
 * @param column x position to start search
 * @param row y position to start search
 * @param width the number of pixels to search on this row
 * @param pattern pattern of counts of number of black and white pixels that are
 *                 being searched for as a pattern
 * @param counters array of counters, as long as pattern, to re-use
 * @return start/end horizontal offset of guard pattern, as an array of two ints.
 */
ArrayRef<int> Detector::findGuardPattern(Ref<BitMatrix> matrix,
                                         int column,
                                         int row,
                                         int width,
                                         bool whiteFirst,
                                         const int pattern[],
                                         int patternSize,
                                         ArrayRef<int>& counters) {
    counters->values().assign(counters->size(), 0);
    int patternLength = patternSize;
    bool isWhite = whiteFirst;
    
    int counterPosition = 0;
    int patternStart = column;
    for (int x = column; x < width; x++) {
        bool pixel = matrix->get(x, row);
        if (pixel ^ isWhite) {
            counters[counterPosition]++;
        }
        else
        {
            if (counterPosition == patternLength - 1) {
                if (patternMatchVariance(counters, pattern,
                                         MAX_INDIVIDUAL_VARIANCE) < MAX_AVG_VARIANCE) {
                    ArrayRef<int> result = new Array<int>(2);
                    result[0] = patternStart;
                    result[1] = x;
                    return result;
                }
                patternStart += counters[0] + counters[1];
                for (int i = 0; i < patternLength - 2; ++i)
                    counters[i] = counters[ i + 2];
                counters[patternLength - 2] = 0;
                counters[patternLength - 1] = 0;
                counterPosition--;
            }
            else
            {
                counterPosition++;
            }
            counters[counterPosition] = 1;
            isWhite = !isWhite;
        }
    }
    return ArrayRef<int>();
}

/**
 * Determines how closely a set of observed counts of runs of black/white
 * values matches a given target pattern. This is reported as the ratio of
 * the total variance from the expected pattern proportions across all
 * pattern elements, to the length of the pattern.
 *
 * @param counters observed counters
 * @param pattern expected pattern
 * @param maxIndividualVariance The most any counter can differ before we give up
 * @return ratio of total variance between counters and pattern compared to
 *         total pattern size, where the ratio has been multiplied by 256.
 *         So, 0 means no variance (perfect match); 256 means the total
 *         variance between counters and patterns equals the pattern length,
 *         higher values mean even more variance
 */
int Detector::patternMatchVariance(ArrayRef<int>& counters,
                                   const int pattern[],
                                   int maxIndividualVariance)
{
    int numCounters = counters->size();
    int total = 0;
    int patternLength = 0;
    for (int i = 0; i < numCounters; i++) {
        total += counters[i];
        patternLength += pattern[i];
    }
    if (total < patternLength || patternLength == 0) {
        // If we don't even have one pixel per unit of bar width, assume this
        // is too small to reliably match, so fail:
        return numeric_limits<int>::max();
    }
    // We're going to fake floating-point math in integers. We just need to use more bits.
    // Scale up patternLength so that intermediate values below like scaledCounter will have
    // more "significant digits".
    int unitBarWidth = (total << 8) / patternLength;
    maxIndividualVariance = (maxIndividualVariance * unitBarWidth) >> 8;
    
    int totalVariance = 0;
    for (int x = 0; x < numCounters; x++) {
        int counter = counters[x] << 8;
        int scaledPattern = pattern[x] * unitBarWidth;
        int variance = counter > scaledPattern ? counter - scaledPattern : scaledPattern - counter;
        if (variance > maxIndividualVariance) {
            return numeric_limits<int>::max();
        }
        totalVariance += variance;
    }
    return totalVariance / total;
}

/**
 * <p>Correct the vertices by searching for top and bottom vertices of wide
 * bars, then locate the intersections between the upper and lower horizontal
 * line and the inner vertices vertical lines.</p>
 *
 * @param matrix the scanned barcode image.
 * @param vertices the vertices vector is extended and the new members are:
 *           vertices[ 8] x, y point on upper border of left wide bar
 *           vertices[ 9] x, y point on lower border of left wide bar
 *           vertices[10] x, y point on upper border of right wide bar
 *           vertices[11] x, y point on lower border of right wide bar
 *           vertices[12] x, y final top left codeword area
 *           vertices[13] x, y final bottom left codeword area
 *           vertices[14] x, y final top right codeword area
 *           vertices[15] x, y final bottom right codeword area
 * @param upsideDown true if rotated by 180 degree.
 */
ErrorHandler Detector::correctVertices(Ref<BitMatrix> matrix,
                                       ArrayRef< Ref<ResultPoint> >& vertices,
                                       bool upsideDown)
{
    ErrorHandler err_handler;
    bool isLowLeft = abs(vertices[4]->getY() - vertices[5]->getY()) < 2.0;
    bool isLowRight = abs(vertices[6]->getY() - vertices[7]->getY()) < 2.0;
    if (isLowLeft || isLowRight) {
        return NotFoundErrorHandler("Cannot find enough PDF417 guard patterns!");
    }
    else
    {
        findWideBarTopBottom(matrix, vertices, 0, 0,  8, 17, upsideDown ? 1 : -1);
        findWideBarTopBottom(matrix, vertices, 1, 0,  8, 17, upsideDown ? -1 : 1);
        findWideBarTopBottom(matrix, vertices, 2, 11, 7, 18, upsideDown ? 1 : -1);
        findWideBarTopBottom(matrix, vertices, 3, 11, 7, 18, upsideDown ? -1 : 1);
        err_handler = findCrossingPoint(vertices, 12, 4, 5, 8, 10, matrix);
        if (err_handler.errCode()) return err_handler;
        err_handler = findCrossingPoint(vertices, 13, 4, 5, 9, 11, matrix);
        if (err_handler.errCode()) return err_handler;
        err_handler = findCrossingPoint(vertices, 14, 6, 7, 8, 10, matrix);
        if (err_handler.errCode()) return err_handler;
        err_handler = findCrossingPoint(vertices, 15, 6, 7, 9, 11, matrix);
        if (err_handler.errCode()) return err_handler;
    }
    
    return err_handler;
}

/**
 * <p>Locate the top or bottom of one of the two wide black bars of a guard pattern.</p>
 *
 * <p>Warning: it only searches along the y axis, so the return points would not be
 * right if the barcode is too curved.</p>
 *
 * @param matrix The bit matrix.
 * @param vertices The 16 vertices located by findVertices(); the result
 *           points are stored into vertices[8], ... , vertices[11].
 * @param offsetVertice The offset of the outer vertice and the inner
 *           vertice (+ 4) to be corrected and (+ 8) where the result is stored.
 * @param startWideBar start of a wide bar.
 * @param lenWideBar length of wide bar.
 * @param lenPattern length of the pattern.
 * @param rowStep +1 if corner should be exceeded towards the bottom, -1 towards the top.
 */
void Detector::findWideBarTopBottom(Ref<BitMatrix> matrix,
                                    ArrayRef< Ref<ResultPoint> > &vertices,
                                    int offsetVertice,
                                    int startWideBar,
                                    int lenWideBar,
                                    int lenPattern,
                                    int rowStep)
{
    Ref<ResultPoint> verticeStart(vertices[offsetVertice]);
    Ref<ResultPoint> verticeEnd(vertices[offsetVertice + 4]);
    
    // Start horizontally at the middle of the bar.
    int endWideBar = startWideBar + lenWideBar;
    float barDiff = verticeEnd->getX() - verticeStart->getX();
    float barStart = verticeStart->getX() + barDiff * static_cast<float>(startWideBar) / static_cast<float>(lenPattern);
    float barEnd = verticeStart->getX() + barDiff * static_cast<float>(endWideBar) / static_cast<float>(lenPattern);
    int x = MathUtils::round((barStart + barEnd) / 2.0f);
    
    // Start vertically between the preliminary vertices.
    int yStart = MathUtils::round(verticeStart->getY());
    int y = yStart;
    
    // Find offset of thin bar to the right as additional safeguard.
    int nextBarX = static_cast<int>(max(barStart, barEnd) + 1);
    for (; nextBarX < matrix->getWidth(); nextBarX++)
        if (!matrix->get(nextBarX - 1, y) && matrix->get(nextBarX, y)) break;
    nextBarX -= x;
    
    bool isEnd = false;
    while (!isEnd) {
        if (matrix->get(x, y))
        {
            // If the thin bar to the right ended, stop as well
            isEnd = !matrix->get(x + nextBarX, y) && !matrix->get(x + nextBarX + 1, y);
            y += rowStep;
            if (y <= 0 || y >= static_cast<int>(matrix->getHeight()) - 1)
            {
                // End of barcode image reached.
                isEnd = true;
            }
        }
        else
        {
            // Look sidewise whether black bar continues? (in the case the image is skewed)
            if (x > 0 && matrix->get(x - 1, y)) {
                x--;
            }
            else if (x < static_cast<int>(matrix->getWidth()) - 1 && matrix->get(x + 1, y))
            {
                x++;
            }
            else
            {
                // End of pattern regarding big bar and big gap reached.
                isEnd = true;
                if (y != yStart) {
                    // Turn back one step, because target has been exceeded.
                    y -= rowStep;
                }
            }
        }
    }
    
    vertices[offsetVertice + 8] = new ResultPoint(static_cast<float>(x), static_cast<float>(y));
}

/**
 * <p>Finds the intersection of two lines.</p>
 *
 * @param vertices The reference of the vertices vector
 * @param idxResult Index of result point inside the vertices vector.
 * @param idxLineA1
 * @param idxLineA2 Indices two points inside the vertices vector that define the first line.
 * @param idxLineB1
 * @param idxLineB2 Indices two points inside the vertices vector that define the second line.
 * @param matrix: bit matrix, here only for testing whether the result is inside the matrix.
 * @return Returns true when the result is valid and lies inside the matrix. Otherwise throws an
 * exception.
 **/
ErrorHandler Detector::findCrossingPoint(ArrayRef< Ref<ResultPoint> >& vertices,
                                         int idxResult,
                                         int idxLineA1, int idxLineA2,
                                         int idxLineB1, int idxLineB2,
                                         Ref<BitMatrix>& matrix)
{
    ErrorHandler err_handler;
    Point p1(vertices[idxLineA1]->getX(), vertices[idxLineA1]->getY());
    Point p2(vertices[idxLineA2]->getX(), vertices[idxLineA2]->getY());
    Point p3(vertices[idxLineB1]->getX(), vertices[idxLineB1]->getY());
    Point p4(vertices[idxLineB2]->getX(), vertices[idxLineB2]->getY());
    
    Point result(intersection(Line(p1, p2), Line(p3, p4)));
    if (result.x == numeric_limits<float>::infinity() ||
        result.y == numeric_limits<float>::infinity()) {
        return NotFoundErrorHandler("PDF:Detector: cannot find the crossing of parallel lines!");
    }
    
    int x = MathUtils::round(result.x);
    int y = MathUtils::round(result.y);
    if (x < 0 || x >= static_cast<int>(matrix->getWidth()) || y < 0 || y >= static_cast<int>(matrix->getHeight())) {
        return NotFoundErrorHandler("PDF:Detector: crossing points out of region!");
    }
    
    vertices[idxResult] = Ref<ResultPoint>(new ResultPoint(result.x, result.y));
    return err_handler;
}

/**
 * Computes the intersection between two lines.
 */
Point Detector::intersection(Line a, Line b) {
    float dxa = a.start.x - a.end.x;
    float dxb = b.start.x - b.end.x;
    float dya = a.start.y - a.end.y;
    float dyb = b.start.y - b.end.y;
    
    float p = a.start.x * a.end.y - a.start.y * a.end.x;
    float q = b.start.x * b.end.y - b.start.y * b.end.x;
    float denom = dxa * dyb - dya * dxb;
    if (abs(denom) < 1e-12)  // Lines don't intersect (replaces "denom == 0")
        return Point(numeric_limits<float>::infinity(),
                     numeric_limits<float>::infinity());
    
    float x = (p * dxb - dxa * q) / denom;
    float y = (p * dyb - dya * q) / denom;
    
    return Point(x, y);
}

/**
 * <p>Estimates module size (pixels in a module) based on the Start and End
 * finder patterns.</p>
 *
 * @param vertices an array of vertices:
 *           vertices[0] x, y top left barcode
 *           vertices[1] x, y bottom left barcode
 *           vertices[2] x, y top right barcode
 *           vertices[3] x, y bottom right barcode
 *           vertices[4] x, y top left codeword area
 *           vertices[5] x, y bottom left codeword area
 *           vertices[6] x, y top right codeword area
 *           vertices[7] x, y bottom right codeword area
 * @return the module size.
 */
float Detector::computeModuleWidth(ArrayRef< Ref<ResultPoint> >& vertices) {
    float pixels1 = ResultPoint::distance(vertices[0], vertices[4]);
    float pixels2 = ResultPoint::distance(vertices[1], vertices[5]);
    float moduleWidth1 = (pixels1 + pixels2) / (17 * 2.0f);
    float pixels3 = ResultPoint::distance(vertices[6], vertices[2]);
    float pixels4 = ResultPoint::distance(vertices[7], vertices[3]);
    float moduleWidth2 = (pixels3 + pixels4) / (18 * 2.0f);
    return (moduleWidth1 + moduleWidth2) / 2.0f;
}

/**
 * Computes the dimension (number of modules in a row) of the PDF417 Code
 * based on vertices of the codeword area and estimated module size.
 *
 * @param topLeft     of codeword area
 * @param topRight    of codeword area
 * @param bottomLeft  of codeword area
 * @param bottomRight of codeword are
 * @param moduleWidth estimated module size
 * @return the number of modules in a row.
 */
int Detector::computeDimension(Ref<ResultPoint> const& topLeft,
                               Ref<ResultPoint> const& topRight,
                               Ref<ResultPoint> const& bottomLeft,
                               Ref<ResultPoint> const& bottomRight,
                               float moduleWidth)
{
    int topRowDimension = MathUtils::round(ResultPoint::distance(topLeft, topRight) / moduleWidth);
    int bottomRowDimension =
    MathUtils::round(ResultPoint::distance(bottomLeft, bottomRight) / moduleWidth);
    return ((((topRowDimension + bottomRowDimension) >> 1) + 8) / 17) * 17;
}

/**
 * Computes the y dimension (number of modules in a column) of the PDF417 Code
 * based on vertices of the codeword area and estimated module size.
 *
 * @param topLeft     of codeword area
 * @param topRight    of codeword area
 * @param bottomLeft  of codeword area
 * @param bottomRight of codeword are
 * @param moduleWidth estimated module size
 * @return the number of modules in a row.
 */
int Detector::computeYDimension(Ref<ResultPoint> const& topLeft,
                                Ref<ResultPoint> const& topRight,
                                Ref<ResultPoint> const& bottomLeft,
                                Ref<ResultPoint> const& bottomRight,
                                float moduleWidth)
{
    int leftColumnDimension =
    MathUtils::round(ResultPoint::distance(topLeft, bottomLeft) / moduleWidth);
    int rightColumnDimension =
    MathUtils::round(ResultPoint::distance(topRight, bottomRight) / moduleWidth);
    return (leftColumnDimension + rightColumnDimension) >> 1;
}

/**
 * Deskew and over-sample image.
 *
 * @param vertices vertices from findVertices()
 * @param dimension x dimension
 * @param yDimension y dimension
 * @return an over-sampled BitMatrix.
 */
Ref<BitMatrix> Detector::sampleLines(ArrayRef< Ref<ResultPoint> > const& vertices,
                                     int dimensionY,
                                     int dimension,
                                     ErrorHandler & err_handler) {
    const int sampleDimensionX = dimension * 8;
    const int sampleDimensionY = dimensionY * 4;
    Ref<PerspectiveTransform> transform(
                                        PerspectiveTransform::quadrilateralToQuadrilateral(
                                                                                           0.0f, 0.0f,
                                                                                           static_cast<float>(sampleDimensionX), 0.0f,
                                                                                           0.0f, static_cast<float>(sampleDimensionY),
                                                                                           static_cast<float>(sampleDimensionX), static_cast<float>(sampleDimensionY),
                                                                                           vertices[12]->getX(), vertices[12]->getY(),
                                                                                           vertices[14]->getX(), vertices[14]->getY(),
                                                                                           vertices[13]->getX(), vertices[13]->getY(),
                                                                                           vertices[15]->getX(), vertices[15]->getY()));
    
    Ref<BitMatrix> bitstmp =  image_->getBlackMatrix(err_handler);
    
    Ref<BitMatrix> linesMatrix = GridSampler::getInstance().sampleGrid(bitstmp, sampleDimensionX, sampleDimensionY, transform, err_handler);
    if (err_handler.errCode()) return Ref<BitMatrix>();
    
    
    return linesMatrix;
}
