// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Detector.cpp
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

#include <map>
#include "../../result_point.hpp"
#include "../../common/grid_sampler.hpp"
#include "../../common/byte_matrix.hpp"
#include "detector.hpp"
#include "../../common/detector/math_utils.hpp"
#include "../../not_found_exception.hpp"
#include <sstream>
#include <cstdlib>

#include<iostream>

using std::abs;
using zxing::Ref;
using zxing::BitMatrix;
using zxing::ByteMatrix;
using zxing::ResultPoint;
using zxing::DetectorResult;
using zxing::PerspectiveTransform;
using zxing::NotFoundException;
using zxing::datamatrix::Detector;
using zxing::datamatrix::ResultPointsAndTransitions;
using zxing::common::detector::MathUtils;
using zxing::ErrorHandler;

namespace {
typedef std::map<Ref<ResultPoint>, int> PointMap;
void increment(PointMap& table, Ref<ResultPoint> const& key) {
    int& value = table[key];
    value += 1;
}
}  // namespace

ResultPointsAndTransitions::ResultPointsAndTransitions() {
    Ref<ResultPoint> ref(new ResultPoint(0, 0));
    from_ = ref;
    to_ = ref;
    transitions_ = 0;
}

ResultPointsAndTransitions::ResultPointsAndTransitions(Ref<ResultPoint> from, Ref<ResultPoint> to,
                                                       int transitions)
: to_(to), from_(from), transitions_(transitions) {
}

Ref<ResultPoint> ResultPointsAndTransitions::getFrom() {
    return from_;
}

Ref<ResultPoint> ResultPointsAndTransitions::getTo() {
    return to_;
}

int ResultPointsAndTransitions::getTransitions() {
    return transitions_;
}

Detector::Detector(Ref<BitMatrix> image)
: image_(image) {
}

Ref<BitMatrix> Detector::getImage() {
    return image_;
}

Ref<ResultPoint> Detector::shiftPoint(Ref<ResultPoint> point, Ref<ResultPoint> to, int div)
{
    float x = (to->getX() - point->getX()) / (div + 1);
    float y = (to->getY() - point->getY()) / (div + 1);
    Ref<ResultPoint> ret_point(new ResultPoint(point->getX() + x, point->getY() + y));
    return ret_point;
}

Ref<ResultPoint> Detector::moveAway(Ref<ResultPoint> point, float fromX, float fromY)
{
    float x = point->getX();
    float y = point->getY();
    
    if (x < fromX)
    {
        x -= 1;
    }
    else
    {
        x += 1;
    }
    
    if (y < fromY)
    {
        y -= 1;
    }
    else
    {
        y += 1;
    }
    
    Ref<ResultPoint> ret_point(new ResultPoint(x, y));
    return ret_point;
}

std::vector<Ref<ResultPoint>> Detector::shiftToModuleCenter(std::vector<Ref<ResultPoint>> points)
{
    // A..D
    // |  :
    // B--C
    Ref<ResultPoint> pointA = points[0];
    Ref<ResultPoint> pointB = points[1];
    Ref<ResultPoint> pointC = points[2];
    Ref<ResultPoint> pointD = points[3];
    
    // calculate pseudo dimensions
    int dimH = transitionsBetweenV2(pointA, pointD) + 1;
    int dimV = transitionsBetweenV2(pointC, pointD) + 1;
    
    // shift points for safe dimension detection
    Ref<ResultPoint> pointAs = shiftPoint(pointA, pointB, dimV * 4);
    Ref<ResultPoint> pointCs = shiftPoint(pointC, pointB, dimH * 4);
    
    //  calculate more precise dimensions
    dimH = transitionsBetweenV2(pointAs, pointD) + 1;
    dimV = transitionsBetweenV2(pointCs, pointD) + 1;
    if ((dimH & 0x01) == 1)
    {
        dimH += 1;
    }
    if ((dimV & 0x01) == 1)
    {
        dimV += 1;
    }
    
    // WhiteRectangleDetector returns points inside of the rectangle.
    // I want points on the edges.
    float centerX = (pointA->getX() + pointB->getX() + pointC->getX() + pointD->getX()) / 4;
    float centerY = (pointA->getY() + pointB->getY() + pointC->getY() + pointD->getY()) / 4;
    pointA = moveAway(pointA, centerX, centerY);
    pointB = moveAway(pointB, centerX, centerY);
    pointC = moveAway(pointC, centerX, centerY);
    pointD = moveAway(pointD, centerX, centerY);
    
    Ref<ResultPoint> pointBs;
    Ref<ResultPoint> pointDs;
    
    // shift points to the center of each modules
    pointAs = shiftPoint(pointA, pointB, dimV * 4);
    pointAs = shiftPoint(pointAs, pointD, dimH * 4);
    pointBs = shiftPoint(pointB, pointA, dimV * 4);
    pointBs = shiftPoint(pointBs, pointC, dimH * 4);
    pointCs = shiftPoint(pointC, pointD, dimV * 4);
    pointCs = shiftPoint(pointCs, pointB, dimH * 4);
    pointDs = shiftPoint(pointD, pointC, dimV * 4);
    pointDs = shiftPoint(pointDs, pointA, dimH * 4);
    
    std::vector<Ref<ResultPoint>> ret(4);
    ret[0].reset(pointAs);
    ret[1].reset(pointBs);
    ret[2].reset(pointCs);
    ret[3].reset(pointDs);
    
    return ret;
}

Ref<DetectorResult> Detector::detectV1(ErrorHandler & err_handler) {
    Ref<WhiteRectangleDetector> rectangleDetector_(new WhiteRectangleDetector(image_, err_handler));
    if (err_handler.ErrCode())   return Ref<DetectorResult>();
    std::vector<Ref<ResultPoint> > ResultPoints = rectangleDetector_->detectNew(err_handler);
    if (err_handler.ErrCode())   return Ref<DetectorResult>();
    Ref<ResultPoint> pointA = ResultPoints[0];
    Ref<ResultPoint> pointB = ResultPoints[1];
    Ref<ResultPoint> pointC = ResultPoints[2];
    Ref<ResultPoint> pointD = ResultPoints[3];
    
    // Point A and D are across the diagonal from one another,
    // as are B and C. Figure out which are the solid black lines
    // by counting transitions
    std::vector<Ref<ResultPointsAndTransitions> > transitions(4);
    transitions[0].reset(transitionsBetween(pointA, pointB));
    transitions[1].reset(transitionsBetween(pointA, pointC));
    transitions[2].reset(transitionsBetween(pointB, pointD));
    transitions[3].reset(transitionsBetween(pointC, pointD));
    insertionSort(transitions);
    
    // Sort by number of transitions. First two will be the two solid sides; last two
    // will be the two alternating black/white sides
    Ref<ResultPointsAndTransitions> lSideOne(transitions[0]);
    Ref<ResultPointsAndTransitions> lSideTwo(transitions[1]);
    
    // Figure out which point is their intersection by tallying up the number of times we see the
    // endpoints in the four endpoints. One will show up twice.
    // typedef std::map<Ref<ResultPoint>, int> PointMap;
    PointMap pointCount;
    increment(pointCount, lSideOne->getFrom());
    increment(pointCount, lSideOne->getTo());
    increment(pointCount, lSideTwo->getFrom());
    increment(pointCount, lSideTwo->getTo());
    
    // Figure out which point is their intersection by tallying up the number of times we see the
    // endpoints in the four endpoints. One will show up twice.
    Ref<ResultPoint> maybeTopLeft;
    Ref<ResultPoint> bottomLeft;
    Ref<ResultPoint> maybeBottomRight;
    for (PointMap::const_iterator entry = pointCount.begin(), end = pointCount.end(); entry != end; ++entry) {
        Ref<ResultPoint> const& point = entry->first;
        int value = entry->second;
        if (value == 2)
        {
            bottomLeft = point;  // this is definitely the bottom left, then -- end of two L sides
        }
        else
        {
            // Otherwise it's either top left or bottom right -- just assign the two arbitrarily now
            if (maybeTopLeft == 0)
            {
                maybeTopLeft = point;
            }
            else
            {
                maybeBottomRight = point;
            }
        }
    }
    
    if (maybeTopLeft == 0 || bottomLeft == 0 || maybeBottomRight == 0)
    {
        err_handler = NotFoundErrorHandler("NotFound datamatrix");
        return Ref<DetectorResult>();
    }
    
    // Bottom left is correct but top left and bottom right might be switched
    std::vector<Ref<ResultPoint> > corners(3);
    corners[0].reset(maybeTopLeft);
    corners[1].reset(bottomLeft);
    corners[2].reset(maybeBottomRight);
    
    // Use the dot product trick to sort them out
    ResultPoint::orderBestPatterns(corners);
    
    // Now we know which is which:
    Ref<ResultPoint> bottomRight(corners[0]);
    bottomLeft = corners[1];
    Ref<ResultPoint> topLeft(corners[2]);
    
    // Which point didn't we find in relation to the "L" sides? that's the top right corner
    Ref<ResultPoint> topRight;
    if (!(pointA->equals(bottomRight) || pointA->equals(bottomLeft) || pointA->equals(topLeft)))
    {
        topRight = pointA;
    }
    else if (!(pointB->equals(bottomRight) || pointB->equals(bottomLeft)
                 || pointB->equals(topLeft)))
    {
        topRight = pointB;
    }
    else if (!(pointC->equals(bottomRight) || pointC->equals(bottomLeft)
                 || pointC->equals(topLeft)))
    {
        topRight = pointC;
    }
    else
    {
        topRight = pointD;
    }
    
    // Next determine the dimension by tracing along the top or right side and counting black/white
    // transitions. Since we start inside a black module, we should see a number of transitions
    // equal to 1 less than the code dimension. Well, actually 2 less, because we are going to
    // end on a black module:
    
    // The top right point is actually the corner of a module, which is one of the two black modules
    // adjacent to the white module at the top right. Tracing to that corner from either the top left
    // or bottom right should work here.
    
    int dimensionTop = transitionsBetween(topLeft, topRight)->getTransitions();
    int dimensionRight = transitionsBetween(bottomRight, topRight)->getTransitions();
    
    if ((dimensionTop & 0x01) == 1)
    {
        // it can't be odd, so, round... up?
        dimensionTop++;
    }
    dimensionTop += 2;
    
    if ((dimensionRight & 0x01) == 1)
    {
        // it can't be odd, so, round... up?
        dimensionRight++;
    }
    dimensionRight += 2;
    
    Ref<BitMatrix> bits;
    Ref<PerspectiveTransform> transform;
    Ref<ResultPoint> correctedTopRight;
    
    
    // Rectanguar symbols are 6x16, 6x28, 10x24, 10x32, 14x32, or 14x44. If one dimension is more
    // than twice the other, it's certainly rectangular, but to cut a bit more slack we accept it as
    // rectangular if the bigger side is at least 7/4 times the other:
    if (4 * dimensionTop >= 7 * dimensionRight || 4 * dimensionRight >= 7 * dimensionTop)
    {
        // The matrix is rectangular
        correctedTopRight = correctTopRightRectangular(bottomLeft, bottomRight, topLeft, topRight,
                                                       dimensionTop, dimensionRight);
        if (correctedTopRight == NULL)
        {
            correctedTopRight = topRight;
        }
        
        dimensionTop = transitionsBetween(topLeft, correctedTopRight)->getTransitions();
        dimensionRight = transitionsBetween(bottomRight, correctedTopRight)->getTransitions();
        
        if ((dimensionTop & 0x01) == 1)
        {
            // it can't be odd, so, round... up?
            dimensionTop++;
        }
        
        if ((dimensionRight & 0x01) == 1)
        {
            // it can't be odd, so, round... up?
            dimensionRight++;
        }
        
        transform = createTransform(topLeft, correctedTopRight, bottomLeft, bottomRight, dimensionTop,
                                    dimensionRight);
        bits = sampleGrid(image_, dimensionTop, dimensionRight, transform, err_handler);
        if (err_handler.ErrCode())   return Ref<DetectorResult>();
    }
    else
    {
        // The matrix is square
        int dimension = min(dimensionRight, dimensionTop);
        
        // correct top right point to match the white module
        correctedTopRight = correctTopRight(bottomLeft, bottomRight, topLeft, topRight, dimension);
        if (correctedTopRight == NULL)
        {
            correctedTopRight = topRight;
        }
        
        // Redetermine the dimension using the corrected top right point
        int dimensionCorrected = (std::max)(transitionsBetween(topLeft, correctedTopRight)->getTransitions(),
                                            transitionsBetween(bottomRight, correctedTopRight)->getTransitions());
        dimensionCorrected++;
        if ((dimensionCorrected & 0x01) == 1)
        {
            dimensionCorrected++;
        }
        
        transform = createTransform(topLeft, correctedTopRight, bottomLeft, bottomRight,
                                    dimensionCorrected, dimensionCorrected);
        bits = sampleGrid(image_, dimensionCorrected, dimensionCorrected, transform, err_handler);
        if (err_handler.ErrCode()) return Ref<DetectorResult>();
    }
    
    ArrayRef< Ref<ResultPoint> > points(new Array< Ref<ResultPoint> >(4));
    points[0].reset(topLeft);
    points[1].reset(bottomLeft);
    points[2].reset(correctedTopRight);
    points[3].reset(bottomRight);
    Ref<DetectorResult> detectorResult(new DetectorResult(bits, points));
    return detectorResult;
}

/**
 * Calculates the position of the white top right module using the output of the rectangle detector
 * for a rectangular matrix
 */
Ref<ResultPoint> Detector::correctTopRightRectangular(Ref<ResultPoint> bottomLeft,
                                                      Ref<ResultPoint> bottomRight, Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                                      int dimensionTop, int dimensionRight) {
    
    float corr = distance(bottomLeft, bottomRight) / static_cast<float>(dimensionTop);
    int norm = distance(topLeft, topRight);
    float cos = (topRight->getX() - topLeft->getX()) / norm;
    float sin = (topRight->getY() - topLeft->getY()) / norm;
    
    Ref<ResultPoint> c1(
                        new ResultPoint(topRight->getX() + corr * cos, topRight->getY() + corr * sin));
    
    corr = distance(bottomLeft, topLeft) / static_cast<float>(dimensionRight);
    norm = distance(bottomRight, topRight);
    cos = (topRight->getX() - bottomRight->getX()) / norm;
    sin = (topRight->getY() - bottomRight->getY()) / norm;
    
    Ref<ResultPoint> c2(
                        new ResultPoint(topRight->getX() + corr * cos, topRight->getY() + corr * sin));
    
    if (!isValid(c1))
    {
        if (isValid(c2))
        {
            return c2;
        }
        return Ref<ResultPoint>(NULL);
    }
    if (!isValid(c2))
    {
        return c1;
    }
    
    int l1 = abs(dimensionTop - transitionsBetween(topLeft, c1)->getTransitions())
    + abs(dimensionRight - transitionsBetween(bottomRight, c1)->getTransitions());
    int l2 = abs(dimensionTop - transitionsBetween(topLeft, c2)->getTransitions())
    + abs(dimensionRight - transitionsBetween(bottomRight, c2)->getTransitions());
    
    return l1 <= l2 ? c1 : c2;
}

/**
 * Calculates the position of the white top right module using the output of the rectangle detector
 * for a square matrix
 */
Ref<ResultPoint> Detector::correctTopRight(Ref<ResultPoint> bottomLeft,
                                           Ref<ResultPoint> bottomRight, Ref<ResultPoint> topLeft, Ref<ResultPoint> topRight,
                                           int dimension) {
    
    float corr = distance(bottomLeft, bottomRight) / static_cast<float>(dimension);
    int norm = distance(topLeft, topRight);
    float cos = (topRight->getX() - topLeft->getX()) / norm;
    float sin = (topRight->getY() - topLeft->getY()) / norm;
    
    Ref<ResultPoint> c1(
                        new ResultPoint(topRight->getX() + corr * cos, topRight->getY() + corr * sin));
    
    corr = distance(bottomLeft, topLeft) / static_cast<float>(dimension);
    norm = distance(bottomRight, topRight);
    cos = (topRight->getX() - bottomRight->getX()) / norm;
    sin = (topRight->getY() - bottomRight->getY()) / norm;
    
    Ref<ResultPoint> c2(
                        new ResultPoint(topRight->getX() + corr * cos, topRight->getY() + corr * sin));
    
    if (!isValid(c1))
    {
        if (isValid(c2))
        {
            return c2;
        }
        return Ref<ResultPoint>(NULL);
    }
    if (!isValid(c2))
    {
        return c1;
    }
    
    int l1 = abs(
                 transitionsBetween(topLeft, c1)->getTransitions()
                 - transitionsBetween(bottomRight, c1)->getTransitions());
    int l2 = abs(
                 transitionsBetween(topLeft, c2)->getTransitions()
                 - transitionsBetween(bottomRight, c2)->getTransitions());
    
    return l1 <= l2 ? c1 : c2;
}

Ref<ResultPoint> Detector::correctTopRightV2(std::vector<Ref<ResultPoint>> & points)
{
    // A..D
    // |  :
    // B--C
    Ref<ResultPoint> pointA = points[0];
    Ref<ResultPoint> pointB = points[1];
    Ref<ResultPoint> pointC = points[2];
    Ref<ResultPoint> pointD = points[3];
    
    // shift points for safe transition detection.
    int trTop = transitionsBetweenV2(pointA, pointD);
    int trRight = transitionsBetweenV2(pointB, pointD);
    Ref<ResultPoint> pointAs = shiftPoint(pointA, pointB, (trRight + 1) * 4);
    Ref<ResultPoint> pointCs = shiftPoint(pointC, pointB, (trTop + 1) * 4);
    
    trTop = transitionsBetweenV2(pointAs, pointD);
    trRight = transitionsBetweenV2(pointCs, pointD);
    
    Ref<ResultPoint> candidate1 (new ResultPoint(
                                                 pointD->getX() + (pointC->getX() - pointB->getX()) / (trTop + 1),
                                                 pointD->getY() + (pointC->getY() - pointB->getY()) / (trTop + 1)));
    Ref<ResultPoint> candidate2 (new ResultPoint(
                                                 pointD->getX() + (pointA->getX() - pointB->getX()) / (trRight + 1),
                                                 pointD->getY() + (pointA->getY() - pointB->getY()) / (trRight + 1)));
    
    if (!isValid(candidate1))
    {
        if (isValid(candidate2))
        {
            return candidate2;
        }
        return Ref<ResultPoint>(NULL);
    }
    if (!isValid(candidate2))
    {
        return candidate1;
    }
    
    int sumc1 = transitionsBetweenV2(pointAs, candidate1) + transitionsBetweenV2(pointCs, candidate1);
    int sumc2 = transitionsBetweenV2(pointAs, candidate2) + transitionsBetweenV2(pointCs, candidate2);
    
    if (sumc1 > sumc2)
    {
        return candidate1;
    }
    else
    {
        return candidate2;
    }
}

bool Detector::isValid(Ref<ResultPoint> p) {
    return p->getX() >= 0 && p->getX() < image_->getWidth() && p->getY() > 0
    && p->getY() < image_->getHeight();
}

int Detector::distance(Ref<ResultPoint> a, Ref<ResultPoint> b) {
    return MathUtils::round(ResultPoint::distance(a, b));
}

Ref<ResultPointsAndTransitions> Detector::transitionsBetween(Ref<ResultPoint> from,
                                                             Ref<ResultPoint> to) {
    // See QR Code Detector, sizeOfBlackWhiteBlackRun()
    int fromX = static_cast<int>(from->getX());
    int fromY = static_cast<int>(from->getY());
    int toX = static_cast<int>(to->getX());
    int toY = static_cast<int>(to->getY());
    bool steep = abs(toY - fromY) > abs(toX - fromX);
    if (steep)
    {
        int temp = fromX;
        fromX = fromY;
        fromY = temp;
        temp = toX;
        toX = toY;
        toY = temp;
    }
    
    int dx = abs(toX - fromX);
    int dy = abs(toY - fromY);
    int error = -dx >> 1;
    int ystep = fromY < toY ? 1 : -1;
    int xstep = fromX < toX ? 1 : -1;
    int transitions = 0;
    bool inBlack = image_->get(steep ? fromY : fromX, steep ? fromX : fromY);
    for (int x = fromX, y = fromY; x != toX; x += xstep) {
        bool isBlack = image_->get(steep ? y : x, steep ? x : y);
        if (isBlack != inBlack)
        {
            transitions++;
            inBlack = isBlack;
        }
        error += dy;
        if (error > 0)
        {
            if (y == toY)
            {
                break;
            }
            y += ystep;
            error -= dx;
        }
    }
    Ref<ResultPointsAndTransitions> result(new ResultPointsAndTransitions(from, to, transitions));
    return result;
}

int Detector::transitionsBetweenV2(Ref<ResultPoint> from, Ref<ResultPoint> to) {
    int fromX = static_cast<int>(from->getX());
    int fromY = static_cast<int>(from->getY());
    int toX = static_cast<int>(to->getX());
    int toY = min(image_->getHeight() - 1, static_cast<int>(to->getY()));
    
    bool steep = abs(toY - fromY) > abs(toX - fromX);
    if (steep)
    {
        int temp = fromX;
        fromX = fromY;
        fromY = temp;
        temp = toX;
        toX = toY;
        toY = temp;
    }
    
    int dx = abs(toX - fromX);
    int dy = abs(toY - fromY);
    int error = -dx / 2;
    int ystep = fromY < toY ? 1 : -1;
    int xstep = fromX < toX ? 1 : -1;
    int transitions = 0;
    bool inBlack = image_->get(steep ? fromY : fromX, steep ? fromX : fromY);
    for (int x = fromX, y = fromY; x != toX; x += xstep) {
        bool isBlack = image_->get(steep ? y : x, steep ? x : y);
        if (isBlack != inBlack)
        {
            transitions++;
            inBlack = isBlack;
        }
        error += dy;
        if (error > 0) {
            if (y == toY)
            {
                break;
            }
            y += ystep;
            error -= dx;
        }
    }
    return transitions;
}

Ref<PerspectiveTransform> Detector::createTransform(Ref<ResultPoint> topLeft,
                                                    Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight,
                                                    int dimensionX, int dimensionY) {
    
    Ref<PerspectiveTransform> transform(
                                        PerspectiveTransform::quadrilateralToQuadrilateral(
                                                                                           0.5f,
                                                                                           0.5f,
                                                                                           dimensionX - 0.5f,
                                                                                           0.5f,
                                                                                           dimensionX - 0.5f,
                                                                                           dimensionY - 0.5f,
                                                                                           0.5f,
                                                                                           dimensionY - 0.5f,
                                                                                           topLeft->getX(),
                                                                                           topLeft->getY(),
                                                                                           topRight->getX(),
                                                                                           topRight->getY(),
                                                                                           bottomRight->getX(),
                                                                                           bottomRight->getY(),
                                                                                           bottomLeft->getX(),
                                                                                           bottomLeft->getY()));
    
    return transform;
}

Ref<PerspectiveTransform> Detector::createTransformV3(Ref<ResultPoint> topLeft,
                                                      Ref<ResultPoint> topRight, Ref<ResultPoint> bottomLeft, Ref<ResultPoint> bottomRight,
                                                      int dimensionX, int dimensionY) {
    Ref<PerspectiveTransform> transform(
                                        PerspectiveTransform::quadrilateralToQuadrilateral(
                                                                                           0.0f,
                                                                                           0.0f,
                                                                                           dimensionX - 0.0f,
                                                                                           0.0f,
                                                                                           dimensionX - 0.0f,
                                                                                           dimensionY - 0.0f,
                                                                                           0.0f,
                                                                                           dimensionY - 0.0f,
                                                                                           topLeft->getX(),
                                                                                           topLeft->getY(),
                                                                                           topRight->getX(),
                                                                                           topRight->getY(),
                                                                                           bottomRight->getX(),
                                                                                           bottomRight->getY(),
                                                                                           bottomLeft->getX(),
                                                                                           bottomLeft->getY()));
    return transform;
}

Ref<BitMatrix> Detector::sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY,
                                    Ref<PerspectiveTransform> transform, ErrorHandler & err_handler) {
    GridSampler &sampler = GridSampler::getInstance();
    
    Ref<BitMatrix> bits =  sampler.sampleGrid(image, dimensionX, dimensionY, transform, err_handler);
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    return bits;
}

void Detector::insertionSort(std::vector<Ref<ResultPointsAndTransitions> > &vector) {
    int max = vector.size();
    bool swapped = true;
    Ref<ResultPointsAndTransitions> value;
    Ref<ResultPointsAndTransitions> valueB;
    do {
        swapped = false;
        for (int i = 1; i < max; i++) {
            value = vector[i - 1];
            if (compare(value, (valueB = vector[i])) > 0){
                swapped = true;
                vector[i - 1].reset(valueB);
                vector[i].reset(value);
            }
        }
    } while (swapped);
}

int Detector::compare(Ref<ResultPointsAndTransitions> a, Ref<ResultPointsAndTransitions> b) {
    return a->getTransitions() - b->getTransitions();
}


///
template <typename T>
struct PointT
{
    using value_t = T;
    T x = 0, y = 0;
    
    constexpr PointT() = default;
    constexpr PointT(T x, T y) : x(x), y(y) {}
    
    template <typename U>
    constexpr explicit PointT(const PointT<U>& p) : x(static_cast<T>(p.x)), y(static_cast<T>(p.y))
    {}
    
    template <typename U>
    PointT& operator+=(const PointT<U>& b)
    {
        x += b.x;
        y += b.y;
        return *this;
    }
};

template <typename T>
bool operator==(const PointT<T>& a, const PointT<T>& b)
{
    return a.x == b.x && a.y == b.y;
}

template <typename T>
bool operator!=(const PointT<T>& a, const PointT<T>& b)
{
    return !(a == b);
}

template <typename T>
auto operator-(const PointT<T>& a) -> PointT<T>
{
    return {-a.x, -a.y};
}

template <typename T, typename U>
auto operator+(const PointT<T>& a, const PointT<U>& b) -> PointT<decltype(a.x + b.x)>
{
    return {a.x + b.x, a.y + b.y};
}

template <typename T, typename U>
auto operator-(const PointT<T>& a, const PointT<U>& b) -> PointT<decltype(a.x - b.x)>
{
    return {a.x - b.x, a.y - b.y};
}

template <typename T, typename U>
auto operator*(const PointT<T>& a, const PointT<U>& b) -> PointT<decltype(a.x * b.x)>
{
    return {a.x * b.x, a.y * b.y};
}

template <typename T, typename U>
PointT<T> operator*(U s, const PointT<T>& a)
{
    return {s * a.x, s * a.y};
}

template <typename T, typename U>
PointT<T> operator/(const PointT<T>& a, U d)
{
    return {a.x / d, a.y / d};
}

template <typename T, typename U>
auto dot(const PointT<T>& a, const PointT<U>& b) -> decltype (a.x * b.x)
{
    return a.x * b.x + a.y * b.y;
}

template <typename T>
auto cross(PointT<T> a, PointT<T> b) -> decltype(a.x * b.x)
{
    return a.x * b.y - b.x * a.y;
}


/// L2 norm
template <typename T>
auto length(PointT<T> p) -> decltype(std::sqrt(dot(p, p)))
{
    return std::sqrt(dot(p, p));
}

/// L-inf norm
template <typename T>
T maxAbsComponent(PointT<T> p)
{
    return (std::max)(std::abs(p.x), std::abs(p.y));
}

template <typename T>
auto distance(PointT<T> a, PointT<T> b) -> decltype(length(a - b))
{
    return length(a - b);
}

using PointF = PointT<float>;

/// Calculate a floating point pixel coordinate representing the 'center' of the pixel.
/// This is sort of the inverse operation of the PointI(PointF) conversion constructor.
/// See also the documentation of the GridSampler API.

inline PointF centered(PointF p)
{
    return {std::floor(p.x) + 0.5f, std::floor(p.y) + 0.5f};
}

template <typename T>
PointF normalized(PointT<T> d)
{
    return PointF(d) / length(PointF(d));
}

template <typename T>
PointT<T> bresenhamDirection(PointT<T> d)
{
    return d / maxAbsComponent(d);
}

template <typename T>
PointT<T> mainDirection(PointT<T> d)
{
    return std::abs(d.x) > std::abs(d.y) ? PointT<T>(d.x, 0) : PointT<T>(0, d.y);
}
/// PointF


#include <numeric>

class RegressionLine
{
protected:
    std::vector<PointF> _points;
    PointF _directionInward;
    PointF::value_t a = NAN, b = NAN, c = NAN;
    
    friend PointF intersect(const RegressionLine& l1, const RegressionLine& l2);
    
    bool evaluate(const std::vector<PointF>& ps)
    {
        auto mean = std::accumulate(ps.begin(), ps.end(), PointF()) / ps.size();
        PointF::value_t sumXX = 0, sumYY = 0, sumXY = 0;
        for (auto& p : ps) {
            auto d = p - mean;
            sumXX += d.x * d.x;
            sumYY += d.y * d.y;
            sumXY += d.x * d.y;
        }
        if (sumYY >= sumXX)
        {
            auto l = std::sqrt(sumYY * sumYY + sumXY * sumXY);
            a = +sumYY / l;
            b = -sumXY / l;
        }
        else
        {
            auto l = std::sqrt(sumXX * sumXX + sumXY * sumXY);
            a = +sumXY / l;
            b = -sumXX / l;
        }
        if (dot(_directionInward, normal()) < 0)
        {
            a = -a;
            b = -b;
        }
        c = dot(normal(), mean);  // (a*mean.x + b*mean.y);
        return dot(_directionInward, normal()) > 0.5f;  // angle between original and new direction is at most 60 degree
    }
    
public:
    RegressionLine() { _points.reserve(16); }  // arbitrary but plausible start size (tiny performance improvement)
    
    const std::vector<PointF>& points() const { return _points; }
    int length() const { return _points.size() >= 2 ? static_cast<int>(distance(_points.front(), _points.back())) : 0; }
    bool isValid() const { return !std::isnan(a); }
    PointF normal() const { return isValid() ? PointF(a, b) : _directionInward; }
    float signedDistance(PointF p) const { return dot(normal(), p) - c; }
    PointF project(PointF p) const { return p - signedDistance(p) * normal(); }
    
    void reset()
    {
        _points.clear();
        _directionInward = {};
        a = b = c = NAN;
    }
    
    void add(PointF p) {
        // assert(_directionInward != PointF());
        if (_directionInward == PointF()) return;
        _points.push_back(p);
        if (_points.size() == 1)
            c = dot(normal(), p);
    }
    
    void pop_back() { _points.pop_back(); }
    
    void setDirectionInward(PointF d) { _directionInward = normalized(d); }
    
    bool evaluate(double maxSignedDist = -1, bool updatePoints = false)
    {
        bool ret = evaluate(_points);
        if (maxSignedDist > 0)
        {
            auto points = _points;
            while (true) {
                auto old_points_size = points.size();
                points.erase(
                             std::remove_if(points.begin(), points.end(),
                                            [this, maxSignedDist](PointF p) { return this->signedDistance(p) > maxSignedDist; }),
                             points.end());
                if (old_points_size == points.size())
                    break;
                
                ret = evaluate(points);
            }
            
            if (updatePoints)
                _points = std::move(points);
        }
        return ret;
    }
};

inline PointF intersect(const RegressionLine& l1, const RegressionLine& l2)
{
    auto d = l1.a * l2.b - l1.b * l2.a;
    auto x = (l1.c * l2.b - l1.b * l2.c) / d;
    auto y = (l1.a * l2.c - l1.c * l2.a) / d;
    return {x, y};
}


class DMRegressionLine : public RegressionLine
{
    template <typename Container, typename Filter>
    static double average(const Container& c, Filter f)
    {
        double sum = 0;
        int num = 0;
        for (const auto& v : c)
            if (f(v)) {
                sum += v;
                ++num;
            }
        if (num == 0) return 0;
        
        return sum / num;
    }
    
public:
    void reverse() { std::reverse(_points.begin(), _points.end()); }
    
    double modules(PointF beg, PointF end)
    {
        // re-evaluate and filter out all points too far away. required for the gapSizes calculation.
        evaluate(1.0, true);
        
        std::vector<double> gapSizes;
        gapSizes.reserve(_points.size());
        
        // calculate the distance between the points projected onto the regression line
        for (size_t i = 1; i < _points.size(); ++i)
            gapSizes.push_back(distance(project(_points[i]), project(_points[i - 1])));
        
        // calculate the (average) distance of two adjacent pixels
        auto unitPixelDist = average(gapSizes, [](double dist){ return 0.75 < dist && dist < 1.5; });
        
        // calculate the width of 2 modules (first black pixel to first black pixel)
        double sum = distance(beg, project(_points.front())) - unitPixelDist;
        auto i = gapSizes.begin();
        for (auto dist : gapSizes) {
            sum += dist;
            if (dist > 1.9 * unitPixelDist){
                *i++ = sum;
                sum = 0.0;
            }
        }
        *i++ = sum + distance(end, project(_points.back()));
        if(i < gapSizes.end())
            gapSizes.erase(i, gapSizes.end());
        
        auto lineLength = distance(beg, end) - unitPixelDist;
        auto meanGapSize = lineLength / gapSizes.size();
        
        meanGapSize = average(gapSizes, [&](double dist){ return std::abs(dist - meanGapSize) < meanGapSize/2; });
        
        return lineLength / meanGapSize;
    }
};

enum class Direction {LEFT = -1, RIGHT = 1};

inline Direction opposite(Direction dir) noexcept
{
    return dir == Direction::LEFT ? Direction::RIGHT : Direction::LEFT;
}

template<typename POINT>
class BitMatrixCursor
{
public:
    Ref<BitMatrix> image_;
    
    POINT p;  // current position
    POINT d;  // current direction
    
    BitMatrixCursor(Ref<BitMatrix> image, POINT p, POINT d) : image_(image), p(p) { setDirection(d); }
    
    class Value
    {
        enum { INVALID = -1, WHITE = 0, BLACK = 1 };
        int v = INVALID;
    public:
        Value() = default;
        Value(bool isBlack) : v(isBlack) {}
        bool isValid() const noexcept { return v != INVALID; }
        bool isWhite() const noexcept { return v == WHITE; }
        bool isBlack() const noexcept { return v == BLACK; }
        
        operator bool() const noexcept { return isValid(); }
        
        bool operator==(Value o) const { return v == o.v; }
        bool operator!=(Value o) const { return v != o.v; }
    };
    
    template <typename T>
    Value testAt(PointT<T> p_) const
    {
        bool is_in = (0 <= p_.x && p_.x < image_->getWidth() && 0 <= p_.y && p_.y < image_->getHeight());
        return is_in ? Value{image_->get(p_.x, p_.y)} : Value{};
    }
    
    bool blackAt(POINT pos) const noexcept { return testAt(pos).isBlack(); }
    bool whiteAt(POINT pos) const noexcept { return testAt(pos).isWhite(); }
    
    bool isIn(POINT p_) const noexcept { return (0 <= p_.x && p_.x < image_->getWidth() && 0 <= p_.y && p_.y < image_->getHeight()); }
    bool isIn() const noexcept { return isIn(p); }
    bool isBlack() const noexcept { return blackAt(p); }
    bool isWhite() const noexcept { return whiteAt(p); }
    
    POINT front() const noexcept { return d; }
    POINT back() const noexcept { return {-d.x, -d.y}; }
    POINT left() const noexcept { return {d.y, -d.x}; }
    POINT right() const noexcept { return {-d.y, d.x}; }
    POINT direction(Direction dir) const noexcept { return static_cast<int>(dir) * right(); }
    
    void turnBack() noexcept { d = back(); }
    void turnLeft() noexcept { d = left(); }
    void turnRight() noexcept { d = right(); }
    void turn(Direction dir) noexcept { d = direction(dir); }
    
    Value edgeAt(POINT d_) const noexcept
    {
        Value v = testAt(p);
        return testAt(p + d_) != v ? v : Value();
    }
    
    Value edgeAtFront() const noexcept { return edgeAt(front()); }
    Value edgeAtBack() const noexcept { return edgeAt(back()); }
    Value edgeAtLeft() const noexcept { return edgeAt(left()); }
    Value edgeAtRight() const noexcept { return edgeAt(right()); }
    Value edgeAt(Direction dir) const noexcept { return edgeAt(direction(dir)); }
    
    void setDirection(PointF dir) { d = bresenhamDirection(dir); }
    
    bool step(typename POINT::value_t s = 1)
    {
        p += s * d;
        return isIn(p);
    }
    
    BitMatrixCursor<POINT> movedBy(POINT d_) const
    {
        auto res = *this;
        res.p += d_;
        return res;
    }
    
    /**
     * @brief stepToEdge advances cursor to one step behind the next (or n-th) edge.
     * @param nth number of edges to pass
     * @param range max number of steps to take
     * @param backup whether or not to backup one step so we land in front of the edge
     * @return number of steps taken or 0 if moved outside of range/image
     */
    int stepToEdge(int nth = 1, int range = 0, bool backup = false)
    {
        int steps = 0;
        auto lv = testAt(p);
        
        while (nth && (!range || steps < range) && lv.isValid()) {
            ++steps;
            auto v = testAt(p + steps * d);
            if (lv != v)
            {
                lv = v;
                --nth;
            }
        }
        if (backup)
            --steps;
        p += steps * d;
        return steps * (nth == 0);
    }
    
    bool stepAlongEdge(Direction dir, bool skipCorner = false)
    {
        if (!edgeAt(dir))
            turn(dir);
        else if (edgeAtFront())
        {
            turn(opposite(dir));
            if (edgeAtFront())
            {
                turn(opposite(dir));
                if (edgeAtFront())
                    return false;
            }
        }
        
        bool ret = step();
        
        if (ret && skipCorner && !edgeAt(dir))
        {
            turn(dir);
            ret = step();
        }
        
        return ret;
    }
    
    int countEdges(int range = 0)
    {
        int res = 0;
        
        while (int steps = stepToEdge(1, range)) {
            range -= steps;
            ++res;
        }
        
        return res;
    }
    
    template<typename ARRAY>
    ARRAY readPattern(int range = 0)
    {
        ARRAY res;
        for (auto& i : res)
            i = stepToEdge(1, range);
        return res;
    }
    
    template<typename ARRAY>
    ARRAY readPatternFromBlack(int maxWhitePrefix, int range = 0)
    {
        if (maxWhitePrefix && isWhite() && !stepToEdge(1, maxWhitePrefix))
            return {};
        return readPattern<ARRAY>(range);
    }
};

using BitMatrixCursorF = BitMatrixCursor<PointF>;

class EdgeTracer : public BitMatrixCursorF
{
    enum class StepResult { FOUND, OPEN_END, CLOSED_END };
    
    StepResult traceStep(PointF dEdge, int maxStepSize, bool goodDirection)
    {
        dEdge = mainDirection(dEdge);
        for (int breadth = 1; breadth <= (goodDirection ? 1 : (maxStepSize == 1 ? 2 : 3)); ++breadth)
            for (int step = 1; step <= maxStepSize; ++step)
                for (int i = 0; i <= 2*(step/4+1) * breadth; ++i) {
                    auto pEdge = p + step * d + (i&1 ? (i+1)/2 : -i/2) * dEdge;
                    // log(pEdge);
                    
                    if (!blackAt(pEdge + dEdge))
                        continue;
                    
                    // found black pixel -> go 'outward' until we hit the b/w border
                    for (int j = 0; j < (std::max)(maxStepSize, 3) && isIn(pEdge); ++j) {
                        if (whiteAt(pEdge))
                        {
                            // if we are not making any progress, we still have another endless loop bug
                            // assert(p != centered(pEdge));
                            if (p == centered(pEdge))
                                return StepResult::CLOSED_END;
                            p = centered(pEdge);
                            
                            if (history && maxStepSize == 1)
                            {
                                if (history->get(p.x, p.y) == state)
                                    return StepResult::CLOSED_END;
                                history->set(p.x, p.y, state);
                            }
                            
                            return StepResult::FOUND;
                        }
                        pEdge = pEdge - dEdge;
                        if (blackAt(pEdge - d))
                            pEdge = pEdge - d;
                        // log(pEdge);
                    }
                    // no valid b/w border found within reasonable range
                    return StepResult::CLOSED_END;
                }
        return StepResult::OPEN_END;
    }
    
public:
    ByteMatrix* history = nullptr;
    int state = 0;
    
    using BitMatrixCursorF::BitMatrixCursor;
    
    bool updateDirectionFromOrigin(PointF origin)
    {
        auto old_d = d;
        setDirection(p - origin);
        // if the new direction is pointing "backward", i.e. angle(new, old) > 90 deg -> break
        if (dot(d, old_d) < 0)
            return false;
        // make sure d stays in the same quadrant to prevent an infinite loop
        if (std::abs(d.x) == std::abs(d.y))
            d = mainDirection(old_d) + 0.99f * (d - mainDirection(old_d));
        else if (mainDirection(d) != mainDirection(old_d))
            d = mainDirection(old_d) + 0.99f * mainDirection(d);
        return true;
    }
    
    bool traceLine(PointF dEdge, RegressionLine& line)
    {
        line.setDirectionInward(dEdge);
        do {
            line.add(p);
            if (line.points().size() % 50 == 10) {
                if (!line.evaluate())
                    return false;
                if (!updateDirectionFromOrigin(p - line.project(p) + line.points().front()))
                    return false;
            }
            auto stepResult = traceStep(dEdge, 1, line.isValid());
            if (stepResult != StepResult::FOUND)
                return stepResult == StepResult::OPEN_END && line.points().size() > 1;
        } while (true);
    }
    
    bool traceGaps(PointF dEdge, RegressionLine& line, int maxStepSize, const RegressionLine& finishLine = {})
    {
        line.setDirectionInward(dEdge);
        int gaps = 0;
        do {
            // detect an endless loop(lack of progress). if encountered, please report.
            // assert(line.points().empty() || p != line.points().back());
            if (!line.points().empty() && p == line.points().back())
                return false;
            
            // if we drifted too far outside of the code, break
            if (line.isValid() && line.signedDistance(p) < -5 && (!line.evaluate() || line.signedDistance(p) < -5))
                return false;
            
            // if we are drifting towards the inside of the code, pull the current position back out onto the line
            if (line.isValid() && line.signedDistance(p) > 3)
            {
                // The current direction d and the line we are tracing are supposed to be roughly parallel.
                // In case the 'go outward' step in traceStep lead us astray, we might end up with a line
                // that is almost perpendicular to d. Then the back-projection below can result in an
                // endless loop. Break if the angle between d and line is greater than 45 deg.
                if (std::abs(dot(normalized(d), line.normal())) > 0.7)  // thresh is approx. sin(45 deg)
                    return false;
                
                auto np = line.project(p);
                // make sure we are making progress even when back-projecting:
                // consider a 90deg corner, rotated 45deg. we step away perpendicular from the line and get
                // back projected where we left off the line.
                // The 'while' instead of 'if' was introduced to fix the issue with #245. It turns out that
                // np can actually be behind the projection of the last line point and we need 2 steps in d
                // to prevent a dead lock. see #245.png
                while (distance(np, line.project(line.points().back())) < 1)
                    np = np + d;
                p = centered(np);
            }
            else
            {
                auto stepLengthInMainDir = line.points().empty() ? 0.0 : dot(mainDirection(d), (p - line.points().back()));
                line.add(p);
                
                if (stepLengthInMainDir > 1)
                {
                    ++gaps;
                    if (gaps >= 2 || line.points().size() > 5)
                    {
                        if (!line.evaluate(1.5))
                            return false;
                        if (!updateDirectionFromOrigin(p - line.project(p) + line.points().front()))
                            return false;
                        // check if the first half of the top-line trace is complete.
                        // the minimum code size is 10x10 -> every code has at least 4 gaps
                        // TODO(sofiawu): maybe switch to termination condition based on bottom line length to get a better
                        // finishLine for the right line trace
                        if (!finishLine.isValid() && gaps == 4)
                        {
                            // undo the last insert, it will be inserted again after the restart
                            line.pop_back();
                            --gaps;
                            return true;
                        }
                    }
                }
                else if (gaps == 0 && line.points().size() >= static_cast<size_t>(2 * maxStepSize))
                    return false;  // no point in following a line that has no gaps
            }
            
            if (finishLine.isValid())
                maxStepSize = (std::min)(maxStepSize, static_cast<int>(finishLine.signedDistance(p)));
            
            auto stepResult = traceStep(dEdge, maxStepSize, line.isValid());
            
            if (stepResult != StepResult::FOUND)
                // we are successful iff we found an open end across a valid finishLine
                return stepResult == StepResult::OPEN_END && finishLine.isValid() &&
                static_cast<int>(finishLine.signedDistance(p)) <= maxStepSize + 1;
        } while (true);
    }
    
    bool traceCorner(PointF dir, PointF& corner)
    {
        step();
        // log(p);
        corner = p;
        std::swap(d, dir);
        traceStep(-1 * dir, 2, false);
        
        return isIn(corner) && isIn(p);
    }
};

static inline PointF movedTowardsBy (PointF& a, PointF b1, PointF b2, float d) {
    return a + d * normalized(normalized(b1 - a) + normalized(b2 - a));
};

static int Scan(EdgeTracer& startTracer, std::vector<DMRegressionLine>& lines, zxing::ArrayRef< Ref<ResultPoint> >& final_points, int& dimT, int& dimR)
{
    while (startTracer.step()) {
        // continue until we cross from black into white
        if (!startTracer.edgeAtBack().isWhite())
            continue;
        
        PointF tl, bl, br, tr;
        
        DMRegressionLine& lineL = lines[0];
        DMRegressionLine& lineB = lines[1];
        DMRegressionLine& lineR = lines[2];
        DMRegressionLine& lineT = lines[3];
        
        for (size_t i = 0; i < lines.size(); i++)
            lines[i].reset();
        
        EdgeTracer t = startTracer;
        
        // follow left leg upwards
        t.turnRight();
        t.state = 1;
        if (!t.traceLine(t.right(), lineL)){
            continue;
        }
        if (!t.traceCorner(t.right(), tl)){
            continue;
        }
        
        lineL.reverse();
        auto tlTracer = t;
        
        // follow left leg downwards
        t = startTracer;
        t.state = 1;
        t.setDirection(tlTracer.right());
        if (!t.traceLine(t.left(), lineL)){
            continue;
        }
        
        if (!lineL.isValid())
            t.updateDirectionFromOrigin(tl);
        auto up = t.back();
        if (!t.traceCorner(t.left(), bl)){
            continue;
        }
        
        // follow bottom leg right
        t.state = 2;
        if (!t.traceLine(t.left(), lineB)){
            continue;
        }
        
        if (!lineB.isValid())
            t.updateDirectionFromOrigin(bl);
        auto right = t.front();
        if (!t.traceCorner(t.left(), br)){
            continue;
        }
        
        auto lenL = distance(tl, bl) - 1;
        auto lenB = distance(bl, br) - 1;
        if (!((lenL >= 8 && lenB >= 10 && lenB >= lenL / 4 && lenB <= lenL * 18))){
            continue;
        }
        
        auto maxStepSize = static_cast<int>(lenB / 5 + 1);  // datamatrix bottom dim is at least 10
        
        // at this point we found a plausible L-shape and are now looking for the b/w pattern at the top and right:
        // follow top row right 'half way' (4 gaps), see traceGaps break condition with 'invalid' line
        tlTracer.setDirection(right);
        if (!(tlTracer.traceGaps(tlTracer.right(), lineT, maxStepSize))){
            continue;
        }
        
        maxStepSize = (std::min)(lineT.length() / 3, static_cast<int>(lenL / 5)) * 2;
        
        // follow up until we reach the top line
        t.setDirection(up);
        t.state = 3;
        if (!(t.traceGaps(t.left(), lineR, maxStepSize, lineT))){
            continue;
        }
        if (!(t.traceCorner(t.left(), tr))){
            continue;
        }
        
        auto lenT = distance(tl, tr) - 1;
        auto lenR = distance(tr, br) - 1;
        
        if (!(std::abs(lenT - lenB) / lenB < 0.5 && std::abs(lenR - lenL) / lenL < 0.5 &&
              lineT.points().size() >= 5 && lineR.points().size() >= 5)){
            continue;
        }
        
        // continue top row right until we cross the right line
        if (!(tlTracer.traceGaps(tlTracer.right(), lineT, maxStepSize, lineR))){
            continue;
        }
        
        for (auto* l : {&lineL, &lineB, &lineT, &lineR})
            l->evaluate(1.0);
        
        // find the bounding box corners of the code with sub-pixel precision by intersecting the 4 border lines
        if (!lineB.isValid() || !lineL.isValid() || !lineT.isValid() || !lineR.isValid())
            continue;
        bl = intersect(lineB, lineL);
        tl = intersect(lineT, lineL);
        tr = intersect(lineT, lineR);
        br = intersect(lineB, lineR);
        
        // int dimT, dimR;
        double fracT, fracR;
        auto splitDouble = [](double d, int* i, double* f) {
            *i = std::isnormal(d) ? static_cast<int>(d + 0.5) : 0;
            *f = std::isnormal(d) ? std::abs(d - *i) : INFINITY;
        };
        if (lineT.points().size() <= 3 ||  lineR.points().size() <= 3)
            continue;
        splitDouble(lineT.modules(tl, tr), &dimT, &fracT);
        splitDouble(lineR.modules(br, tr), &dimR, &fracR);
        
        // if we have an almost square (invalid rectangular) data matrix dimension, we try to parse it by assuming a
        // square. we use the dimension that is closer to an integral value. all valid rectangular symbols differ in
        // their dimension by at least 10 (here 5, see doubling below). Note: this is currently not required for the
        // black-box tests to complete.
        if (std::abs(dimT - dimR) < 5)
            dimT = dimR = fracR < fracT ? dimR : dimT;
        
        // the dimension is 2x the number of black/white transitions
        dimT *= 2;
        dimR *= 2;
        
        if (!(dimT >= 10 && dimT <= 144 && dimR >= 8 && dimR <= 144)){
            continue;
        }
        
        // shrink shape by half a pixel to go from center of white pixel outside of code to the edge between white and black
        tl = movedTowardsBy(tl, tr, bl, 0.5f);
        tr = movedTowardsBy(tr, br, tl, 0.3f);
        br = movedTowardsBy(br, bl, tr, 0.5f);
        bl = movedTowardsBy(bl, tl, br, 0.5f);
        
        Ref<ResultPoint> topLeft(new ResultPoint(static_cast<float>(tl.x), static_cast<float>(tl.y)));
        Ref<ResultPoint> bottomLeft(new ResultPoint(static_cast<float>(bl.x), static_cast<float>(bl.y)));
        Ref<ResultPoint> bottomRight(new ResultPoint(static_cast<float>(br.x), static_cast<float>(br.y)));
        Ref<ResultPoint> topRight(new ResultPoint(static_cast<float>(tr.x), static_cast<float>(tr.y)));
        
        final_points[0].reset(topLeft);
        final_points[1].reset(bottomLeft);
        final_points[2].reset(bottomRight);
        final_points[3].reset(topRight);
        
        return 0;
    }
    
    return -1;
}

#include <array>
Ref<DetectorResult> Detector::detectV3(ErrorHandler &err_handler)
{
    // instantiate RegressionLine objects outside of Scan function to prevent repetitive std::vector allocations
    std::vector<DMRegressionLine> lines(4);
    
    constexpr int minSymbolSize = 8 * 2;  // minimum realistic size in pixel: 8 modules x 2 pixels per module
    
    PointF center = PointF(image_->getWidth() / 2, image_->getHeight() / 2);
    for (auto dir : {PointF(-1, 0), PointF(1, 0), PointF(0, -1), PointF(0, 1)}) {
        auto startPos = centered(center - center * dir + minSymbolSize / 2 * dir);
        
        EdgeTracer tracer(image_, startPos, dir);
        
        for (int i = 1;; ++i) {
            tracer.p = startPos + i / 2 * minSymbolSize * (i & 1 ? -1 : 1) * tracer.right();
            
            if (!tracer.isIn())
            {
                break;
            }
            
            ArrayRef< Ref<ResultPoint> > final_points(new Array< Ref<ResultPoint> >(4));
            int dimT, dimR;
            
            int ret = Scan(tracer, lines, final_points, dimT, dimR);
            if (ret == 0)
            {
                Ref<BitMatrix> bits;
                Ref<PerspectiveTransform> transform;
                Ref<ResultPoint> topLeft = final_points[0];
                Ref<ResultPoint> bottomLeft = final_points[1];
                Ref<ResultPoint> bottomRight = final_points[2];
                Ref<ResultPoint> topRight = final_points[3];
                
                transform = createTransformV3(topLeft, topRight, bottomLeft, bottomRight, dimT,
                                              dimR);
                bits = sampleGrid(image_, dimT, dimR, transform, err_handler);
                if (err_handler.ErrCode())   return Ref<DetectorResult>();
                Ref<DetectorResult> detectorResult(new DetectorResult(bits, final_points));
                return detectorResult;
            }
            
            break;
        }
    }
    
    return Ref<DetectorResult>();
}

Ref<DetectorResult> Detector::detectV2(ErrorHandler &err_handler)
{
    Ref<WhiteRectangleDetector> rectangleDetector_(new WhiteRectangleDetector(image_, err_handler));
    if (err_handler.ErrCode())   return Ref<DetectorResult>();
    std::vector<Ref<ResultPoint> > ResultPoints = rectangleDetector_->detectNew(err_handler);
    if (err_handler.ErrCode())   return Ref<DetectorResult>();
    
    // 0  2
    // 1  3
    
    // detect solid 1
    Ref<ResultPoint> pointA = ResultPoints[0];
    Ref<ResultPoint> pointB = ResultPoints[1];
    Ref<ResultPoint> pointC = ResultPoints[3];
    Ref<ResultPoint> pointD = ResultPoints[2];
    
    int trAB = transitionsBetweenV2(pointA, pointB);
    int trBC = transitionsBetweenV2(pointB, pointC);
    int trCD = transitionsBetweenV2(pointC, pointD);
    int trDA = transitionsBetweenV2(pointD, pointA);
    
    
    int min = trAB;
    std::vector<Ref<ResultPoint>> points(4);
    points[0].reset(pointD);
    points[1].reset(pointA);
    points[2].reset(pointB);
    points[3].reset(pointC);
    
    if (min > trBC)
    {
        min = trBC;
        points[0] = pointA;
        points[1] = pointB;
        points[2] = pointC;
        points[3] = pointD;
    }
    if (min > trCD)
    {
        min = trCD;
        points[0] = pointB;
        points[1] = pointC;
        points[2] = pointD;
        points[3] = pointA;
    }
    if (min > trDA)
    {
        points[0] = pointC;
        points[1] = pointD;
        points[2] = pointA;
        points[3] = pointB;
    }
    
    // detect solid 2
    // A..D
    // :  :
    // B--C
    pointA = points[0];
    pointB = points[1];
    pointC = points[2];
    pointD = points[3];
    
    // Transition detection on the edge is not stable.
    // To safely detect, shift the points to the module center.
    int tr = transitionsBetweenV2(pointA, pointD);
    Ref<ResultPoint> pointBs = shiftPoint(pointB, pointC, (tr + 1) * 4);
    Ref<ResultPoint> pointCs = shiftPoint(pointC, pointB, (tr + 1) * 4);
    int trBA = transitionsBetweenV2(pointBs, pointA);
    trCD = transitionsBetweenV2(pointCs, pointD);
    
    // 0..3
    // |  :
    // 1--2
    if (trBA < trCD)
    {
        // solid sides: A-B-C
        points[0] = pointA;
        points[1] = pointB;
        points[2] = pointC;
        points[3] = pointD;
    }
    else
    {
        // solid sides: B-C-D
        points[0] = pointB;
        points[1] = pointC;
        points[2] = pointD;
        points[3] = pointA;
    }
    
    points[3] = correctTopRightV2(points);
    if (points[3] == NULL)
    {
        err_handler = NotFoundErrorHandler("NotFound Instance");
        return Ref<DetectorResult>();
    }
    
    points = shiftToModuleCenter(points);
    
    Ref<ResultPoint> topLeft = points[0];
    Ref<ResultPoint> bottomLeft = points[1];
    Ref<ResultPoint> bottomRight = points[2];
    Ref<ResultPoint> topRight = points[3];
    
    int dimensionTop = transitionsBetweenV2(topLeft, topRight) + 1;
    int dimensionRight = transitionsBetweenV2(bottomRight, topRight) + 1;
    if ((dimensionTop & 0x01) == 1)
    {
        dimensionTop += 1;
    }
    if ((dimensionRight & 0x01) == 1)
    {
        dimensionRight += 1;
    }
    
    Ref<ResultPoint> correctedTopRight;
    if (4 * dimensionTop < 6 * dimensionRight && 4 * dimensionRight < 6 * dimensionTop)
    {
        // The matrix is square
        dimensionTop = dimensionRight = (std::max)(dimensionTop, dimensionRight);
    }
    
    Ref<BitMatrix> bits;
    Ref<PerspectiveTransform> transform;
    transform = createTransform(topLeft, topRight, bottomLeft, bottomRight, dimensionTop,
                                dimensionRight);
    bits = sampleGrid(image_, dimensionTop, dimensionRight, transform, err_handler);
    if (err_handler.ErrCode())   return Ref<DetectorResult>();
    
    ArrayRef< Ref<ResultPoint> > final_points(new Array< Ref<ResultPoint> >(4));
    final_points[0].reset(topLeft);
    final_points[1].reset(bottomLeft);
    final_points[2].reset(bottomRight);
    final_points[3].reset(topRight);
    Ref<DetectorResult> detectorResult(new DetectorResult(bits, final_points));
    
    return detectorResult;
}

Ref<DetectorResult> Detector::detect(bool use_v2, bool use_v3, ErrorHandler &err_handler)
{
    Ref<DetectorResult> result;
    
    if (use_v3)
    {
        try
        {
            result = detectV3(err_handler);
        }
        catch (...) {
            err_handler = ErrorHandler("detectV3 error!");
        }
    }
    if (err_handler.ErrCode() || result == NULL)
    {
        if (use_v2) {
            err_handler.Reset();
            result = detectV2(err_handler);
        }
    }
    
    return result;
}
