// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

//
//  dmtxdecode.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#include "dmtxdecode.hpp"
#include "dmtxmessage.hpp"
#include "dmtximage.hpp"
#include "dmtxmatrix3.hpp"
#include "dmtxsymbol.hpp"
#include "dmtxplacemod.hpp"
#include "dmtxbytelist.hpp"
#include "dmtxreedsol.hpp"
#include "dmtxvector2.hpp"


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>


namespace dmtx {

int DmtxDecode::dmtxDecodeCreate(unsigned char *pxl, int width, int height)
{
    if (this->image.dmtxImageCreate(pxl, width, height) < 0)
        return -1;
    
    this->fnc1 = DmtxUndefined;
    this->edgeMin = DmtxUndefined;
    this->edgeMax = DmtxUndefined;
    this->scanGap = 1;
    this->squareDevn = cos(50 * (M_PI/180));
    this->sizeIdxExpected = DmtxSymbolShapeAuto;
    this->edgeThresh = 10;
    
    this->xMin = 0;
    this->xMax = width - 1;
    this->yMin = 0;
    this->yMax = height - 1;
    
    this->cache = (unsigned char *)calloc(width * height, sizeof(unsigned char));
    if (this->cache == NULL)
    {
        return -1;
    }
    
    if (initScanGrid() < 0) return -1;
    
    return 0;
}


DmtxDecode::~DmtxDecode()
{
    if (this->cache != NULL)
        free(this->cache);
    this->cache = NULL;
}

int DmtxDecode::dmtxDecodeGetProp(int prop)
{
    switch (prop) {
        case DmtxPropEdgeMin:
            return this->edgeMin;
        case DmtxPropEdgeMax:
            return this->edgeMax;
        case DmtxPropScanGap:
            return this->scanGap;
        case DmtxPropFnc1:
            return this->fnc1;
        case DmtxPropSquareDevn:
            return static_cast<int>(acos(this->squareDevn) * 180.0/M_PI);
        case DmtxPropSymbolSize:
            return this->sizeIdxExpected;
        case DmtxPropEdgeThresh:
            return this->edgeThresh;
        case DmtxPropXmin:
            return this->xMin;
        case DmtxPropXmax:
            return this->xMax;
        case DmtxPropYmin:
            return this->yMin;
        case DmtxPropYmax:
            return this->yMax;
        case DmtxPropWidth:
            return this->image.dmtxImageGetProp(DmtxPropWidth);
        case DmtxPropHeight:
            return this->image.dmtxImageGetProp(DmtxPropHeight);
        default:
            break;
    }
    
    return DmtxUndefined;
}

unsigned char * DmtxDecode::dmtxDecodeGetCache(int x, int y)
{
    int width = dmtxDecodeGetProp(DmtxPropWidth);
    int height = dmtxDecodeGetProp(DmtxPropHeight);
    
    if (x < 0 || x >= width || y < 0 || y >= height)
        return NULL;
    
    return &(this->cache[y * width + x]);
}

unsigned int DmtxDecode::dmtxDecodeGetPixelValue(int x, int y, int *value)
{
    unsigned int err = this->image.dmtxImageGetPixelValue(x, y, value);
    
    return err;
}

int DmtxDecode::dmtxRegionFindNext()
{
    int locStatus;
    DmtxPixelLoc loc;
    
    /* Continue until we find a region or run out of chances */
    int count = 0;
    for (;;) {
        locStatus = popGridLocation(&loc);
        if (locStatus == DmtxRangeEnd)
            break;
        count += 1;
        
        if (count > 500) return -1;
        /* Scan location for presence of valid barcode region */
        int ret = dmtxRegionScanPixel(loc.X, loc.Y);
        if (ret == 0)
            return 0;
    }
    
    return -1;
}

///

int DmtxDecode::popGridLocation( DmtxPixelLoc *locPtr)
{
    int locStatus;
    
    do {
        locStatus = getGridCoordinates(locPtr);
        
        /* Always leave grid pointing at next available location */
        grid.pixelCount++;
        
    } while (locStatus == DmtxRangeBad);
    
    return locStatus;
}

int DmtxDecode::getGridCoordinates(DmtxPixelLoc *locPtr)
{
    int count, half, quarter;
    DmtxPixelLoc loc;
    
    /* Initially pixelCount may fall beyond acceptable limits. Update grid
     * state before testing coordinates */
    
    /* Jump to next cross pattern horizontally if current column is done */
    if (grid.pixelCount >= grid.pixelTotal)
    {
        grid.pixelCount = 0;
        grid.xCenter += grid.jumpSize;
    }
    
    /* Jump to next cross pattern vertically if current row is done */
    if (grid.xCenter > grid.maxExtent)
    {
        grid.xCenter = grid.startPos;
        grid.yCenter += grid.jumpSize;
    }
    
    /* Increment level when vertical step goes too far */
    if (grid.yCenter > grid.maxExtent)
    {
        grid.total *= 4;
        grid.extent /= 2;
        
        //SetDerivedFields(grid);
        
        grid.jumpSize = grid.extent + 1;
        grid.pixelTotal = 2 * grid.extent - 1;
        grid.startPos = grid.extent / 2;
        grid.pixelCount = 0;
        grid.xCenter = grid.yCenter = grid.startPos;
    }
    
    if (grid.extent == 0 || grid.extent < grid.minExtent)
    {
        locPtr->X = locPtr->Y = -1;
        return DmtxRangeEnd;
    }
    
    count = grid.pixelCount;
    
    if (count >= grid.pixelTotal) return DmtxRangeEnd;
    
    if (count == grid.pixelTotal - 1)
    {
        /* center pixel */
        loc.X = grid.xCenter;
        loc.Y = grid.yCenter;
    }
    else
    {
        half = grid.pixelTotal / 2;
        quarter = half / 2;
        
        /* horizontal portion */
        if (count < half)
        {
            loc.X = grid.xCenter + ((count < quarter) ? (count - quarter) : (half - count));
            loc.Y = grid.yCenter;
        }
        /* vertical portion */
        else
        {
            count -= half;
            loc.X = grid.xCenter;
            loc.Y = grid.yCenter + ((count < quarter) ? (count - quarter) : (half - count));
        }
    }
    
    loc.X += grid.xOffset;
    loc.Y += grid.yOffset;
    
    *locPtr = loc;
    
    if (loc.X < grid.xMin || loc.X > grid.xMax ||
       loc.Y < grid.yMin || loc.Y > grid.yMax)
        return DmtxRangeBad;
    
    return DmtxRangeGood;
}

///

int DmtxDecode::dmtxRegionScanPixel(int x, int y)
{
    unsigned char *cache_;
    DmtxPointFlow flowBegin;
    DmtxPixelLoc loc;
    
    loc.X = x;
    loc.Y = y;
    
    cache_ = dmtxDecodeGetCache(loc.X, loc.Y);
    if (cache_ == NULL)
        return -1;
    
    if (static_cast<int>(*cache_ & 0x80) != 0x00)
        return -1;
    
    /* Test for presence of any reasonable edge at this location */
    flowBegin = matrixRegionSeekEdge(loc);
    if (flowBegin.mag < static_cast<int>(this->edgeThresh * 7.65 + 0.5))
        return -1;
    
    memset(&region, 0x00, sizeof(DmtxRegion));
    
    /* Determine barcode orientation */
    if (matrixRegionOrientation(flowBegin) == DmtxFail)
        return -1;
    if (dmtxRegionUpdateXfrms() == DmtxFail)
        return -1;
    
    /* Define top edge */
    if (matrixRegionAlignCalibEdge(DmtxEdgeTop) == DmtxFail)
        return -1;
    if (dmtxRegionUpdateXfrms() == DmtxFail)
        return -1;
    
    /* Define right edge */
    if (matrixRegionAlignCalibEdge(DmtxEdgeRight) == DmtxFail)
        return -1;
    if (dmtxRegionUpdateXfrms() == DmtxFail)
        return -1;
    
    /* Calculate the best fitting symbol size */
    if (matrixRegionFindSize() == DmtxFail)
        return -1;
    
    /* Found a valid matrix region */
    return 0;
}

DmtxPointFlow DmtxDecode::matrixRegionSeekEdge(DmtxPixelLoc loc)
{
    DmtxPointFlow flow, flowPlane;
    DmtxPointFlow flowPos, flowPosBack;
    DmtxPointFlow flowNeg, flowNegBack;
    
    /* Find whether red, green, or blue shows the strongest edge */
    flowPlane = GetPointFlow(loc, dmtxNeighborNone);
    
    if (flowPlane.mag < 10)
        return dmtxBlankEdge;
    
    flow = flowPlane;
    
    flowPos = findStrongestNeighbor(flow, +1);
    flowNeg = findStrongestNeighbor(flow, -1);
    if (flowPos.mag != 0 && flowNeg.mag != 0) {
        flowPosBack = findStrongestNeighbor(flowPos, -1);
        flowNegBack = findStrongestNeighbor(flowNeg, +1);
        if (flowPos.arrive == (flowPosBack.arrive + 4) % 8 &&
           flowNeg.arrive == (flowNegBack.arrive + 4) % 8) {
            flow.arrive = dmtxNeighborNone;
            return flow;
        }
    }
    
    return dmtxBlankEdge;
}

DmtxPointFlow DmtxDecode::GetPointFlow(DmtxPixelLoc loc, int arrive)
{
    static const int coefficient[] = {  0,  1,  2,  1,  0, -1, -2, -1 };
    int err;
    int coefficientIdx;
    int compassMax;
    int mag[4] = { 0 };
    int xAdjust, yAdjust;
    int color, colorPattern[8];
    DmtxPointFlow flow;
    
    for (int patternIdx = 0; patternIdx < 8; patternIdx++) {
        xAdjust = loc.X + dmtxPatternX[patternIdx];
        yAdjust = loc.Y + dmtxPatternY[patternIdx];
        err = dmtxDecodeGetPixelValue(xAdjust, yAdjust, &colorPattern[patternIdx]);
        if (err == DmtxFail)
            return dmtxBlankEdge;
    }
    
    /* Calculate this pixel's flow intensity for each direction (-45, 0, 45, 90) */
    compassMax = 0;
    for (int compass = 0; compass < 4; compass++) {
        
        /* Add portion from each position in the convolution matrix pattern */
        for (int patternIdx = 0; patternIdx < 8; patternIdx++) {
            
            coefficientIdx = (patternIdx - compass + 8) % 8;
            if (coefficient[coefficientIdx] == 0)
                continue;
            
            color = colorPattern[patternIdx];
            
            switch (coefficient[coefficientIdx]) {
                case 2:
                    mag[compass] += color;
                    /* Fall through */
                case 1:
                    mag[compass] += color;
                    break;
                case -2:
                    mag[compass] -= color;
                    /* Fall through */
                case -1:
                    mag[compass] -= color;
                    break;
            }
        }
        
        /* Identify strongest compass flow */
        if (compass != 0 && abs(mag[compass]) > abs(mag[compassMax]))
            compassMax = compass;
    }
    
    /* Convert signed compass direction into unique flow directions (0-7) */
    flow.arrive = arrive;
    flow.depart = (mag[compassMax] > 0) ? compassMax + 4 : compassMax;
    flow.mag = abs(mag[compassMax]);
    flow.loc = loc;
    
    return flow;
}

unsigned int DmtxDecode:: matrixRegionFindSize()
{
    int row, col;
    int sizeIdxBeg, sizeIdxEnd;
    int sizeIdx, bestSizeIdx;
    int symbolRows, symbolCols;
    int jumpCount, errors;
    int color;
    int colorOnAvg, bestColorOnAvg;
    int colorOffAvg, bestColorOffAvg;
    int contrast, bestContrast;
    
    bestSizeIdx = DmtxUndefined;
    bestContrast = 0;
    bestColorOnAvg = bestColorOffAvg = 0;
    
    if (this->sizeIdxExpected == DmtxSymbolShapeAuto)
    {
        sizeIdxBeg = 0;
        sizeIdxEnd = DmtxSymbolSquareCount + DmtxSymbolRectCount;
    }
    else if (this->sizeIdxExpected == DmtxSymbolSquareAuto)
    {
        sizeIdxBeg = 0;
        sizeIdxEnd = DmtxSymbolSquareCount;
    }
    else if (this->sizeIdxExpected == DmtxSymbolRectAuto)
    {
        sizeIdxBeg = DmtxSymbolSquareCount;
        sizeIdxEnd = DmtxSymbolSquareCount + DmtxSymbolRectCount;
    }
    else
    {
        sizeIdxBeg = this->sizeIdxExpected;
        sizeIdxEnd = this->sizeIdxExpected + 1;
    }
    
    /* Test each barcode size to find best contrast in calibration modules */
    for (sizeIdx = sizeIdxBeg; sizeIdx < sizeIdxEnd; sizeIdx++) {
        
        symbolRows = dmtxGetSymbolAttribute(DmtxSymAttribSymbolRows, sizeIdx);
        symbolCols = dmtxGetSymbolAttribute(DmtxSymAttribSymbolCols, sizeIdx);
        colorOnAvg = colorOffAvg = 0;
        
        /* Sum module colors along horizontal calibration bar */
        row = symbolRows - 1;
        for (col = 0; col < symbolCols; col++) {
            color = readModuleColor(row, col, sizeIdx);
            if (color == -1) return DmtxFail;
            
            if ((col & 0x01) != 0x00)
                colorOffAvg += color;
            else
                colorOnAvg += color;
        }
        
        /* Sum module colors along vertical calibration bar */
        col = symbolCols - 1;
        for (row = 0; row < symbolRows; row++) {
            color = readModuleColor(row, col, sizeIdx);
            if (color == -1) return DmtxFail;
            
            if ((row & 0x01) != 0x00)
                colorOffAvg += color;
            else
                colorOnAvg += color;
        }
        
        colorOnAvg = (colorOnAvg * 2)/(symbolRows + symbolCols);
        colorOffAvg = (colorOffAvg * 2)/(symbolRows + symbolCols);
        
        contrast = abs(colorOnAvg - colorOffAvg);
        if (contrast < 20)
            continue;
        
        if (contrast > bestContrast)
        {
            bestContrast = contrast;
            bestSizeIdx = sizeIdx;
            bestColorOnAvg = colorOnAvg;
            bestColorOffAvg = colorOffAvg;
        }
    }
    
    /* If no sizes produced acceptable contrast then call it quits */
    if (bestSizeIdx == DmtxUndefined || bestContrast < 20)
        return DmtxFail;
    
    region.sizeIdx = bestSizeIdx;
    region.onColor = bestColorOnAvg;
    region.offColor = bestColorOffAvg;
    
    region.symbolRows = dmtxGetSymbolAttribute(DmtxSymAttribSymbolRows, region.sizeIdx);
    region.symbolCols = dmtxGetSymbolAttribute(DmtxSymAttribSymbolCols, region.sizeIdx);
    region.mappingRows = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixRows, region.sizeIdx);
    region.mappingCols = dmtxGetSymbolAttribute(DmtxSymAttribMappingMatrixCols, region.sizeIdx);
    
    /* Tally jumps on horizontal calibration bar to verify sizeIdx */
    jumpCount = countJumpTally(0, region.symbolRows - 1, DmtxDirRight);
    errors = abs(1 + jumpCount - region.symbolCols);
    if (jumpCount < 0 || errors > 2)
        return DmtxFail;
    
    /* Tally jumps on vertical calibration bar to verify sizeIdx */
    jumpCount = countJumpTally(region.symbolCols - 1, 0, DmtxDirUp);
    errors = abs(1 + jumpCount - region.symbolRows);
    if (jumpCount < 0 || errors > 2)
        return DmtxFail;
    
    /* Tally jumps on horizontal finder bar to verify sizeIdx */
    errors = countJumpTally(0, 0, DmtxDirRight);
    if (jumpCount < 0 || errors > 2)
        return DmtxFail;
    
    /* Tally jumps on vertical finder bar to verify sizeIdx */
    errors = countJumpTally(0, 0, DmtxDirUp);
    if (errors < 0 || errors > 2)
        return DmtxFail;
    
    /* Tally jumps on surrounding whitespace, else fail */
    errors = countJumpTally(0, -1, DmtxDirRight);
    if (errors < 0 || errors > 2)
        return DmtxFail;
    
    errors = countJumpTally(-1, 0, DmtxDirUp);
    if (errors < 0 || errors > 2)
        return DmtxFail;
    
    errors = countJumpTally(0, region.symbolRows, DmtxDirRight);
    if (errors < 0 || errors > 2)
        return DmtxFail;
    
    errors = countJumpTally(region.symbolCols, 0, DmtxDirUp);
    if (errors < 0 || errors > 2)
        return DmtxFail;
    
    return DmtxPass;
}

int DmtxDecode::countJumpTally(int xStart, int yStart, DmtxDirection dir)
{
    int x, xInc = 0;
    int y, yInc = 0;
    int state = DmtxModuleOn;
    int jumpCount = 0;
    int jumpThreshold;
    int tModule, tPrev;
    int darkOnLight;
    int color;
    
    // assert(xStart == 0 || yStart == 0);
    // assert(dir == DmtxDirRight || dir == DmtxDirUp);
    if (xStart != 0 && yStart != 0) return -1;
    if (dir != DmtxDirRight && dir != DmtxDirUp) return -1;
    
    if (dir == DmtxDirRight)
        xInc = 1;
    else
        yInc = 1;
    
    if (xStart == -1 || xStart == region.symbolCols ||
       yStart == -1 || yStart == region.symbolRows)
        state = DmtxModuleOff;
    
    darkOnLight = static_cast<int>(region.offColor > region.onColor);
    jumpThreshold = abs(static_cast<int>(0.4 * (region.onColor - region.offColor) + 0.5));
    color = readModuleColor(yStart, xStart, region.sizeIdx);
    if (color == -1) return -1;
    
    tModule = (darkOnLight) ? region.offColor - color : color - region.offColor;
    
    for (x = xStart + xInc, y = yStart + yInc;
        (dir == DmtxDirRight && x < region.symbolCols) ||
        (dir == DmtxDirUp && y < region.symbolRows);
        x += xInc, y += yInc) {
        
        tPrev = tModule;
        color = readModuleColor(y, x, region.sizeIdx);
        if (color == -1) return -1;
        tModule = (darkOnLight) ? region.offColor - color : color - region.offColor;
        
        if (state == DmtxModuleOff)
        {
            if (tModule > tPrev + jumpThreshold)
            {
                jumpCount++;
                state = DmtxModuleOn;
            }
        }
        else
        {
            if (tModule < tPrev - jumpThreshold)
            {
                jumpCount++;
                state = DmtxModuleOff;
            }
        }
    }
    
    return jumpCount;
}

DmtxPointFlow DmtxDecode::findStrongestNeighbor(DmtxPointFlow center, int sign)
{
    int strongIdx;
    int attemptDiff;
    unsigned char *cache_;
    DmtxPixelLoc loc;
    DmtxPointFlow flow[8];
    
    int attempt = (sign < 0) ? center.depart : (center.depart + 4) % 8;
    
    int occupied = 0;
    strongIdx = DmtxUndefined;
    for (int i = 0; i < 8; i++) {
        
        loc.X = center.loc.X + dmtxPatternX[i];
        loc.Y = center.loc.Y + dmtxPatternY[i];
        
        cache_ = dmtxDecodeGetCache(loc.X, loc.Y);
        if (cache_ == NULL)
            continue;
        
        if (static_cast<int>(*cache_ & 0x80) != 0x00)
        {
            if (++occupied > 2)
                return dmtxBlankEdge;
            else
                continue;
        }
        
        attemptDiff = abs(attempt - i);
        if (attemptDiff > 4)
            attemptDiff = 8 - attemptDiff;
        if (attemptDiff > 1)
            continue;
        
        flow[i] = GetPointFlow(loc, i);
        
        if (strongIdx == DmtxUndefined || flow[i].mag > flow[strongIdx].mag ||
           (flow[i].mag == flow[strongIdx].mag && ((i & 0x01) != 0))) {
            strongIdx = i;
        }
    }
    
    return (strongIdx == DmtxUndefined) ? dmtxBlankEdge : flow[strongIdx];
}

unsigned int DmtxDecode::matrixRegionOrientation(DmtxPointFlow begin)
{
    int cross;
    int minArea;
    
    int symbolShape;
    int maxDiagonal;
    unsigned int err;
    DmtxBestLine line1x, line2x;
    DmtxBestLine line2n, line2p;
    DmtxFollow fTmp;
    
    if (this->sizeIdxExpected == DmtxSymbolSquareAuto ||
       (this->sizeIdxExpected >= DmtxSymbol10x10 &&
        this->sizeIdxExpected <= DmtxSymbol144x144))
        symbolShape = DmtxSymbolSquareAuto;
    else if (this->sizeIdxExpected == DmtxSymbolRectAuto ||
            (this->sizeIdxExpected >= DmtxSymbol8x18 &&
             this->sizeIdxExpected <= DmtxSymbol16x48))
        symbolShape = DmtxSymbolRectAuto;
    else
        symbolShape = DmtxSymbolShapeAuto;
    
    if (this->edgeMax != DmtxUndefined) {
        if (symbolShape == DmtxSymbolRectAuto)
            maxDiagonal = static_cast<int>(1.23 * this->edgeMax + 0.5); /* sqrt(5/4) + 10% */
        else
            maxDiagonal = static_cast<int>(1.56 * this->edgeMax + 0.5); /* sqrt(2) + 10% */
    }
    else
    {
        maxDiagonal = DmtxUndefined;
    }
    
    /* Follow to end in both directions */
    err = trailBlazeContinuous(begin, maxDiagonal);
    if (err == DmtxFail || region.stepsTotal < 40)
    {
        trailClear(0x40);
        return DmtxFail;
    }
    
    /* Filter out region candidates that are smaller than expected */
    if (this->edgeMin != DmtxUndefined)
    {
        if (symbolShape == DmtxSymbolSquareAuto)
            minArea = (this->edgeMin * this->edgeMin);
        else
            minArea = (2 * this->edgeMin * this->edgeMin);
        
        if ((region.boundMax.X - region.boundMin.X) * (region.boundMax.Y - region.boundMin.Y) < minArea) {
            trailClear(0x40);
            return DmtxFail;
        }
    }
    
    unsigned int passFail;
    line1x = findBestSolidLine(0, 0, +1, DmtxUndefined, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    if (line1x.mag < 5)
    {
        trailClear(0x40);
        return DmtxFail;
    }
    
    err = findTravelLimits(&line1x);
    if (line1x.distSq < 100 || line1x.devn * 10 >= sqrt(static_cast<double>(line1x.distSq)))
    {
        trailClear(0x40);
        return DmtxFail;
    }
    if (line1x.stepPos < line1x.stepNeg) return DmtxFail;
    
    fTmp = followSeek(line1x.stepPos + 5, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    line2p = findBestSolidLine(fTmp.step, line1x.stepNeg, +1, line1x.angle, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    
    fTmp = followSeek(line1x.stepNeg - 5, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    line2n = findBestSolidLine(fTmp.step, line1x.stepPos, -1, line1x.angle, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    if (fmax(line2p.mag, line2n.mag) < 5)
        return DmtxFail;
    
    if (line2p.mag > line2n.mag)
    {
        line2x = line2p;
        err = findTravelLimits(&line2x);
        if (line2x.distSq < 100 || line2x.devn * 10 >= sqrt(static_cast<double>(line2x.distSq)))
            return DmtxFail;
        
        cross = ((line1x.locPos.X - line1x.locNeg.X) * (line2x.locPos.Y - line2x.locNeg.Y)) -
        ((line1x.locPos.Y - line1x.locNeg.Y) * (line2x.locPos.X - line2x.locNeg.X));
        if (cross > 0)
        {
            /* Condition 2 */
            region.polarity = +1;
            region.locR = line2x.locPos;
            region.stepR = line2x.stepPos;
            region.locT = line1x.locNeg;
            region.stepT = line1x.stepNeg;
            region.leftLoc = line1x.locBeg;
            region.leftAngle = line1x.angle;
            region.bottomLoc = line2x.locBeg;
            region.bottomAngle = line2x.angle;
            region.leftLine = line1x;
            region.bottomLine = line2x;
        }
        else
        {
            /* Condition 3 */
            region.polarity = -1;
            region.locR = line1x.locNeg;
            region.stepR = line1x.stepNeg;
            region.locT = line2x.locPos;
            region.stepT = line2x.stepPos;
            region.leftLoc = line2x.locBeg;
            region.leftAngle = line2x.angle;
            region.bottomLoc = line1x.locBeg;
            region.bottomAngle = line1x.angle;
            region.leftLine = line2x;
            region.bottomLine = line1x;
        }
    }
    else
    {
        line2x = line2n;
        err = findTravelLimits(&line2x);
        if (line2x.distSq < 100 || line2x.devn / sqrt(static_cast<double>(line2x.distSq)) >= 0.1)
            return DmtxFail;
        
        cross = ((line1x.locNeg.X - line1x.locPos.X) * (line2x.locNeg.Y - line2x.locPos.Y)) -
        ((line1x.locNeg.Y - line1x.locPos.Y) * (line2x.locNeg.X - line2x.locPos.X));
        if (cross > 0)
        {
            /* Condition 1 */
            region.polarity = -1;
            region.locR = line2x.locNeg;
            region.stepR = line2x.stepNeg;
            region.locT = line1x.locPos;
            region.stepT = line1x.stepPos;
            region.leftLoc = line1x.locBeg;
            region.leftAngle = line1x.angle;
            region.bottomLoc = line2x.locBeg;
            region.bottomAngle = line2x.angle;
            region.leftLine = line1x;
            region.bottomLine = line2x;
        }
        else
        {
            /* Condition 4 */
            region.polarity = +1;
            region.locR = line1x.locPos;
            region.stepR = line1x.stepPos;
            region.locT = line2x.locNeg;
            region.stepT = line2x.stepNeg;
            region.leftLoc = line2x.locBeg;
            region.leftAngle = line2x.angle;
            region.bottomLoc = line1x.locBeg;
            region.bottomAngle = line1x.angle;
            region.leftLine = line2x;
            region.bottomLine = line1x;
        }
    }
    
    region.leftKnown = region.bottomKnown = 1;
    
    return DmtxPass;
}

DmtxFollow DmtxDecode::followSeek(int seek, unsigned int* passFail)
{
    int i;
    int sign;
    DmtxFollow follow;
    
    follow.loc = region.flowBegin.loc;
    follow.step = 0;
    follow.ptr = dmtxDecodeGetCache(follow.loc.X, follow.loc.Y);
    if (follow.ptr == NULL){
        *passFail = DmtxFail;
        return follow;
    }
    follow.neighbor = *follow.ptr;
    
    if (passFail == DmtxFail)
    {
        // *passFail = DmtxFail;
        return follow;
    }
    
    sign = (seek > 0) ? +1 : -1;
    for (i = 0; i != seek; i += sign) {
        follow = followStep(follow, sign, passFail);
        if (passFail == DmtxFail)
        {
            // *passFail = DmtxFail;
            return follow;
        }
        if (follow.ptr == NULL)
        {
            *passFail = DmtxFail;
            return follow;
        }
        if (abs(follow.step) > region.stepsTotal){
            *passFail = DmtxFail;
            return follow;
        }
    }
    
    *passFail = DmtxPass;
    return follow;
}

DmtxFollow DmtxDecode::followStep(DmtxFollow followBeg, int sign, unsigned int *passFail)
{
    int patternIdx;
    int stepMod;
    int factor;
    DmtxFollow follow;
    
    if (abs(sign) != 1)
    {
        *passFail = DmtxFail;
        return follow;
    }
    if (static_cast<int>(followBeg.neighbor & 0x40) == 0x00)
    {
        *passFail = DmtxFail;
        return follow;
    }
    
    factor = region.stepsTotal + 1;
    if (sign > 0)
        stepMod = (factor + (followBeg.step % factor)) % factor;
    else
        stepMod = (factor - (followBeg.step % factor)) % factor;
    
    /* End of positive trail -- magic jump */
    if (sign > 0 && stepMod == region.jumpToNeg)
    {
        follow.loc = region.finalNeg;
    }
    /* End of negative trail -- magic jump */
    else if (sign < 0 && stepMod == region.jumpToPos)
    {
        follow.loc = region.finalPos;
    }
    /* Trail in progress -- normal jump */
    else
    {
        patternIdx = (sign < 0) ? followBeg.neighbor & 0x07 : ((followBeg.neighbor & 0x38) >> 3);
        follow.loc.X = followBeg.loc.X + dmtxPatternX[patternIdx];
        follow.loc.Y = followBeg.loc.Y + dmtxPatternY[patternIdx];
    }
    
    follow.step = followBeg.step + sign;
    follow.ptr = dmtxDecodeGetCache(follow.loc.X, follow.loc.Y);
    if (follow.ptr == NULL)
    {
        *passFail = DmtxFail;
        return follow;
    }
    follow.neighbor = *follow.ptr;
    
    return follow;
}


unsigned int DmtxDecode::trailBlazeContinuous(DmtxPointFlow flowBegin, int maxDiagonal)
{
    int posAssigns, negAssigns, clears;
    int sign;
    int steps;
    unsigned char *cache_, *cacheNext, *cacheBeg;
    DmtxPointFlow flow, flowNext;
    DmtxPixelLoc boundMin, boundMax;
    
    boundMin = boundMax = flowBegin.loc;
    cacheBeg = dmtxDecodeGetCache(flowBegin.loc.X, flowBegin.loc.Y);
    if (cacheBeg == NULL)
        return DmtxFail;
    *cacheBeg = (0x80 | 0x40); /* Mark location as visited and assigned */
    
    region.flowBegin = flowBegin;
    
    posAssigns = negAssigns = 0;
    for (sign = 1; sign >= -1; sign -= 2) {
        
        flow = flowBegin;
        cache_ = cacheBeg;
        
        for (steps = 0;; steps++) {
            
            if (maxDiagonal != DmtxUndefined && (boundMax.X - boundMin.X > maxDiagonal ||
                                                boundMax.Y - boundMin.Y > maxDiagonal))
                break;
            
            /* Find the strongest eligible neighbor */
            flowNext = findStrongestNeighbor(flow, sign);
            if (flowNext.mag < 50)
                break;
            
            /* Get the neighbor's cache location */
            cacheNext = dmtxDecodeGetCache(flowNext.loc.X, flowNext.loc.Y);
            if (cacheNext == NULL)
                break;
            if ((*cacheNext & 0x80)) return DmtxFail;
            
            /* Mark departure from current location. If flowing downstream
             * (sign < 0) then departure vector here is the arrival vector
             * of the next location. Upstream flow uses the opposite rule. */
            *cache_ |= (sign < 0) ? flowNext.arrive : flowNext.arrive << 3;
            
            /* Mark known direction for next location */
            /* If testing downstream (sign < 0) then next upstream is opposite of next arrival */
            /* If testing upstream (sign > 0) then next downstream is opposite of next arrival */
            *cacheNext = (sign < 0) ? (((flowNext.arrive + 4)%8) << 3) : ((flowNext.arrive + 4)%8);
            *cacheNext |= (0x80 | 0x40); /* Mark location as visited and assigned */
            if (sign > 0)
                posAssigns++;
            else
                negAssigns++;
            cache_ = cacheNext;
            flow = flowNext;
            
            if (flow.loc.X > boundMax.X)
                boundMax.X = flow.loc.X;
            else if (flow.loc.X < boundMin.X)
                boundMin.X = flow.loc.X;
            if (flow.loc.Y > boundMax.Y)
                boundMax.Y = flow.loc.Y;
            else if (flow.loc.Y < boundMin.Y)
                boundMin.Y = flow.loc.Y;
        }
        
        if (sign > 0)
        {
            region.finalPos = flow.loc;
            region.jumpToNeg = steps;
        }
        else
        {
            region.finalNeg = flow.loc;
            region.jumpToPos = steps;
        }
    }
    region.stepsTotal = region.jumpToPos + region.jumpToNeg;
    region.boundMin = boundMin;
    region.boundMax = boundMax;
    
    /* Clear "visited" bit from trail */
    clears = trailClear(0x80);
    if (clears < 0) return DmtxFail;
    if (posAssigns + negAssigns != clears - 1) return DmtxFail;
    
    /* XXX clean this up ... redundant test above */
    if (maxDiagonal != DmtxUndefined && (boundMax.X - boundMin.X > maxDiagonal ||
                                        boundMax.Y - boundMin.Y > maxDiagonal))
        return DmtxFail;
    
    return DmtxPass;
}

int DmtxDecode::trailClear(int clearMask)
{
    int clears;
    DmtxFollow follow;
    unsigned int passFail;
    
    if ((clearMask | 0xff) != 0xff) return -1;
    
    /* Clear "visited" bit from trail */
    clears = 0;
    follow = followSeek(0, &passFail);
    if (passFail == DmtxFail) return -1;
    while (abs(follow.step) <= region.stepsTotal) {
        if (static_cast<int>(*follow.ptr & clearMask) == 0x00) return -1;
        *follow.ptr &= (clearMask ^ 0xff);
        follow = followStep(follow, +1, &passFail);
        if (passFail == DmtxFail) return -1;
        clears++;
    }
    
    return clears;
}

unsigned int DmtxDecode::matrixRegionAlignCalibEdge(int edgeLoc)
{
    int streamDir;
    int steps;
    int avoidAngle;
    int symbolShape;
    DmtxVector2 pTmp;
    DmtxPixelLoc loc0, loc1, locOrigin;
    DmtxBresLine line;
    DmtxFollow follow;
    DmtxBestLine bestLine;
    unsigned int passFail;
    
    /* Determine pixel coordinates of origin */
    pTmp.X = 0.0;
    pTmp.Y = 0.0;
    if (dmtxMatrix3VMultiplyBy(&pTmp, region.fit2raw) == DmtxFail)
        return DmtxFail;
    locOrigin.X = static_cast<int>(pTmp.X + 0.5);
    locOrigin.Y = static_cast<int>(pTmp.Y + 0.5);
    
    if (this->sizeIdxExpected == DmtxSymbolSquareAuto ||
       (this->sizeIdxExpected >= DmtxSymbol10x10 &&
        this->sizeIdxExpected <= DmtxSymbol144x144))
        symbolShape = DmtxSymbolSquareAuto;
    else if (this->sizeIdxExpected == DmtxSymbolRectAuto ||
            (this->sizeIdxExpected >= DmtxSymbol8x18 &&
             this->sizeIdxExpected <= DmtxSymbol16x48))
        symbolShape = DmtxSymbolRectAuto;
    else
        symbolShape = DmtxSymbolShapeAuto;
    
    /* Determine end locations of test line */
    if (edgeLoc == DmtxEdgeTop) {
        streamDir = region.polarity * -1;
        avoidAngle = region.leftLine.angle;
        follow = followSeekLoc(region.locT, &passFail);
        if (passFail == DmtxFail) return DmtxFail;
        pTmp.X = 0.8;
        pTmp.Y = (symbolShape == DmtxSymbolRectAuto) ? 0.2 : 0.6;
    }
    else
    {
        if (edgeLoc != DmtxEdgeRight) return DmtxFail;
        streamDir = region.polarity;
        avoidAngle = region.bottomLine.angle;
        follow = followSeekLoc(region.locR, &passFail);
        if (passFail == DmtxFail) return DmtxFail;
        pTmp.X = (symbolShape == DmtxSymbolSquareAuto) ? 0.7 : 0.9;
        pTmp.Y = 0.8;
    }
    
    if (dmtxMatrix3VMultiplyBy(&pTmp, region.fit2raw) == DmtxFail)
        return DmtxFail;
    loc1.X = static_cast<int>(pTmp.X + 0.5);
    loc1.Y = static_cast<int>(pTmp.Y + 0.5);
    
    loc0 = follow.loc;
    line = bresLineInit(loc0, loc1, locOrigin);
    steps = trailBlazeGapped(line, streamDir);
    if (steps < 0) return DmtxFail;
    
    bestLine = findBestSolidLine2(loc0, steps, streamDir, avoidAngle, &passFail);
    if (passFail == DmtxFail)
    {
        return DmtxFail;
    }
    // if (bestLine.mag < 5)
    // {
    // }

    if (edgeLoc == DmtxEdgeTop)
    {
        region.topKnown = 1;
        region.topAngle = bestLine.angle;
        region.topLoc = bestLine.locBeg;
    }
    else
    {
        region.rightKnown = 1;
        region.rightAngle = bestLine.angle;
        region.rightLoc = bestLine.locBeg;
    }
    
    return DmtxPass;
}

DmtxFollow DmtxDecode::followSeekLoc(DmtxPixelLoc loc, unsigned int* passFail)
{
    DmtxFollow follow;
    
    follow.loc = loc;
    follow.step = 0;
    follow.ptr = dmtxDecodeGetCache(follow.loc.X, follow.loc.Y);
    if (follow.ptr == NULL)
    {
        *passFail = DmtxFail;
        return follow;
    }
    
    follow.neighbor = *follow.ptr;
    *passFail = DmtxPass;
    return follow;
}

int DmtxDecode::trailBlazeGapped(DmtxBresLine line, int streamDir)
{
    unsigned char *beforeCache, *afterCache;
    DmtxBoolean onEdge;
    int distSq, distSqMax;
    int travel, outward;
    int xDiff, yDiff;
    int steps;
    int stepDir, dirMap[] = { 0, 1, 2, 7, 8, 3, 6, 5, 4 };
    unsigned int err;
    DmtxPixelLoc beforeStep, afterStep;
    DmtxPointFlow flow, flowNext;
    DmtxPixelLoc loc0;
    int xStep, yStep;
    
    loc0 = line.loc;
    flow = GetPointFlow(loc0, dmtxNeighborNone);
    distSqMax = (line.xDelta * line.xDelta) + (line.yDelta * line.yDelta);
    steps = 0;
    onEdge = DmtxTrue;
    
    beforeStep = loc0;
    beforeCache = dmtxDecodeGetCache(loc0.X, loc0.Y);
    if (beforeCache == NULL)
        return DmtxFail;
    else
        *beforeCache = 0x00; /* probably should just overwrite one direction */
    
    do {
        if (onEdge == DmtxTrue) {
            flowNext = findStrongestNeighbor(flow, streamDir);
            if (flowNext.mag == DmtxUndefined)
                break;
            
            err = bresLineGetStep(line, flowNext.loc, &travel, &outward);
            if (err == DmtxFail) { return DmtxFail; }
            
            if (flowNext.mag < 50 || outward < 0 || (outward == 0 && travel < 0)) {
                onEdge = DmtxFalse;
            }
            else
            {
                bresLineStep(&line, travel, outward);
                flow = flowNext;
            }
        }
        
        if (onEdge == DmtxFalse)
        {
            bresLineStep(&line, 1, 0);
            flow = GetPointFlow(line.loc, dmtxNeighborNone);
            if (flow.mag > 50)
                onEdge = DmtxTrue;
        }
        
        afterStep = line.loc;
        afterCache = dmtxDecodeGetCache(afterStep.X, afterStep.Y);
        if (afterCache == NULL)
            break;
        
        /* Determine step direction using pure magic */
        xStep = afterStep.X - beforeStep.X;
        yStep = afterStep.Y - beforeStep.Y;
        if (abs(xStep) > 1 || abs(yStep) > 1) return -1;
        stepDir = dirMap[3 * yStep + xStep + 4];
        if (stepDir == 8) return -1;
        
        if (streamDir < 0)
        {
            *beforeCache |= (0x40 | stepDir);
            *afterCache = (((stepDir + 4)%8) << 3);
        }
        else
        {
            *beforeCache |= (0x40 | (stepDir << 3));
            *afterCache = ((stepDir + 4)%8);
        }
        
        /* Guaranteed to have taken one step since top of loop */
        xDiff = line.loc.X - loc0.X;
        yDiff = line.loc.Y - loc0.Y;
        distSq = (xDiff * xDiff) + (yDiff * yDiff);
        
        beforeStep = line.loc;
        beforeCache = afterCache;
        steps++;
        
    } while (distSq < distSqMax);
    
    return steps;
}

DmtxBestLine DmtxDecode::findBestSolidLine2(DmtxPixelLoc loc0, int tripSteps, int sign, int houghAvoid, unsigned int* passFail)
{
    int hough[3][DMTX_HOUGH_RES] = { { 0 } };
    int houghMin, houghMax;
    char houghTest[DMTX_HOUGH_RES];
    int i;
    int step;
    int angleBest;
    int hOffset, hOffsetBest;
    int xDiff, yDiff;
    int dH;
    DmtxRay2 rH;
    DmtxBestLine line;
    DmtxPixelLoc rHp;
    DmtxFollow follow;
    
    memset(&line, 0x00, sizeof(DmtxBestLine));
    memset(&rH, 0x00, sizeof(DmtxRay2));
    angleBest = 0;
    hOffset = hOffsetBest = 0;
    
    follow = followSeekLoc(loc0, passFail);
    if (*passFail == DmtxFail) {
        return line;
    }
    rHp = line.locBeg = line.locPos = line.locNeg = follow.loc;
    line.stepBeg = line.stepPos = line.stepNeg = 0;
    
    /* Predetermine which angles to test */
    for (i = 0; i < DMTX_HOUGH_RES; i++) {
        if (houghAvoid == DmtxUndefined) {
            houghTest[i] = 1;
        }
        else
        {
            houghMin = (houghAvoid + DMTX_HOUGH_RES/6) % DMTX_HOUGH_RES;
            houghMax = (houghAvoid - DMTX_HOUGH_RES/6 + DMTX_HOUGH_RES) % DMTX_HOUGH_RES;
            if (houghMin > houghMax)
                houghTest[i] = (i > houghMin || i < houghMax) ? 1 : 0;
            else
                houghTest[i] = (i > houghMin && i < houghMax) ? 1 : 0;
        }
    }
    
    /* Test each angle for steps along path */
    for (step = 0; step < tripSteps; step++)
    {
        
        xDiff = follow.loc.X - rHp.X;
        yDiff = follow.loc.Y - rHp.Y;
        
        /* Increment Hough accumulator */
        for (i = 0; i < DMTX_HOUGH_RES; i++) {
            
            if (static_cast<int>(houghTest[i]) == 0)
                continue;
            
            dH = (rHvX[i] * yDiff) - (rHvY[i] * xDiff);
            if (dH >= -384 && dH <= 384) {
                if (dH > 128)
                    hOffset = 2;
                else if (dH >= -128)
                    hOffset = 1;
                else
                    hOffset = 0;
                
                hough[hOffset][i]++;
                
                /* New angle takes over lead */
                if (hough[hOffset][i] > hough[hOffsetBest][angleBest])
                {
                    angleBest = i;
                    hOffsetBest = hOffset;
                }
            }
        }
        
        follow = followStep2(follow, sign, passFail);
        if (*passFail == DmtxFail) {
            return line;
        }
    }
    
    line.angle = angleBest;
    line.hOffset = hOffsetBest;
    line.mag = hough[hOffsetBest][angleBest];
    
    return line;
}

DmtxFollow DmtxDecode:: followStep2(DmtxFollow followBeg, int sign, unsigned int *passFail)
{
    int patternIdx;
    DmtxFollow follow;
    
    if (abs(sign) != 1){
        *passFail = DmtxFail;
        return follow;
    }
    if (static_cast<int>(followBeg.neighbor & 0x40) == 0x00) {
        *passFail = DmtxFail;
        return follow;
    }
    
    patternIdx = (sign < 0) ? followBeg.neighbor & 0x07 : ((followBeg.neighbor & 0x38) >> 3);
    follow.loc.X = followBeg.loc.X + dmtxPatternX[patternIdx];
    follow.loc.Y = followBeg.loc.Y + dmtxPatternY[patternIdx];
    
    follow.step = followBeg.step + sign;
    follow.ptr = dmtxDecodeGetCache(follow.loc.X, follow.loc.Y);
    if (follow.ptr == NULL){
        *passFail = DmtxFail;
        return follow;
    }
    follow.neighbor = *follow.ptr;
    
    return follow;
}

///

int DmtxDecode::dmtxDecodeMatrixRegion(int fix, DmtxMessage& msg)
{
    DmtxVector2 topLeft, topRight, bottomLeft, bottomRight;
    DmtxPixelLoc pxTopLeft, pxTopRight, pxBottomLeft, pxBottomRight;
    
    if (msg.init(this->region.sizeIdx, DmtxFormatMatrix) < 0)
        return -1;
    
    if (populateArrayFromMatrix(&msg) != DmtxPass) {
        return -1;
    }
    
    msg.fnc1 = this->fnc1;
    
    topLeft.X = bottomLeft.X = topLeft.Y = topRight.Y = -0.1;
    topRight.X = bottomRight.X = bottomLeft.Y = bottomRight.Y = 1.1;
    
    if (dmtxMatrix3VMultiplyBy(&topLeft, this->region.fit2raw) == DmtxFail)
        return NULL;
    if (dmtxMatrix3VMultiplyBy(&topRight, this->region.fit2raw) == DmtxFail)
        return NULL;
    if (dmtxMatrix3VMultiplyBy(&bottomLeft, this->region.fit2raw) == DmtxFail)
        return NULL;
    if (dmtxMatrix3VMultiplyBy(&bottomRight, this->region.fit2raw) == DmtxFail)
        return NULL;
    
    pxTopLeft.X = static_cast<int>(0.5 + topLeft.X);
    pxTopLeft.Y = static_cast<int>(0.5 + topLeft.Y);
    pxBottomLeft.X = static_cast<int>(0.5 + bottomLeft.X);
    pxBottomLeft.Y = static_cast<int>(0.5 + bottomLeft.Y);
    pxTopRight.X = static_cast<int>(0.5 + topRight.X);
    pxTopRight.Y = static_cast<int>(0.5 + topRight.Y);
    pxBottomRight.X = static_cast<int>(0.5 + bottomRight.X);
    pxBottomRight.Y = static_cast<int>(0.5 + bottomRight.Y);
    
    if (cacheFillQuad(pxTopLeft, pxTopRight, pxBottomRight, pxBottomLeft) == DmtxFail)
        return NULL;
    
    msg.points.clear();
    msg.points.push_back(pxTopLeft);
    msg.points.push_back(pxBottomLeft);
    msg.points.push_back(pxBottomRight);
    msg.points.push_back(pxTopRight);
    
    return dmtxDecodePopulatedArray(this->region.sizeIdx, msg, fix);
}

unsigned int DmtxDecode::cacheFillQuad(DmtxPixelLoc p0, DmtxPixelLoc p1, DmtxPixelLoc p2, DmtxPixelLoc p3)
{
    DmtxBresLine lines[4];
    DmtxPixelLoc pEmpty = { 0, 0 };
    unsigned char *cache_;
    int *scanlineMin, *scanlineMax;
    int minY, maxY, sizeY, posY, posX;
    int i, idx;
    
    lines[0] = bresLineInit(p0, p1, pEmpty);
    lines[1] = bresLineInit(p1, p2, pEmpty);
    lines[2] = bresLineInit(p2, p3, pEmpty);
    lines[3] = bresLineInit(p3, p0, pEmpty);
    
    minY = this->yMax;
    maxY = 0;
    
    minY = fmin(minY, p0.Y); maxY = fmax(maxY, p0.Y);
    minY = fmin(minY, p1.Y); maxY = fmax(maxY, p1.Y);
    minY = fmin(minY, p2.Y); maxY = fmax(maxY, p2.Y);
    minY = fmin(minY, p3.Y); maxY = fmax(maxY, p3.Y);
    
    sizeY = maxY - minY + 1;
    
    scanlineMin = (int *)malloc(sizeY * sizeof(int));
    scanlineMax = (int *)calloc(sizeY, sizeof(int));
    
    if (scanlineMin == NULL) return DmtxFail; /* XXX handle this better */
    if (scanlineMax == NULL) {
        free(scanlineMin);
        return DmtxFail; /* XXX handle this better */
    }
    
    for (i = 0; i < sizeY; i++)
        scanlineMin[i] = this->xMax;
    
    for (i = 0; i < 4; i++) {
        while (lines[i].loc.X != lines[i].loc1.X || lines[i].loc.Y != lines[i].loc1.Y) {
            idx = lines[i].loc.Y - minY;
            scanlineMin[idx] = fmin(scanlineMin[idx], lines[i].loc.X);
            scanlineMax[idx] = fmax(scanlineMax[idx], lines[i].loc.X);
            bresLineStep(lines + i, 1, 0);
        }
    }
    
    for (posY = minY; posY < maxY && posY < this->yMax; posY++) {
        idx = posY - minY;
        for (posX = scanlineMin[idx]; posX < scanlineMax[idx] && posX < this->xMax; posX++) {
            cache_ = dmtxDecodeGetCache(posX, posY);
            if (cache_ != NULL)
                *cache_ |= 0x80;
        }
    }
    
    free(scanlineMin);
    free(scanlineMax);
    
    return DmtxPass;
}

DmtxBresLine DmtxDecode::bresLineInit(DmtxPixelLoc loc0, DmtxPixelLoc loc1, DmtxPixelLoc locInside)
{
    int cp;
    DmtxBresLine line;
    DmtxPixelLoc *locBeg, *locEnd;
    
    /* XXX Verify that loc0 and loc1 are inbounds */
    
    /* Values that stay the same after initialization */
    line.loc0 = loc0;
    line.loc1 = loc1;
    line.xStep = (loc0.X < loc1.X) ? +1 : -1;
    line.yStep = (loc0.Y < loc1.Y) ? +1 : -1;
    line.xDelta = abs(loc1.X - loc0.X);
    line.yDelta = abs(loc1.Y - loc0.Y);
    line.steep = static_cast<int>(line.yDelta > line.xDelta);
    
    /* Take cross product to determine outward step */
    if (line.steep != 0)
    {
        /* Point first vector up to get correct sign */
        if (loc0.Y < loc1.Y)
        {
            locBeg = &loc0;
            locEnd = &loc1;
        }
        else
        {
            locBeg = &loc1;
            locEnd = &loc0;
        }
        cp = (((locEnd->X - locBeg->X) * (locInside.Y - locEnd->Y)) -
              ((locEnd->Y - locBeg->Y) * (locInside.X - locEnd->X)));
        
        line.xOut = (cp > 0) ? +1 : -1;
        line.yOut = 0;
    }
    else
    {
        /* Point first vector left to get correct sign */
        if (loc0.X > loc1.X)
        {
            locBeg = &loc0;
            locEnd = &loc1;
        }
        else
        {
            locBeg = &loc1;
            locEnd = &loc0;
        }
        cp = (((locEnd->X - locBeg->X) * (locInside.Y - locEnd->Y)) -
              ((locEnd->Y - locBeg->Y) * (locInside.X - locEnd->X)));
        
        line.xOut = 0;
        line.yOut = (cp > 0) ? +1 : -1;
    }
    
    /* Values that change while stepping through line */
    line.loc = loc0;
    line.travel = 0;
    line.outward = 0;
    line.error = (line.steep) ? line.yDelta/2 : line.xDelta/2;
    
    return line;
}

unsigned int DmtxDecode::bresLineStep(DmtxBresLine *line, int travel, int outward)
{
    int i;
    DmtxBresLine lineNew;
    
    lineNew = *line;
    
    if (abs(travel) >= 2) return DmtxFail;
    if (abs(outward) < 0) return DmtxFail;
    
    /* Perform forward step */
    if (travel > 0)
    {
        lineNew.travel++;
        if (lineNew.steep != 0)
        {
            lineNew.loc.Y += lineNew.yStep;
            lineNew.error -= lineNew.xDelta;
            if (lineNew.error < 0)
            {
                lineNew.loc.X += lineNew.xStep;
                lineNew.error += lineNew.yDelta;
            }
        }
        else
        {
            lineNew.loc.X += lineNew.xStep;
            lineNew.error -= lineNew.yDelta;
            if (lineNew.error < 0)
            {
                lineNew.loc.Y += lineNew.yStep;
                lineNew.error += lineNew.xDelta;
            }
        }
    }
    else if (travel < 0)
    {
        lineNew.travel--;
        if (lineNew.steep != 0)
        {
            lineNew.loc.Y -= lineNew.yStep;
            lineNew.error += lineNew.xDelta;
            if (lineNew.error >= lineNew.yDelta)
            {
                lineNew.loc.X -= lineNew.xStep;
                lineNew.error -= lineNew.yDelta;
            }
        }
        else
        {
            lineNew.loc.X -= lineNew.xStep;
            lineNew.error += lineNew.yDelta;
            if (lineNew.error >= lineNew.xDelta)
            {
                lineNew.loc.Y -= lineNew.yStep;
                lineNew.error -= lineNew.xDelta;
            }
        }
    }
    
    for (i = 0; i < outward; i++) {
        /* Outward steps */
        lineNew.outward++;
        lineNew.loc.X += lineNew.xOut;
        lineNew.loc.Y += lineNew.yOut;
    }
    
    *line = lineNew;
    
    return DmtxPass;
}

DmtxBestLine DmtxDecode::findBestSolidLine(int step0, int step1, int streamDir, int houghAvoid, unsigned int* passFail)
{
    int hough[3][DMTX_HOUGH_RES] = { { 0 } };
    int houghMin, houghMax;
    char houghTest[DMTX_HOUGH_RES];
    int i;
    int step;
    int sign;
    int tripSteps = 0;
    int angleBest;
    int hOffset, hOffsetBest;
    int xDiff, yDiff;
    int dH;
    DmtxRay2 rH;
    DmtxFollow follow;
    DmtxBestLine line;
    DmtxPixelLoc rHp;
    
    memset(&line, 0x00, sizeof(DmtxBestLine));
    memset(&rH, 0x00, sizeof(DmtxRay2));
    angleBest = 0;
    hOffset = hOffsetBest = 0;
    
    sign = 0;
    
    /* Always follow path flowing away from the trail start */
    if (step0 != 0)
    {
        if (step0 > 0)
        {
            sign = +1;
            tripSteps = (step1 - step0 + region.stepsTotal) % region.stepsTotal;
        }
        else
        {
            sign = -1;
            tripSteps = (step0 - step1 + region.stepsTotal) % region.stepsTotal;
        }
        if (tripSteps == 0)
            tripSteps = region.stepsTotal;
    }
    else if (step1 != 0)
    {
        sign = (step1 > 0) ? +1 : -1;
        tripSteps = abs(step1);
    }
    else if (step1 == 0)
    {
        sign = +1;
        tripSteps = region.stepsTotal;
    }
    if (sign != streamDir)
    {
        *passFail = DmtxFail;
        return line;
    }
    
    follow = followSeek(step0, passFail);
    if (*passFail == DmtxFail)
    {
        // *passFail = DmtxFail;
        return line;
    }
    rHp = follow.loc;
    
    line.stepBeg = line.stepPos = line.stepNeg = step0;
    line.locBeg = follow.loc;
    line.locPos = follow.loc;
    line.locNeg = follow.loc;
    
    /* Predetermine which angles to test */
    for (i = 0; i < DMTX_HOUGH_RES; i++) {
        if (houghAvoid == DmtxUndefined)
        {
            houghTest[i] = 1;
        }
        else
        {
            houghMin = (houghAvoid + DMTX_HOUGH_RES/6) % DMTX_HOUGH_RES;
            houghMax = (houghAvoid - DMTX_HOUGH_RES/6 + DMTX_HOUGH_RES) % DMTX_HOUGH_RES;
            if (houghMin > houghMax)
                houghTest[i] = (i > houghMin || i < houghMax) ? 1 : 0;
            else
                houghTest[i] = (i > houghMin && i < houghMax) ? 1 : 0;
        }
    }
    
    /* Test each angle for steps along path */
    for (step = 0; step < tripSteps; step++) {
        
        xDiff = follow.loc.X - rHp.X;
        yDiff = follow.loc.Y - rHp.Y;
        
        /* Increment Hough accumulator */
        for (i = 0; i < DMTX_HOUGH_RES; i++) {
            
            if (static_cast<int>(houghTest[i]) == 0)
                continue;
            
            dH = (rHvX[i] * yDiff) - (rHvY[i] * xDiff);
            if (dH >= -384 && dH <= 384)
            {
                if (dH > 128)
                    hOffset = 2;
                else if (dH >= -128)
                    hOffset = 1;
                else
                    hOffset = 0;
                
                hough[hOffset][i]++;
                
                /* New angle takes over lead */
                if (hough[hOffset][i] > hough[hOffsetBest][angleBest])
                {
                    angleBest = i;
                    hOffsetBest = hOffset;
                }
            }
        }
        follow = followStep(follow, sign, passFail);
        if (*passFail == DmtxFail)
        {
            // *passFail = DmtxFail;
            return line;
        }
    }
    
    line.angle = angleBest;
    line.hOffset = hOffsetBest;
    line.mag = hough[hOffsetBest][angleBest];
    
    return line;
}

unsigned int DmtxDecode::findTravelLimits(DmtxBestLine *line)
{
    int i;
    int distSq, distSqMax;
    int xDiff, yDiff;
    int posRunning, negRunning;
    int posTravel, negTravel;
    int posWander, posWanderMin, posWanderMax, posWanderMinLock, posWanderMaxLock;
    int negWander, negWanderMin, negWanderMax, negWanderMinLock, negWanderMaxLock;
    int cosAngle, sinAngle;
    DmtxFollow followPos, followNeg;
    DmtxPixelLoc loc0, posMax, negMax;
    unsigned int passFail;
    
    /* line->stepBeg is already known to sit on the best Hough line */
    followPos = followNeg = followSeek(line->stepBeg, &passFail);
    if (passFail == DmtxFail) return DmtxFail;
    loc0 = followPos.loc;
    
    cosAngle = rHvX[line->angle];
    sinAngle = rHvY[line->angle];
    
    distSqMax = 0;
    posMax = negMax = followPos.loc;
    
    posTravel = negTravel = 0;
    posWander = posWanderMin = posWanderMax = posWanderMinLock = posWanderMaxLock = 0;
    negWander = negWanderMin = negWanderMax = negWanderMinLock = negWanderMaxLock = 0;
    
    for (i = 0; i < region.stepsTotal/2; i++) {
        
        posRunning = static_cast<int>(i < 10 || abs(posWander) < abs(posTravel));
        negRunning = static_cast<int>(i < 10 || abs(negWander) < abs(negTravel));
        
        if (posRunning != 0) {
            xDiff = followPos.loc.X - loc0.X;
            yDiff = followPos.loc.Y - loc0.Y;
            posTravel = (cosAngle * xDiff) + (sinAngle * yDiff);
            posWander = (cosAngle * yDiff) - (sinAngle * xDiff);
            
            if (posWander >= -3*256 && posWander <= 3*256) {
                distSq = static_cast<int>(distanceSquared(followPos.loc, negMax));
                if (distSq > distSqMax) {
                    posMax = followPos.loc;
                    distSqMax = distSq;
                    line->stepPos = followPos.step;
                    line->locPos = followPos.loc;
                    posWanderMinLock = posWanderMin;
                    posWanderMaxLock = posWanderMax;
                }
            }
            else
            {
                posWanderMin = fmin(posWanderMin, posWander);
                posWanderMax = fmax(posWanderMax, posWander);
            }
        }
        else if (!negRunning)
        {
            break;
        }
        
        if (negRunning != 0)
        {
            xDiff = followNeg.loc.X - loc0.X;
            yDiff = followNeg.loc.Y - loc0.Y;
            negTravel = (cosAngle * xDiff) + (sinAngle * yDiff);
            negWander = (cosAngle * yDiff) - (sinAngle * xDiff);
            
            if (negWander >= -3*256 && negWander < 3*256)
            {
                distSq = static_cast<int>(distanceSquared(followNeg.loc, posMax));
                if (distSq > distSqMax) {
                    negMax = followNeg.loc;
                    distSqMax = distSq;
                    line->stepNeg = followNeg.step;
                    line->locNeg = followNeg.loc;
                    negWanderMinLock = negWanderMin;
                    negWanderMaxLock = negWanderMax;
                }
            }
            else
            {
                negWanderMin = fmin(negWanderMin, negWander);
                negWanderMax = fmax(negWanderMax, negWander);
            }
        }
        else if (!posRunning)
        {
            break;
        }
        
        followPos = followStep(followPos, +1, &passFail);
        if (passFail == DmtxFail) return DmtxFail;
        followNeg = followStep(followNeg, -1, &passFail);
        if (passFail == DmtxFail) return DmtxFail;
    }
    line->devn = fmax(posWanderMaxLock - posWanderMinLock, negWanderMaxLock - negWanderMinLock)/256;
    line->distSq = distSqMax;
    
    return DmtxPass;
}

long DmtxDecode::distanceSquared(DmtxPixelLoc a, DmtxPixelLoc b)
{
    long xDelta, yDelta;
    
    xDelta = a.X - b.X;
    yDelta = a.Y - b.Y;
    
    return (xDelta * xDelta) + (yDelta * yDelta);
}


unsigned int DmtxDecode::dmtxRegionUpdateXfrms()
{
    double radians;
    DmtxRay2 rLeft, rBottom, rTop, rRight;
    DmtxVector2 p00, p10, p11, p01;
    
    if (region.leftKnown == 0 || region.bottomKnown == 0)
        return DmtxFail;
    
    /* Build ray representing left edge */
    rLeft.p.X = static_cast<double>(region.leftLoc.X);
    rLeft.p.Y = static_cast<double>(region.leftLoc.Y);
    radians = region.leftAngle * (M_PI/DMTX_HOUGH_RES);
    rLeft.v.X = cos(radians);
    rLeft.v.Y = sin(radians);
    rLeft.tMin = 0.0;
    rLeft.tMax = dmtxVector2Norm(&rLeft.v);
    
    /* Build ray representing bottom edge */
    rBottom.p.X = static_cast<double>(region.bottomLoc.X);
    rBottom.p.Y = static_cast<double>(region.bottomLoc.Y);
    radians = region.bottomAngle * (M_PI/DMTX_HOUGH_RES);
    rBottom.v.X = cos(radians);
    rBottom.v.Y = sin(radians);
    rBottom.tMin = 0.0;
    rBottom.tMax = dmtxVector2Norm(&rBottom.v);
    
    /* Build ray representing top edge */
    if (region.topKnown != 0) {
        rTop.p.X = static_cast<double>(region.topLoc.X);
        rTop.p.Y = static_cast<double>(region.topLoc.Y);
        radians = region.topAngle * (M_PI/DMTX_HOUGH_RES);
        rTop.v.X = cos(radians);
        rTop.v.Y = sin(radians);
        rTop.tMin = 0.0;
        rTop.tMax = dmtxVector2Norm(&rTop.v);
    }
    else
    {
        rTop.p.X = static_cast<double>(region.locT.X);
        rTop.p.Y = static_cast<double>(region.locT.Y);
        radians = region.bottomAngle * (M_PI/DMTX_HOUGH_RES);
        rTop.v.X = cos(radians);
        rTop.v.Y = sin(radians);
        rTop.tMin = 0.0;
        rTop.tMax = rBottom.tMax;
    }
    
    /* Build ray representing right edge */
    if (region.rightKnown != 0)
    {
        rRight.p.X = static_cast<double>(region.rightLoc.X);
        rRight.p.Y = static_cast<double>(region.rightLoc.Y);
        radians = region.rightAngle * (M_PI/DMTX_HOUGH_RES);
        rRight.v.X = cos(radians);
        rRight.v.Y = sin(radians);
        rRight.tMin = 0.0;
        rRight.tMax = dmtxVector2Norm(&rRight.v);
    }
    else
    {
        rRight.p.X = static_cast<double>(region.locR.X);
        rRight.p.Y = static_cast<double>(region.locR.Y);
        radians = region.leftAngle * (M_PI/DMTX_HOUGH_RES);
        rRight.v.X = cos(radians);
        rRight.v.Y = sin(radians);
        rRight.tMin = 0.0;
        rRight.tMax = rLeft.tMax;
    }
    
    /* Calculate 4 corners, real or imagined */
    if (dmtxRay2Intersect(&p00, &rLeft, &rBottom) == DmtxFail)
        return DmtxFail;
    
    if (dmtxRay2Intersect(&p10, &rBottom, &rRight) == DmtxFail)
        return DmtxFail;
    
    if (dmtxRay2Intersect(&p11, &rRight, &rTop) == DmtxFail)
        return DmtxFail;
    
    if (dmtxRay2Intersect(&p01, &rTop, &rLeft) == DmtxFail)
        return DmtxFail;
    
    if (dmtxRegionUpdateCorners(p00, p10, p11, p01) != DmtxPass)
        return DmtxFail;
    
    return DmtxPass;
}

unsigned int DmtxDecode::dmtxRegionUpdateCorners(DmtxVector2 p00, DmtxVector2 p10, DmtxVector2 p11, DmtxVector2 p01)
{
    double xMax_, yMax_;
    double tx, ty, phi, shx, scx, scy, skx, sky;
    double dimOT, dimOR, dimTX, dimRX, ratio;
    DmtxVector2 vOT, vOR, vTX, vRX, vTmp;
    DmtxMatrix3 m, mtxy, mphi, mshx, mscx, mscy, mscxy, msky, mskx;
    
    xMax_ = static_cast<double>(dmtxDecodeGetProp(DmtxPropWidth) - 1);
    yMax_ = static_cast<double>(dmtxDecodeGetProp(DmtxPropHeight) - 1);
    
    if (p00.X < 0.0 || p00.Y < 0.0 || p00.X > xMax_ || p00.Y > yMax_ ||
       p01.X < 0.0 || p01.Y < 0.0 || p01.X > xMax_ || p01.Y > yMax_ ||
       p10.X < 0.0 || p10.Y < 0.0 || p10.X > xMax_ || p10.Y > yMax_)
        return DmtxFail;
    
    dimOT = dmtxVector2Mag(dmtxVector2Sub(&vOT, &p01, &p00)); /* XXX could use MagSquared() */
    dimOR = dmtxVector2Mag(dmtxVector2Sub(&vOR, &p10, &p00));
    dimTX = dmtxVector2Mag(dmtxVector2Sub(&vTX, &p11, &p01));
    dimRX = dmtxVector2Mag(dmtxVector2Sub(&vRX, &p11, &p10));
    
    /* Verify that sides are reasonably long */
    if (dimOT <= 8.0 || dimOR <= 8.0 || dimTX <= 8.0 || dimRX <= 8.0)
        return DmtxFail;
    
    /* Verify that the 4 corners define a reasonably fat quadrilateral */
    ratio = dimOT / dimRX;
    if (ratio <= 0.5 || ratio >= 2.0)
        return DmtxFail;
    
    ratio = dimOR / dimTX;
    if (ratio <= 0.5 || ratio >= 2.0)
        return DmtxFail;
    
    /* Verify this is not a bowtie shape */
    if (dmtxVector2Cross(&vOR, &vRX) <= 0.0 ||
       dmtxVector2Cross(&vOT, &vTX) >= 0.0)
        return DmtxFail;
    
    if (rightAngleTrueness(p00, p10, p11, M_PI_2) <= this->squareDevn)
        return DmtxFail;
    if (rightAngleTrueness(p10, p11, p01, M_PI_2) <= this->squareDevn)
        return DmtxFail;
    
    /* Calculate values needed for transformations */
    tx = -1 * p00.X;
    ty = -1 * p00.Y;
    dmtxMatrix3Translate(mtxy, tx, ty);
    
    phi = atan2(vOT.X, vOT.Y);
    dmtxMatrix3Rotate(mphi, phi);
    dmtxMatrix3Multiply(m, mtxy, mphi);
    
    if (dmtxMatrix3VMultiply(&vTmp, &p10, m) == DmtxFail)
        return DmtxFail;
    shx = -vTmp.Y / vTmp.X;
    dmtxMatrix3Shear(mshx, 0.0, shx);
    dmtxMatrix3MultiplyBy(m, mshx);
    
    scx = 1.0/vTmp.X;
    dmtxMatrix3Scale(mscx, scx, 1.0);
    dmtxMatrix3MultiplyBy(m, mscx);
    
    if (dmtxMatrix3VMultiply(&vTmp, &p11, m) == DmtxFail)
        return DmtxFail;
    scy = 1.0/vTmp.Y;
    dmtxMatrix3Scale(mscy, 1.0, scy);
    dmtxMatrix3MultiplyBy(m, mscy);
    
    if (dmtxMatrix3VMultiply(&vTmp, &p11, m) == DmtxFail)
        return DmtxFail;
    skx = vTmp.X;
    if (dmtxMatrix3LineSkewSide(mskx, 1.0, skx, 1.0) == DmtxFail)
        return DmtxFail;
    dmtxMatrix3MultiplyBy(m, mskx);
    
    if (dmtxMatrix3VMultiply(&vTmp, &p01, m) == DmtxFail)
        return DmtxFail;
    sky = vTmp.Y;
    if (dmtxMatrix3LineSkewTop(msky, sky, 1.0, 1.0) == DmtxFail)
        return DmtxFail;
    dmtxMatrix3Multiply(region.raw2fit, m, msky);
    
    /* Create inverse matrix by reverse (avoid straight matrix inversion) */
    if (dmtxMatrix3LineSkewTopInv(msky, sky, 1.0, 1.0) == DmtxFail)
        return DmtxFail;
    if (dmtxMatrix3LineSkewSideInv(mskx, 1.0, skx, 1.0) == DmtxFail)
        return DmtxFail;
    dmtxMatrix3Multiply(m, msky, mskx);
    
    dmtxMatrix3Scale(mscxy, 1.0/scx, 1.0/scy);
    dmtxMatrix3MultiplyBy(m, mscxy);
    
    dmtxMatrix3Shear(mshx, 0.0, -shx);
    dmtxMatrix3MultiplyBy(m, mshx);
    
    dmtxMatrix3Rotate(mphi, -phi);
    dmtxMatrix3MultiplyBy(m, mphi);
    
    dmtxMatrix3Translate(mtxy, -tx, -ty);
    dmtxMatrix3Multiply(region.fit2raw, m, mtxy);
    
    return DmtxPass;
}

double DmtxDecode::rightAngleTrueness(DmtxVector2 c0, DmtxVector2 c1, DmtxVector2 c2, double angle)
{
    DmtxVector2 vA, vB;
    DmtxMatrix3 m;
    
    dmtxVector2Norm(dmtxVector2Sub(&vA, &c0, &c1));
    dmtxVector2Norm(dmtxVector2Sub(&vB, &c2, &c1));
    
    dmtxMatrix3Rotate(m, angle);
    if (dmtxMatrix3VMultiplyBy(&vB, m) == DmtxFail)
        return -1.0;
    
    return dmtxVector2Dot(&vA, &vB);
}

unsigned int DmtxDecode::bresLineGetStep(DmtxBresLine line, DmtxPixelLoc target, int *travel, int *outward)
{
    /* Determine necessary step along and outward from Bresenham line */
    if (line.steep != 0) {
        *travel = (line.yStep > 0) ? target.Y - line.loc.Y : line.loc.Y - target.Y;
        bresLineStep(&line, *travel, 0);
        *outward = (line.xOut > 0) ? target.X - line.loc.X : line.loc.X - target.X;
        if (line.yOut != 0) return DmtxFail;
    }
    else
    {
        *travel = (line.xStep > 0) ? target.X - line.loc.X : line.loc.X - target.X;
        bresLineStep(&line, *travel, 0);
        *outward = (line.yOut > 0) ? target.Y - line.loc.Y : line.loc.Y - target.Y;
        if (line.xOut != 0) return DmtxFail;
    }
    
    return DmtxPass;
}

///

unsigned int DmtxDecode::populateArrayFromMatrix(DmtxMessage *msg)
{
    int weightFactor;
    int mapWidth, mapHeight;
    int xRegionTotal, yRegionTotal;
    int xRegionCount, yRegionCount;
    int xOrigin, yOrigin;
    int mapCol, mapRow;
    int colTmp, rowTmp, idx;
    int tally[24][24]; /* Large enough to map largest single region */
    
    /* Capture number of regions present in barcode */
    xRegionTotal = dmtxGetSymbolAttribute(DmtxSymAttribHorizDataRegions, region.sizeIdx);
    yRegionTotal = dmtxGetSymbolAttribute(DmtxSymAttribVertDataRegions, region.sizeIdx);
    
    /* Capture region dimensions (not including border modules) */
    mapWidth = dmtxGetSymbolAttribute(DmtxSymAttribDataRegionCols, region.sizeIdx);
    mapHeight = dmtxGetSymbolAttribute(DmtxSymAttribDataRegionRows, region.sizeIdx);
    
    weightFactor = 2 * (mapHeight + mapWidth + 2);
    if (weightFactor <= 0) return DmtxFail;
    
    /* Tally module changes for each region in each direction */
    for (yRegionCount = 0; yRegionCount < yRegionTotal; yRegionCount++) {
        
        /* Y location of mapping region origin in symbol coordinates */
        yOrigin = yRegionCount * (mapHeight + 2) + 1;
        
        for (xRegionCount = 0; xRegionCount < xRegionTotal; xRegionCount++) {
            
            /* X location of mapping region origin in symbol coordinates */
            xOrigin = xRegionCount * (mapWidth + 2) + 1;
            
            memset(tally, 0x00, 24 * 24 * sizeof(int));
            if (tallyModuleJumps(tally, xOrigin, yOrigin, mapWidth, mapHeight, DmtxDirUp) == DmtxFail)
                return DmtxFail;
            if (tallyModuleJumps(tally, xOrigin, yOrigin, mapWidth, mapHeight, DmtxDirLeft) == DmtxFail)
                return DmtxFail;
            if (tallyModuleJumps(tally, xOrigin, yOrigin, mapWidth, mapHeight, DmtxDirDown) == DmtxFail)
                return DmtxFail;
            if (tallyModuleJumps(tally, xOrigin, yOrigin, mapWidth, mapHeight, DmtxDirRight) == DmtxFail)
                return DmtxFail;
            
            /* Decide module status based on final tallies */
            for (mapRow = 0; mapRow < mapHeight; mapRow++) {
                for (mapCol = 0; mapCol < mapWidth; mapCol++) {
                    rowTmp = (yRegionCount * mapHeight) + mapRow;
                    rowTmp = yRegionTotal * mapHeight - rowTmp - 1;
                    colTmp = (xRegionCount * mapWidth) + mapCol;
                    idx = (rowTmp * xRegionTotal * mapWidth) + colTmp;
                    
                    if (tally[mapRow][mapCol] / static_cast<double>(weightFactor) >= 0.5)
                    {
                        msg->array[idx] = DmtxModuleOnRGB;
                    }
                    else
                    {
                        msg->array[idx] = DmtxModuleOff;
                    }
                    
                    msg->array[idx] |= DmtxModuleAssigned;
                }
            }
        }
    }
    
    return DmtxPass;
}

unsigned int DmtxDecode::tallyModuleJumps(int tally[][24], int xOrigin, int yOrigin, int mapWidth, int mapHeight, DmtxDirection dir)
{
    int extent, weight;
    int travelStep;
    int symbolRow, symbolCol;
    int mapRow, mapCol;
    int lineStart, lineStop;
    int travelStart, travelStop;
    int *line, *travel;
    int jumpThreshold;
    int darkOnLight;
    int color;
    int statusPrev, statusModule;
    int tPrev, tModule;
    
    if (dir != DmtxDirUp && dir != DmtxDirLeft && dir != DmtxDirDown && dir != DmtxDirRight)
        return DmtxFail;
    
    travelStep = (dir == DmtxDirUp || dir == DmtxDirRight) ? 1 : -1;
    
    /* Abstract row and column progress using pointers to allow grid
     traversal in all 4 directions using same logic */
    
    if ((dir & DmtxDirHorizontal) != 0x00)
    {
        line = &symbolRow;
        travel = &symbolCol;
        extent = mapWidth;
        lineStart = yOrigin;
        lineStop = yOrigin + mapHeight;
        travelStart = (travelStep == 1) ? xOrigin - 1 : xOrigin + mapWidth;
        travelStop = (travelStep == 1) ? xOrigin + mapWidth : xOrigin - 1;
    }
    else
    {
        if ( !(dir & DmtxDirVertical)) return DmtxFail;
        
        line = &symbolCol;
        travel = &symbolRow;
        extent = mapHeight;
        lineStart = xOrigin;
        lineStop = xOrigin + mapWidth;
        travelStart = (travelStep == 1) ? yOrigin - 1: yOrigin + mapHeight;
        travelStop = (travelStep == 1) ? yOrigin + mapHeight : yOrigin - 1;
    }
    
    
    darkOnLight = static_cast<int>(region.offColor > region.onColor);
    jumpThreshold = abs(static_cast<int>(0.4 * (region.offColor - region.onColor) + 0.5));
    
    if (jumpThreshold < 0) return DmtxFail;
    
    for (*line = lineStart; *line < lineStop; (*line)++) {
        
        /* Capture tModule for each leading border module as normal but
         decide status based on predictable barcode border pattern */
        
        *travel = travelStart;
        color = readModuleColor(symbolRow, symbolCol, region.sizeIdx);
        if (color == -1) return DmtxFail;
        
        tModule = (darkOnLight) ? region.offColor - color : color - region.offColor;
        
        statusModule = (travelStep == 1 || (*line & 0x01) == 0) ? DmtxModuleOnRGB : DmtxModuleOff;
        
        weight = extent;
        
        while ((*travel += travelStep) != travelStop) {
            
            tPrev = tModule;
            statusPrev = statusModule;
            
            /* For normal data-bearing modules capture color and decide
             module status based on comparison to previous "known" module */
            
            color = readModuleColor(symbolRow, symbolCol, region.sizeIdx);
            if (color == -1) return DmtxFail;
            
            tModule = (darkOnLight) ? region.offColor - color : color - region.offColor;
            
            if (statusPrev == DmtxModuleOnRGB)
            {
                if (tModule < tPrev - jumpThreshold)
                {
                    statusModule = DmtxModuleOff;
                }
                else
                {
                    statusModule = DmtxModuleOnRGB;
                }
            }
            else if (statusPrev == DmtxModuleOff)
            {
                if (tModule > tPrev + jumpThreshold)
                {
                    statusModule = DmtxModuleOnRGB;
                }
                else
                {
                    statusModule = DmtxModuleOff;
                }
            }
            
            mapRow = symbolRow - yOrigin;
            mapCol = symbolCol - xOrigin;
            // assert(mapRow < 24 && mapCol < 24);
            if (mapRow >= 24 || mapCol >= 24) return DmtxFail;
            
            if (statusModule == DmtxModuleOnRGB)
            {
                tally[mapRow][mapCol] += (2 * weight);
            }
            
            weight--;
        }
        
        if (weight != 0) return DmtxFail;
    }
    
    return DmtxPass;
}

int DmtxDecode::readModuleColor(int symbolRow, int symbolCol, int sizeIdx)
{
    int colorTmp;
    double sampleX[] = { 0.5, 0.4, 0.5, 0.6, 0.5 };
    double sampleY[] = { 0.5, 0.5, 0.4, 0.5, 0.6 };
    DmtxVector2 p;
    
    int symbolRows = dmtxGetSymbolAttribute(DmtxSymAttribSymbolRows, sizeIdx);
    int symbolCols = dmtxGetSymbolAttribute(DmtxSymAttribSymbolCols, sizeIdx);
    
    int color = 0;
    for (int i = 0; i < 5; i++) {
        
        p.X = (1.0/symbolCols) * (symbolCol + sampleX[i]);
        p.Y = (1.0/symbolRows) * (symbolRow + sampleY[i]);
        
        if (dmtxMatrix3VMultiplyBy(&p, region.fit2raw) == DmtxFail)
            return -1;
        
        dmtxDecodeGetPixelValue(static_cast<int>(p.X + 0.5), static_cast<int>(p.Y + 0.5), &colorTmp);
        color += colorTmp;
    }
    
    return color/5;
}

int DmtxDecode::dmtxDecodePopulatedArray(int sizeIdx, DmtxMessage& msg, int fix)
{
    /*
     * Example msg->array indices for a 12x12 datamatrix.
     *  also, the 'L' color (usually black) is defined as 'DmtxModuleOnRGB'
     *
     * XX    XX    XX    XX    XX    XX
     * XX 0   1  2  3  4  5  6  7  8  9 XX
     * XX 10 11 12 13 14 15 16 17 18 19
     * XX 20 21 22 23 24 25 26 27 28 29 XX
     * XX 30 31 32 33 34 35 36 37 38 39
     * XX 40 41 42 43 44 45 46 47 48 49 XX
     * XX 50 51 52 53 54 55 56 57 58 59
     * XX 60 61 62 63 64 65 66 67 68 69 XX
     * XX 70 71 72 73 74 75 76 77 78 79
     * XX 80 81 82 83 84 85 86 87 88 89 XX
     * XX 90 91 92 93 94 95 96 97 98 99
     * XX XX XX XX XX XX XX XX XX XX XX XX
     *
     */
    
    if (modulePlacementEcc200(msg.array, msg.code, sizeIdx, DmtxModuleOnRed | DmtxModuleOnGreen | DmtxModuleOnBlue) < 0)
        return -1;
    
    if (rsDecode(msg.code, sizeIdx, fix) == DmtxFail){
        return -1;
    }
    
    if (msg.decodeDataStream(sizeIdx, NULL) == DmtxFail)
    {
        return -1;
    }
    
    return 0;
}

///

int DmtxDecode::initScanGrid()
{
    memset(&grid, 0x00, sizeof(DmtxScanGrid));
    
    int smallestFeature = this->scanGap;  // dmtxDecodeGetProp(DmtxPropScanGap);
    
    grid.xMin = this->xMin;  // dmtxDecodeGetProp(DmtxPropXmin);
    grid.xMax = this->xMax;  // dmtxDecodeGetProp(DmtxPropXmax);
    grid.yMin = this->yMin;  // dmtxDecodeGetProp(DmtxPropYmin);
    grid.yMax = this->yMax;  // dmtxDecodeGetProp(DmtxPropYmax);
    
    /* Values that get set once */
    int xExtent = grid.xMax - grid.xMin;
    int yExtent = grid.yMax - grid.yMin;
    int maxExtent = (xExtent > yExtent) ? xExtent : yExtent;
    
    if (maxExtent <= 1) return -1;
    
    int extent;
    for (extent = 1; extent < maxExtent; extent = ((extent + 1) * 2) - 1)
        if (extent <= smallestFeature)
            grid.minExtent = extent;
    
    grid.maxExtent = extent;
    
    grid.xOffset = (grid.xMin + grid.xMax - grid.maxExtent) / 2;
    grid.yOffset = (grid.yMin + grid.yMax - grid.maxExtent) / 2;
    
    /* Values that get reset for every level */
    grid.total = 1;
    grid.extent = grid.maxExtent;
    
    //SetDerivedFields();
    grid.jumpSize = grid.extent + 1;
    grid.pixelTotal = 2 * grid.extent - 1;
    grid.startPos = grid.extent / 2;
    grid.pixelCount = 0;
    grid.xCenter = grid.yCenter = grid.startPos;
    
    return 0;
}

}  // namespace dmtx


