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
//  dmtxdecode.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_DMTXDECODE_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_DMTXDECODE_HPP__

#include <stdio.h>
#include "common.hpp"
#include "dmtximage.hpp"
#include "dmtxmatrix3.hpp"
#include "dmtxbytelist.hpp"


#define DMTX_HOUGH_RES 180

namespace dmtx {

class DmtxMessage;

class DmtxDecode {
public:
    DmtxDecode(){}
    ~DmtxDecode();
    
    int dmtxDecodeCreate(unsigned char *pxl, int width, int height);
    
    int dmtxDecodeGetProp(int prop);
    unsigned char *dmtxDecodeGetCache(int x, int y);
    unsigned int dmtxDecodeGetPixelValue(int x, int y, /*@out@*/ int *value);
    
    int dmtxDecodeMatrixRegion(int fix, DmtxMessage& msg);
    int dmtxDecodePopulatedArray(int sizeIdx, DmtxMessage& msg, int fix);
    DmtxMessage *dmtxDecodeMosaicRegion(int fix);
    unsigned char *dmtxDecodeCreateDiagnostic(/*@out@*/ int *totalBytes, /*@out@*/ int *headerBytes, int style);
    
    int dmtxRegionFindNext();
private:
    int initScanGrid();
    
    unsigned int populateArrayFromMatrix(DmtxMessage *msg);
    unsigned int cacheFillQuad(DmtxPixelLoc p0, DmtxPixelLoc p1, DmtxPixelLoc p2, DmtxPixelLoc p3);
    unsigned int tallyModuleJumps(int tally[][24], int xOrigin, int yOrigin, int mapWidth, int mapHeight, DmtxDirection dir);
    int readModuleColor(int symbolRow, int symbolCol, int sizeIdx);
    
    DmtxBresLine bresLineInit(DmtxPixelLoc loc0, DmtxPixelLoc loc1, DmtxPixelLoc locInside);
    unsigned int bresLineStep(DmtxBresLine *line, int travel, int outward);
    unsigned int bresLineGetStep(DmtxBresLine line, DmtxPixelLoc target, int *travel, int *outward);
    
    int popGridLocation( DmtxPixelLoc *locPtr);
    int getGridCoordinates(DmtxPixelLoc *locPtr);
    int dmtxRegionScanPixel(int x, int y);
    
    DmtxPointFlow matrixRegionSeekEdge(DmtxPixelLoc loc);
    DmtxPointFlow GetPointFlow(DmtxPixelLoc loc, int arrive);
    DmtxPointFlow findStrongestNeighbor(DmtxPointFlow center, int sign);
    unsigned int matrixRegionOrientation(DmtxPointFlow begin);
    
    unsigned int trailBlazeContinuous(DmtxPointFlow flowBegin, int maxDiagonal);
    int trailClear(int clearMask);
    int trailBlazeGapped(DmtxBresLine line, int streamDir);
    
    DmtxBestLine findBestSolidLine(int step0, int step1, int streamDir, int houghAvoid, unsigned int* passFail);
    DmtxBestLine findBestSolidLine2(DmtxPixelLoc loc0, int tripSteps, int sign, int houghAvoid, unsigned int* passFail);
    
    unsigned int findTravelLimits(DmtxBestLine *line);
    unsigned int dmtxRegionUpdateXfrms();
    unsigned int matrixRegionAlignCalibEdge(int edgeLoc);
    unsigned int matrixRegionFindSize();
    int countJumpTally(int xStart, int yStart, DmtxDirection dir);
    DmtxFollow followSeek(int seek, unsigned int* passFail);
    DmtxFollow followStep(DmtxFollow followBeg, int sign, unsigned int* passFail);
    DmtxFollow followSeekLoc(DmtxPixelLoc loc, unsigned int* passFail);
    DmtxFollow followStep2(DmtxFollow followBeg, int sign, unsigned int* passFail);
    
    long distanceSquared(DmtxPixelLoc a, DmtxPixelLoc b);
    
    unsigned int dmtxRegionUpdateCorners(DmtxVector2 p00, DmtxVector2 p10, DmtxVector2 p11, DmtxVector2 p01);
    double rightAngleTrueness(DmtxVector2 c0, DmtxVector2 c1, DmtxVector2 c2, double angle);
    
    
private:
    /* Options */
    int             edgeMin;
    int             edgeMax;
    int             scanGap;
    int             fnc1;
    double          squareDevn;
    int             sizeIdxExpected;
    int             edgeThresh;
    
    /* Image modifiers */
    int             xMin;
    int             xMax;
    int             yMin;
    int             yMax;
    
    /* Internals */
    /* int             cacheComplete; */
    unsigned char  *cache;
    DmtxImage       image;
    DmtxScanGrid    grid;
    DmtxRegion      region;
};

}  // namespace dmtx
#endif // __ZXING_DATAMATRIX_LIBDMTX_DMTXDECODE_HPP__
