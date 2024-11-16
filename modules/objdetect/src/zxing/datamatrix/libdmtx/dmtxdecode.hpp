//
//  dmtxdecode.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef dmtxdecode_hpp
#define dmtxdecode_hpp

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
    int InitScanGrid();
    
    unsigned int PopulateArrayFromMatrix(DmtxMessage *msg);
    unsigned int CacheFillQuad(DmtxPixelLoc p0, DmtxPixelLoc p1, DmtxPixelLoc p2, DmtxPixelLoc p3);
    unsigned int TallyModuleJumps(int tally[][24], int xOrigin, int yOrigin, int mapWidth, int mapHeight, DmtxDirection dir);
    int ReadModuleColor(int symbolRow, int symbolCol, int sizeIdx);
    
    DmtxBresLine BresLineInit(DmtxPixelLoc loc0, DmtxPixelLoc loc1, DmtxPixelLoc locInside);
    unsigned int BresLineStep(DmtxBresLine *line, int travel, int outward);
    unsigned int BresLineGetStep(DmtxBresLine line, DmtxPixelLoc target, int *travel, int *outward);
    
    int PopGridLocation( DmtxPixelLoc *locPtr);
    int GetGridCoordinates(DmtxPixelLoc *locPtr);
    int dmtxRegionScanPixel(int x, int y);
    
    DmtxPointFlow MatrixRegionSeekEdge(DmtxPixelLoc loc);
    DmtxPointFlow GetPointFlow(DmtxPixelLoc loc, int arrive);
    DmtxPointFlow FindStrongestNeighbor(DmtxPointFlow center, int sign);
    unsigned int MatrixRegionOrientation(DmtxPointFlow begin);
    
    unsigned int TrailBlazeContinuous(DmtxPointFlow flowBegin, int maxDiagonal);
    int TrailClear(int clearMask);
    int TrailBlazeGapped(DmtxBresLine line, int streamDir);
    
    DmtxBestLine FindBestSolidLine(int step0, int step1, int streamDir, int houghAvoid, unsigned int* passFail);
    DmtxBestLine FindBestSolidLine2(DmtxPixelLoc loc0, int tripSteps, int sign, int houghAvoid, unsigned int* passFail);
    
    unsigned int FindTravelLimits(DmtxBestLine *line);
    unsigned int dmtxRegionUpdateXfrms();
    unsigned int MatrixRegionAlignCalibEdge(int edgeLoc);
    unsigned int MatrixRegionFindSize();
    int CountJumpTally(int xStart, int yStart, DmtxDirection dir);
    DmtxFollow FollowSeek(int seek, unsigned int* passFail);
    DmtxFollow FollowStep(DmtxFollow followBeg, int sign, unsigned int* passFail);
    DmtxFollow FollowSeekLoc(DmtxPixelLoc loc, unsigned int* passFail);
    DmtxFollow FollowStep2(DmtxFollow followBeg, int sign, unsigned int* passFail);
    
    long DistanceSquared(DmtxPixelLoc a, DmtxPixelLoc b);
    
    unsigned int dmtxRegionUpdateCorners(DmtxVector2 p00, DmtxVector2 p10, DmtxVector2 p11, DmtxVector2 p01);
    double RightAngleTrueness(DmtxVector2 c0, DmtxVector2 c1, DmtxVector2 c2, double angle);
    
    
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
#endif /* dmtxdecode_hpp */
