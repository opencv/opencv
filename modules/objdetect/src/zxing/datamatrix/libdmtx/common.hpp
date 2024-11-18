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
//  common.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#ifndef __ZXING_DATAMATRIX_LIBDMTX_COMMON_HPP__
#define __ZXING_DATAMATRIX_LIBDMTX_COMMON_HPP__

namespace dmtx {
#define NN                      255
#define MAX_ERROR_WORD_COUNT     68

#ifndef M_PI
#define M_PI      3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2    1.57079632679489661923
#endif

#define DmtxPassFail        unsigned int
#define DmtxPass                       1
#define DmtxFail                       0

#define DmtxBoolean         unsigned int
#define DmtxTrue                       1
#define DmtxFalse                      0

#define DmtxModuleOff               0x00
#define DmtxModuleOnRed             0x01
#define DmtxModuleOnGreen           0x02
#define DmtxModuleOnBlue            0x04
#define DmtxModuleOnRGB             0x07  /* OnRed | OnGreen | OnBlue */
#define DmtxModuleOn                0x07
#define DmtxModuleUnsure            0x08
#define DmtxModuleAssigned          0x10
#define DmtxModuleVisited           0x20
#define DmtxModuleData              0x40

#define DmtxFormatMatrix               0
#define DmtxFormatMosaic               1
#define DmtxSymbolSquareCount         24
#define DmtxSymbolRectCount            6
#define DmtxUndefined                 -1

#define DmtxValueC40Latch            230
#define DmtxValueTextLatch           239
#define DmtxValueX12Latch            238
#define DmtxValueEdifactLatch        240
#define DmtxValueBase256Latch        231

#define DmtxValueCTXUnlatch   254
#define DmtxValueEdifactUnlatch       31

#define DmtxValueAsciiPad            129
#define DmtxValueAsciiUpperShift     235
#define DmtxValueCTXShift1      0
#define DmtxValueCTXShift2      1
#define DmtxValueCTXShift3      2
#define DmtxValueFNC1                232
#define DmtxValueStructuredAppend    233
#define DmtxValueReaderProgramming   234
#define DmtxValue05Macro             236
#define DmtxValue06Macro             237
#define DmtxValueECI                 241

#define DmtxC40TextBasicSet            0
#define DmtxC40TextShift1              1
#define DmtxC40TextShift2              2
#define DmtxC40TextShift3              3

///

typedef double DmtxMatrix3[3][3];

///

#define DmtxAlmostZero          0.000001


typedef struct DmtxVector2_struct {
    double          X;
    double          Y;
} DmtxVector2;

typedef struct DmtxRay2_struct {
    double          tMin;
    double          tMax;
    DmtxVector2     p;
    DmtxVector2     v;
} DmtxRay2;

///

typedef enum {
    DmtxFlipNone               = 0x00,
    DmtxFlipX                  = 0x01 << 0,
    DmtxFlipY                  = 0x01 << 1
} DmtxFlip;

typedef enum {
    /* Encoding properties */
    DmtxPropScheme            = 100,
    DmtxPropSizeRequest,
    DmtxPropMarginSize,
    DmtxPropModuleSize,
    DmtxPropFnc1,
    /* Decoding properties */
    DmtxPropEdgeMin           = 200,
    DmtxPropEdgeMax,
    DmtxPropScanGap,
    DmtxPropSquareDevn,
    DmtxPropSymbolSize,
    DmtxPropEdgeThresh,
    
    /* Image properties */
    DmtxPropWidth             = 300,
    DmtxPropHeight,
    // dmtxPropPixelPacking,
    DmtxPropBitsPerPixel,
    DmtxPropBytesPerPixel,
    DmtxPropRowPadBytes,
    DmtxPropRowSizeBytes,
    DmtxPropImageFlip,
    // dmtxPropChannelCount,
    
    /* Image modifiers */
    DmtxPropXmin              = 400,
    DmtxPropXmax,
    DmtxPropYmin,
    DmtxPropYmax,
    // dmtxPropScale
} DmtxProperty;
///

///
typedef struct DmtxScanGrid_struct {
    /* set once */
    int             minExtent;     /* Smallest cross size used in scan */
    int             maxExtent;     /* Size of bounding grid region (2^N - 1) */
    int             xOffset;       /* Offset to obtain image X coordinate */
    int             yOffset;       /* Offset to obtain image Y coordinate */
    int             xMin;          /* Minimum X in image coordinate system */
    int             xMax;          /* Maximum X in image coordinate system */
    int             yMin;          /* Minimum Y in image coordinate system */
    int             yMax;          /* Maximum Y in image coordinate system */
    
    /* reset for each level */
    int             total;         /* Total number of crosses at this size */
    int             extent;        /* Length/width of cross in pixels */
    int             jumpSize;      /* Distance in pixels between cross centers */
    int             pixelTotal;    /* Total pixel count within an individual cross path */
    int             startPos;      /* X and Y coordinate of first cross center in pattern */
    
    /* reset for each cross */
    int             pixelCount;    /* Progress (pixel count) within current cross pattern */
    int             xCenter;       /* X center of current cross pattern */
    int             yCenter;       /* Y center of current cross pattern */
} DmtxScanGrid;

typedef struct DmtxPixelLoc_struct {
    int X;
    int Y;
} DmtxPixelLoc;

typedef struct DmtxPointFlow_struct {
    // int             plane;
    int             arrive;
    int             depart;
    int             mag;
    DmtxPixelLoc    loc;
} DmtxPointFlow;

typedef struct DmtxBestLine_struct {
    int             angle;
    int             hOffset;
    int             mag;
    int             stepBeg;
    int             stepPos;
    int             stepNeg;
    int             distSq;
    double          devn;
    DmtxPixelLoc    locBeg;
    DmtxPixelLoc    locPos;
    DmtxPixelLoc    locNeg;
} DmtxBestLine;

typedef struct DmtxRegion_struct {
    
    /* Trail blazing values */
    int             jumpToPos;     /* */
    int             jumpToNeg;     /* */
    int             stepsTotal;    /* */
    DmtxPixelLoc    finalPos;      /* */
    DmtxPixelLoc    finalNeg;      /* */
    DmtxPixelLoc    boundMin;      /* */
    DmtxPixelLoc    boundMax;      /* */
    DmtxPointFlow   flowBegin;     /* */
    
    /* Orientation values */
    int             polarity;      /* */
    int             stepR;
    int             stepT;
    DmtxPixelLoc    locR;          /* remove if stepR works above */
    DmtxPixelLoc    locT;          /* remove if stepT works above */
    
    /* Region fitting values */
    int             leftKnown;     /* known == 1; unknown == 0 */
    int             leftAngle;     /* hough angle of left edge */
    DmtxPixelLoc    leftLoc;       /* known (arbitrary) location on left edge */
    DmtxBestLine    leftLine;      /* */
    int             bottomKnown;   /* known == 1; unknown == 0 */
    int             bottomAngle;   /* hough angle of bottom edge */
    DmtxPixelLoc    bottomLoc;     /* known (arbitrary) location on bottom edge */
    DmtxBestLine    bottomLine;    /* */
    int             topKnown;      /* known == 1; unknown == 0 */
    int             topAngle;      /* hough angle of top edge */
    DmtxPixelLoc    topLoc;        /* known (arbitrary) location on top edge */
    int             rightKnown;    /* known == 1; unknown == 0 */
    int             rightAngle;    /* hough angle of right edge */
    DmtxPixelLoc    rightLoc;      /* known (arbitrary) location on right edge */
    
    /* Region calibration values */
    int             onColor;       /* */
    int             offColor;      /* */
    int             sizeIdx;       /* Index of arrays that store Data Matrix constants */
    int             symbolRows;    /* Number of total rows in symbol including alignment patterns */
    int             symbolCols;    /* Number of total columns in symbol including alignment patterns */
    int             mappingRows;   /* Number of data rows in symbol */
    int             mappingCols;   /* Number of data columns in symbol */
    
    /* Transform values */
    DmtxMatrix3     raw2fit;       /* 3x3 transformation from raw image to fitted barcode grid */
    DmtxMatrix3     fit2raw;       /* 3x3 transformation from fitted barcode grid to raw image */
} DmtxRegion;
///

typedef unsigned char DmtxByte;

typedef struct DmtxByteList_struct DmtxByteList;
struct DmtxByteList_struct
{
    int length;
    int capacity;
    DmtxByte *b;
};

typedef enum {
    DmtxSymbolRectAuto        = -3,
    DmtxSymbolSquareAuto      = -2,
    DmtxSymbolShapeAuto       = -1,
    DmtxSymbol10x10           =  0,
    DmtxSymbol12x12,
    DmtxSymbol14x14,
    DmtxSymbol16x16,
    DmtxSymbol18x18,
    DmtxSymbol20x20,
    DmtxSymbol22x22,
    DmtxSymbol24x24,
    DmtxSymbol26x26,
    DmtxSymbol32x32,
    DmtxSymbol36x36,
    DmtxSymbol40x40,
    DmtxSymbol44x44,
    DmtxSymbol48x48,
    DmtxSymbol52x52,
    DmtxSymbol64x64,
    DmtxSymbol72x72,
    DmtxSymbol80x80,
    DmtxSymbol88x88,
    DmtxSymbol96x96,
    DmtxSymbol104x104,
    DmtxSymbol120x120,
    DmtxSymbol132x132,
    DmtxSymbol144x144,
    DmtxSymbol8x18,
    DmtxSymbol8x32,
    DmtxSymbol12x26,
    DmtxSymbol12x36,
    DmtxSymbol16x36,
    DmtxSymbol16x48
} DmtxSymbolSize;

typedef struct DmtxBresLine_struct {
    int             xStep;
    int             yStep;
    int             xDelta;
    int             yDelta;
    int             steep;
    int             xOut;
    int             yOut;
    int             travel;
    int             outward;
    int             error;
    DmtxPixelLoc    loc;
    DmtxPixelLoc    loc0;
    DmtxPixelLoc    loc1;
} DmtxBresLine;

typedef enum {
    DmtxDirNone               = 0x00,
    DmtxDirUp                 = 0x01 << 0,
    DmtxDirLeft               = 0x01 << 1,
    DmtxDirDown               = 0x01 << 2,
    DmtxDirRight              = 0x01 << 3,
    DmtxDirHorizontal         = DmtxDirLeft  | DmtxDirRight,
    DmtxDirVertical           = DmtxDirUp    | DmtxDirDown,
    DmtxDirRightUp            = DmtxDirRight | DmtxDirUp,
    DmtxDirLeftDown           = DmtxDirLeft  | DmtxDirDown
} DmtxDirection;

typedef enum {
    DmtxSymAttribSymbolRows,
    DmtxSymAttribSymbolCols,
    DmtxSymAttribDataRegionRows,
    DmtxSymAttribDataRegionCols,
    DmtxSymAttribHorizDataRegions,
    DmtxSymAttribVertDataRegions,
    DmtxSymAttribMappingMatrixRows,
    DmtxSymAttribMappingMatrixCols,
    DmtxSymAttribInterleavedBlocks,
    DmtxSymAttribBlockErrorWords,
    DmtxSymAttribBlockMaxCorrectable,
    DmtxSymAttribSymbolDataWords,
    DmtxSymAttribSymbolErrorWords,
    DmtxSymAttribSymbolMaxCorrectable
} DmtxSymAttribute;

typedef enum {
    DmtxMaskBit8              = 0x01 << 0,
    DmtxMaskBit7              = 0x01 << 1,
    DmtxMaskBit6              = 0x01 << 2,
    DmtxMaskBit5              = 0x01 << 3,
    DmtxMaskBit4              = 0x01 << 4,
    DmtxMaskBit3              = 0x01 << 5,
    DmtxMaskBit2              = 0x01 << 6,
    DmtxMaskBit1              = 0x01 << 7
} DmtxMaskBit;

typedef enum {
    DmtxRangeGood,
    DmtxRangeBad,
    DmtxRangeEnd
} DmtxRange;

typedef enum {
    DmtxEdgeTop               = 0x01 << 0,
    DmtxEdgeBottom            = 0x01 << 1,
    DmtxEdgeLeft              = 0x01 << 2,
    DmtxEdgeRight             = 0x01 << 3
} DmtxEdge;

typedef struct DmtxFollow_struct {
    unsigned char  *ptr;
    unsigned char   neighbor;
    int             step;
    DmtxPixelLoc    loc;
} DmtxFollow;

static const int dmtxNeighborNone = 8;
static const int dmtxPatternX[] = { -1,  0,  1,  1,  1,  0, -1, -1 };
static const int dmtxPatternY[] = { -1, -1, -1,  0,  1,  1,  1,  0 };
static const DmtxPointFlow dmtxBlankEdge = { 0, 0, DmtxUndefined, { -1, -1 } };


static int rHvX[] =
{  256,  256,  256,  256,  255,  255,  255,  254,  254,  253,  252,  251,  250,  249,  248,
    247,  246,  245,  243,  242,  241,  239,  237,  236,  234,  232,  230,  228,  226,  224,
    222,  219,  217,  215,  212,  210,  207,  204,  202,  199,  196,  193,  190,  187,  184,
    181,  178,  175,  171,  168,  165,  161,  158,  154,  150,  147,  143,  139,  136,  132,
    128,  124,  120,  116,  112,  108,  104,  100,   96,   92,   88,   83,   79,   75,   71,
    66,   62,   58,   53,   49,   44,   40,   36,   31,   27,   22,   18,   13,    9,    4,
    0,   -4,   -9,  -13,  -18,  -22,  -27,  -31,  -36,  -40,  -44,  -49,  -53,  -58,  -62,
    -66,  -71,  -75,  -79,  -83,  -88,  -92,  -96, -100, -104, -108, -112, -116, -120, -124,
    -128, -132, -136, -139, -143, -147, -150, -154, -158, -161, -165, -168, -171, -175, -178,
    -181, -184, -187, -190, -193, -196, -199, -202, -204, -207, -210, -212, -215, -217, -219,
    -222, -224, -226, -228, -230, -232, -234, -236, -237, -239, -241, -242, -243, -245, -246,
    -247, -248, -249, -250, -251, -252, -253, -254, -254, -255, -255, -255, -256, -256, -256 };

static int rHvY[] =
{    0,    4,    9,   13,   18,   22,   27,   31,   36,   40,   44,   49,   53,   58,   62,
    66,   71,   75,   79,   83,   88,   92,   96,  100,  104,  108,  112,  116,  120,  124,
    128,  132,  136,  139,  143,  147,  150,  154,  158,  161,  165,  168,  171,  175,  178,
    181,  184,  187,  190,  193,  196,  199,  202,  204,  207,  210,  212,  215,  217,  219,
    222,  224,  226,  228,  230,  232,  234,  236,  237,  239,  241,  242,  243,  245,  246,
    247,  248,  249,  250,  251,  252,  253,  254,  254,  255,  255,  255,  256,  256,  256,
    256,  256,  256,  256,  255,  255,  255,  254,  254,  253,  252,  251,  250,  249,  248,
    247,  246,  245,  243,  242,  241,  239,  237,  236,  234,  232,  230,  228,  226,  224,
    222,  219,  217,  215,  212,  210,  207,  204,  202,  199,  196,  193,  190,  187,  184,
    181,  178,  175,  171,  168,  165,  161,  158,  154,  150,  147,  143,  139,  136,  132,
    128,  124,  120,  116,  112,  108,  104,  100,   96,   92,   88,   83,   79,   75,   71,
    66,   62,   58,   53,   49,   44,   40,   36,   31,   27,   22,   18,   13,    9,    4 };


typedef struct C40TextState_struct {
    int             shift;
    DmtxBoolean     upperShift;
} C40TextState;

typedef enum {
    DmtxSchemeAutoFast        = -2,
    DmtxSchemeAutoBest        = -1,
    DmtxSchemeAscii           =  0,
    DmtxSchemeC40,
    DmtxSchemeText,
    DmtxSchemeX12,
    DmtxSchemeEdifact,
    DmtxSchemeBase256
} DmtxScheme;

}
#endif // __ZXING_DATAMATRIX_LIBDMTX_COMMON_HPP__
