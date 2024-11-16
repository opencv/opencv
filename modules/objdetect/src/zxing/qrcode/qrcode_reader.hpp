// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __QR_CODE_READER_H__
#define __QR_CODE_READER_H__

/*
 *  QRCodeReader.hpp
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

#include "../reader.hpp"
#include "decoder/decoder.hpp"
#include "decoder/qrcode_decoder_meta_data.hpp"
#include "detector/detector.hpp"
#include "../error_handler.hpp"
#include "../decode_hints.hpp"

namespace zxing {
namespace qrcode {

struct QBAR_QRCODE_DETECT_INFO
{
    int possibleFixIndex;
    unsigned int possibleAPType;
    
    // QRCodeReader Info
    float possibleFix;
    float patternPossibleFix;
    int pyramidLev;
    float possibleModuleSize;
    std::vector<Ref<ResultPoint>> qrcodeBorder;
    
    QBAR_QRCODE_DETECT_INFO()
    {
        clear();
    }
    
    void clear()
    {
        possibleFixIndex = -1;
        possibleAPType = 0;
        possibleModuleSize = 0;
        
        possibleFix = 0;
        patternPossibleFix = 0;
        pyramidLev = 0;
        qrcodeBorder.clear();
    }
    
    std::string DebugString()
    {
        std::string sDebugString;
        sDebugString.resize(1024);
        if (qrcodeBorder.size() == 4)
        {
            snprintf(const_cast<char *>(sDebugString.data()), sDebugString.size(), "possiblefix: %.4f\n\
    point_0 (%.0f, %.0f)\n\
    point_1 (%.0f, %.0f)\n\
    point_2 (%.0f, %.0f)\n\
    point_3 (%.0f, %.0f)\n", \
                    possibleFix,  \
                    qrcodeBorder[0]->getX(), qrcodeBorder[0]->getY(), \
                    qrcodeBorder[1]->getX(), qrcodeBorder[1]->getY(), \
                    qrcodeBorder[2]->getX(), qrcodeBorder[2]->getY(), \
                    qrcodeBorder[3]->getX(), qrcodeBorder[3]->getY());
        }
        return sDebugString;
    }
};

class QRCodeReader : public Reader {
public:
    enum  ReaderState{
        READER_START = -1,
        DETECT_START = 0,
        DETECT_FINDFINDERPATTERN = 1,
        DETECT_FINDALIGNPATTERN = 2,
        DETECT_FAILD = 3,
        DECODE_START = 4,
        DECODE_READVERSION = 5,
        DECODE_READERRORCORRECTIONLEVEL = 6,
        DECODE_READCODEWORDSORRECTIONLEVEL = 7,
        DECODE_FINISH = 8
    };
    
private:
    Decoder decoder_;
    int detectedDimension_;
    ReaderState readerState_;
    DecodeHints nowHints_;
    
protected:
    Decoder& getDecoder();
    
public:
    QRCodeReader();
    virtual ~QRCodeReader();
    std::string name() { return "qrcode";  }
    
    Ref<Result> decode(Ref<BinaryBitmap> image);
    Ref<Result> decode(Ref<BinaryBitmap> image, DecodeHints hints);
    
    Ref<Result> decodeMore(Ref<BinaryBitmap> image, Ref<BitMatrix> imageBitMatrix, DecodeHints hints, ErrorHandler & err_handler);
    
    Ref<Result> moduleSize(ArrayRef<int> leftTopBlack,ArrayRef<int> leftRightBlack, ArrayRef<int> bottomRightBlack, ArrayRef<int> bottomLeftBlack, Ref<BitMatrix> imageBitMatrix);
    
#ifdef CALC_CODE_AREA_SCORE
    float getCodeAreaScore(BitMatrix& bits, const string& content, const string& ecLevel,
                           int version, const string& encoding, int maskType, ArrayRef<char> bytes);
#endif
    
private:
    QBAR_QRCODE_DETECT_INFO possibleQrcodeInfo_;
    
protected:
    void setPossibleAPCountByVersion(unsigned int version);
    int getRecommendedImageSizeTypeInteral();
    static void initIntegralOld(unsigned int *integral, Ref<BitMatrix> input);
    static void initIntegral(unsigned int *integral, Ref<BitMatrix> input);
    static int smooth(unsigned int *integral, Ref<BitMatrix> input, Ref<BitMatrix> output, int window);
    unsigned int lastDecodeTime_;
    unsigned int lastDecodeID_;
    unsigned int decodeID_;
    int lastPossibleAPCount_;
    int possibleAPCount_;
    float possibleModuleSize_;
    unsigned int lastSamePossibleAPCountTimes_;
    unsigned int samePossibleAPCountTimes_;
    unsigned int lastRecommendedImageSizeType_;
    unsigned int recommendedImageSizeType_;
    unsigned int smoothMaxMultiple_;
    
public:
    virtual unsigned int getDecodeID();
    virtual void setDecodeID(unsigned int id);
    virtual float getPossibleFix();
    virtual unsigned int getPossibleAPType();
    virtual int getPossibleFixType();
    
    void setReaderState(Detector::DetectorState state);
    void setReaderState(Decoder::DecoderState state);
    
    void setPatternFix(float possibleFix);
    void setDecoderFix(float possibleFix, ArrayRef< Ref<ResultPoint> > border);
    void setSuccFix(ArrayRef< Ref<ResultPoint> > border);
    
    ReaderState getReaderState(){return this->readerState_;}
    float calQrcodeArea(Ref<DetectorResult> detectorResult);
    float calTriangleArea(Ref<ResultPoint> centerA, Ref<ResultPoint> centerB, Ref<ResultPoint> centerC);
    
    int getQrcodeInfo(const void * &pQBarQrcodeInfo);
    
    // Added by Valiantliu
    std::vector<int> getPossibleDimentions(int detectDimension);
};

}  // namespace qrcode
}  // namespace zxing

#endif  // __QR_CODE_READER_H__
