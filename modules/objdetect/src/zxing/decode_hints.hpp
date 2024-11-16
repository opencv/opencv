// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __DECODEHINTS_H_
#define __DECODEHINTS_H_
/*
 *  DecodeHintType.hpp
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

#include "barcode_format.hpp"
#include "result_point_callback.hpp"
#include "error_handler.hpp"

#ifdef USE_LANGUAGEICONFIG
#include <string>
#endif

#include <vector>

namespace zxing {

typedef unsigned int DecodeHintType;
class DecodeHints;
DecodeHints operator | (DecodeHints const&, DecodeHints const&);

struct CORNER_POINT
{
    float x,y;
    CORNER_POINT() {
        x = 0;
        y = 0;
    }
    CORNER_POINT(float x_, float y_): x(x_), y(y_) {}
};

class DecodeHints {
public:
    std::vector<CORNER_POINT> qbar_points;
    
private:
    DecodeHintType hints;
    Ref<ResultPointCallback> callback;
    int iPyramidLev;
    bool tryVideo;
    bool useAI_;
    int frameCnt_;
    
    bool useLibdmtx_;
    
#ifdef USE_LANGUAGEICONFIG
    // Add for config language
    // By Skylook
    std::string inputCharset;
    std::string outputCharset;
#endif
    
public:
    static const DecodeHintType AZTEC_HINT = 1 << BarcodeFormat::AZTEC;
    static const DecodeHintType CODABAR_HINT = 1 << BarcodeFormat::CODABAR;
    static const DecodeHintType CODE_25_HINT = 1 << BarcodeFormat::CODE_25;
    static const DecodeHintType CODE_39_HINT = 1 << BarcodeFormat::CODE_39;
    static const DecodeHintType CODE_93_HINT = 1 << BarcodeFormat::CODE_93;
    static const DecodeHintType CODE_128_HINT = 1 << BarcodeFormat::CODE_128;
    static const DecodeHintType DATA_MATRIX_HINT = 1 << BarcodeFormat::DATA_MATRIX;
    static const DecodeHintType EAN_8_HINT = 1 << BarcodeFormat::EAN_8;
    static const DecodeHintType EAN_13_HINT = 1 << BarcodeFormat::EAN_13;
    static const DecodeHintType ITF_HINT = 1 << BarcodeFormat::ITF;
    static const DecodeHintType MAXICODE_HINT = 1 << BarcodeFormat::MAXICODE;
    static const DecodeHintType PDF_417_HINT = 1 << BarcodeFormat::PDF_417;
    static const DecodeHintType QR_CODE_HINT = 1 << BarcodeFormat::QR_CODE;
    static const DecodeHintType RSS_14_HINT = 1 << BarcodeFormat::RSS_14;
    static const DecodeHintType RSS_EXPANDED_HINT = 1 << BarcodeFormat::RSS_EXPANDED;
    static const DecodeHintType UPC_A_HINT = 1 << BarcodeFormat::UPC_A;
    static const DecodeHintType UPC_E_HINT = 1 << BarcodeFormat::UPC_E;
    static const DecodeHintType UPC_EAN_EXTENSION_HINT = 1 << BarcodeFormat::UPC_EAN_EXTENSION;
    
    static const DecodeHintType TRYHARDER_HINT = 1 << 31;
    static const DecodeHintType CHARACTER_SET = 1 << 30;
    static const DecodeHintType  ASSUME_GS1 = 1 << 27;
    
    static const DecodeHintType PURE_BARCODE  = 1 << 25;
    
    static const DecodeHints PRODUCT_HINT;
    static const DecodeHints ONED_HINT;
    static const DecodeHints DEFAULT_HINT;
    
    DecodeHints();
    DecodeHints(DecodeHintType init, int iPyramid = 0);
    
    //  -- Code Detector
    void addDecodeHints(DecodeHints hints_);
    
    void addFormat(BarcodeFormat toadd);
    
    bool containsFormat(BarcodeFormat tocheck) const;
    bool isEmpty() const {return (hints == 0); }
    void clear() {hints = 0;}
    void setTryHarder(bool toset);
    bool getTryHarder() const;
    void setPureBarcode(bool toset);
    bool getPureBarcode() const;
    void setTryVideo(bool tryVideo);
    bool getTryVideo() const;
    
    inline void setPyramidLev(int iLev) { iPyramidLev = iLev;  };
    inline int  getPyramidLev() { return iPyramidLev;  };
    
    inline void setFrameCnt(int frameCnt) { frameCnt_ = frameCnt; }
    inline int getFrameCnt() { return frameCnt_;}
    
    inline bool isUseAI() const { return useAI_; }
    inline void setUseAI(bool useAI) { useAI_ = useAI; }
    
    bool isUseLibdmtx() const { return useLibdmtx_;}
    void setUseLibdmtx(bool useLibdmtx) { useLibdmtx_ = useLibdmtx;}
    
#ifdef USE_LANGUAGEICONFIG
    // Add for config language
    // By Skylook
    void setInputCharset(string charset);
    string getIutputCharset();
    void setOutputCharset(string charset);
    string getOutputCharset();
#endif
    
    void setResultPointCallback(Ref<ResultPointCallback> const&);
    Ref<ResultPointCallback> getResultPointCallback() const;
    
    friend DecodeHints operator | (DecodeHints const&, DecodeHints const&);
};

}  // namespace zxing

#endif
