// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
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

#include "zxing.hpp"
#include "multi_format_reader.hpp"
#include "qrcode/qrcode_reader.hpp"
#include "error_handler.hpp"

#ifndef USE_QRCODE_ONLY
#include "datamatrix/data_matrix_reader.hpp"
#include "pdf417/pdf417reader.hpp"
#include "oned/multi_format_one_dreader.hpp"
#endif

#include "reader_exception.hpp"

#include <iostream>


using zxing::Ref;
using zxing::Result;
using zxing::MultiFormatReader;
using std::string;
using std::cout;
using std::endl;

// VC++
using zxing::DecodeHints;
using zxing::BinaryBitmap;


MultiFormatReader::MultiFormatReader() {
    fixIndex_=-1;
    OnlyDecodeIndex_ = -1;
    
    qrcodeCount = 0;
    decodeID_ = 0;
}

Ref<Result> MultiFormatReader::decode(Ref<BinaryBitmap> image) {
    setHints(DecodeHints::DEFAULT_HINT);
    return decodeInternal(image);
}

Ref<Result> MultiFormatReader::decode(Ref<BinaryBitmap> image, 
                                      DecodeHints hints) {
    reader_call_path_ = "";
    
    setHints(hints);
    ++decodeID_;
    
    for (size_t i = 0; i < readers_.size(); ++i)
    {
        reader_call_path_ += readers_[i]->name();
        readers_[i]->setDecodeID(decodeID_);
        Ref<Result> rst = readers_[i]->decode(image, hints);
        reader_call_path_ += readers_[i]->reader_call_path_;
        if (rst == NULL)
        {
            continue;
        }
        return rst;
        
    }
    
    return Ref<Result>();
}

int MultiFormatReader::getQRCodeCount()
{
    return qrcodeCount;
}

Ref<Result> MultiFormatReader::decodeWithState(Ref<BinaryBitmap> image) {
    // Make sure to set up the default state so we don't crash
    if (readers_.size() == 0)
    {
        setHints(DecodeHints::DEFAULT_HINT);
    }
    return decodeInternal(image);
}

void MultiFormatReader::setHints(DecodeHints hints) {
    hints_ = hints;
    readers_.clear();
    
    if (hints.containsFormat(BarcodeFormat::QR_CODE))
        readers_.push_back(Ref<Reader>(new zxing::qrcode::QRCodeReader()));
    
    bool addOneDReader = hints.containsFormat(BarcodeFormat::UPC_E) ||
    hints.containsFormat(BarcodeFormat::UPC_A) ||
    hints.containsFormat(BarcodeFormat::UPC_E) ||
    hints.containsFormat(BarcodeFormat::EAN_13) ||
    hints.containsFormat(BarcodeFormat::EAN_8) ||
    hints.containsFormat(BarcodeFormat::CODABAR) ||
    hints.containsFormat(BarcodeFormat::CODE_39) ||
    hints.containsFormat(BarcodeFormat::CODE_93) ||
    hints.containsFormat(BarcodeFormat::CODE_128) ||
    hints.containsFormat(BarcodeFormat::CODE_25) ||
    hints.containsFormat(BarcodeFormat::ITF) ||
    hints.containsFormat(BarcodeFormat::RSS_14) ||
    hints.containsFormat(BarcodeFormat::RSS_EXPANDED);
    if (addOneDReader)
    {
        readers_.push_back(Ref<Reader>(new zxing::oned::MultiFormatOneDReader(hints)));
    }
    
    if (hints.containsFormat(BarcodeFormat::PDF_417))
    {
        readers_.push_back(Ref<Reader>(new zxing::pdf417::PDF417Reader()));
    }
    if (hints.containsFormat(BarcodeFormat::DATA_MATRIX))
    {
        readers_.push_back(Ref<Reader>(new zxing::datamatrix::DataMatrixReader()));
    }
}

Ref<Result> MultiFormatReader::decodeInternal(Ref<BinaryBitmap> image) {
    for (unsigned int i = 0; i < readers_.size(); i++) {
        Ref<Result> rst = readers_[i]->decode(image, hints_);
        if (rst) return rst;
    }
    
    return Ref<Result>();
}

float  MultiFormatReader::findPossibleFix(){
    float maxFix = 0.0f;
    for (size_t i = 0; i < readers_.size(); ++i) {
        if (maxFix < readers_[i]->getPossibleFix())
        {
            maxFix = readers_[i]->getPossibleFix();
            fixIndex_ = i;
        }
    }
    return maxFix;
}

int MultiFormatReader::getFixIndex(){
    return fixIndex_;
}

void MultiFormatReader::setFixIndex(int index){
    fixIndex_=index;
}

int MultiFormatReader::getQrcodeInfo(const void * &pQBarQrcodeInfo)
{
    if (fixIndex_ <0 || fixIndex_ >= static_cast<int>(readers_.size()))
        return readers_[0]->getQrcodeInfo(pQBarQrcodeInfo);
    else
        return readers_[fixIndex_]->getQrcodeInfo(pQBarQrcodeInfo);
}
