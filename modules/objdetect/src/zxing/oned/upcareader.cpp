// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  UPCAReader.cpp
 *  ZXing
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

#include "upcareader.hpp"
#include "../format_exception.hpp"

using zxing::oned::UPCAReader;
using zxing::Ref;
using zxing::Result;

// VC++
using zxing::BitArray;
using zxing::BinaryBitmap;
using zxing::DecodeHints;
using zxing::FormatException;

UPCAReader::UPCAReader() : ean13Reader() {}

Ref<Result> UPCAReader::decodeRow(int rowNumber, Ref<BitArray> row) {
    return maybeReturnResult(ean13Reader.decodeRow(rowNumber, row));
}

Ref<Result> UPCAReader::decodeRow(int rowNumber,
                                  Ref<BitArray> row,
                                  Range const& startGuardRange) {
    return maybeReturnResult(ean13Reader.decodeRow(rowNumber, row, startGuardRange));
}

Ref<Result> UPCAReader::decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    return maybeReturnResult(ean13Reader.decode(image, hints));
}

int UPCAReader::decodeMiddle(Ref<BitArray> row,
                             Range const& startRange,
                             std::string& resultString) {
    
    if (_onedReaderData->ean13_checked){
        resultString = _onedReaderData->ean13_decode_middle_final_string;
        return _onedReaderData->ean13_decode_middle_final_offset;
    }
    
    return ean13Reader.decodeMiddle(row, startRange, resultString);
}

Ref<Result> UPCAReader::maybeReturnResult(Ref<Result> result) {
    const std::string& text = (result->getText())->getText();
    if (text[0] == '0')
    {
        Ref<String> resultString(new String(text.substr(1)));
        Ref<Result> res(new Result(resultString, result->getRawBytes(), result->getResultPoints(),
                                   BarcodeFormat::UPC_A));
        return res;
    }
    else
    {
        return  Ref<Result>();
    }
}

zxing::BarcodeFormat UPCAReader::getBarcodeFormat(){
    return BarcodeFormat::UPC_A;
}
