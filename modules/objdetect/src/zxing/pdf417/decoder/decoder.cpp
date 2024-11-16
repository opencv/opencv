// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 * Copyright 2010, 2012 ZXing authors All rights reserved.
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
 *
 * 2012-06-27 hfn: PDF417 Reed-Solomon error correction, using following Java
 * source code:
 * http:// code.google.com/p/zxing/issues/attachmentText?id=817&aid=8170033000&name = pdf417-java-reed-solomon-error-correction-2.patch&token = 0819f5d7446ae2814fd91385eeec6a11
 */

#include "../pdf417reader.hpp"
#include "decoder.hpp"
#include "bit_matrix_parser.hpp"
#include "decoded_bit_stream_parser.hpp"
#include "../../reader_exception.hpp"
#include "../../common/reedsolomon/reed_solomon_exception.hpp"


using zxing::pdf417::decoder::Decoder;
using zxing::pdf417::decoder::ec::ErrorCorrection;
using zxing::Ref;
using zxing::DecoderResult;
using zxing::FormatErrorHandler;
using zxing::ErrorHandler;

// VC++

using zxing::BitMatrix;
using zxing::DecodeHints;
using zxing::ArrayRef;

const int Decoder::MAX_ERRORS = 3;
const int Decoder::MAX_EC_CODEWORDS = 512;

Ref<DecoderResult> Decoder::decode(Ref<BitMatrix> bits, DecodeHints const& hints, ErrorHandler & err_handler) {
    (void)hints;
    
    // Construct a parser to read the data codewords and error-correction level
    BitMatrixParser parser(bits);
   
    ArrayRef<int> codewords;
    err_handler = parser.readCodewords(codewords);
    if (err_handler.ErrCode() || codewords == NULL) {
        err_handler = FormatErrorHandler("PDF:Decoder:decode: cannot read codewords");
        return Ref<DecoderResult>();
    }
    
    int ecLevel = parser.getECLevel();
    int numECCodewords = 1 << (ecLevel + 1);
    ArrayRef<int> erasures = parser.getErasures();
    
    correctErrors(codewords, erasures, numECCodewords, err_handler);
    if (err_handler.ErrCode())  return Ref<DecoderResult>();
    
    verifyCodewordCount(codewords, numECCodewords, err_handler);
    if (err_handler.ErrCode()) return Ref<DecoderResult>();
    
    // Decode the codewords
    DecodedBitStreamParser dbs_parser;
    Ref<DecoderResult> rst = dbs_parser.decode(codewords, err_handler);
    if (err_handler.ErrCode() || NULL == rst){
        err_handler = FormatErrorHandler("PDF:Decoder:decode: cannot read codewords");
        return Ref<DecoderResult>();
    }
    return rst;
}

/**
 * Verify that all is OK with the codeword array.
 *
 * @param codewords
 * @return an index to the first data codeword.
 * @throws FormatException
 */
void Decoder::verifyCodewordCount(ArrayRef<int> codewords, int numECCodewords, ErrorHandler & err_handler) {
    int cwsize = codewords->size();
    if (cwsize < 4) {
        // Codeword array size should be at least 4 allowing for
        // Count CW, At least one Data CW, Error Correction CW, Error Correction CW
        err_handler = FormatErrorHandler("PDF:Decoder:verifyCodewordCount: codeword array too small!");
        return;
    }
    // The first codeword, the Symbol Length Descriptor, shall always encode the total number of data
    // codewords in the symbol, including the Symbol Length Descriptor itself, data codewords and pad
    // codewords, but excluding the number of error correction codewords.
    int numberOfCodewords = codewords[0];
    if (numberOfCodewords > cwsize) {
        err_handler = FormatErrorHandler("PDF:Decoder:verifyCodewordCount: bad codeword number descriptor!");
        return;
    }
    if (numberOfCodewords == 0) {
        // Reset to the length of the array - 8 (Allow for at least level 3 Error Correction (8 Error Codewords)
        if (numECCodewords < cwsize) {
            codewords[0] = cwsize - numECCodewords;
        }
        else
        {
            err_handler = FormatErrorHandler("PDF:Decoder:verifyCodewordCount: bad error correction cw number!");
            return;
        }
    }
}

/**
 * Correct errors whenever it is possible using Reed-Solomom algorithm
 *
 * @param codewords, erasures, numECCodewords
 * @return 0.
 * @throws FormatException
 */
void Decoder::correctErrors(ArrayRef<int> codewords,
                            ArrayRef<int> erasures, int numECCodewords, ErrorHandler & err_handler) {
    if (erasures->size() > numECCodewords / 2 + MAX_ERRORS ||
        numECCodewords < 0 || numECCodewords > MAX_EC_CODEWORDS) {
        err_handler = FormatErrorHandler("PDF:Decoder:correctErrors: Too many errors or EC Codewords corrupted");
        return;
    }
    
    Ref<ErrorCorrection> errorCorrection(new ErrorCorrection);
    errorCorrection->decode(codewords, numECCodewords, erasures, err_handler);
    if (err_handler.ErrCode())
        return;
    
    // 2012-06-27 HFN if, despite of error correction, there are still codewords with invalid
    // value, throw an exception here:
    for (int i = 0; i < codewords->size(); i++) {
        if (codewords[i] < 0) {
            err_handler = FormatErrorHandler("PDF:Decoder:correctErrors: Error correction did not succeed!");
            return;
        }
    }
}
