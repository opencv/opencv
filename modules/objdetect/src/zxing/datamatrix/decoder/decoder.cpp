// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

/*
 *  Decoder.cpp
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

#include "decoder.hpp"
#include "bit_matrix_parser.hpp"
#include "data_block.hpp"
#include "decoded_bit_stream_parser.hpp"
#include "../version.hpp"
#include "../../reader_exception.hpp"
#include "../../checksum_exception.hpp"
#include "../../common/reedsolomon/reed_solomon_exception.hpp"
#include "../../error_handler.hpp"

using zxing::Ref;
using zxing::DecoderResult;
using zxing::datamatrix::Decoder;

// VC++
using zxing::ArrayRef;
using zxing::BitMatrix;
using zxing::ErrorHandler;

Decoder::Decoder() : rsDecoder_(Ref<GenericGF>(GF_DATA_MATRIX_FIELD_256)) {}

void Decoder::correctErrors(ArrayRef<char> codewordBytes, int numDataCodewords, ErrorHandler & err_handler) {
    int numCodewords = codewordBytes->size();
    ArrayRef<int> codewordInts(numCodewords);
    for (int i = 0; i < numCodewords; i++) {
        codewordInts[i] = codewordBytes[i] & 0xff;
    }
    int numECCodewords = numCodewords - numDataCodewords;
    
    rsDecoder_.decode(codewordInts, numECCodewords, err_handler);
    if (err_handler.errCode()) return;
    
    // Copy back into array of bytes -- only need to worry about the bytes that were data
    // We don't care about errors in the error-correction codewords
    for (int i = 0; i < numDataCodewords; i++) {
        codewordBytes[i] = static_cast<char>(codewordInts[i]);
    }
}

Ref<DecoderResult> Decoder::decode(Ref<BitMatrix> bits, ErrorHandler & err_handler) {
    // Construct a parser and read version, error-correction level
    
    BitMatrixParser parser(bits, err_handler);
    if (err_handler.errCode()) return Ref<DecoderResult>();
    
    Version *version = parser.readVersion(bits, err_handler);
    if (err_handler.errCode()) return Ref<DecoderResult>();
    
    // Read codewords
    ArrayRef<char> codewords(parser.readCodewords(err_handler));
    if (err_handler.errCode()) return Ref<DecoderResult>();
    
    // Separate into data blocks
    std::vector<Ref<DataBlock> > dataBlocks = DataBlock::getDataBlocks_new(codewords, version, err_handler);
    if (err_handler.errCode()) return Ref<DecoderResult>();
    
    int dataBlocksCount = dataBlocks.size();
    
    // Count total number of data bytes
    int totalBytes = 0;
    for (int i = 0; i < dataBlocksCount; i++) {
        totalBytes += dataBlocks[i]->getNumDataCodewords();
    }
    ArrayRef<char> resultBytes(totalBytes);
    
    // Error-correct and copy data blocks together into a stream of bytes
    for (int j = 0; j < dataBlocksCount; j++) {
        Ref<DataBlock> dataBlock(dataBlocks[j]);
        ArrayRef<char> codewordBytes = dataBlock->getCodewords();
        int numDataCodewords = dataBlock->getNumDataCodewords();
        correctErrors(codewordBytes, numDataCodewords, err_handler);
        for (int i = 0; i < numDataCodewords; i++) {
            // De-interlace data blocks.
            resultBytes[i * dataBlocksCount + j] = codewordBytes[i];
        }
    }
    // Decode the contents of that stream of bytes
    DecodedBitStreamParser decodedBSParser;
    Ref<DecoderResult> decoderResult = decodedBSParser.decode(resultBytes, err_handler);
    if (err_handler.errCode()) return Ref<DecoderResult> ();
    
    return Ref<DecoderResult> (decoderResult);
}
