// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Decoder.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 20/05/2008.
 *  Copyright 2008 ZXing authors All rights reserved.
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
#include "qrcode_decoder_meta_data.hpp"
#include "../error_correction_level.hpp"
#include "../version.hpp"
#include "data_block.hpp"
#include "decoded_bit_stream_parser.hpp"
#include "../../reader_exception.hpp"
#include "../../checksum_exception.hpp"
#include "../../format_exception.hpp"
#include "../../common/reedsolomon/reed_solomon_exception.hpp"
#include <iostream>

using zxing::qrcode::Decoder;
using zxing::DecoderResult;
using zxing::Ref;
using std::cout;
using std::endl;

// VC++
// The main class which implements QR Code decoding -- as opposed to locating and extracting
// the QR Code from an image.
using zxing::ArrayRef;
using zxing::BitMatrix;
using zxing::DetectorResult;
using zxing::ErrorHandler;

Decoder::Decoder() :rsDecoder_(Ref<GenericGF>(GF_QR_CODE_FIELD_256))
{
    possibleVersion_ = 0;
    possibleFix_ = 0;
    decoderState_ = NOTSTART;
}

// Convenience method that can decode a QR Code represented as a 2D array of booleans.
// "true" is taken to mean a black module.
Ref<DecoderResult> Decoder::decode(Ref<BitMatrix> bits, ErrorHandler & err_handler)
{
    std::string errMsg = "";
    
    // Used for mirrored qrcode
    int width = bits->getWidth();
    int height = bits->getHeight();

    std::cout << width << ' ' << height << '\n';
    
    Ref<BitMatrix> bits2(new BitMatrix(width, height, reinterpret_cast<bool*>(bits->getPtr()), err_handler));
    if (err_handler.ErrCode())   {
        bits2 = NULL;
    }
    
    Ref<DecoderResult> rst = decode(bits, false, err_handler);
    if (err_handler.ErrCode() || rst == NULL)
    {
        errMsg = err_handler.ErrMsg();
    }
    else
    {
        return rst;
    }
    
    if (bits2 == NULL)
    {
        return Ref<DecoderResult>();
    }
    
    // flip add by sofiawu
    err_handler.Reset();
    if (bits->copyOf2(bits2))
    {
        bits->randomFlipRegion(static_cast<int>(3.0 / 8.0 * width), static_cast<int>(3.0 / 8.0 * height), static_cast<int>(1.0 / 4.0 * width) + 1, static_cast<int>(1.0 / 4.0 * height) + 1, err_handler);
        rst = decode(bits, false, err_handler);
        if (err_handler.ErrCode() || rst == NULL)
        {
            errMsg = err_handler.ErrMsg();
        }
        else
        {
            return rst;
        }
    }
    err_handler.Reset();
    if (bits->copyOf2(bits2))
    {
        bits->randomFlipRegion(static_cast<int>(3.0 / 8.0 * width), static_cast<int>(3.0 / 8.0 * height), static_cast<int>(1.0 / 4.0 * width) + 1, static_cast<int>(1.0 / 4.0 * height) + 1, err_handler);
        rst = decode(bits, false, err_handler);
        if (err_handler.ErrCode() || rst == NULL)
        {
            errMsg = err_handler.ErrMsg();
        }
        else
        {
            return rst;
        }
    }
    
    err_handler.Reset();
    Ref<DecoderResult> result = decode(bits2, true, err_handler);
    if (err_handler.ErrCode())
    {
        return Ref<DecoderResult>();
    }
    else
    {
        // Success! Notify the caller that the code was mirrored.
        result->setOther(Ref<QRCodeDecoderMetaData>(new QRCodeDecoderMetaData(true)));
        return result;
    }
    
    err_handler = zxing::ReaderErrorHandler("Decoder::decode mirror & no mirror");
    return Ref<DecoderResult>();
};

Ref<DecoderResult> Decoder::decode(Ref<BitMatrix> bits, bool isMirror, ErrorHandler & err_handler)
{
    // Construct a parser and read version, error-correction level
    BitMatrixParser parser(bits, err_handler);
    if (err_handler.ErrCode())   return Ref<DecoderResult>();
    
    if (isMirror == true)
    {
        // Revert the bit matrix
        parser.remask();
        
        // Will be attempting a mirrored reading of the version and format info.
        parser.setMirror(true);
        
        // Preemptively read the version.
        parser.readVersion(err_handler);
        if (err_handler.ErrCode())
        {
            err_handler = zxing::ReaderErrorHandler("Decoder::decode mirror & no mirror");
            return Ref<DecoderResult>();
        }
        
        // Preemptively read the format information.
        parser.readFormatInformation(err_handler);
        if (err_handler.ErrCode())  return Ref<DecoderResult>();
        
        /*
         * Since we're here, this means we have successfully detected some kind
         * of version and format information when mirrored. This is a good sign,
         * that the QR code may be mirrored, and we should try once more with a
         * mirrored content.
         */
        // Prepare for a mirrored reading.
        parser.mirror();
    }
    
    decoderState_ = START; possibleFix_ = 0;
    Version *version = parser.readVersion(err_handler);
    if (err_handler.ErrCode() || version == NULL)
    {
        err_handler = ReaderErrorHandler("Decoder::decode mirror & no mirror");
        return Ref<DecoderResult>();
    }
    
    decoderState_ = READVERSION;
    float fixedPatternScore = estimateFixedPattern(bits, version, err_handler);
    if (err_handler.ErrCode())
    {
        return Ref<DecoderResult>();
    }
    
    Ref<FormatInformation> formatInfo = parser.readFormatInformation(err_handler);
    if (err_handler.ErrCode())
    {
        return Ref<DecoderResult>();
    }
    ErrorCorrectionLevel &ecLevel = formatInfo->getErrorCorrectionLevel();
    
    decoderState_ = READERRORCORRECTIONLEVEL;
    
    // Read codewords
    ArrayRef<char> codewords(parser.readCodewords(err_handler));  // add more tricks by sofiawu
    if (err_handler.ErrCode())
    {
        err_handler.Reset();
        codewords = parser.readCodewords(err_handler);
        err_handler = zxing::ReaderErrorHandler("Decoder::decode mirror & no mirror");
        return Ref<DecoderResult>();
    }
    
    decoderState_ = READCODEWORDSORRECTIONLEVEL;
    possibleFix_ = fixedPatternScore;
    
    // Separate into data blocks
    std::vector<Ref<DataBlock> > dataBlocks(DataBlock::getDataBlocks(codewords, version, ecLevel, err_handler));
    if (err_handler.ErrCode())
    {
        return Ref<DecoderResult>();
    }
    
    // Count total number of data bytes
    int totalBytes = 0;
    for (size_t i = 0; i < dataBlocks.size(); i++) {
        totalBytes += dataBlocks[i]->getNumDataCodewords();
    }
    ArrayRef<char> resultBytes(totalBytes);
    int resultOffset = 0;
    
    // Error-correct and copy data blocks together into a stream of bytes
    for (size_t j = 0; j < dataBlocks.size(); j++)
    {
        err_handler.Reset();
        Ref<DataBlock> dataBlock(dataBlocks[j]);
        ArrayRef<char> codewordBytes = dataBlock->getCodewords();
        int numDataCodewords = dataBlock->getNumDataCodewords();
        
        correctErrors(codewordBytes, numDataCodewords, err_handler);
        if (err_handler.ErrCode())
        {
            return Ref<DecoderResult>();
        }
        
        for (int i = 0; i < numDataCodewords; i++)
        {
            resultBytes[resultOffset++] = codewordBytes[i];
        }
    }
    
    decoderState_ = FINISH;
    DecodedBitStreamParser dbs_parser;
    Ref<DecoderResult> rst = dbs_parser.decode(resultBytes,
                                               version,
                                               ecLevel,
                                               DecodedBitStreamParser::Hashtable(),
                                               err_handler,
                                               version->getVersionNumber());
    
    if (err_handler.ErrCode())
    {
        return Ref<DecoderResult>();
    }
    
    return rst;
}

// Given data and error-correction codewords received, possibly corrupted by errors, attempts to
// correct the errors in-place using Reed-Solomon error correction.</p>
// codewordBytes: data and error correction codewords
// numDataCodewords: number of codewords that are data bytes
void Decoder::correctErrors(ArrayRef<char> codewordBytes, int numDataCodewords, ErrorHandler & err_handler)
{
    // First read into an arrya of ints
    int numCodewords = codewordBytes->size();
    ArrayRef<int> codewordInts(numCodewords);
    for (int i = 0; i < numCodewords; i++)
    {
        codewordInts[i] = codewordBytes[i] & 0xff;
    }
    int numECCodewords = numCodewords - numDataCodewords;
    
    rsDecoder_.decode(codewordInts, numECCodewords, err_handler);
    if (err_handler.ErrCode())
    {
        return;
    }
    
    // Copy back into array of bytes -- only need to worry about the bytes that were data
    // We don't care about errors in the error-correction codewords
    {
        for (int i = 0; i < numDataCodewords; i++)
        {
            codewordBytes[i] = static_cast<char>(codewordInts[i]);
        }
    }
}

unsigned int Decoder::getPossibleVersion()
{
    return possibleVersion_;
} 

float Decoder::estimateFixedPattern(Ref<BitMatrix> bits, zxing::qrcode::Version * version, ErrorHandler & err_handler)
{
    Ref<BitMatrix> fixedPatternValue = version->buildFixedPatternValue(err_handler);
    if (err_handler.ErrCode()){
        err_handler = zxing::ReaderErrorHandler("Decoder::decode mirror & no mirror");
        return -1.0;
    }
    
    Ref<BitMatrix> fixedPatternTemplate = version->buildFixedPatternTemplate(err_handler);
    if (err_handler.ErrCode())
    {
        err_handler = zxing::ReaderErrorHandler("Decoder::decode mirror & no mirror");
        return -1.0;
    }
    
    int iSum = 0;
    int iCount = 0;
    for (int i = 0; i < bits->getHeight(); ++i)
    {
        for (int j = 0; j < bits->getWidth(); ++j)
        {
            if (fixedPatternTemplate->get(i, j))
            {
                iSum++;
                if (bits->get(i, j) == fixedPatternValue->get(i, j)) iCount++;
            }
        }
    }
    
    if (iSum == 0){
        err_handler = zxing::ReaderErrorHandler("Decoder::isum is zero");
        return -1.0;
    }
    
    float possbielFix = 2.0 * iCount / iSum - 1;
    return possbielFix > 0 ? possbielFix : 0;
}
