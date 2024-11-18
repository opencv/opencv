// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 * Copyright 2010 ZXing authors All rights reserved.
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

#include "pdf417reader.hpp"
#include "detector/detector.hpp"
#include "detector/lines_sampler.hpp"

using zxing::Ref;
using zxing::Result;
using zxing::BitMatrix;
using zxing::pdf417::PDF417Reader;
using zxing::pdf417::detector::Detector;

// VC++
using zxing::ArrayRef;
using zxing::BinaryBitmap;
using zxing::DecodeHints;
using zxing::ErrorHandler;

const int MAX_DECODE_FAIL_RETRY = 100;

Ref<Result> PDF417Reader::decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    Ref<DecoderResult> decoderResult;
    
    reader_call_path_ = "";
    // PDF417  only process pyramid level 0
    if (hints.getPyramidLev() > 0)
    {
        return Ref<Result>();
    }
    
    Detector detector(image);
    Ref<DetectorResult> detectorResult;
    ErrorHandler err_handler;
    err_handler = detector.detect(hints, detectorResult);
    if (err_handler.errCode() || detectorResult == NULL) {
        reader_call_path_ += "1";  // detect fail
        return Ref<Result>();
    }
    
    ArrayRef< Ref<ResultPoint> > points(detectorResult->getPoints());
    
    if (!hints.isEmpty()) {
        Ref<ResultPointCallback> rpcb = hints.getResultPointCallback();
        if (rpcb != NULL) {
            for (int i = 0; i < points->size(); i++) {
                rpcb->foundPossibleResultPoint(*points[i]);
            }
        }
    }
    
    zxing::pdf417::detector::LinesSampler oLinesSampler(detectorResult->getBits(), detectorResult->dimension_);
    
    Ref<BitMatrix> linesGrid;
    err_handler = oLinesSampler.sample(linesGrid);
    if (err_handler.errCode()) {
        reader_call_path_ += "2";  // sample fail
        return Ref<Result>();
    }
    
    if (!linesGrid)
    {
        Ref<BitMatrix> linesMatrixRotate(new BitMatrix(detectorResult->getBits()->getWidth(), detectorResult->getBits()->getHeight(), err_handler));
        if (err_handler.errCode()) return Ref<Result>();
        for (int i = 0; i<detectorResult->getBits()->getHeight(); i++)
        {
            for (int j = 0; j < linesMatrixRotate->getWidth(); j++)
            {
                if (detectorResult->getBits()->get(j, i))
                    linesMatrixRotate->set(j, detectorResult->getBits()->getHeight() - i - 1);
            }
        }
        oLinesSampler.setLineMatrix(linesMatrixRotate);
        
        err_handler = oLinesSampler.sample(linesGrid);
        if (!linesGrid || err_handler.errCode()) {
            err_handler = NotFoundErrorHandler("LinesSampler Faileds!");
            reader_call_path_ += "3";  // sample fail
            return Ref<Result>();
        }
    }
    
    for (int i = 0; i < MAX_DECODE_FAIL_RETRY; ++i)
    {
        // decoderResult = decoder.decode(detectorResult->getBits(), hints);
        ErrorHandler err_handler_;
        decoderResult = decoder.decode(linesGrid, hints, err_handler_);
        if (err_handler_.errCode()) {
            std::string cell_result = "zxing::ReaderException: " + err_handler_.errMsg();
            err_handler_.reset();
            linesGrid = oLinesSampler.getNextPossibleGrid(err_handler_);
            if (err_handler_.errCode()) return Ref<Result>();
            // retry logic
            if (!linesGrid || i == MAX_DECODE_FAIL_RETRY - 1){
                err_handler_ = ReaderErrorHandler(cell_result);
                reader_call_path_ += "4";  // decode fail
                return Ref<Result>();
            }
            continue;
        }
        break;
    }
    
    Ref<Result> r(new Result(decoderResult->getText(), decoderResult->getRawBytes(), points,
                             BarcodeFormat::PDF_417));
    reader_call_path_ += "0";  // ok
    return r;
}

void PDF417Reader::reset() {
}

Ref<BitMatrix> PDF417Reader::extractPureBits(Ref<BitMatrix> image, ErrorHandler & err_handler) {
    ArrayRef<int> leftTopBlack = image->getTopLeftOnBit();
    ArrayRef<int> rightBottomBlack = image->getBottomRightOnBit();
    
    int nModuleSize = moduleSize(leftTopBlack, image, err_handler);
    if (err_handler.errCode()) return Ref<BitMatrix>();
    
    int top = leftTopBlack[1];
    int bottom = rightBottomBlack[1];
    int left = findPatternStart(leftTopBlack[0], top, image, err_handler);
    if (err_handler.errCode()) Ref<BitMatrix>();
    int right = findPatternEnd(leftTopBlack[0], top, image, err_handler);
    if (err_handler.errCode()) Ref<BitMatrix>();
    
    int matrixWidth = (right - left + 1) / nModuleSize;
    int matrixHeight = (bottom - top + 1) / nModuleSize;
    if (matrixWidth <= 0 || matrixHeight <= 0) {
        err_handler = NotFoundErrorHandler("PDF417Reader::extractPureBits: no matrix found!");
        return Ref<BitMatrix>();
    }
    
    // Push in the "border" by half the module width so that we start
    // sampling in the middle of the module. Just in case the image is a
    // little off, this will help recover.
    int nudge = nModuleSize >> 1;
    top += nudge;
    left += nudge;
    
    // Now just read off the bits
    Ref<BitMatrix> bits(new BitMatrix(matrixWidth, matrixHeight, err_handler));
    if (err_handler.errCode()) return Ref<BitMatrix>();
    for (int y = 0; y < matrixHeight; y++) {
        int iOffset = top + y * nModuleSize;
        for (int x = 0; x < matrixWidth; x++) {
            if (image->get(left + x * nModuleSize, iOffset)) {
                bits->set(x, y);
            }
        }
    }
    return bits;
}

int PDF417Reader::moduleSize(ArrayRef<int> leftTopBlack, Ref<BitMatrix> image, ErrorHandler & err_handler) {
    int x = leftTopBlack[0];
    int y = leftTopBlack[1];
    int width = image->getWidth();
    while (x < width && image->get(x, y)) {
        x++;
    }
    if (x == width) {
        err_handler = NotFoundErrorHandler("PDF417Reader::moduleSize: not found!");
        return -1;
    }
    
    int moduleSize = static_cast<int>(((unsigned)(x - leftTopBlack[0])) >> 3);  // We've crossed left first bar, which is 8x
    if (moduleSize == 0) {
        err_handler = NotFoundErrorHandler("PDF417Reader::moduleSize: is zero!");
        return -1;
    }
    
    return moduleSize;
}

int PDF417Reader::findPatternStart(int x, int y, Ref<BitMatrix> image, ErrorHandler & err_handler) {
    int width = image->getWidth();
    int start = x;
    // start should be on black
    int transitions = 0;
    bool black = true;
    while (start < width - 1 && transitions < 8) {
        start++;
        bool newBlack = image->get(start, y);
        if (black != newBlack) {
            transitions++;
        }
        black = newBlack;
    }
    if (start == width - 1) {
        err_handler = NotFoundErrorHandler("PDF417Reader::findPatternStart: no pattern start found!");
        return -1;
    }
    return start;
}

int PDF417Reader::findPatternEnd(int x, int y, Ref<BitMatrix> image, ErrorHandler& err_handler) {
    int width = image->getWidth();
    int end = width - 1;
    // end should be on black
    while (end > x && !image->get(end, y)) {
        end--;
    }
    int transitions = 0;
    bool black = true;
    while (end > x && transitions < 9) {
        end--;
        bool newBlack = image->get(end, y);
        if (black != newBlack) {
            transitions++;
        }
        black = newBlack;
    }
    if (end == x) {
        err_handler = NotFoundErrorHandler("PDF417Reader::findPatternEnd: no pattern end found!");
        return -1;
    }
    return end;
}
