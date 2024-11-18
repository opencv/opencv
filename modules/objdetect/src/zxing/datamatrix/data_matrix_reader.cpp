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
 *  DataMatrixReader.cpp
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

#include "data_matrix_reader.hpp"
#include "detector/detector.hpp"
#include "../error_handler.hpp"

#include "libdmtx/dmtxdecode.hpp"
#include "libdmtx/dmtxmessage.hpp"

#include <string.h>

#include <iostream>

namespace zxing {
namespace datamatrix {

DataMatrixReader::DataMatrixReader() :
decoder_() {
}

Ref<Result> DataMatrixReader::decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    (void)hints;
    reader_call_path_ = "";
    
    
    ErrorHandler err_handler;
    Ref<BitMatrix> imageBitMatrix=image->getBlackMatrix(err_handler);
    
    Ref<Result> rst = decodeMore(image, imageBitMatrix, hints, err_handler);
    if (err_handler.errCode() || rst == NULL)
    {
        // black white mirro!!!
        err_handler.reset();
        Ref<BitMatrix> invertedMatrix = image->getInvertedMatrix(err_handler);
        if (err_handler.errCode() || invertedMatrix == NULL)   return Ref<Result>();
        Ref<Result> rst_ = decodeMore(image, invertedMatrix, hints, err_handler);
        if (err_handler.errCode() || rst_ == NULL) {
            if (!hints.getTryVideo() && hints.isUseLibdmtx()) {
                Ref<LuminanceSource> gray_img = image->getLuminanceSource();
                dmtx::DmtxDecode dec;
                int ret = dec.dmtxDecodeCreate((unsigned char *)gray_img->getMatrix().data(), gray_img->getWidth(), gray_img->getHeight());
                if (ret == 0)
                {
                    ret = dec.dmtxRegionFindNext();
                    if (ret == 0)
                    {
                        dmtx::DmtxMessage msg;
                        ret = dec.dmtxDecodeMatrixRegion(DmtxUndefined, msg);
                        if (ret == 0 && msg.points.size() == 4)
                        {
                            std::string output = (char*)msg.output;
                            Ref<String> text(new String(output));
                            
                            ArrayRef< Ref<ResultPoint> > points_(4);
                            points_[0] = Ref<ResultPoint>(new ResultPoint(msg.points[0].X, msg.points[0].Y));
                            points_[1] = Ref<ResultPoint>(new ResultPoint(msg.points[1].X, msg.points[1].Y));
                            points_[2] = Ref<ResultPoint>(new ResultPoint(msg.points[2].X, msg.points[2].Y));
                            points_[3] = Ref<ResultPoint>(new ResultPoint(msg.points[3].X, msg.points[3].Y));
                            Ref<Result> result(new Result(text, ArrayRef<char>(), points_, BarcodeFormat::DATA_MATRIX));
                            return result;
                        }
                    }
                }
            }
            
            return Ref<Result>();
        }
        reader_call_path_ += "0";  // ok
        return rst_;
    }
    
    reader_call_path_ += "0";  // ok
    return rst;
}

Ref<Result> DataMatrixReader::decodeMore(Ref<BinaryBitmap> image, Ref<BitMatrix> imageBitMatrix, DecodeHints hints, ErrorHandler & err_handler)
{
    (void)image;
    (void)hints;

    Detector detector(imageBitMatrix);
    if (err_handler.errCode()) return Ref<Result>();
    
    Ref<DetectorResult> detectorResult(detector.detect(true, true, err_handler));
    
    if (err_handler.errCode() || detectorResult == NULL)
    {
        reader_call_path_ += "1";  // detect fail
        return Ref<Result>();
    }
    
    ArrayRef< Ref<ResultPoint> > points(detectorResult->getPoints());
    
    Ref<DecoderResult> decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
    if (err_handler.errCode())
    {
        reader_call_path_ += "2";  // decode fail
        return Ref<Result>();
    }
    
    Ref<Result> result(new Result(decoderResult->getText(), decoderResult->getRawBytes(), points, BarcodeFormat::DATA_MATRIX));
    return result;
}

DataMatrixReader::~DataMatrixReader() {
}

}  // namespace datamatrix
}  // namespace zxing
