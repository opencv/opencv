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
 *  QRCodeReader.cpp
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


#include "qrcode_reader.hpp"
#include "detector/detector.hpp"
#include "../reader_exception.hpp"
#include "../common/bit_array.hpp"
#include <time.h>

#include <set>

#ifdef STRAT_TIMER
#include <windows.h>
#endif

#include "opencv2/core.hpp"

using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

QRCodeReader::QRCodeReader() :decoder_()
{
    readerState_ = QRCodeReader::READER_START;
    detectedDimension_ = -1;
    lastDecodeTime_ = 0;
    lastDecodeID_ = 0;
    decodeID_ = 0;
    lastPossibleAPCount_ = 0;
    possibleAPCount_ = 0;
    lastSamePossibleAPCountTimes_ = 0;
    samePossibleAPCountTimes_ = 0;
    lastRecommendedImageSizeType_ = 0;
    recommendedImageSizeType_ = 0;
    smoothMaxMultiple_ = 40;
    
    possibleModuleSize_ = -1;
}

Ref<Result> QRCodeReader::decode(Ref<BinaryBitmap> image)
{
    return decode(image, DecodeHints::DEFAULT_HINT);
}

Ref<Result> QRCodeReader::decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    reader_call_path_ = "";
    
    // Binarize image using the Histogram Binarized method and be binarized
    ErrorHandler err_handler;
    Ref<BitMatrix> imageBitMatrix = image->getBlackMatrix(err_handler);
    if (err_handler.errCode() || imageBitMatrix == NULL)
        return Ref<Result>();
    
    Ref<Result> rst = decodeMore(image, imageBitMatrix, hints, err_handler);
    if (err_handler.errCode() || rst == NULL)
    {
        reader_call_path_ += "1";  // enter mirro
        
        // black white mirro!!!
        Ref<BitMatrix> invertedMatrix = image->getInvertedMatrix(err_handler);
        if (err_handler.errCode() || invertedMatrix == NULL)
            return Ref<Result>();
        
        Ref<Result> rst_ = decodeMore(image, invertedMatrix, hints, err_handler);
        if (err_handler.errCode() || rst == NULL)
            return Ref<Result>();
        return rst_;
    }
    reader_call_path_ += "0";  // ok
    
    return rst;
}

Ref<Result> QRCodeReader::decodeMore(Ref<BinaryBitmap> image, Ref<BitMatrix> imageBitMatrix, DecodeHints hints, ErrorHandler & err_handler)
{
    if (imageBitMatrix == NULL)
    {
        return Ref<Result>();
    }
    
    nowHints_ = hints;
    std::string ept;
    
    image->m_poUnicomBlock->init();
    image->m_poUnicomBlock->reset(imageBitMatrix);
    
    // detect
    err_handler.reset();
    Ref<Detector> detector(new Detector(imageBitMatrix, image->m_poUnicomBlock));
    detector->detect(hints, err_handler);
    setReaderState(detector->getState());
    if (err_handler.errCode())
    {
        err_handler = zxing::ReaderErrorHandler("error detect");
        ept = err_handler.errMsg();
        reader_call_path_ += "2";  // detect fail
        return Ref<Result>();
    }
    
    int possiblePatternCount = detector->getPossiblePatternCount();
    if (possiblePatternCount <= 0)
    {
        reader_call_path_ += "3";  // get pattern fail
        return Ref<Result>();
    }
    
    for (int i = 0; i < possiblePatternCount; i++)
    {
        Ref<FinderPatternInfo> patternInfo = detector->getFinderPatternInfo(i);
        
        setPatternFix(patternInfo->getPossibleFix());
        if (!hints.getPureBarcode() && patternInfo->getAnglePossibleFix() < 0.6 && i)
            continue;
        
        // find alignment points
        int possibleAlignmentCount = 0;
        possibleAlignmentCount = detector->getPossibleAlignmentCount(i, hints);
        if (possibleAlignmentCount < 0)
            continue;
        
        detectedDimension_ = detector->getDimension(i);
        possibleModuleSize_ = detector->getPossibleModuleSize(i);
        setPossibleAPCountByVersion(detector->getPossibleVersion(i));
        
        std::vector<bool> needTryVariousDeimensions(possibleAlignmentCount, false);
        std::vector<cv::Point2f> fixed_alignment_pts_src, pattern_pts_src, corner_pts_src;
        std::vector<cv::Point2f> fixed_alignment_pts_dst, pattern_pts_dst, corner_pts_dst;
        bool found_pattern_points = false, found_fixed_alignment_points = false, found_corner_points = false;
        for (int j = 0; j < possibleAlignmentCount; j++)
        {
            ArrayRef< Ref<ResultPoint> > points;
            err_handler.reset();
            
            Ref<AlignmentPattern> alignpattern = detector->getAlignmentPattern(i, j);
            
            Ref<DetectorResult> detectorResult = detector->getResultViaAlignment(i, j, detectedDimension_, err_handler);
            if (err_handler.errCode())
            {
                ept = err_handler.errCode();
                setDecoderFix(decoder_.getPossibleFix(), points);
                setReaderState(decoder_.getState());
                
                if ((patternInfo->getPossibleFix() > 0.9 &&  decoder_.getPossibleFix() < 0.1) || detectedDimension_ == Version::MICRO_DEMITION)
                {
                    needTryVariousDeimensions[j] = true;
                }
                continue;
            }
            
            points = detectorResult->getPoints();
            
            Ref<DecoderResult> decoderResult(decoder_.decode(detectorResult->getBits(), err_handler));
            if (err_handler.errCode())
            {
                ept = err_handler.errCode();
                setDecoderFix(decoder_.getPossibleFix(), points);
                setReaderState(decoder_.getState());
                
                if ((patternInfo->getPossibleFix() > 0.9 &&  decoder_.getPossibleFix() < 0.1) || detectedDimension_ == Version::MICRO_DEMITION){
                    needTryVariousDeimensions[j] = true;
                }
                
                // More Try
                bool use_curve_opt = true;
                if (hints.getTryVideo()) {
                    if (hints.getFrameCnt() % 2 == 1 && i < 2 && j < 2) use_curve_opt = true;
                    else use_curve_opt = false;
                }
                if (use_curve_opt)
                {
                    bool is_decode_success = false;
                    
                    if (!found_pattern_points) {
                        detector->getPatternPoints(i, j, detectedDimension_, pattern_pts_src, pattern_pts_dst);
                        found_pattern_points = true;
                    }
                    
                    std::vector<cv::Point2f> flexible_alignment_pts_src;
                    std::vector<cv::Point2f> flexible_alignment_pts_dst;
                    int version= (detectedDimension_ - 21) / 4 + 1;
                    
                    //
                    if (version >= 7) {
                        if (!found_fixed_alignment_points) {
                            detector->getFixedAlignmentPoints(i, detectedDimension_, possibleModuleSize_, fixed_alignment_pts_src, fixed_alignment_pts_dst);
                            found_fixed_alignment_points = true;
                        }
                        detector->getFlexibleAlignmentPoints(i, j, detectedDimension_, possibleModuleSize_, flexible_alignment_pts_src, flexible_alignment_pts_dst);
                        
                        {
                            std::vector<cv::Point2f> pts_src, pts_dst;
                            if (pattern_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), pattern_pts_src.begin(), pattern_pts_src.end());
                                pts_dst.insert(pts_dst.end(), pattern_pts_dst.begin(), pattern_pts_dst.end());
                            }
                            if (fixed_alignment_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), fixed_alignment_pts_src.begin(), fixed_alignment_pts_src.end());
                                pts_dst.insert(pts_dst.end(), fixed_alignment_pts_dst.begin(), fixed_alignment_pts_dst.end());
                            }
                            if (flexible_alignment_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), flexible_alignment_pts_src.begin(), flexible_alignment_pts_src.end());
                                pts_dst.insert(pts_dst.end(), flexible_alignment_pts_dst.begin(), flexible_alignment_pts_dst.end());
                            }
                            
                            // decode
                            err_handler.reset();
                            detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                            if (detectorResult != NULL && err_handler.errCode() == 0) {
                                decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                if (decoderResult != NULL && err_handler.errCode() == 0) {
                                    is_decode_success = true;
                                }
                            }
                            if (!is_decode_success) {
                                err_handler.reset();
                                detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                if (detectorResult != NULL && err_handler.errCode() == 0) {
                                    decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                    if (decoderResult != NULL && err_handler.errCode() == 0) {
                                        is_decode_success = true;
                                    }
                                }
                            }
                        }
                    }
                    
                    bool video_more_try = true;
                    if (hints.getTryVideo() && hints.getFrameCnt() % 5 == 1) video_more_try = false;
                    //
                    if (!is_decode_success && video_more_try) {
                        if (!found_corner_points) {
                            detector->getCornerPoints(i, detectedDimension_, possibleModuleSize_, corner_pts_src, corner_pts_dst);
                            found_corner_points = true;
                        }
                        
                        int segment_len = static_cast<int>(corner_pts_src.size()) / 3;
                        {
                            std::vector<cv::Point2f> pts_src, pts_dst;
                            if (pattern_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), pattern_pts_src.begin(), pattern_pts_src.end());
                                pts_dst.insert(pts_dst.end(), pattern_pts_dst.begin(), pattern_pts_dst.end());
                            }
                            if (corner_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), corner_pts_src.begin(), corner_pts_src.begin() + segment_len);
                                pts_dst.insert(pts_dst.end(), corner_pts_dst.begin(), corner_pts_dst.end());
                            }
                            
                            {
                                err_handler.reset();
                                detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                                if (detectorResult != NULL && err_handler.errCode() == 0) {
                                    decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                    if (decoderResult != NULL && err_handler.errCode() == 0) {
                                        is_decode_success = true;
                                    }
                                }
                            }
                            
                            {
                                if (!is_decode_success) {
                                    err_handler.reset();
                                    detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                    if (detectorResult != NULL && err_handler.errCode() == 0) {
                                        decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                        if (decoderResult != NULL && err_handler.errCode() == 0) {
                                            is_decode_success = true;
                                        }
                                    }
                                }
                            }
                            
                            if (!is_decode_success)
                            {
                                // try different offset
                                if ((!hints.getTryVideo() || (hints.getFrameCnt() % 4 == 1)))
                                {
                                    if (!is_decode_success) {
                                        size_t begin_idx = pattern_pts_src.size();
                                        for (size_t kk = begin_idx; kk < pts_src.size(); kk++) {
                                            pts_src[kk] = corner_pts_src[segment_len + kk - begin_idx];
                                        }
                                        
                                        err_handler.reset();
                                        detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                                        if (detectorResult != NULL && err_handler.errCode() == 0) {
                                            decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                            if (decoderResult != NULL && err_handler.errCode() == 0) {
                                                is_decode_success = true;
                                            }
                                        }
                                    }
                                    if (!is_decode_success) {
                                        err_handler.reset();
                                        detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                        if (detectorResult != NULL && err_handler.errCode() == 0) {
                                            decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                            if (decoderResult != NULL && err_handler.errCode() == 0) {
                                                is_decode_success = true;
                                            }
                                        }
                                    }
                                }
                                
                                // more
                                if (!hints.getTryVideo() && i < 2 && j < 2) {
                                    if (!is_decode_success) {
                                        size_t begin_idx = pattern_pts_src.size();
                                        for (size_t kk = begin_idx; kk < pts_src.size(); kk++) {
                                            pts_src[kk] = corner_pts_src[2 * segment_len + kk - begin_idx];
                                        }
                                        
                                        err_handler.reset();
                                        detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                                        if (detectorResult != NULL && err_handler.errCode() == 0) {
                                            decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                            if (decoderResult != NULL && err_handler.errCode() == 0) {
                                                is_decode_success = true;
                                            }
                                        }
                                    }
                                    if (!is_decode_success) {
                                        err_handler.reset();
                                        detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                        if (detectorResult != NULL && err_handler.errCode() == 0) {
                                            decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                            if (decoderResult != NULL && err_handler.errCode() == 0) {
                                                is_decode_success = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (video_more_try &&  hints.getTryVideo() && hints.getFrameCnt() % 7 == 2) video_more_try = false;
                    if (!is_decode_success && version >= 7 && corner_pts_src.size() > 0 && video_more_try) {
                        std::vector<cv::Point2f> pts_src, pts_dst;
                        int segment_len = static_cast<int>(corner_pts_src.size()) / 3;
                        {
                            if (pattern_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), pattern_pts_src.begin(), pattern_pts_src.end());
                                pts_dst.insert(pts_dst.end(), pattern_pts_dst.begin(), pattern_pts_dst.end());
                            }
                            if (fixed_alignment_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), fixed_alignment_pts_src.begin(), fixed_alignment_pts_src.end());
                                pts_dst.insert(pts_dst.end(), fixed_alignment_pts_dst.begin(), fixed_alignment_pts_dst.end());
                            }
                            if (flexible_alignment_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), flexible_alignment_pts_src.begin(), flexible_alignment_pts_src.end());
                                pts_dst.insert(pts_dst.end(), flexible_alignment_pts_dst.begin(), flexible_alignment_pts_dst.end());
                            }
                            if (corner_pts_src.size() > 0) {
                                pts_src.insert(pts_src.end(), corner_pts_src.begin(), corner_pts_src.begin() + segment_len);
                                pts_dst.insert(pts_dst.end(), corner_pts_dst.begin(), corner_pts_dst.end());
                            }
                        }
                        
                        {
                            err_handler.reset();
                            detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                            if (detectorResult != NULL && err_handler.errCode() == 0) {
                                decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                if (decoderResult != NULL && err_handler.errCode() == 0) {
                                    is_decode_success = true;
                                }
                            }
                        }
                        
                        {
                            if (!is_decode_success) {
                                err_handler.reset();
                                detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                if (detectorResult != NULL && err_handler.errCode() == 0) {
                                    decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                    if (decoderResult != NULL && err_handler.errCode() == 0) {
                                        is_decode_success = true;
                                    }
                                }
                            }
                        }
                        
                        if (!is_decode_success)
                        {
                            // try different offset
                            if (!hints.getTryVideo() || (hints.getFrameCnt() % 6 == 3))
                            {
                                if (!is_decode_success) {
                                    size_t begin_idx = pattern_pts_src.size() + fixed_alignment_pts_src.size() + flexible_alignment_pts_src.size();
                                    
                                    for (size_t kk = begin_idx; kk < pts_src.size(); kk ++) {
                                        pts_src[kk] = corner_pts_src[segment_len + kk - begin_idx];
                                    }
                                    err_handler.reset();
                                    detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                                    if (detectorResult != NULL && err_handler.errCode() == 0) {
                                        decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                        if (decoderResult != NULL && err_handler.errCode() == 0) {
                                            is_decode_success = true;
                                        }
                                    }
                                }
                                if (!is_decode_success) {
                                    err_handler.reset();
                                    detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                    if (detectorResult != NULL && err_handler.errCode() == 0) {
                                        decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                        if (decoderResult != NULL && err_handler.errCode() == 0) {
                                            is_decode_success = true;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // more
                        if (!hints.getTryVideo() && i < 2 && j < 2) {
                            if (!is_decode_success) {
                                size_t begin_idx = pattern_pts_src.size() + fixed_alignment_pts_src.size() + flexible_alignment_pts_src.size();
                                
                                for (size_t kk = begin_idx; kk < pts_src.size(); kk ++) {
                                    pts_src[kk] = corner_pts_src[2 * segment_len + kk - begin_idx];
                                }
                                err_handler.reset();
                                detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, true, err_handler);
                                if (detectorResult != NULL && err_handler.errCode() == 0) {
                                    decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                    if (decoderResult != NULL && err_handler.errCode() == 0) {
                                        is_decode_success = true;
                                    }
                                }
                            }
                            if (!is_decode_success) {
                                err_handler.reset();
                                detectorResult = detector->getResultViaPoints(hints, detectedDimension_, pts_src, pts_dst, false, err_handler);
                                if (detectorResult != NULL && err_handler.errCode() == 0) {
                                    decoderResult = decoder_.decode(detectorResult->getBits(), err_handler);
                                    if (decoderResult != NULL && err_handler.errCode() == 0) {
                                        is_decode_success = true;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (err_handler.errCode() || detectorResult == NULL || decoderResult == NULL)
                    continue;
            }
            
            // If the code was mirrored: swap the bottom-left and the top-right points.
            if (decoderResult->getOtherClassName() == "QRCodeDecoderMetaData")
            {
                decoderResult->getOther()->applyMirroredCorrection(points);
            }
            
            setDecoderFix(decoder_.getPossibleFix(), points);
            setReaderState(decoder_.getState());
            
            Ref<Result> result(new Result(decoderResult->getText(),
                                           decoderResult->getRawBytes(),
                                           points, BarcodeFormat::QR_CODE,
                                           decoderResult->getCharset(),
                                           decoderResult->getQRCodeVersion(),
                                           decoderResult->getEcLevel(),
                                           decoderResult->getCharsetMode()
                                          ));
            setSuccFix(points);
            
            return result;
        }
    }
    return Ref<Result>();
}

std::vector<int> QRCodeReader::getPossibleDimentions(int detectDimension)
{
    std::vector<int> possibleDimentions;
    possibleDimentions.clear();
    
    if (detectDimension < 0)
    {
        return possibleDimentions;
    }
    
    possibleDimentions.push_back(detectDimension);
    
    if (detectDimension <= 169 &&detectDimension >= 73)
    {
        possibleDimentions.push_back(detectDimension + 4);
        possibleDimentions.push_back(detectDimension - 4);
        possibleDimentions.push_back(detectDimension - 8);
        possibleDimentions.push_back(detectDimension + 8);
    }
    else if (detectDimension <= 69 &&detectDimension >= 45)
    {
        possibleDimentions.push_back(detectDimension + 4);
        possibleDimentions.push_back(detectDimension - 4);
    }
    
    if (detectDimension == 19)
    {
        possibleDimentions.push_back(21);
    }
    
    return possibleDimentions;
}

void QRCodeReader::setPossibleAPCountByVersion(unsigned int version)
{
    if (version < 2)
        possibleAPCount_ = 0;
    else if (version < 7)
        possibleAPCount_ = 1;
    else if (version < 14)
        possibleAPCount_ = 2;
    else if (version < 21)
        possibleAPCount_ = 3;
    else if (version < 28)
        possibleAPCount_ = 4;
    else if (version < 35)
        possibleAPCount_ = 5;
    else
        possibleAPCount_ = 6;
}

float QRCodeReader::getPossibleFix()
{
    return possibleQrcodeInfo_.possibleFix;
} 

int QRCodeReader::smooth(unsigned int *integral, Ref<BitMatrix> input, Ref<BitMatrix> output, int window)
{
    BitMatrix &imatrix = *input;
    BitMatrix &omatrix = *output;
    window >>= 1;
    int count = 0;
    int width = input->getWidth();
    int height = input->getHeight();
    
    int bitsize=imatrix.getRowBitsSize();
    
    bool* jrowtoset = new bool[bitsize];
    
    bool* jrow = NULL;
    
    jrow = NULL;
    
    unsigned int size = window * window;
    
    for (int j = (window + 1); j < (height - 1 - window); ++j)
    {
        int y1 = j - window - 1;
        int y2 = j + window;
        
        int offset1 = y1*width;
        int offset2 = y2*width;
        
        jrow = imatrix.getRowBoolPtr(j);
        
        memcpy(jrowtoset, jrow, bitsize*sizeof(bool));
        
        for (int i = (window + 1); i < (width - 1 - window); ++i){
            int x1 = i - window - 1;
            int x2 = i + window;
            
            unsigned int sum = integral[offset2 + x2] -
            integral[offset2 + x1] +
            integral[offset1 + x2] -
            integral[offset1 + x1];
            
            bool b = jrow[i];
            bool result;
            // the middle 1/3 contains informations of corner, these informations is useful for finding the finder pattern
            int sum3 = 3*sum;
            if ((unsigned int)sum3 <= size)
            {
                result = false;
            }
            else if ((unsigned int)sum3 >= size * 2)
            {
                result = true;
            }
            else
            {
                result = b;
            }
            
            if (result)
            {
                jrowtoset[i] = true;
            }
            count += (result ^ b) == 1 ? 1U : 0U;
        }
        omatrix.setRowBool(j, jrowtoset);
    }
    
    delete [] jrowtoset;
    return count;
} 

void QRCodeReader::initIntegralOld(unsigned int *integral, Ref<BitMatrix> input){
    BitMatrix &matrix = *input;
    int width = input->getWidth();
    int height = input->getHeight();
    
    bool* therow = NULL;
    
    therow = matrix.getRowBoolPtr(0);
    
    
    integral[0] = therow[0];
    
    int* s = new int[width];
    
    memset(s, 0, width*sizeof(int));
    
    integral[0]= therow[0];
    for (int j = 1; j < width; j++){
        integral[j] = integral[j - 1] + therow[j];
        s[j] += therow[j];
    }
    
    int offset = width;
    unsigned int prevSum = 0;
    
    for (int i=1; i < height; i++) {
        offset = i*width;
        therow = matrix.getRowBoolPtr(i);
        
        integral[offset] = integral[offset - width] + therow[0];
        offset++;
        
        for (int j=1; j < width; j++) {
            s[j]+=therow[j];
            integral[offset] = prevSum+s[j];
            prevSum = integral[offset];
            offset++;
        }
    }
    
    delete [] s;
    
    return;
}

void QRCodeReader::initIntegral(unsigned int *integral, Ref<BitMatrix> input){
    BitMatrix &matrix = *input;
    int width = input->getWidth();
    int height = input->getHeight();
    
    bool* therow = NULL;
    
    therow = matrix.getRowBoolPtr(0);
    
    // first row only
    int rs = 0;
    for (int j = 0; j < width; j++)
    {
        rs += therow[j];
        integral[j] = rs;
    }
    
    // remaining cells are sum above and to the left
    int offset = 0;
    
    for (int i=1; i < height; ++i)
    {
        therow = matrix.getRowBoolPtr(i);
        
        rs = 0;
        
        offset += width;
        
        for (int j = 0; j < width; ++j)
        {
            rs += therow[j];
            integral[offset+j] = rs + integral[offset-width+j];
        }
    }
    
    
    return;
}

int QRCodeReader::getRecommendedImageSizeTypeInteral()
{
    if (time(0) - lastDecodeTime_ > 30)
        recommendedImageSizeType_ = 0;
    return recommendedImageSizeType_;
}

unsigned int QRCodeReader::getDecodeID()
{
    return decodeID_;
} 

void QRCodeReader::setDecodeID(unsigned int id)
{
    lastDecodeTime_ = time(0);
    
    decodeID_ = id;
    if (decodeID_ != lastDecodeID_)
    {
        lastDecodeID_ = decodeID_;
        lastPossibleAPCount_ = possibleAPCount_;
        lastSamePossibleAPCountTimes_ = samePossibleAPCountTimes_;
        lastRecommendedImageSizeType_ = getRecommendedImageSizeTypeInteral();
        possibleAPCount_ = 0;
        recommendedImageSizeType_ = 0;
    }
} 

QRCodeReader::~QRCodeReader() {
}

Decoder& QRCodeReader::getDecoder() {
    return decoder_;
}

unsigned int QRCodeReader::getPossibleAPType()
{
    int version=(detectedDimension_ - 21) / 4 + 1;
    setPossibleAPCountByVersion(version);
    return possibleAPCount_;
}
int QRCodeReader::getPossibleFixType()
{
    return possibleQrcodeInfo_.possibleFix > 0.0 ? 1 :0;
}

void QRCodeReader::setPatternFix(float possibleFix)
{
    possibleQrcodeInfo_.patternPossibleFix = possibleFix;
}

void QRCodeReader::setDecoderFix(float possibleFix, ArrayRef< Ref<ResultPoint> > border)
{
    float realFix = possibleFix;
    if (possibleQrcodeInfo_.possibleFix < realFix)
    {
        possibleQrcodeInfo_.possibleFix = realFix;
        possibleQrcodeInfo_.pyramidLev = nowHints_.getPyramidLev();
        possibleQrcodeInfo_.qrcodeBorder.clear();
        possibleQrcodeInfo_.possibleModuleSize = possibleModuleSize_;
        if (border)
        {
            for (int i = 0; i < 4; ++i)
            {
                possibleQrcodeInfo_.qrcodeBorder.push_back(border[i]);
            }
        }
    }
}
void QRCodeReader::setSuccFix(ArrayRef< Ref<ResultPoint> > border)
{
#ifndef CALC_CODE_AREA_SCORE 
    possibleQrcodeInfo_.possibleFix = 1.0;
#endif
    possibleQrcodeInfo_.pyramidLev = nowHints_.getPyramidLev();
    possibleQrcodeInfo_.qrcodeBorder.clear();
    possibleQrcodeInfo_.possibleModuleSize = possibleModuleSize_;
    if (border)
    {
        for (int i = 0; i < 4; ++i)
        {
            possibleQrcodeInfo_.qrcodeBorder.push_back(border[i]);
        }
    }
}

void QRCodeReader::setReaderState(Detector::DetectorState state)
{
    switch (state){
        case Detector::START:
            this->readerState_ = QRCodeReader::DETECT_START;
            break;
        case Detector::FINDFINDERPATTERN:
            this->readerState_ = QRCodeReader::DETECT_FINDFINDERPATTERN;
            break;
        case Detector::FINDALIGNPATTERN:
            this->readerState_ = QRCodeReader::DETECT_FINDALIGNPATTERN;
            break;
    }
    return;
}

void QRCodeReader::setReaderState(Decoder::DecoderState state)
{
    switch (state){
        case Decoder::NOTSTART:
            this->readerState_ = QRCodeReader::DETECT_FAILD;
            break;
        case Decoder::START:
            if (this->readerState_<QRCodeReader::DECODE_START){
                this->readerState_ = QRCodeReader::DECODE_START;
            }
            break;
        case  Decoder::READVERSION:
            if (this->readerState_<QRCodeReader::DECODE_READVERSION){
                this->readerState_ =QRCodeReader::DECODE_READVERSION;
            }
            break;
        case Decoder::READERRORCORRECTIONLEVEL:
            if (this->readerState_<QRCodeReader::DECODE_READERRORCORRECTIONLEVEL){
                this->readerState_ =QRCodeReader::DECODE_READERRORCORRECTIONLEVEL;
            }
            break;
        case  Decoder::READCODEWORDSORRECTIONLEVEL:
            if (this->readerState_< QRCodeReader::DECODE_READCODEWORDSORRECTIONLEVEL){
                this->readerState_ = QRCodeReader::DECODE_READCODEWORDSORRECTIONLEVEL;
            }
            break;
        case Decoder::FINISH:
            if (this->readerState_< QRCodeReader::DECODE_FINISH){
                this->readerState_ = QRCodeReader::DECODE_FINISH;
            }
            break;
    }
    return;
}
int QRCodeReader::getQrcodeInfo(const void * &pQBarQrcodeInfo)
{
    pQBarQrcodeInfo = &possibleQrcodeInfo_;
    return 1;
}
}  // namespace qrcode
}  // namespace zxing
