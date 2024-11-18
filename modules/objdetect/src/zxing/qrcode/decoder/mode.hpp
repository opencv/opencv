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
#ifndef __ZXING_QRCODE_DECODER_MODE_HPP__
#define __ZXING_QRCODE_DECODER_MODE_HPP__

/*
 *  Mode.hpp
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

#include "../../common/counted.hpp"
#include "../version.hpp"
#include "../../error_handler.hpp"

namespace zxing {
namespace qrcode {

class Mode {
private:
    int characterCountBitsForVersions0To9_;
    int characterCountBitsForVersions10To26_;
    int characterCountBitsForVersions27AndHigher_;
    int bits_;
    std::string name_;
    
    Mode(int cbv0_9, int cbv10_26, int cbv27, int bits, char const* name);
    
public:
    static Mode TERMINATOR;
    static Mode NUMERIC;
    static Mode ALPHANUMERIC;
    static Mode STRUCTURED_APPEND;
    static Mode BYTE;
    static Mode ECI;
    static Mode KANJI;
    static Mode FNC1_FIRST_POSITION;
    static Mode FNC1_SECOND_POSITION;
    static Mode HANZI;
    
    static Mode& forBits(int bits, ErrorHandler & err_handler);
   
    int getCharacterCountBits(Version *version)const;
    int getBits() const;
    std::string getName() const;
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __MODE_H__
