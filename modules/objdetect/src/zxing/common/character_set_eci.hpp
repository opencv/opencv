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

#ifndef __ZXING_COMMON_CHARACTER_SET_ECI_HPP__
#define __ZXING_COMMON_CHARACTER_SET_ECI_HPP__

/*
 * Copyright 2008-2011 ZXing authors
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

#include <map>
#include "../decode_hints.hpp"

namespace zxing {
namespace common {

class CharacterSetECI : public Counted{
private:
    static std::map<int, zxing::Ref<CharacterSetECI> > VALUE_TO_ECI;
    static std::map<std::string, zxing::Ref<CharacterSetECI> > NAME_TO_ECI;
    static const bool inited;
    static bool init_tables();
    
    int const* const values_;
    char const* const* const names_;
    
    CharacterSetECI(int const* values, char const* const* names);
    
    static void addCharacterSet(int const* value, char const* const* encodingNames);
    
public:
    char const* name() const;
    int getValue() const;
    
    static CharacterSetECI* getCharacterSetECIByValueFind(int value);
    static CharacterSetECI* getCharacterSetECIByName(std::string const & name);
};

}  // namespace common
}  // namespace zxing

#endif  // __ZXING_COMMON_CHARACTER_SET_ECI_HPP__
