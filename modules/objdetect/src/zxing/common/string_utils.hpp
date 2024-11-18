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

#ifndef __ZXING_COMMON_STRING_UTILS_HPP__
#define __ZXING_COMMON_STRING_UTILS_HPP__

/*
 * Copyright (C) 2010-2011 ZXing authors
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

#include <string>
#include <map>
#include "../decode_hints.hpp"
#include "../zxing.hpp"

namespace zxing {
  namespace common {
    class StringUtils;
  }  // namespace common
}  // namespace zxing

class zxing::common::StringUtils {
private:
  static char const* const PLATFORM_DEFAULT_ENCODING;

public:
  static char const* const ASCII;
  static char const* const SHIFT_JIS;
  static char const* const GB2312;
  static char const* const EUC_JP;
  static char const* const UTF8;
  static char const* const ISO88591;
  static char const* const GBK;  
  static char const* const GB18030;  
  static char const* const BIG5;  

  static const bool ASSUME_SHIFT_JIS;

  typedef std::map<DecodeHintType, std::string> Hashtable;

  static std::string guessEncoding(char* bytes, int length);
  static std::string guessEncoding(char* bytes, int length, Hashtable const& hints);
  static std::string guessEncodingZXing(char* bytes, int length, Hashtable const& hints);

#ifdef USE_UCHARDET
  static std::string guessEncodingUCharDet(char* bytes, int length, Hashtable const& hints);
#endif

  static int is_utf8_special_byte(unsigned char c);
  // static int is_utf8_code(const string& str);
  static int is_utf8_code(char* str, int length);
  static int is_gb2312_code(char* str, int length);
  static int is_big5_code(char* str, int length);
  static int is_gbk_code(char* str, int length);
  static int is_gb18030_code(char* str, int length);
  static int is_gb18030_code_one(char* str, int length);
  static int is_shiftjis_code(char* str, int length);
  static int is_ascii_code(char* str, int length);
  static int shift_jis_to_jis (const unsigned char * may_be_shift_jis, int * jis_first_ptr, int * jis_second_ptr);

  static std::string convertString(const char* rawData, int length, const char* fromCharset, const char* toCharset);
};

#endif // __ZXING_COMMON_STRING_UTILS_HPP__
