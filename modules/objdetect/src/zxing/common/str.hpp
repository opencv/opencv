// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __STR_H__
#define __STR_H__



/*
 *  Str.hpp
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

#include <string>
#include <iostream>
#include <sstream>
#include "counted.hpp"

#include <stdio.h>
#include <stdlib.h>


namespace zxing {

class String;
std::ostream& operator << (std::ostream& out, String const& s);

class String : public Counted {
private:
    std::string text_;
public:
    explicit String(const std::string &text);
    explicit String(int);
    char charAt(int) const;
    Ref<String> substring(int) const;
    // Added by Valiantliu
    Ref<String> substring(int, int) const;
    const std::string& getText() const;
    int size() const;
    void append(std::string const& tail);
    void append(char c);
    void append(int d);
    void append(Ref<String> str);
    int length() const;
    friend std::ostream& zxing::operator << (std::ostream& out, String const& s);
};

class StrUtil
{
public:
    static std::string COMBINE_STRING(std::string str1, std::string str2);
    static std::string COMBINE_STRING(std::string str1, char c);
    static std::string COMBINE_STRING(std::string str1, int d);
    static Ref<String> COMBINE_STRING(char c1, Ref<String> content, char c2);
    
    template <typename T>
    static std::string numberToString ( T Number);
    
    template <typename T>
    static T stringToNumber (const std::string &Text);
    
    static int indexOf(const char* str, char c);
}; 

}  // namespace zxing

#endif  // QBAR_AI_QBAR_ZXING_COMMON_STR_H_
