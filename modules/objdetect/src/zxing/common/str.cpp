// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  String.cpp
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

#include "str.hpp"
#include <cstring>

using std::string;
using zxing::String;
using zxing::StrUtil;
using zxing::Ref;


String::String(const std::string &text) :
text_(text) {
}

String::String(int capacity) {
    text_.reserve(capacity);
}

const std::string& String::getText() const {
    return text_;
}

char String::charAt(int i) const { return text_[i]; }

int String::size() const { return text_.size(); }

int String::length() const { return text_.size(); }

Ref<String> String::substring(int i) const {
    return Ref<String>(new String(text_.substr(i)));
}

// Added by Valiantliu
Ref<String> String::substring(int start, int end) const {
    return Ref<String>(new String(text_.substr(start, (end - start))));
}

void String::append(const std::string &tail) {
    text_.append(tail);
}

void String::append(char c) {
    text_.append(1, c);
}

void String::append(int d) {
    std::string str = StrUtil::numberToString(d);
    text_.append(str);
}

void String::append(Ref<String> str)
{
    append(str->getText());
}

std::ostream& zxing::operator << (std::ostream& out, String const& s) {
    out << s.text_;
    return out;
}

std::string StrUtil::COMBINE_STRING(std::string str1, std::string str2)
{
    std::string str = str1;
    str += str2;
    return str;
}

std::string StrUtil::COMBINE_STRING(std::string str1, char c)
{
    std::string str = str1;
    str += c;
    return str;
}

string StrUtil::COMBINE_STRING(string str1, int d)
{
    string str = str1;
    str += numberToString(d);
    return str;
}

Ref<String> StrUtil::COMBINE_STRING(char c1, Ref<String> content, char c2)
{
    Ref<String> str(new String(0));
    str->append(c1);
    str->append(content);
    str->append(c2);
    
    return str;
}

template <typename T>
std::string StrUtil::numberToString ( T Number)
{
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

template <typename T>
T StrUtil::stringToNumber (const std::string &Text)
{
    std::istringstream ss(Text);
    T result;
    return ss >> result ? result : 0;
}

int StrUtil::indexOf(const char* str, char c)
{
    int len = strlen(str);
    
    for (int i = 0; i<len; i++)
    {
        if (str[i] == c)
        {
            return i;
        }
    }
    
    return -1;
}
