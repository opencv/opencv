// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
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

#include "character_set_eci.hpp"
#include "illegal_argument_exception.hpp"
#include "../format_exception.hpp"

using std::string;

using zxing::common::CharacterSetECI;
using zxing::IllegalArgumentException;

// Fix memory leak : Valiantliu
// https:// github.com/ukeller/zxing-cpp/commit/c632ffe47ca7342f894ae533263be249cbdfd37e
std::map<int, zxing::Ref<CharacterSetECI> > CharacterSetECI::VALUE_TO_ECI;
std::map<std::string, zxing::Ref<CharacterSetECI> > CharacterSetECI::NAME_TO_ECI;

const bool CharacterSetECI::inited = CharacterSetECI::init_tables();

#define ADD_CHARACTER_SET(VALUES, STRINGS) \
{ static int values[] = {VALUES, -1}; \
static char const* strings[] = {STRINGS, 0}; \
addCharacterSet(values, strings); }

#define XC ,

bool CharacterSetECI::init_tables() {
    ADD_CHARACTER_SET(0 XC 2, "Cp437");
    ADD_CHARACTER_SET(1 XC 3, "ISO8859_1" XC "ISO-8859-1");
    ADD_CHARACTER_SET(4, "ISO8859_2" XC "ISO-8859-2");
    ADD_CHARACTER_SET(5, "ISO8859_3" XC "ISO-8859-3");
    ADD_CHARACTER_SET(6, "ISO8859_4" XC "ISO-8859-4");
    ADD_CHARACTER_SET(7, "ISO8859_5" XC "ISO-8859-5");
    ADD_CHARACTER_SET(8, "ISO8859_6" XC "ISO-8859-6");
    ADD_CHARACTER_SET(9, "ISO8859_7" XC "ISO-8859-7");
    ADD_CHARACTER_SET(10, "ISO8859_8" XC "ISO-8859-8");
    ADD_CHARACTER_SET(11, "ISO8859_9" XC "ISO-8859-9");
    ADD_CHARACTER_SET(12, "ISO8859_10" XC "ISO-8859-10");
    ADD_CHARACTER_SET(13, "ISO8859_11" XC "ISO-8859-11");
    ADD_CHARACTER_SET(15, "ISO8859_13" XC "ISO-8859-13");
    ADD_CHARACTER_SET(16, "ISO8859_14" XC "ISO-8859-14");
    ADD_CHARACTER_SET(17, "ISO8859_15" XC "ISO-8859-15");
    ADD_CHARACTER_SET(18, "ISO8859_16" XC "ISO-8859-16");
    ADD_CHARACTER_SET(20, "SJIS" XC "Shift_JIS");
    ADD_CHARACTER_SET(21, "Cp1250" XC "windows-1250");
    ADD_CHARACTER_SET(22, "Cp1251" XC "windows-1251");
    ADD_CHARACTER_SET(23, "Cp1252" XC "windows-1252");
    ADD_CHARACTER_SET(24, "Cp1256" XC "windows-1256");
    ADD_CHARACTER_SET(25, "UnicodeBigUnmarked" XC "UTF-16BE" XC "UnicodeBig");
    ADD_CHARACTER_SET(26, "UTF8" XC "UTF-8");
    ADD_CHARACTER_SET(27 XC 170, "ASCII" XC "US-ASCII");
    ADD_CHARACTER_SET(28, "Big5");
    ADD_CHARACTER_SET(29, "GB18030" XC "GB2312" XC "EUC_CN" XC "GBK");
    ADD_CHARACTER_SET(30, "EUC_KR" XC "EUC-KR");
    return true;
}

#undef XC

CharacterSetECI::CharacterSetECI(int const* values,
                                 char const* const* names)
: values_(values), names_(names) {
    zxing::Ref<CharacterSetECI> this_ref(this);
    for (int const* value = values_; *value != -1; value++) {
        VALUE_TO_ECI[*value] = this_ref;
    }
    for (char const* const* name = names_; *name; name++) {
        NAME_TO_ECI[string(*name)] = this_ref;
    }
}

char const* CharacterSetECI::name() const {
    return names_[0];
}

int CharacterSetECI::getValue() const {
    return values_[0];
}

void CharacterSetECI::addCharacterSet(int const* values, char const* const* names) {
    new CharacterSetECI(values, names);
}

CharacterSetECI* CharacterSetECI::getCharacterSetECIByValueFind(int value) {
    if (value < 0 || value >= 900)
    {
        return zxing::Ref<CharacterSetECI>(0);
    }
    
    std::map<int, zxing::Ref<CharacterSetECI> >::iterator iter;
    iter = VALUE_TO_ECI.find(value);
    
    if (iter != VALUE_TO_ECI.end())
    {
        return iter->second;
    }
    else
    {
        return zxing::Ref<CharacterSetECI>(0);
    }
    return zxing::Ref<CharacterSetECI>(0);
}

CharacterSetECI* CharacterSetECI::getCharacterSetECIByName(string const& name) {
    std::map<std::string, zxing::Ref<CharacterSetECI> >::iterator iter;
    iter = NAME_TO_ECI.find(name);
    
    if (iter != NAME_TO_ECI.end())
    {
        return iter->second;
    }
    else
    {
        return zxing::Ref<CharacterSetECI>(0);
    }
    return zxing::Ref<CharacterSetECI>(0);
}
