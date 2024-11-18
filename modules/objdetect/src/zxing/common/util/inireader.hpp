// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info:
//
// https:// github.com/benhoyt/inih

#ifndef ____ZXING_COMMON_UTIL_INIREADER_HPP__
#define ____ZXING_COMMON_UTIL_INIREADER_HPP__

#include <map>
#include <string>

// Read an INI file into easy-to-access name/value pairs. (Note that I've gone
// for simplicity here rather than speed, but it should be pretty decent.)
class INIReader
{
public:
    // Construct INIReader and parse given filename. See ini.hpp for more info
    // about the parsing.
    INIReader(const std::string& filename);
    
    // Return the result of ini_parse(), i.e., 0 on success, line number of
    // first error on parse error, or -1 on file open error.
    int parseError() const;
    
    // Get a string value from INI file, returning default_value if not found.
    std::string get(const std::string& section, const std::string& name,
                    const std::string& default_value) const;
    
    // Get an integer (long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    long getInteger(const std::string& section, const std::string& name, long default_value) const;
    
    // Get a real (floating point double) value from INI file, returning
    // default_value if not found or not a valid floating point value
    // according to strtod().
    double getReal(const std::string& section, const std::string& name, double default_value) const;
    
    // Get a boolean value from INI file, returning default_value if not found or if
    // not a valid true/false value. Valid true values are "true", "yes", "on", "1",
    // and valid false values are "false", "no", "off", "0" (not case sensitive).
    bool getBoolean(const std::string& section, const std::string& name, bool default_value) const;
    
private:
    int _error;
    std::map<std::string, std::string> _values;
    static std::string makeKey(const std::string& section, const std::string& name);
    static int valueHandler(void* user, const char* section, const char* name,
                            const char* value);
};

static inline INIReader * GetIniParser()
{
    static INIReader iniReader("./global.ini");
    return &iniReader;
}


#endif  // ____ZXING_COMMON_UTIL_INIREADER_HPP__
