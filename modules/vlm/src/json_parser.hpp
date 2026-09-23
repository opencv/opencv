// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_JSON_PARSER_HPP
#define OPENCV_VLM_JSON_PARSER_HPP

#include "opencv2/core.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cv { namespace vlm {

/** @brief A read-only node of a parsed JSON document.

Exists because a provider returns a whole page of text as one JSON string, and FileStorage's
JSON reader caps a string at CV_FS_MAX_LEN. Lookups never throw: a missing key, a bad index,
a lookup on the wrong type and an explicit null all give an empty node.
*/
class JsonValue
{
public:
    enum Type
    {
        NONE = 0,  //!< Missing node, or an explicit JSON null.
        BOOL,
        NUMBER,
        STRING,
        ARRAY,
        OBJECT
    };

    JsonValue() : type_(NONE), bool_(false), number_(0.0) {}

    Type type() const { return type_; }

    bool empty() const { return type_ == NONE; }

    //! Element count of an ARRAY or OBJECT; 0 for every other type.
    size_t size() const;

    //! Member of an OBJECT, or an empty node if absent or this is not an OBJECT.
    const JsonValue& operator[](const std::string& key) const;

    //! Element of an ARRAY, or an empty node if out of range or this is not an ARRAY.
    const JsonValue& operator[](size_t index) const;

    //! STRING value, or an empty string for every other type.
    const std::string& asString() const;

    //! NUMBER rounded to int, or BOOL as 0/1; @p fallback for every other type.
    int asInt(int fallback) const;

private:
    friend struct JsonBuilder;

    static const JsonValue& none();

    Type type_;
    bool bool_;
    double number_;
    std::string string_;
    std::vector<JsonValue> array_;
    std::vector<std::pair<std::string, JsonValue> > members_;
};

/** @brief Parses one complete JSON document.

@param text Exactly one JSON value, optionally surrounded by whitespace.
@return The root node. Throws Error::StsParseError on malformed input, on nesting deeper
        than 64 levels, and on trailing content. String length is not limited.
*/
JsonValue jsonParse(const std::string& text);

}} // namespace cv::vlm

#endif // OPENCV_VLM_JSON_PARSER_HPP
