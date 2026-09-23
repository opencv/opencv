// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "json_parser.hpp"

#include <cmath>
#include <cstring>

namespace cv { namespace vlm {

// Recursive descent over the document. Nested inside cv::vlm rather than an unnamed
// namespace so that JsonValue can befriend it; every member is defined inline, so nothing
// here gains external linkage.
struct JsonBuilder
{
    // Guards against a hostile or malformed response recursing the stack to death. Core's
    // own parsers grew the same limit (see modules/core/src/persistence*.cpp).
    static const int MAX_DEPTH = 64;

    const char* begin;
    const char* p;
    const char* end;

    explicit JsonBuilder(const std::string& text)
        : begin(text.c_str()), p(text.c_str()), end(text.c_str() + text.size())
    {
    }

    void fail(const char* what) const
    {
        CV_Error(Error::StsParseError,
                 cv::format("vlm: malformed JSON response: %s at offset %d",
                            what, (int)(p - begin)));
    }

    static bool isDigit(char c) { return c >= '0' && c <= '9'; }

    void skipWs()
    {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            p++;
    }

    void expectLiteral(const char* literal)
    {
        const size_t n = strlen(literal);
        if ((size_t)(end - p) < n || strncmp(p, literal, n) != 0)
            fail("unknown literal");
        p += n;
    }

    static void appendUtf8(std::string& out, unsigned cp)
    {
        if (cp < 0x80)
        {
            out += (char)cp;
        }
        else if (cp < 0x800)
        {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000)
        {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
        else
        {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    unsigned readHex4()
    {
        unsigned value = 0;
        for (int k = 0; k < 4; k++)
        {
            if (p >= end)
                fail("truncated \\u escape");
            const char c = *p++;
            unsigned digit = 0;
            if (c >= '0' && c <= '9')
                digit = (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f')
                digit = (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                digit = (unsigned)(c - 'A' + 10);
            else
                fail("bad hex digit in \\u escape");
            value = (value << 4) | digit;
        }
        return value;
    }

    // Providers escape non-ASCII as \uXXXX, and anything outside the BMP arrives as a
    // surrogate pair. An unpaired surrogate becomes U+FFFD so the result is always valid
    // UTF-8 rather than a byte sequence no consumer can decode.
    unsigned readEscapedCodePoint()
    {
        unsigned cp = readHex4();
        if (cp >= 0xD800 && cp <= 0xDBFF)
        {
            if (end - p >= 2 && p[0] == '\\' && p[1] == 'u')
            {
                const char* save = p;
                p += 2;
                const unsigned low = readHex4();
                if (low >= 0xDC00 && low <= 0xDFFF)
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                else
                {
                    p = save;
                    cp = 0xFFFD;
                }
            }
            else
                cp = 0xFFFD;
        }
        else if (cp >= 0xDC00 && cp <= 0xDFFF)
        {
            cp = 0xFFFD;
        }
        return cp;
    }

    void parseString(std::string& out)
    {
        if (p >= end || *p != '"')
            fail("expected a string");
        p++;
        out.clear();
        for (;;)
        {
            if (p >= end)
                fail("unterminated string");
            const char c = *p++;
            if (c == '"')
                return;
            if (c != '\\')
            {
                out += c;
                continue;
            }
            if (p >= end)
                fail("unterminated escape");
            const char escape = *p++;
            switch (escape)
            {
            case '"':  out += '"';  break;
            case '\\': out += '\\'; break;
            case '/':  out += '/';  break;
            case 'b':  out += '\b'; break;
            case 'f':  out += '\f'; break;
            case 'n':  out += '\n'; break;
            case 'r':  out += '\r'; break;
            case 't':  out += '\t'; break;
            case 'u':  appendUtf8(out, readEscapedCodePoint()); break;
            default:   fail("unknown escape");
            }
        }
    }

    // Assembled by hand rather than with strtod, whose decimal separator follows the
    // process locale -- a caller that set LC_NUMERIC to a comma locale would otherwise
    // truncate every fractional value in a response.
    void parseNumber(JsonValue& out)
    {
        bool negative = false;
        if (p < end && (*p == '-' || *p == '+'))
        {
            negative = *p == '-';
            p++;
        }

        if (p >= end || !isDigit(*p))
            fail("expected a digit");

        double mantissa = 0.0;
        while (p < end && isDigit(*p))
            mantissa = mantissa * 10.0 + (double)(*p++ - '0');

        int exponent = 0;
        if (p < end && *p == '.')
        {
            p++;
            if (p >= end || !isDigit(*p))
                fail("expected a digit after '.'");
            while (p < end && isDigit(*p))
            {
                mantissa = mantissa * 10.0 + (double)(*p++ - '0');
                exponent--;
            }
        }

        if (p < end && (*p == 'e' || *p == 'E'))
        {
            p++;
            bool exponentNegative = false;
            if (p < end && (*p == '-' || *p == '+'))
            {
                exponentNegative = *p == '-';
                p++;
            }
            if (p >= end || !isDigit(*p))
                fail("expected a digit in the exponent");
            int magnitude = 0;
            while (p < end && isDigit(*p))
            {
                if (magnitude < 10000)
                    magnitude = magnitude * 10 + (*p - '0');
                p++;
            }
            exponent += exponentNegative ? -magnitude : magnitude;
        }

        out.type_ = JsonValue::NUMBER;
        out.number_ = (negative ? -mantissa : mantissa) * std::pow(10.0, (double)exponent);
    }

    void parseArray(JsonValue& out, int depth)
    {
        p++; // '['
        out.type_ = JsonValue::ARRAY;
        skipWs();
        if (p < end && *p == ']')
        {
            p++;
            return;
        }
        for (;;)
        {
            JsonValue element;
            parseValue(element, depth + 1);
            out.array_.push_back(std::move(element));

            skipWs();
            if (p >= end)
                fail("unterminated array");
            if (*p == ',')
            {
                p++;
                continue;
            }
            if (*p == ']')
            {
                p++;
                return;
            }
            fail("expected ',' or ']'");
        }
    }

    void parseObject(JsonValue& out, int depth)
    {
        p++; // '{'
        out.type_ = JsonValue::OBJECT;
        skipWs();
        if (p < end && *p == '}')
        {
            p++;
            return;
        }
        for (;;)
        {
            skipWs();
            std::string key;
            parseString(key);
            skipWs();
            if (p >= end || *p != ':')
                fail("expected ':' after a member name");
            p++;

            JsonValue value;
            parseValue(value, depth + 1);
            out.members_.push_back(
                std::pair<std::string, JsonValue>(std::move(key), std::move(value)));

            skipWs();
            if (p >= end)
                fail("unterminated object");
            if (*p == ',')
            {
                p++;
                continue;
            }
            if (*p == '}')
            {
                p++;
                return;
            }
            fail("expected ',' or '}'");
        }
    }

    void parseValue(JsonValue& out, int depth)
    {
        if (depth > MAX_DEPTH)
            CV_Error(Error::StsParseError,
                     cv::format("vlm: JSON response nested deeper than %d levels", MAX_DEPTH));

        skipWs();
        if (p >= end)
            fail("unexpected end of document");

        switch (*p)
        {
        case '{': parseObject(out, depth); break;
        case '[': parseArray(out, depth); break;
        case '"': out.type_ = JsonValue::STRING; parseString(out.string_); break;
        case 't': expectLiteral("true");  out.type_ = JsonValue::BOOL; out.bool_ = true;  break;
        case 'f': expectLiteral("false"); out.type_ = JsonValue::BOOL; out.bool_ = false; break;
        case 'n': expectLiteral("null");  out.type_ = JsonValue::NONE; break;
        default:  parseNumber(out); break;
        }
    }
};

const JsonValue& JsonValue::none()
{
    static const JsonValue emptyNode;
    return emptyNode;
}

size_t JsonValue::size() const
{
    if (type_ == ARRAY)
        return array_.size();
    if (type_ == OBJECT)
        return members_.size();
    return 0;
}

const JsonValue& JsonValue::operator[](const std::string& key) const
{
    if (type_ == OBJECT)
    {
        for (size_t i = 0; i < members_.size(); i++)
            if (members_[i].first == key)
                return members_[i].second;
    }
    return none();
}

const JsonValue& JsonValue::operator[](size_t index) const
{
    if (type_ == ARRAY && index < array_.size())
        return array_[index];
    return none();
}

const std::string& JsonValue::asString() const
{
    static const std::string emptyString;
    return type_ == STRING ? string_ : emptyString;
}

int JsonValue::asInt(int fallback) const
{
    if (type_ == NUMBER)
        return cvRound(number_);
    if (type_ == BOOL)
        return bool_ ? 1 : 0;
    return fallback;
}

JsonValue jsonParse(const std::string& text)
{
    JsonBuilder builder(text);

    JsonValue root;
    builder.parseValue(root, 0);
    builder.skipWs();
    if (builder.p != builder.end)
        builder.fail("unexpected trailing content after the top-level value");

    return root;
}

}} // namespace cv::vlm
