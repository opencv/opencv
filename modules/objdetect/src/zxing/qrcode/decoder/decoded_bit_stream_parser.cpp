// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  DecodedBitStreamParser.cpp
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

#include "../../zxing.hpp"
#include "decoded_bit_stream_parser.hpp"
#include "../../common/character_set_eci.hpp"
#include "../../format_exception.hpp"
#include "../../common/string_utils.hpp"
#include <iostream>
#ifndef NO_ICONV_INSIDE
#include <iconv.h>
#endif
#include <iomanip>

// Required for compatibility. 
#undef ICONV_CONST
#define ICONV_CONST const

#ifndef ICONV_CONST
#define ICONV_CONST
#endif

using zxing::ErrorHandler;

// Add this to fix both Mac and Windows compilers
// by Skylook
template<class T>
class sloppy {};

// convert between T** and const T**
template<class T>
class sloppy<T**>
{
    T** t;
public:
    sloppy(T** mt) : t(mt) {}
    sloppy(const T** mt) : t(const_cast<T**>(mt)) {}
    
    operator T** () const { return t; }
    operator const T** () const { return const_cast<const T**>(t); }
};

using namespace zxing;
using namespace zxing::qrcode;
using namespace zxing::common;

const char DecodedBitStreamParser::ALPHANUMERIC_CHARS[] =
{ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', ' ', '$', '%', '*', '+', '-', '.', '/', ':'
};

namespace {
int GB2312_SUBSET = 1;
}  // namespace

void DecodedBitStreamParser::append(std::string &result,
                                    std::string const& in,
                                    const char* src,
                                    ErrorHandler & err_handler) {
    append(result, (char const*)in.c_str(), in.length(), src, err_handler);
}

void DecodedBitStreamParser::append(std::string &result,
                                    const char *bufIn,
                                    size_t nIn,
                                    const char* src,
                                    ErrorHandler & err_handler) {
    (void)src;

    if (nIn == 0) {
        return;
    }
    if (bufIn == NULL) {
        err_handler = zxing::ReaderErrorHandler("error converting characters");
        return;
    }
#ifndef NO_ICONV_INSIDE
    
    iconv_t cd;
    
    cd = iconv_open(StringUtils::UTF8, src);
    
    if (cd == (iconv_t)-1) {
        result.append((const char *)bufIn, nIn);
        return;
    }
    
    const int maxOut = 4 * nIn + 1;
    char* bufOut = new char[maxOut];
    
    ICONV_CONST char *fromPtr = (ICONV_CONST char *)bufIn;
    size_t nFrom = nIn;
    char *toPtr = (char *)bufOut;
    size_t nTo = maxOut;
    
    while (nFrom > 0) {
        size_t oneway = iconv(cd, sloppy<char**>(&fromPtr), &nFrom, sloppy<char**>(&toPtr), &nTo);
        
        if (oneway == (size_t)(-1)) {
            iconv_close(cd);
            delete[] bufOut;
            err_handler = zxing::ReaderErrorHandler("error converting characters");
            return;
        }
    }
    iconv_close(cd);
    
    int nResult = maxOut - nTo;
    bufOut[nResult] = '\0';
    result.append((const char *)bufOut);
    delete[] bufOut;
#else
    result.append((const char *)bufIn, nIn);
#endif
}

void DecodedBitStreamParser::decodeHanziSegment(Ref<BitSource> bits_,
                                                std::string& result,
                                                int count,
                                                ErrorHandler & err_handler) {
    BitSource& bits(*bits_);
    // Don't crash trying to read more bits than we have available.
    if (count * 13 > bits.available())
    {
        err_handler = zxing::FormatErrorHandler("decodeKanjiSegment");
        return;
    }
    
    // Each character will require 2 bytes. Read the characters as 2-byte pairs
    // and decode as GB2312 afterwards
    size_t nBytes = 2 * count;
    char* buffer = new char[nBytes];
    int offset = 0;
    while (count > 0) {
        // Each 13 bits encodes a 2-byte character
        int twoBytes = bits.readBits(13, err_handler);
        if (err_handler.ErrCode())   return;
        int assembledTwoBytes = ((twoBytes / 0x060) << 8) | (twoBytes % 0x060);
        // if (assembledTwoBytes < 0x003BF) {
        // mod by sofiawu
        if (assembledTwoBytes < 0x00A00)
        {
            // In the 0xA1A1 to 0xAAFE range
            assembledTwoBytes += 0x0A1A1;
        }
        else
        {
            // In the 0xB0A1 to 0xFAFE range
            assembledTwoBytes += 0x0A6A1;
        }
        buffer[offset] = static_cast<char>((assembledTwoBytes >> 8) & 0xFF);
        buffer[offset + 1] = static_cast<char>(assembledTwoBytes & 0xFF);
        offset += 2;
        count--;
    }
    
    append(result, buffer, nBytes, StringUtils::GB2312, err_handler);
    if (err_handler.ErrCode()){
        delete [] buffer;
        return;
    }
    
    delete [] buffer;
}

void DecodedBitStreamParser::decodeKanjiSegment(Ref<BitSource> bits, std::string &result, int count, ErrorHandler & err_handler)
{
    // Each character will require 2 bytes. Read the characters as 2-byte pairs
    // and decode as Shift_JIS afterwards
    size_t nBytes = 2 * count;
    char* buffer = new char[nBytes];
    int offset = 0;
    while (count > 0) {
        // Each 13 bits encodes a 2-byte character
        int twoBytes = bits->readBits(13, err_handler);
        if (err_handler.ErrCode())   return;
        int assembledTwoBytes = ((twoBytes / 0x0C0) << 8) | (twoBytes % 0x0C0);
        if (assembledTwoBytes < 0x01F00)
        {
            // In the 0x8140 to 0x9FFC range
            assembledTwoBytes += 0x08140;
        }
        else
        {
            // In the 0xE040 to 0xEBBF range
            assembledTwoBytes += 0x0C140;
        }
        buffer[offset] = static_cast<char>(assembledTwoBytes >> 8);
        buffer[offset + 1] = static_cast<char>(assembledTwoBytes);
        offset += 2;
        count--;
    }
    
    append(result, buffer, nBytes, StringUtils::SHIFT_JIS, err_handler);
    if (err_handler.ErrCode()){
        delete [] buffer;
        return;
    }
    
    delete[] buffer;
}

void DecodedBitStreamParser::decodeByteSegment(Ref<BitSource> bits_,
                                               std::string& result,
                                               int count,
                                               CharacterSetECI* currentCharacterSetECI,
                                               ArrayRef< ArrayRef<char> >& byteSegments,
                                               Hashtable const& hints,
                                               ErrorHandler & err_handler) {
    (void)hints;
    
    BitSource& bits(*bits_);
    
    int available = bits.available();
    // try to repair count data if count data is invalid
    if (count * 8 > available)
    {
        count = (available + 7) / 8;
    }
    int nBytes = count;
    
    ArrayRef<char> bytes_(count);
    char* readBytes = &(*bytes_)[0];
    for (int i = 0; i < count; i++) {
        int readBits = available < 8 ? available : 8;
        readBytes[i] = static_cast<char>(bits.readBits(readBits, err_handler));
    }
    if (err_handler.ErrCode()) return;
    std::string encoding;
    
    if (currentCharacterSetECI == 0)
    {
        // The spec isn't clear on this mode; see
        // section 6.4.5: t does not say which encoding to assuming
        // upon decoding. I have seen ISO-8859-1 used as well as
        // Shift_JIS -- without anything like an ECI designator to
        // give a hint.
        // string guessCharset = StringUtils::guessEncoding(readBytes, count, hints);
        
#ifndef NO_ICONV_INSIDE
        outputCharset = StringUtils::UTF8;
#else
        outputCharset = StringUtils::UTF8;
#endif
        
        encoding = outputCharset;
    }
    else
    {
        encoding = currentCharacterSetECI->name();
    }
    
    append(result, readBytes, nBytes, encoding.c_str(), err_handler);
    if (err_handler.ErrCode())  return;
    
    byteSegments->values().push_back(bytes_);
}

void DecodedBitStreamParser::decodeNumericSegment(Ref<BitSource> bits, std::string &result, int count, ErrorHandler & err_handler) {
    int nBytes = count;
    ArrayRef<char> bytes = new Array<char>(nBytes);
    int i = 0;
    // Read three digits at a time
    while (count >= 3) {
        // Each 10 bits encodes three digits
        if (bits->available() < 10)
        {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        int threeDigitsBits = bits->readBits(10, err_handler);
        if (err_handler.ErrCode()) return;
        if (threeDigitsBits >= 1000)
        {
            std::ostringstream s;
            s << "Illegal value for 3-digit unit: " << threeDigitsBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[threeDigitsBits / 100];
        bytes[i++] = ALPHANUMERIC_CHARS[(threeDigitsBits / 10) % 10];
        bytes[i++] = ALPHANUMERIC_CHARS[threeDigitsBits % 10];
        count -= 3;
    }
    if (count == 2)
    {
        if (bits->available() < 7)
        {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        // Two digits left over to read, encoded in 7 bits
        int twoDigitsBits = bits->readBits(7, err_handler);
        if (err_handler.ErrCode()) return;
        if (twoDigitsBits >= 100)
        {
            std::ostringstream s;
            s << "Illegal value for 2-digit unit: " << twoDigitsBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[twoDigitsBits / 10];
        bytes[i++] = ALPHANUMERIC_CHARS[twoDigitsBits % 10];
    }
    else if (count == 1)
    {
        if (bits->available() < 4)
        {
            err_handler = zxing::ReaderErrorHandler("format exception");
            return;
        }
        // One digit left over to read
        int digitBits = bits->readBits(4, err_handler);
        if (err_handler.ErrCode()) return;
        if (digitBits >= 10)
        {
            std::ostringstream s;
            s << "Illegal value for digit unit: " << digitBits;
            err_handler = zxing::ReaderErrorHandler(s.str().c_str());
            return;
        }
        bytes[i++] = ALPHANUMERIC_CHARS[digitBits];
    }
    append(result, bytes->data(), nBytes, StringUtils::ASCII, err_handler);
    if (err_handler.ErrCode()) return;
}

char DecodedBitStreamParser::toAlphaNumericChar(size_t value, ErrorHandler & err_handler) {
    if (value >= sizeof(DecodedBitStreamParser::ALPHANUMERIC_CHARS))
    {
        err_handler = zxing::FormatErrorHandler("toAlphaNumericChar");
        return 0;
    }
    return ALPHANUMERIC_CHARS[value];
}

void DecodedBitStreamParser::decodeAlphanumericSegment(Ref<BitSource> bits_,
                                                       std::string& result,
                                                       int count,
                                                       bool fc1InEffect,
                                                       ErrorHandler & err_handler) {
    BitSource& bits(*bits_);
    std::ostringstream bytes;
    // Read two characters at a time
    while (count > 1) {
        if (bits.available() < 11)
        {
            err_handler = zxing::FormatErrorHandler("decodeAlphanumericSegment");
            return;
        }
        int nextTwoCharsBits = bits.readBits(11, err_handler);
        bytes << toAlphaNumericChar(nextTwoCharsBits / 45, err_handler);
        bytes << toAlphaNumericChar(nextTwoCharsBits % 45, err_handler);
        if (err_handler.ErrCode())   return;
        count -= 2;
    }
    if (count == 1) {
        // special case: one character left
        if (bits.available() < 6)
        {
            err_handler = zxing::FormatErrorHandler("decodeAlphanumericSegment");
            return;
        }
        bytes << toAlphaNumericChar(bits.readBits(6, err_handler), err_handler);
        if (err_handler.ErrCode())   return;
    }
    // See section 6.4.8.1, 6.4.8.2
    std::string s = bytes.str();
    if (fc1InEffect)
    {
        // We need to massage the result a bit if in an FNC1 mode:
        std::ostringstream r;
        for (size_t i = 0; i < s.length(); i++) {
            if (s[i] != '%')
            {
                r << s[i];
            }
            else
            {
                if (i < s.length() - 1 && s[i + 1] == '%')
                {
                    // %% is rendered as %
                    r << s[i++];
                }
                else
                {
                    // In alpha mode, % should be converted to FNC1 separator 0x1D
                    r << static_cast<char>(0x1D);
                }
            }
        }
        s = r.str();
    }
    append(result, s, StringUtils::ASCII, err_handler);
    if (err_handler.ErrCode())   return;
}

namespace {
int parseECIValue(BitSource& bits, ErrorHandler &err_handler) {
    int firstByte = bits.readBits(8, err_handler);
    if (err_handler.ErrCode())   return 0;
    if ((firstByte & 0x80) == 0)
    {
        // just one byte
        return firstByte & 0x7F;
    }
    if ((firstByte & 0xC0) == 0x80)
    {
        // two bytes
        int secondByte = bits.readBits(8, err_handler);
        if (err_handler.ErrCode())   return 0;
        return ((firstByte & 0x3F) << 8) | secondByte;
    }
    if ((firstByte & 0xE0) == 0xC0)
    {
        // three bytes
        int secondThirdBytes = bits.readBits(16, err_handler);
        if (err_handler.ErrCode())   return 0;
        return ((firstByte & 0x1F) << 16) | secondThirdBytes;
    }
    
    err_handler = zxing::FormatErrorHandler("parseECIValue");
    return 0;
}
}  // namespace 

Ref<DecoderResult>
DecodedBitStreamParser::decode(ArrayRef<char> bytes,
                               Version* version,
                               ErrorCorrectionLevel const& ecLevel,
                               Hashtable const& hints,
                               ErrorHandler & err_handler,
                               int iVersion)
{
    Ref<BitSource> bits_(new BitSource(bytes));
    BitSource& bits(*bits_);
    std::string result;
    result.reserve(50);
    Mode* mode = 0;
    std::string modeName;
    ArrayRef< ArrayRef<char> > byteSegments(0);
    
    int symbolSequence = -1;
    int parityData = -1;
    int symbologyModifier;
    
    CharacterSetECI* currentCharacterSetECI = 0;
    bool fc1InEffect = false;
    // mod by sofiawu
    bool hasFNC1first = false;
    bool hasFNC1second = false;
    
    // Added by valiantliu
    outputCharset = "UTF-8";
    do {
        // While still another segment to read...
        if (bits.available() < 4)
        {
            // OK, assume we're done. Really, a TERMINATOR mode should have been recorded here
            mode = &Mode::TERMINATOR;
        }
        else
        {
            mode = &Mode::forBits(bits.readBits(4, err_handler), err_handler);  // mode is encoded by 4 bits
            if (err_handler.ErrCode())   return Ref<DecoderResult>();
        }
        
        if (mode != &Mode::TERMINATOR)
        {
            if (mode == &Mode::FNC1_FIRST_POSITION)
            {
                hasFNC1first = true;
                fc1InEffect = true;
            }
            else if (mode == &Mode::FNC1_SECOND_POSITION)
            {
                hasFNC1second = true;
                fc1InEffect = true;
            }
            else if (mode == &Mode::STRUCTURED_APPEND)
            {
                if (bits.available() < 16)
                {
                    err_handler = zxing::FormatErrorHandler("decode");
                    return Ref<DecoderResult>();
                }
                // not really supported; all we do is ignore it
                // Read next 8 bits(symbol sequence #) and 8 bits(parity data), then continue
                
                symbolSequence = bits.readBits(8, err_handler);
                if (err_handler.ErrCode()) return Ref<DecoderResult>();
                parityData = bits.readBits(8, err_handler);
    
                if (err_handler.ErrCode()) return Ref<DecoderResult>();
            }
            else if (mode == &Mode::ECI)
            {
                // Count doesn't apply to ECI
                int value = parseECIValue(bits, err_handler);
                if (err_handler.ErrCode()) Ref<DecoderResult>();
                currentCharacterSetECI = CharacterSetECI::getCharacterSetECIByValueFind(value);
                if (currentCharacterSetECI == 0)
                {
                    err_handler = zxing::FormatErrorHandler("decode");
                    return Ref<DecoderResult>();
                }
            }
            else
            {
                // First handle Hanzi mode which does not start with character count
                if (mode == &Mode::HANZI)
                {
                    // chinese mode contains a sub set indicator right after mode indicator
                    int subset = bits.readBits(4, err_handler);
                    int countHanzi = bits.readBits(mode->getCharacterCountBits(version), err_handler);
                    if (err_handler.ErrCode()) return Ref<DecoderResult>();
                    if (subset == GB2312_SUBSET)
                    {
                        decodeHanziSegment(bits_, result, countHanzi, err_handler);
                        if (err_handler.ErrCode())   Ref<DecoderResult>();
                        outputCharset = "GB2312";
                        modeName = mode->getName();
                    }
                }
                else
                {
                    // "Normal" QR code modes:
                    // How many characters will follow, encoded in this mode?
                    int count = bits.readBits(mode->getCharacterCountBits(version), err_handler);
                    if (err_handler.ErrCode()) return Ref<DecoderResult>();
                    
                    if (mode == &Mode::NUMERIC)
                    {
                        decodeNumericSegment(bits_, result, count, err_handler);
                        if (err_handler.ErrCode())
                        {
                            err_handler = zxing::FormatErrorHandler("decode");
                            return Ref<DecoderResult>();
                        }
                        modeName = mode->getName();
                    }
                    else if (mode == &Mode::ALPHANUMERIC)
                    {
                        decodeAlphanumericSegment(bits_, result, count, fc1InEffect, err_handler);
                        if (err_handler.ErrCode()) Ref<DecoderResult>();
                        modeName = mode->getName();
                    }
                    else if (mode == &Mode::BYTE)
                    {
                        decodeByteSegment(bits_, result, count, currentCharacterSetECI, byteSegments, hints, err_handler);
                        if (err_handler.ErrCode())
                        {
                            err_handler = zxing::FormatErrorHandler("decode");
                            return Ref<DecoderResult>();
                        }
                        
                        modeName = mode->getName();
                    }
                    else if (mode == &Mode::KANJI)
                    {
                        decodeKanjiSegment(bits_, result, count, err_handler);
                        if (err_handler.ErrCode()) Ref<DecoderResult>();
                        modeName = mode->getName();
                    }
                    else
                    {
                        err_handler = zxing::FormatErrorHandler("decode");
                        return Ref<DecoderResult>();
                    }
                }
            }
        }
    } while (mode != &Mode::TERMINATOR);
    
    if (currentCharacterSetECI != NULL)
    {
        if (hasFNC1first)
        {
            symbologyModifier = 4;
        }
        else if (hasFNC1second)
        {
            symbologyModifier = 6;
        }
        else
        {
            symbologyModifier = 2;
        }
    }
    else
    {
        if (hasFNC1first)
        {
            symbologyModifier = 3;
        }
        else if (hasFNC1second)
        {
            symbologyModifier = 5;
        }
        else
        {
            symbologyModifier = 1;
        }
    }
    
    return Ref<DecoderResult>(new DecoderResult(bytes,
                                                Ref<String>(new String(result)),
                                                byteSegments,
                                                (std::string)ecLevel,
                                                (std::string)outputCharset,
                                                symbolSequence,
                                                parityData,
                                                symbologyModifier,
                                                iVersion,
                                                modeName));
}
