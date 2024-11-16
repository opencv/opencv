// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  DecoderResult.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 20/05/2008.
 *  Copyright 2008-2011 ZXing authors All rights reserved.
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

#include "decoder_result.hpp"

using namespace zxing;

DecoderResult::DecoderResult(ArrayRef<char> rawBytes,
                             Ref<String> text,
                             ArrayRef< ArrayRef<char> >& byteSegments,
                             std::string const& ecLevel) :
rawBytes_(rawBytes),
text_(text),
byteSegments_(byteSegments),
ecLevel_(ecLevel) {
    outputCharset_ = "UTF-8";
    otherClassName = "";
    qrcodeVersion_ = -1;
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes,
                             Ref<String> text,
                             ArrayRef< ArrayRef<char> >& byteSegments,
                             std::string const& ecLevel,
                             std::string outputCharset) :
rawBytes_(rawBytes),
text_(text),
byteSegments_(byteSegments),
ecLevel_(ecLevel),
outputCharset_(outputCharset){
    otherClassName = "";
    qrcodeVersion_ = -1;
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes,
                             Ref<String> text,
                             ArrayRef< ArrayRef<char> >& byteSegments,
                             std::string const& ecLevel,
                             std::string outputCharset,
                             int qrcodeVersion,
                             std::string & charsetMode):
rawBytes_(rawBytes),
text_(text),
byteSegments_(byteSegments),
ecLevel_(ecLevel),
outputCharset_(outputCharset),
qrcodeVersion_(qrcodeVersion),
charsetMode_(charsetMode){
    otherClassName = "";
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes,
                             Ref<String> text,
                             ArrayRef< ArrayRef<char> >& byteSegments,
                             std::string const& ecLevel,
                             std::string outputCharset,
                             int saSequence,
                             int saParity,
                             int symbologyModifier,
                             int qrcodeVersion,
                             std::string & charsetMode):
rawBytes_(rawBytes),
text_(text),
byteSegments_(byteSegments),
ecLevel_(ecLevel),
outputCharset_(outputCharset),
structuredAppendSequenceNumber_(saSequence),
structuredAppendParity_(saParity),
symbologyModifier_(symbologyModifier),
qrcodeVersion_(qrcodeVersion),
charsetMode_(charsetMode){
    otherClassName = "";
}


DecoderResult::DecoderResult(ArrayRef<char> rawBytes,
                             Ref<String> text) :
rawBytes_(rawBytes),
text_(text)
{
    outputCharset_ = "UTF-8";
    otherClassName = "";
}

DecoderResult::DecoderResult(ArrayRef<char> rawBytes, Ref<String> text, std::string outputCharset) : 
rawBytes_(rawBytes),
text_(text),
outputCharset_(outputCharset){
    otherClassName = "";
}

ArrayRef<char> DecoderResult::getRawBytes() {
    return rawBytes_;
}

Ref<String> DecoderResult::getText() {
    return text_;
}

std::string DecoderResult::getCharset()
{
    return outputCharset_;
}
