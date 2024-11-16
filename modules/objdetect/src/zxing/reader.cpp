/*
 *  Reader.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 13/05/2008.
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

#include "reader.hpp"

namespace zxing {

Reader::~Reader() { }

Ref<Result> Reader::decode(Ref<BinaryBitmap> image) {
    return decode(image, DecodeHints::DEFAULT_HINT);
}

unsigned int Reader::getDecodeID(){
    return 0;
}

void Reader::setDecodeID(unsigned int){
}

float Reader::getPossibleFix() {
    return 0.0;
}

int Reader::getQrcodeInfo(const void * &pQBarQrcodeInfo)
{
    (void)pQBarQrcodeInfo;
    
    return 0;
}

std::string Reader::name() {
    return "unknow";
}

}  // namespace zxing
