// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
#ifndef __MULTI_FORMAT_UPC_EAN_READER_H__
#define __MULTI_FORMAT_UPC_EAN_READER_H__
/*
 *  MultiFormatUPCEANReader.hpp
 *  ZXing
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

#include "one_dreader.hpp"
#include "upceanreader.hpp"

namespace zxing {
namespace oned {

class UPCEANReader;

class MultiFormatUPCEANReader : public OneDReader {
private:
    std::vector< Ref<UPCEANReader> > readers;
    
public:
    MultiFormatUPCEANReader(DecodeHints hints);
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
    void setData(ONED_READER_DATA* onedReaderData);
};

}  // namespace oned
}  // namespace zxing

#endif
