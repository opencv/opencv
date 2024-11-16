// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  MultiFormatUPCEANReader.cpp
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

#include "../zxing.hpp"
#include "multi_format_upceanreader.hpp"
#include "ean13reader.hpp"
#include "ean8reader.hpp"
#include "upcereader.hpp"
#include "upcareader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../not_found_exception.hpp"
#include <math.h>

using zxing::NotFoundException;
using zxing::Ref;
using zxing::Result;
using zxing::oned::MultiFormatUPCEANReader;

// VC++
using zxing::DecodeHints;
using zxing::BitArray;

MultiFormatUPCEANReader::MultiFormatUPCEANReader(DecodeHints hints) : readers() {
    if (hints.containsFormat(BarcodeFormat::EAN_13))
    {
        readers.push_back(Ref<UPCEANReader>(new EAN13Reader()));
    }
    else if (hints.containsFormat(BarcodeFormat::UPC_A))
    {
        readers.push_back(Ref<UPCEANReader>(new UPCAReader()));
    }
    if (hints.containsFormat(BarcodeFormat::EAN_8))
    {
        readers.push_back(Ref<UPCEANReader>(new EAN8Reader()));
    }
    if (hints.containsFormat(BarcodeFormat::UPC_E))
    {
        readers.push_back(Ref<UPCEANReader>(new UPCEReader()));
    }
    if (readers.size() == 0)
    {
        readers.push_back(Ref<UPCEANReader>(new EAN13Reader()));
        // UPC-A is covered by EAN-13
        readers.push_back(Ref<UPCEANReader>(new EAN8Reader()));
        readers.push_back(Ref<UPCEANReader>(new UPCEReader()));
    }
}

void MultiFormatUPCEANReader::setData(ONED_READER_DATA* onedReaderData)
{
    for (size_t k = 0; k < readers.size(); k++)
    {
        readers[k]->setData(onedReaderData);
    }
    
    _onedReaderData = onedReaderData;
}


#include <typeinfo>

Ref<Result> MultiFormatUPCEANReader::decodeRow(int rowNumber, Ref<BitArray> row) {
    // Compute this location once and reuse it on multiple implementations
    ErrorHandler err_handler;
    UPCEANReader::Range startGuardPattern = UPCEANReader::findStartGuardPattern(row, _onedReaderData, err_handler);
    if (err_handler.ErrCode()) return Ref<Result>(NULL);
    
    if (startGuardPattern.isValid() == false) {
        return Ref<Result>(NULL);
    }
    
    _onedReaderData->ean13_checked = false;
    
    
#ifdef USE_PRE_BESTMATCH
    UPCEANReader::initbestMatchDigit(row, _onedReaderData);
#endif
    
    for (int i = 0, e = readers.size(); i < e; i++) {
        
        Ref<UPCEANReader> reader = readers[i];
        Ref<Result> result;
        result = reader->decodeRow(rowNumber, row, startGuardPattern);
        
        if (result == NULL) {
            continue;
        }
        
        // Special case: a 12-digit code encoded in UPC-A is identical
        // to a "0" followed by those 12 digits encoded as EAN-13. Each
        // will recognize such a code, UPC-A as a 12-digit string and
        // EAN-13 as a 13-digit string starting with "0".  Individually
        // these are correct and their readers will both read such a
        // code and correctly call it EAN-13, or UPC-A, respectively.
        //
        // In this case, if we've been looking for both types, we'd like
        // to call it a UPC-A code. But for efficiency we only run the
        // EAN-13 decoder to also read UPC-A. So we special case it
        // here, and convert an EAN-13 result to a UPC-A result if
        // appropriate.
        bool ean13MayBeUPCA =
        result->getBarcodeFormat() == BarcodeFormat::EAN_13 &&
        result->getText()->charAt(0) == '0';
        
        // Note: doesn't match Java which uses hints
        
        bool canReturnUPCA = true;
        
        if (ean13MayBeUPCA && canReturnUPCA) {
            // Transfer the metdata across
            Ref<Result> resultUPCA (new Result(result->getText()->substring(1),
                                               result->getRawBytes(),
                                               result->getResultPoints(),
                                               BarcodeFormat::UPC_A));
            // needs java metadata stuff
            return resultUPCA;
        }
        return result;
    }
    
    return Ref<Result>(NULL);
}
