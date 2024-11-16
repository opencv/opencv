// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
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
#include "multi_format_one_dreader.hpp"
#include "multi_format_upceanreader.hpp"
#include "code39reader.hpp"
#include "code128reader.hpp"
#include "code93reader.hpp"
#include "code25reader.hpp"
#include "coda_bar_reader.hpp"
#include "itfreader.hpp"
#include "../reader_exception.hpp"
#include "../not_found_exception.hpp"

using zxing::Ref;
using zxing::Result;
using zxing::oned::MultiFormatOneDReader;

// VC++
using zxing::DecodeHints;
using zxing::BitArray;

// changoran-20160102- add oned reader base on hints 
MultiFormatOneDReader::MultiFormatOneDReader(DecodeHints hints) : readers() {
    if (hints.containsFormat(BarcodeFormat::EAN_13) ||
        hints.containsFormat(BarcodeFormat::EAN_8) ||
        hints.containsFormat(BarcodeFormat::UPC_A) ||
        hints.containsFormat(BarcodeFormat::UPC_E)) {
        readers.push_back(Ref<OneDReader>(new MultiFormatUPCEANReader(hints)));
    }
    
    if (hints.containsFormat(BarcodeFormat::CODE_93)) {
        readers.push_back(Ref<OneDReader>(new Code93Reader()));
    }
    
    if (hints.containsFormat(BarcodeFormat::CODABAR)) {
        readers.push_back(Ref<OneDReader>(new CodaBarReader()));
    }
    if (hints.containsFormat(BarcodeFormat::CODE_39)) {
        readers.push_back(Ref<OneDReader>(new Code39Reader()));
    }
    if (hints.containsFormat(BarcodeFormat::CODE_128)) {
        readers.push_back(Ref<OneDReader>(new Code128Reader()));
    }
    
    // ITF === CODE_25 , so we not use it
    // if (hints.containsFormat(BarcodeFormat::ITF)) {
    // Some mistake in ITF, preserve for later check : valiantliu
    // readers.push_back(Ref<OneDReader>(new ITFReader()));
    // }
    
    if (hints.containsFormat(BarcodeFormat::CODE_25)) {
        readers.push_back(Ref<OneDReader>(new Code25Reader()));
    }
    
    /*
     if (hints.containsFormat(BarcodeFormat::RSS_14)) {
     readers.push_back(Ref<OneDReader>(new RSS14Reader()));
     }
     */
    /*
     if (hints.containsFormat(BarcodeFormat::RSS_EXPANDED)) {
     readers.push_back(Ref<OneDReader>(new RSS14ExpandedReader()));
     }
     */
    
    if (readers.size() == 0) {
        readers.push_back(Ref<OneDReader>(new MultiFormatUPCEANReader(hints)));
        readers.push_back(Ref<OneDReader>(new Code39Reader()));
        readers.push_back(Ref<OneDReader>(new CodaBarReader()));
        readers.push_back(Ref<OneDReader>(new Code93Reader()));
        readers.push_back(Ref<OneDReader>(new Code128Reader()));
        
        // Some mistake in ITF, preserve for later check : valiantliu
        // readers.push_back(Ref<OneDReader>(new ITFReader()));
        // readers.push_back(Ref<OneDReader>(new RSS14Reader()));
        // readers.push_back(Ref<OneDReader>(new RSS14ExpandedReader()));
        readers.push_back(Ref<OneDReader>(new Code25Reader()));
    }
    
    //  -- Add Reader Data : Start
    readerData = new ONED_READER_DATA;
    
    readerData->all_counters = std::vector<int>(0);
    readerData->all_counters_offsets = std::vector<int>(0);
    readerData->first_is_white = false;
    readerData->counter_size = 0;
    
    readerData->ean13_checked = false;
    readerData->ean13_lg_pattern_found = 0;
    readerData->ean13_decode_middle_middle_offset = 0;
    readerData->ean13_decode_middle_final_offset = 0;
    readerData->ean13_decode_middle_middle_string = "";
    readerData->ean13_decode_middle_final_string = "";
    
    for (unsigned int k=0; k < readers.size(); k++)
    {
        readers[k]->setData(readerData);
    }
    
    _onedReaderData = readerData;
}

MultiFormatOneDReader::~MultiFormatOneDReader()
{
    delete readerData;
}

#include <typeinfo>

Ref<Result> MultiFormatOneDReader::decodeRow(int rowNumber, Ref<BitArray> row) {
    
    // Just need to check
    if (bNeedCheck == true)
    {
        return decodeRow(rowNumber, row, _lastReaderIdx);
    }
    
    int size = readers.size();
    
    for (int i = 0; i < size; i++) {
        OneDReader* reader = readers[i];
        
        Ref<Result> result = reader->decodeRow(rowNumber, row);
        
        if (result==NULL) {
            continue;
        }
        
        _lastReaderIdx = i;
        return result;
    }
    
    return Ref<Result>(NULL);
}

Ref<Result> MultiFormatOneDReader::decodeRow(int rowNumber, Ref<BitArray> row, int readerIdx) {
    
    OneDReader* reader = readers[readerIdx];
    
    Ref<Result> result = reader->decodeRow(rowNumber, row);
    
    _lastReaderIdx = readerIdx;
    return result;
    
    return Ref<Result>(NULL);
}
