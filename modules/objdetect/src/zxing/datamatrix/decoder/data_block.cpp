/*
 *  DataBlock.cpp
 *  zxing
 *
 *  Created by Luiz Silva on 09/02/2010.
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

#include "data_block.hpp"
#include "../../common/illegal_argument_exception.hpp"

namespace zxing {
namespace datamatrix {

DataBlock::DataBlock(int numDataCodewords, ArrayRef<char> codewords) :
numDataCodewords_(numDataCodewords), codewords_(codewords) {
}

int DataBlock::getNumDataCodewords() {
    return numDataCodewords_;
}

ArrayRef<char> DataBlock::getCodewords() {
    return codewords_;
}

std::vector<Ref<DataBlock> > DataBlock::getDataBlocks(ArrayRef<char> rawCodewords, Version *version, ErrorHandler & err_handler) {
    // Figure out the number and size of data blocks used by this version and
    // error correction level
    ECBlocks* ecBlocks = version->getECBlocks();
    
    // First count the total number of data blocks
    int totalBlocks = 0;
    std::vector<ECB*> ecBlockArray = ecBlocks->getECBlocks();
    for (size_t i = 0; i < ecBlockArray.size(); i++) {
        totalBlocks += ecBlockArray[i]->getCount();
    }
    
    // Now establish DataBlocks of the appropriate size and number of data codewords
    std::vector<Ref<DataBlock> > result(totalBlocks);
    int numResultBlocks = 0;
    for (size_t j = 0; j < ecBlockArray.size(); j++) {
        ECB *ecBlock = ecBlockArray[j];
        for (int i = 0; i < ecBlock->getCount(); i++) {
            int numDataCodewords = ecBlock->getDataCodewords();
            int numBlockCodewords = ecBlocks->getECCodewords() + numDataCodewords;
            ArrayRef<char> buffer(numBlockCodewords);
            Ref<DataBlock> blockRef(new DataBlock(numDataCodewords, buffer));
            result[numResultBlocks++] = blockRef;
        }
    }
    
    // All blocks have the same amount of data, except that the last n
    // (where n may be 0) have 1 more byte. Figure out where these start.
    int shorterBlocksTotalCodewords = result[0]->codewords_->size();
    int longerBlocksStartAt = result.size() - 1;
    while (longerBlocksStartAt >= 0) {
        int numCodewords = result[longerBlocksStartAt]->codewords_->size();
        if (numCodewords == shorterBlocksTotalCodewords)
        {
            break;
        }
        if (numCodewords != shorterBlocksTotalCodewords + 1)
        {
            err_handler = IllegalArgumentErrorHandler("Data block sizes differ by more than 1");
            return std::vector<Ref<DataBlock> >();
        }
        longerBlocksStartAt--;
    }
    longerBlocksStartAt++;
    
    int shorterBlocksNumDataCodewords = shorterBlocksTotalCodewords - ecBlocks->getECCodewords();
    // The last elements of result may be 1 element longer;
    // first fill out as many elements as all of them have
    int rawCodewordsOffset = 0;
    for (int i = 0; i < shorterBlocksNumDataCodewords; i++) {
        for (int j = 0; j < numResultBlocks; j++) {
            result[j]->codewords_[i] = rawCodewords[rawCodewordsOffset++];
        }
    }
    // Fill out the last data block in the longer ones
    for (int j = longerBlocksStartAt; j < numResultBlocks; j++) {
        result[j]->codewords_[shorterBlocksNumDataCodewords] = rawCodewords[rawCodewordsOffset++];
    }
    // Now add in error correction blocks
    int max = result[0]->codewords_->size();
    for (int i = shorterBlocksNumDataCodewords; i < max; i++) {
        for (int j = 0; j < numResultBlocks; j++) {
            int iOffset = j < longerBlocksStartAt ? i : i + 1;
            result[j]->codewords_[iOffset] = rawCodewords[rawCodewordsOffset++];
        }
    }
    
    if (rawCodewordsOffset != rawCodewords->size())
    {
        err_handler = IllegalArgumentErrorHandler("rawCodewordsOffset != rawCodewords.length");
        return std::vector<Ref<DataBlock> >();
    }
    
    return result;
}

std::vector<Ref<DataBlock> > DataBlock::getDataBlocks_new(ArrayRef<char> rawCodewords, Version *version, ErrorHandler & err_handler) {
    // Figure out the number and size of data blocks used by this version and
    // error correction level
    ECBlocks* ecBlocks = version->getECBlocks();
    
    // First count the total number of data blocks
    int totalBlocks = 0;
    std::vector<ECB*> ecBlockArray = ecBlocks->getECBlocks();
    for (size_t i = 0; i < ecBlockArray.size(); i++) {
        totalBlocks += ecBlockArray[i]->getCount();
    }
    
    // Now establish DataBlocks of the appropriate size and number of data codewords
    std::vector<Ref<DataBlock> > result(totalBlocks);
    int numResultBlocks = 0;
    for (size_t j = 0; j < ecBlockArray.size(); j++) {
        ECB *ecBlock = ecBlockArray[j];
        for (int i = 0; i < ecBlock->getCount(); i++) {
            int numDataCodewords = ecBlock->getDataCodewords();
            int numBlockCodewords = ecBlocks->getECCodewords() + numDataCodewords;
            ArrayRef<char> buffer(numBlockCodewords);
            Ref<DataBlock> blockRef(new DataBlock(numDataCodewords, buffer));
            result[numResultBlocks++] = blockRef;
        }
    }
    
    if (numResultBlocks == 0)
    {
        err_handler = IllegalArgumentErrorHandler("number Result Blocks is zero");
        return std::vector<Ref<DataBlock> >();
    }
    
    // All blocks have the same amount of data, except that the last n
    // (where n may be 0) have 1 more byte. Figure out where these start.
    int longerBlocksTotalCodewords = result[0]->codewords_->size();
    int longerBlocksNumDataCodewords = longerBlocksTotalCodewords - ecBlocks->getECCodewords();
    int shorterBlocksNumDataCodewords = longerBlocksNumDataCodewords - 1;
    
    // The last elements of result may be 1 element longer;
    // first fill out as many elements as all of them have
    int rawCodewordsOffset = 0;
    for (int i = 0; i < shorterBlocksNumDataCodewords; i++) {
        for (int j = 0; j < numResultBlocks; j++) {
            result[j]->codewords_[i] = rawCodewords[rawCodewordsOffset++];
        }
    }
    // Fill out the last data block in the longer ones
    bool specialVersion = version->getVersionNumber() == 24;
    int numLongerBlocks = specialVersion ? 8 : numResultBlocks;
    for (int j = 0; j < numLongerBlocks; j++) {
        result[j]->codewords_[longerBlocksNumDataCodewords - 1] = rawCodewords[rawCodewordsOffset++];
    }
    // Now add in error correction blocks
    int max = result[0]->codewords_->size();
    for (int i = longerBlocksNumDataCodewords; i < max; i++) {
        for (int j = 0; j < numResultBlocks; j++) {
            int jOffset = specialVersion ? (j + 8) % numResultBlocks : j;
            int iOffset = specialVersion && jOffset > 7 ? i - 1 : i;
            result[jOffset]->codewords_[iOffset] = rawCodewords[rawCodewordsOffset++];
        }
    }
    
    if (rawCodewordsOffset != rawCodewords->size())
    {
        err_handler = IllegalArgumentErrorHandler("rawCodewordsOffset != rawCodewords.length");
        return std::vector<Ref<DataBlock> >();
    }
    
    return result;
}

}  // namespace datamatrix
}  // namespace zxing
