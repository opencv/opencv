// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 *  Copyright 2010 ZXing authors. All rights reserved.
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

#include "bit_array.hpp"

using std::vector;
using zxing::BitArray;
using zxing::ArrayRef;
using zxing::ErrorHandler;
// VC++
using zxing::Ref;

# if __WORDSIZE == 64
// typedef long int int64_t;
# else
typedef long long int int64_t;
#endif

BitArray::BitArray(int size_)
: size(size_), bits(size_), nextSets(size_), nextUnSets(size_) {}

void BitArray::setUnchar(int i, unsigned char newBits) {
    bits[i] = newBits;
}

bool BitArray::isRange(int start, int end, bool value, ErrorHandler & err_handler) {
    if (end < start)
    {
        err_handler = IllegalArgumentErrorHandler("isRange");
        return false;
    }
    if (start < 0 || end >= bits->size())
    {
        err_handler = IllegalArgumentErrorHandler("isRange");
        return false;
    }
    if (end == start)
    {
        return true;  // empty range matches
    }
    
    bool startBool = static_cast<bool>(bits[start]);
    
    int end2 = start;
    
    if (startBool)
    {
        end2 = getNextUnset(start);
    }
    else
    {
        end2 = getNextSet(start);
    }
    
    if (startBool == value)
    {
        if (end2 < end)
        {
            return false;
        }
    }
    else
    {
        return false;
    }
    
    return true;
}

void BitArray::reverse()
{
    bool* rowBits = getRowBoolPtr();
    bool tempBit;
    
    for (int i = 0; i < size/2; i++)
    {
        tempBit = rowBits[i];
        rowBits[i] = rowBits[size - i -1];
        rowBits[size - i -1] = tempBit;
    }
}

void BitArray::initAllNextSets()
{
    bool* rowBits = getRowBoolPtr();
    
    int* nextSetArray = nextSets->data();
    int* nextUnsetArray = nextUnSets->data();
    
    // Init the last one
    if (rowBits[size-1])
    {
        nextSetArray[size-1] = size-1;
        nextUnsetArray[size-1] = size;
    }
    else
    {
        nextUnsetArray[size-1] = size-1;
        nextSetArray[size-1] = size;
    }
    
    // do inits
    for (int i = size-2; i >= 0; i--) {
        if (rowBits[i])
        {
            nextSetArray[i] = i;
            nextUnsetArray[i] = nextUnsetArray[i+1];
        }
        else
        {
            nextUnsetArray[i] = i;
            nextSetArray[i] = nextSetArray[i+1];
        }
    }
}

void BitArray::initAllNextSetsFromCounters(std::vector<int> counters)
{
    bool* rowBits = getRowBoolPtr();
    bool isWhite = rowBits[0];
    int c = 0;
    int offset = 0;
    int count = 0;
    int prevCount = 0;
    int currCount = 0;
    int size_ = counters.size();
    
    int* nextSetArray = nextSets->data();
    int* nextUnsetArray = nextUnSets->data();
    
    int* countersArray = &counters[0];
    
    while (c < size_)
    {
        currCount = countersArray[c];
        
        count += currCount;
        
        if (isWhite)
        {
            for (int i = 0; i < currCount; i++)
            {
                offset = prevCount+i;
                nextSetArray[offset]=prevCount+i;
                nextUnsetArray[offset]=count;
            }
        }
        else
        {
            for (int i = 0; i < currCount; i++)
            {
                offset = prevCount+i;
                nextSetArray[offset]=count;
                nextUnsetArray[offset]=prevCount+i;
            }
        }
        
        isWhite = !isWhite;
        
        prevCount += currCount;
        
        c++;
    }
}

int BitArray::getNextSet(int from) {
    if (from >= size)
    {
        return size;
    }
    return nextSets[from];
}

int BitArray::getNextUnset(int from) {
    if (from >= size)
    {
        return size;
    }
    return nextUnSets[from];
}

BitArray::~BitArray() {
}

int BitArray::getSize() const {
    return size;
}

void BitArray::clear() {
    int max = bits->size();
    for (int i = 0; i < max; i++) {
        bits[i] = 0;
    }
}

BitArray::Reverse::Reverse(Ref<BitArray> array_) : array(array_) {
    array->reverse();
}

BitArray::Reverse::~Reverse() {
    array->reverse();
}

void BitArray::appendBit(bool value) {
    ArrayRef<unsigned char>newBits(size + 1);
    for (int i = 0; i < size; i++) {
        newBits[i] = bits[i];
    }
    bits = newBits;
    if (value)
    {
        set(size);
    }
    ++size;
}

int BitArray::getSizeInBytes() const {
    return size;
}

// Appends the least-significant bits, from value, in order from
// most-significant to least-significant. For example, appending 6 bits
// from 0x000001E will append the bits 0, 1, 1, 1, 1, 0 in that order.
void BitArray::appendBits(int value, int numBits, ErrorHandler & err_handler) {
    if (numBits < 0 || numBits > 32)
    {
        err_handler = IllegalArgumentErrorHandler("Number of bits must be between 0 and 32");
        return;
    }
    ArrayRef<unsigned char> newBits(size + numBits);
    for (int i = 0; i < size; i++)
        newBits[i]=bits[i];
    bits=newBits;
    for (int numBitsLeft = numBits; numBitsLeft>0; numBitsLeft--){
        if (((value >> (numBitsLeft - 1)) & 0x01) == 1)
        {
            set(size);
        }
        ++size;
    }
    return;
}

void BitArray::appendBitArray(const BitArray& array) {
    ArrayRef<unsigned char> newBits(size + array.getSize());
    for (int i = 0; i < size; ++i) {
        newBits[i] = bits[i];
    }
    bits = newBits;
    for (int i = 0; i < array.getSize(); ++i) {
        if (array.get(i))
        {
            set(size);
        }
        ++size;
    }
}

void BitArray::toBytes(int bitOffset, ArrayRef<int>& array, int offset, int numBytes) {
    for (int i = 0; i < numBytes; i++){
        int theByte = 0;
        if (get(bitOffset)){
            theByte = 1;
        }
        bitOffset++;
        array[offset+i] = theByte;
    }
}

void BitArray::bitXOR(const BitArray& other, ErrorHandler & err_handler) {
    if (size != other.size)
    {
        err_handler = IllegalArgumentErrorHandler("Sizes don't match");
        return;
    }
 
    for (int i = 0; i < bits->size(); i++){
        bits[i] = (bits[i] == other.bits[i]) ? 0: 1;
    }
}
