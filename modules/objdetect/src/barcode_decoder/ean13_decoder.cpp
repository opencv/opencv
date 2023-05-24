// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "../precomp.hpp"
#include "ean13_decoder.hpp"

// three digit decode method from https://baike.baidu.com/item/EAN-13

namespace cv {
namespace barcode {

static constexpr size_t EAN13BITS_NUM = 95;
static constexpr size_t EAN13DIGIT_NUM = 13;
// default thought that mat is a matrix after binary-transfer.
/**
* decode EAN-13
* @prama: data: the input array,
* @prama: start: the index of start order, begin at 0, max-value is data.size()-1
* it scan begin at the data[start]
*/
Result Ean13Decoder::decode(const vector<uchar> &data) const
{
    string result;
    char decode_result[EAN13DIGIT_NUM + 1]{'\0'};
    if (data.size() < EAN13BITS_NUM)
    {
        return Result("Wrong Size", BarcodeType::NONE);
    }
    pair<uint, uint> pattern;
    if (!findStartGuardPatterns(data, pattern))
    {
        return Result("Begin Pattern Not Found", BarcodeType::NONE);
    }
    uint start = pattern.second;
    Counter counter(vector<int>{0, 0, 0, 0});
    size_t end = data.size();
    int first_char_bit = 0;
    // [1,6] are left part of EAN, [7,12] are right part, index 0 is calculated by left part
    for (int i = 1; i < 7 && start < end; ++i)
    {
        int bestMatch = decodeDigit(data, counter, start, get_AB_Patterns());
        if (bestMatch == -1)
        {
            return Result("Decode Error", BarcodeType::NONE);
        }
        decode_result[i] = static_cast<char>('0' + bestMatch % 10);
        start = counter.sum + start;
        first_char_bit += (bestMatch >= 10) << i;
    }
    decode_result[0] = static_cast<char>(FIRST_CHAR_ARRAY()[first_char_bit >> 2] + '0');
    // why there need >> 2?
    // first, the i in for-cycle is begin in 1
    // second, the first i = 1 is always
    Counter middle_counter(vector<int>(MIDDLE_PATTERN().size()));
    if (!findGuardPatterns(data, start, true, MIDDLE_PATTERN(), middle_counter, pattern))
    {
        return Result("Middle Pattern Not Found", BarcodeType::NONE);

    }
    start = pattern.second;
    for (int i = 0; i < 6 && start < end; ++i)
    {
        int bestMatch = decodeDigit(data, counter, start, get_A_or_C_Patterns());
        if (bestMatch == -1)
        {
            return Result("Decode Error", BarcodeType::NONE);
        }
        decode_result[i + 7] = static_cast<char>('0' + bestMatch);
        start = counter.sum + start;
    }
    Counter end_counter(vector<int>(BEGIN_PATTERN().size()));
    if (!findGuardPatterns(data, start, false, BEGIN_PATTERN(), end_counter, pattern))
    {
        return Result("End Pattern Not Found", BarcodeType::NONE);
    }
    result = string(decode_result);
    if (!isValid(result))
    {
        return Result("Wrong: " + result.append(string(EAN13DIGIT_NUM - result.size(), ' ')), BarcodeType::NONE);
    }
    return Result(result, BarcodeType::EAN_13);
}

Ean13Decoder::Ean13Decoder()
{
    this->bits_num = EAN13BITS_NUM;
    this->digit_number = EAN13DIGIT_NUM;
}
}
}
