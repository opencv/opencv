// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "../precomp.hpp"
#include "upcean_decoder.hpp"
#include <map>

namespace cv {
namespace barcode {

static constexpr int DIVIDE_PART = 15;
static constexpr int BIAS_PART = 2;

#if 0
void UPCEANDecoder::drawDebugLine(Mat &debug_img, const Point2i &begin, const Point2i &end) const
{
    Result result;
    std::vector<uchar> middle;
    LineIterator line = LineIterator(debug_img, begin, end);
    middle.reserve(line.count);
    for (int cnt = 0; cnt < line.count; cnt++, line++)
    {
        middle.push_back(debug_img.at<uchar>(line.pos()));
    }
    std::pair<int, int> start_range;
    if (findStartGuardPatterns(middle, start_range))
    {
        circle(debug_img, Point2i(begin.x + start_range.second, begin.y), 2, Scalar(0), 2);
    }
    result = this->decode(middle);
    if (result.format == Result::BARCODE_NONE)
    {
        result = this->decode(std::vector<uchar>(middle.crbegin(), middle.crend()));
    }
    if (result.format == Result::BARCODE_NONE)
    {
        cv::line(debug_img, begin, end, Scalar(0), 2);
        cv::putText(debug_img, result.result, begin, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
    }
}
#endif

bool UPCEANDecoder::findGuardPatterns(const std::vector<uchar> &row, uint rowOffset, uchar whiteFirst,
                                      const std::vector<int> &pattern, Counter &counter, std::pair<uint, uint> &result)
{
    size_t patternLength = pattern.size();
    size_t width = row.size();
    uchar color = whiteFirst ? WHITE : BLACK;
    rowOffset = (int) (std::find(row.cbegin() + rowOffset, row.cend(), color) - row.cbegin());
    uint counterPosition = 0;
    uint patternStart = rowOffset;
    for (uint x = rowOffset; x < width; x++)
    {
        if (row[x] == color)
        {
            counter.pattern[counterPosition]++;
            counter.sum++;
        }
        else
        {
            if (counterPosition == patternLength - 1)
            {
                if (patternMatch(counter, pattern, MAX_INDIVIDUAL_VARIANCE) < MAX_AVG_VARIANCE)
                {
                    result.first = patternStart;
                    result.second = x;
                    return true;
                }
                patternStart += counter.pattern[0] + counter.pattern[1];
                counter.sum -= counter.pattern[0] + counter.pattern[1];

                std::copy(counter.pattern.begin() + 2, counter.pattern.end(), counter.pattern.begin());

                counter.pattern[patternLength - 2] = 0;
                counter.pattern[patternLength - 1] = 0;
                counterPosition--;
            }
            else
            {
                counterPosition++;
            }
            counter.pattern[counterPosition] = 1;
            counter.sum++;
            color = (std::numeric_limits<uchar>::max() - color);
        }
    }
    return false;
}

bool UPCEANDecoder::findStartGuardPatterns(const std::vector<uchar> &row, std::pair<uint, uint> &start_range)
{
    bool is_find = false;
    int next_start = 0;
    while (!is_find)
    {
        Counter guard_counters(std::vector<int>{0, 0, 0});
        if (!findGuardPatterns(row, next_start, BLACK, BEGIN_PATTERN(), guard_counters, start_range))
        {
            return false;
        }
        int start = static_cast<int>(start_range.first);
        next_start = static_cast<int>(start_range.second);
        int quiet_start = max(start - (next_start - start), 0);
        is_find = (quiet_start != start) &&
                  (std::find(std::begin(row) + quiet_start, std::begin(row) + start, BLACK) == std::begin(row) + start);
    }
    return true;
}

int UPCEANDecoder::decodeDigit(const std::vector<uchar> &row, Counter &counters, uint rowOffset,
                               const std::vector<std::vector<int>> &patterns)
{
    fillCounter(row, rowOffset, counters);
    int bestMatch = -1;
    uint bestVariance = MAX_AVG_VARIANCE; // worst variance we'll accept
    int i = 0;
    for (const auto &pattern : patterns)
    {
        uint variance = patternMatch(counters, pattern, MAX_INDIVIDUAL_VARIANCE);
        if (variance < bestVariance)
        {
            bestVariance = variance;
            bestMatch = i;
        }
        i++;
    }
    return std::max(-1, bestMatch);
    // -1 is Mismatch or means error.
}

/*Input a ROI mat return result */
std::pair<Result, float> UPCEANDecoder::decodeROI(const Mat &bar_img) const
{
    if ((size_t) bar_img.cols < this->bits_num)
    {
        return std::make_pair(Result{string(), Result::BARCODE_NONE}, 0.0F);
    }

    std::map<std::string, int> result_vote;
    std::map<Result::BarcodeType, int> format_vote;
    int vote_cnt = 0;
    int total_vote = 0;
    std::string max_result;
    Result::BarcodeType max_type = Result::BARCODE_NONE;

    const int step = bar_img.rows / (DIVIDE_PART + BIAS_PART);
    Result result;
    int row_num;
    for (int i = 0; i < DIVIDE_PART; ++i)
    {
        row_num = (i + BIAS_PART / 2) * step;
        if (row_num < 0 || row_num > bar_img.rows)
        {
            continue;
        }
        const auto *ptr = bar_img.ptr<uchar>(row_num);
        vector<uchar> line(ptr, ptr + bar_img.cols);
        result = decodeLine(line);
        if (result.format != Result::BARCODE_NONE)
        {
            total_vote++;
            result_vote[result.result] += 1;
            if (result_vote[result.result] > vote_cnt)
            {
                vote_cnt = result_vote[result.result];
                max_result = result.result;
                max_type = result.format;
            }
        }
    }
    if (total_vote == 0 || (vote_cnt << 2) < total_vote)
    {
        return std::make_pair(Result(string(), Result::BARCODE_NONE), 0.0f);
    }

    float confidence = (float) vote_cnt / (float) DIVIDE_PART;
    //Check if it is UPC-A format
    if (max_type == Result::BARCODE_EAN_13 && max_result[0] == '0')
    {
        max_result = max_result.substr(1, 12); //UPC-A length 12
        max_type = Result::BARCODE_UPC_A;
    }
    return std::make_pair(Result(max_result, max_type), confidence);
}


Result UPCEANDecoder::decodeLine(const vector<uchar> &line) const
{
    Result result = this->decode(line);
    if (result.format == Result::BARCODE_NONE)
    {
        result = this->decode(std::vector<uchar>(line.crbegin(), line.crend()));
    }
    return result;
}

bool UPCEANDecoder::isValid(const string &result) const
{
    if (result.size() != digit_number)
    {
        return false;
    }
    int sum = 0;
    for (int index = (int) result.size() - 2, i = 1; index >= 0; index--, i++)
    {
        int temp = result[index] - '0';
        sum += (temp + ((i & 1) != 0 ? temp << 1 : 0));
    }
    return (result.back() - '0') == ((10 - (sum % 10)) % 10);
}

// right for A
const std::vector<std::vector<int>> &get_A_or_C_Patterns()
{
    static const std::vector<std::vector<int>> A_or_C_Patterns{{3, 2, 1, 1}, // 0
                                                               {2, 2, 2, 1}, // 1
                                                               {2, 1, 2, 2}, // 2
                                                               {1, 4, 1, 1}, // 3
                                                               {1, 1, 3, 2}, // 4
                                                               {1, 2, 3, 1}, // 5
                                                               {1, 1, 1, 4}, // 6
                                                               {1, 3, 1, 2}, // 7
                                                               {1, 2, 1, 3}, // 8
                                                               {3, 1, 1, 2}  // 9
    };
    return A_or_C_Patterns;
}

const std::vector<std::vector<int>> &get_AB_Patterns()
{
    static const std::vector<std::vector<int>> AB_Patterns = [] {
        constexpr uint offset = 10;
        auto AB_Patterns_inited = std::vector<std::vector<int>>(offset << 1, std::vector<int>(PATTERN_LENGTH, 0));
        std::copy(get_A_or_C_Patterns().cbegin(), get_A_or_C_Patterns().cend(), AB_Patterns_inited.begin());
        //AB pattern is
        for (uint i = 0; i < offset; ++i)
        {
            for (uint j = 0; j < PATTERN_LENGTH; ++j)
            {
                AB_Patterns_inited[i + offset][j] = AB_Patterns_inited[i][PATTERN_LENGTH - j - 1];
            }
        }
        return AB_Patterns_inited;
    }();
    return AB_Patterns;
}

const std::vector<int> &BEGIN_PATTERN()
{
    // it just need it's 1:1:1(black:white:black)
    static const std::vector<int> BEGIN_PATTERN_(3, 1);
    return BEGIN_PATTERN_;
}

const std::vector<int> &MIDDLE_PATTERN()
{
    // it just need it's 1:1:1:1:1(white:black:white:black:white)
    static const std::vector<int> MIDDLE_PATTERN_(5, 1);
    return MIDDLE_PATTERN_;
}

const std::array<char, 32> &FIRST_CHAR_ARRAY()
{
    // use array to simulation a Hashmap,
    // because the data's size is small,
    // use a hashmap or brute-force search 10 times both can not accept
    static const std::array<char, 32> pattern{
            '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x06', '\x00', '\x00', '\x00', '\x09', '\x00',
            '\x08', '\x03', '\x00', '\x00', '\x00', '\x00', '\x05', '\x00', '\x07', '\x02', '\x00', '\x00', '\x04',
            '\x01', '\x00', '\x00', '\x00', '\x00', '\x00'};
    // length is 32 to ensure the security
    // 0x00000 -> 0  -> 0
    // 0x11010 -> 26 -> 1
    // 0x10110 -> 22 -> 2
    // 0x01110 -> 14 -> 3
    // 0x11001 -> 25 -> 4
    // 0x10011 -> 19 -> 5
    // 0x00111 -> 7  -> 6
    // 0x10101 -> 21 -> 7
    // 0x01101 -> 13 -> 8
    // 0x01011 -> 11 -> 9
    // delete the 1-13's 2 number's bit,
    // it always be A which do not need to count.
    return pattern;
}
}

} // namespace cv
