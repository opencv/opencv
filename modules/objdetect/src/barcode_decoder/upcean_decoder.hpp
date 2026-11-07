// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef OPENCV_BARCODE_UPCEAN_DECODER_HPP
#define OPENCV_BARCODE_UPCEAN_DECODER_HPP

#include "abs_decoder.hpp"

/**
 *   upcean_decoder the abstract basic class for decode formats,
 *   it will have ean13/8,upc_a,upc_e , etc.. class extend this class
*/
namespace cv {
namespace barcode {
using std::string;
using std::vector;

class UPCEANDecoder : public AbsDecoder
{

public:
    ~UPCEANDecoder() override = default;

    std::pair<Result, float> decodeROI(const Mat &bar_img) const override;

protected:
    static int decodeDigit(const std::vector<uchar> &row, Counter &counters, uint rowOffset,
                           const std::vector<std::vector<int>> &patterns);

    static bool
    findGuardPatterns(const std::vector<uchar> &row, uint rowOffset, uchar whiteFirst, const std::vector<int> &pattern,
                      Counter &counter, std::pair<uint, uint> &result);

    static bool findStartGuardPatterns(const std::vector<uchar> &row, std::pair<uint, uint> &start_range);

    Result decodeLine(const vector<uchar> &line) const;

    Result decode(const vector<uchar> &bar) const override = 0;

    bool isValid(const string &result) const override;

private:
    #if 0
    void drawDebugLine(Mat &debug_img, const Point2i &begin, const Point2i &end) const;
    #endif
};

const std::vector<std::vector<int>> &get_A_or_C_Patterns();

const std::vector<std::vector<int>> &get_AB_Patterns();

const std::vector<int> &BEGIN_PATTERN();

const std::vector<int> &MIDDLE_PATTERN();

const std::array<char, 32> &FIRST_CHAR_ARRAY();

constexpr static uint PATTERN_LENGTH = 4;
constexpr static uint MAX_AVG_VARIANCE = static_cast<uint>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.48f);
constexpr static uint MAX_INDIVIDUAL_VARIANCE = static_cast<uint>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 0.7f);

}
} // namespace cv

#endif // OPENCV_BARCODE_UPCEAN_DECODER_HPP
