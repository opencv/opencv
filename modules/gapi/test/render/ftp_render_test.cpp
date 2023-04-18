// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

#ifdef HAVE_FREETYPE

#include <random>

#include <opencv2/core/utils/configuration.private.hpp>

#include "backends/render/ft_render.hpp"

namespace opencv_test
{
    static std::string getFontPath()
    {
        static std::string path = cv::utils::getConfigurationParameterString("OPENCV_TEST_FREETYPE_FONT_PATH",
                                                                         "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc");
        return path;
    }

    inline void RunTest(const std::string& font,
                        size_t num_iters,
                        size_t lower_char_code,
                        size_t upper_char_code)
    {
        cv::gapi::wip::draw::FTTextRender ftpr(font);

        std::mt19937 gen{std::random_device()()};
        std::uniform_int_distribution<int> dist(lower_char_code, upper_char_code);
        std::uniform_int_distribution<int> dist_size(2, 200);

        for (size_t i = 0; i < num_iters; ++i)
        {
            size_t text_size = dist_size(gen);
            std::wstring text;

            for (size_t j = 0; j < text_size; ++j)
            {
                wchar_t c = dist(gen);
                text += c;
            }

            int fh       = dist_size(gen);
            int baseline = 0;
            cv::Size size;

            ASSERT_NO_THROW(size = ftpr.getTextSize(text, fh, &baseline));

            cv::Mat bmp(size, CV_8UC1, cv::Scalar::all(0));
            cv::Point org(0, bmp.rows - baseline);

            ASSERT_NO_THROW(ftpr.putText(bmp, text, org, fh));
        }
    }

    TEST(FTTextRenderTest, Smoke_Test_Ascii)
    {
        RunTest(getFontPath(), 2000, 32, 126);
    }

    TEST(FTTextRenderTest, Smoke_Test_Unicode)
    {
        RunTest(getFontPath(), 2000, 20320, 30000);
    }
} // namespace opencv_test

#endif // HAVE_FREETYPE
