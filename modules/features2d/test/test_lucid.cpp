// LUCID test case by Str3iber

/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "test_precomp.hpp"

using namespace cv;

class CV_LUCIDTest : public cvtest::BaseTest {
    public:
        CV_LUCIDTest();
        ~CV_LUCIDTest();

    protected:
        void run(int);
};

CV_LUCIDTest::CV_LUCIDTest() {}
CV_LUCIDTest::~CV_LUCIDTest() {}

void CV_LUCIDTest::run(int) {
    Mat image = imread(std::string(ts->get_data_path()) + "inpaint/orig.png");

    if (image.empty()) {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);

        return;
    }

    Mat buf;
    cvtColor(image, buf, COLOR_BGR2GRAY);

    std::vector<KeyPoint> kpt;
    std::vector<std::vector<int> > dsc;

    FAST(buf, kpt, 9, 1);
    if (kpt.empty()) {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
    }
    KeyPointsFilter::retainBest(kpt, 100);

    LUCID(image, kpt, dsc, 1, 2);

    if (!dsc.empty()) {
        for (std::size_t i = 0; i < dsc.size(); ++i) {
            if (dsc[i].size() != 27) {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);

                return;
            }

            for (std::size_t x = 0; x < 27; ++x) {
                if (x > 0 && dsc[i][x] < dsc[i][x-1]) {
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);

                    return;
                }
            }
        }
    }
    else {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);

        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Features2d_LUCID, regression) { CV_LUCIDTest test; test.safe_run(); }
