/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

using namespace cv;

class CV_LshTableBadArgTest : public cvtest::BadArgTest
{
protected:
    void run(int);
    void run_func(void) {};

    struct Caller
    {
        int table_number, key_size, multi_probe_level;
        Mat features;

        void operator()() const
        {
            flann::LshIndexParams indexParams(table_number, key_size, multi_probe_level);
            flann::Index lsh(features, indexParams);
        }
    };
};

void CV_LshTableBadArgTest::run( int /* start_from */ )
{
    RNG &rng = ts->get_rng();

    Caller caller;
    Size featuresSize = cvtest::randomSize(rng, 10.0);
    caller.features = cvtest::randomMat(rng, featuresSize, CV_8UC1, 0, 255, false);
    caller.table_number = 12;
    caller.multi_probe_level = 2;

    int errors = 0;
    caller.key_size = 0;
    errors += run_test_case(CV_StsBadArg, "key_size is zero", caller);

    caller.key_size = static_cast<int>(sizeof(size_t) * CHAR_BIT);
    errors += run_test_case(CV_StsBadArg, "key_size is too big", caller);

    caller.key_size += cvtest::randInt(rng) % 100;
    errors += run_test_case(CV_StsBadArg, "key_size is too big", caller);

    if (errors != 0)
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    else
        ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Flann_LshTable, badarg) { CV_LshTableBadArgTest test; test.safe_run(); }
