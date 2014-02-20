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
using namespace std;

class Core_RotatedRectConstructorTest : public cvtest::BaseTest
{
public:
    Core_RotatedRectConstructorTest();
protected:
    int prepare_test_case( int );
    void run_func();
    int validate_test_results( int );
    const static int MAX_COORD_VAL = 1000;
    Point2f a, b, c;
    RotatedRect rec;
};

Core_RotatedRectConstructorTest::Core_RotatedRectConstructorTest()
{
    test_case_count = 100;
}

int Core_RotatedRectConstructorTest::prepare_test_case( int test_case_idx )
{
    cvtest::BaseTest::prepare_test_case( test_case_idx );
    RNG& rng = ts->get_rng();
    a = Point2f( (float) (cvtest::randInt(rng) % MAX_COORD_VAL), (float) (cvtest::randInt(rng) % MAX_COORD_VAL) );
    b = Point2f( (float) (cvtest::randInt(rng) % MAX_COORD_VAL) , (float) (cvtest::randInt(rng) % MAX_COORD_VAL) );
    // to ensure a != b
    while( norm(a - b) == 0 ) {
        b = Point2f( (float) (cvtest::randInt(rng) % MAX_COORD_VAL) , (float) (cvtest::randInt(rng) % MAX_COORD_VAL) );
    }
    Vec2f along(a - b);
    Vec2f perp = Vec2f(-along[1], along[0]);
    float d = (float) (cvtest::randInt(rng) % MAX_COORD_VAL) + 1.0f;  // c can't be same as b, so d must be > 0
    c = Point2f( b.x + d * perp[0], b.y + d * perp[1] );
    return 1;
}

void Core_RotatedRectConstructorTest::run_func()
{
    rec = RotatedRect(a, b, c);
}

int Core_RotatedRectConstructorTest::validate_test_results( int )
{
    int code = cvtest::TS::OK;
    Point2f vertices[4];
    rec.points(vertices);

    int count_match = 0;
    for( int i = 0; i < 4; i++ )
    {
        if( norm(vertices[i] - a) <= 0.1 ) count_match++;
        else if( norm(vertices[i] - b) <= 0.1 ) count_match++;
        else if( norm(vertices[i] - c) <= 0.1 ) count_match++;
    }
    if( count_match == 3 )
        return code;
    ts->printf( cvtest::TS::LOG, "RotatedRect end points don't match those supplied in constructor");
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    return code;
}

TEST(Core_RotatedRect, three_point_constructor) { Core_RotatedRectConstructorTest test; test.safe_run(); }
