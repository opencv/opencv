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

#include "cvtest.h"


class CV_SubdivTest : public CvTest
{
public:
    CV_SubdivTest();
    ~CV_SubdivTest();
    int write_default_params(CvFileStorage* fs);
    void clear();

protected:
    int read_params( CvFileStorage* fs );
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void run_func();

    int min_log_img_size, max_log_img_size;
    CvSize img_size;
    int min_log_point_count;
    int max_log_point_count;
    int point_count;
    CvSubdiv2D* subdiv;
    CvMemStorage* storage;
};


CV_SubdivTest::CV_SubdivTest() :
    CvTest( "subdiv",
    "cvCreateSubdivDelaunay2D, cvSubdivDelaunay2DInsert, cvSubdiv2DLocate, "
    "cvCalcSubdivVoronoi2D, cvFindNearestPoint2D" )
{
    test_case_count = 100;
    min_log_point_count = 1;
    max_log_point_count = 10;
    min_log_img_size = 1;
    max_log_img_size = 10;

    storage = 0;
}


CV_SubdivTest::~CV_SubdivTest()
{
    clear();
}


void CV_SubdivTest::clear()
{
    CvTest::clear();
    cvReleaseMemStorage( &storage );
}


int CV_SubdivTest::write_default_params( CvFileStorage* fs )
{
    CvTest::write_default_params( fs );
    if( ts->get_testing_mode() != CvTS::TIMING_MODE )
    {
        write_param( fs, "test_case_count", test_case_count );
        write_param( fs, "min_log_point_count", min_log_point_count );
        write_param( fs, "max_log_point_count", max_log_point_count );
        write_param( fs, "min_log_img_size", min_log_img_size );
        write_param( fs, "max_log_img_size", max_log_img_size );
    }
    return 0;
}


int CV_SubdivTest::read_params( CvFileStorage* fs )
{
    int code = CvTest::read_params( fs );
    int t;

    if( code < 0 )
        return code;

    test_case_count = cvReadInt( find_param( fs, "test_case_count" ), test_case_count );
    min_log_point_count = cvReadInt( find_param( fs, "min_log_point_count" ), min_log_point_count );
    max_log_point_count = cvReadInt( find_param( fs, "max_log_point_count" ), max_log_point_count );
    min_log_img_size = cvReadInt( find_param( fs, "min_log_img_size" ), min_log_img_size );
    max_log_img_size = cvReadInt( find_param( fs, "max_log_img_size" ), max_log_img_size );
    
    min_log_point_count = cvTsClipInt( min_log_point_count, 1, 10 );
    max_log_point_count = cvTsClipInt( max_log_point_count, 1, 10 );
    if( min_log_point_count > max_log_point_count )
        CV_SWAP( min_log_point_count, max_log_point_count, t );

    min_log_img_size = cvTsClipInt( min_log_img_size, 1, 10 );
    max_log_img_size = cvTsClipInt( max_log_img_size, 1, 10 );
    if( min_log_img_size > max_log_img_size )
        CV_SWAP( min_log_img_size, max_log_img_size, t );

    return 0;
}


int CV_SubdivTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CvTest::prepare_test_case( test_case_idx );
    if( code < 0 )
        return code;
    
    clear();

    point_count = cvRound(exp((cvTsRandReal(rng)*
        (max_log_point_count - min_log_point_count) + min_log_point_count)*CV_LOG2));
    img_size.width = cvRound(exp((cvTsRandReal(rng)*
        (max_log_img_size - min_log_img_size) + min_log_img_size)*CV_LOG2));
    img_size.height = cvRound(exp((cvTsRandReal(rng)*
        (max_log_img_size - min_log_img_size) + min_log_img_size)*CV_LOG2));

    storage = cvCreateMemStorage( 1 << 10 );
    return 1;
}


void CV_SubdivTest::run_func()
{
}


// the whole testing is done here, run_func() is not utilized in this test
int CV_SubdivTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = CvTS::OK;
    CvRNG* rng = ts->get_rng();
    int j, k, real_count = point_count;
    double xrange = img_size.width*(1 - FLT_EPSILON);
    double yrange = img_size.height*(1 - FLT_EPSILON);
    
    subdiv = subdiv = cvCreateSubdivDelaunay2D(
        cvRect( 0, 0, img_size.width, img_size.height ), storage );
    
    CvSeq* seq = cvCreateSeq( 0, sizeof(*seq), sizeof(CvPoint2D32f), storage );
    CvSeqWriter writer;
    cvStartAppendToSeq( seq, &writer );

    // insert random points
    for( j = 0; j < point_count; j++ )
    {
        CvPoint2D32f pt;
        CvSubdiv2DPoint* point;

        pt.x = (float)(cvTsRandReal(rng)*xrange);
        pt.y = (float)(cvTsRandReal(rng)*yrange);

        CvSubdiv2DPointLocation loc = 
            cvSubdiv2DLocate( subdiv, pt, 0, &point );

        if( loc == CV_PTLOC_VERTEX )
        {
            int index = cvSeqElemIdx( (CvSeq*)subdiv, point );
            CvPoint2D32f* pt1;
            cvFlushSeqWriter( &writer );
            pt1 = (CvPoint2D32f*)cvGetSeqElem( seq, index - 3 );

            if( !pt1 ||
                fabs(pt1->x - pt.x) > FLT_EPSILON ||
                fabs(pt1->y - pt.y) > FLT_EPSILON )
            {
                ts->printf( CvTS::LOG, "The point #%d: (%.1f,%.1f) is said to coinside with a subdivision vertex, "
                    "however it could be found in a sequence of inserted points\n", j, pt.x, pt.y );
                code = CvTS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
            real_count--;
        }

        point = cvSubdivDelaunay2DInsert( subdiv, pt );
        if( point->pt.x != pt.x || point->pt.y != pt.y )
        {
            ts->printf( CvTS::LOG, "The point #%d: (%.1f,%.1f) has been incorrectly added\n", j, pt.x, pt.y );
            code = CvTS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }

        if( (j + 1) % 10 == 0 || j == point_count - 1 )
        {
            if( !icvSubdiv2DCheck( subdiv ))
            {
                ts->printf( CvTS::LOG, "Subdivision consistency check failed after inserting the point #%d\n", j );
                code = CvTS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
        }
        
        if( loc != CV_PTLOC_VERTEX )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }

    if( code < 0 )
        goto _exit_;

    cvCalcSubdivVoronoi2D( subdiv );
    seq = cvEndWriteSeq( &writer );

    if( !icvSubdiv2DCheck( subdiv ))
    {
        ts->printf( CvTS::LOG, "The subdivision failed consistency check after building the Voronoi tesselation\n" );
        code = CvTS::FAIL_INVALID_OUTPUT;
        goto _exit_;
    }

    for( j = 0; j < MAX((point_count - 5)/10 + 5, 10); j++ )
    {
        CvPoint2D32f pt;
        double minDistance;

        pt.x = (float)(cvTsRandReal(rng)*xrange);
        pt.y = (float)(cvTsRandReal(rng)*yrange);

        CvSubdiv2DPoint* point = cvFindNearestPoint2D( subdiv, pt );
        CvSeqReader reader;

        if( !point )
        {
            ts->printf( CvTS::LOG, "There is no nearest point (?!) for the point (%.1f, %.1f) in the subdivision\n",
                pt.x, pt.y );
            code = CvTS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }

        cvStartReadSeq( seq, &reader );
        minDistance = icvSqDist2D32f( pt, point->pt );

        for( k = 0; k < seq->total; k++ )
        {
            CvPoint2D32f ptt;
            CV_READ_SEQ_ELEM( ptt, reader );

            double distance = icvSqDist2D32f( pt, ptt );
            if( minDistance > distance && icvSqDist2D32f(ptt, point->pt) > FLT_EPSILON*1000 )
            {
                ts->printf( CvTS::LOG, "The triangulation vertex (%.3f,%.3f) was said to be nearest to (%.3f,%.3f),\n"
                    "whereas another vertex (%.3f,%.3f) is closer\n",
                    point->pt.x, point->pt.y, pt.x, pt.y, ptt.x, ptt.y );
                code = CvTS::FAIL_BAD_ACCURACY;
                goto _exit_;
            }
        }
    }

_exit_:
    if( code < 0 )
        ts->set_failed_test_info( code );

    return code;
}

//CV_SubdivTest subdiv_test;

/* End of file. */

