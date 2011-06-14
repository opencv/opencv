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

class CV_SubdivTest : public cvtest::BaseTest
{
public:
    CV_SubdivTest();
    ~CV_SubdivTest();
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


CV_SubdivTest::CV_SubdivTest()
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
    cvtest::BaseTest::clear();
    cvReleaseMemStorage( &storage );
}


int CV_SubdivTest::read_params( CvFileStorage* fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    int t;

    if( code < 0 )
        return code;

    test_case_count = cvReadInt( find_param( fs, "test_case_count" ), test_case_count );
    min_log_point_count = cvReadInt( find_param( fs, "min_log_point_count" ), min_log_point_count );
    max_log_point_count = cvReadInt( find_param( fs, "max_log_point_count" ), max_log_point_count );
    min_log_img_size = cvReadInt( find_param( fs, "min_log_img_size" ), min_log_img_size );
    max_log_img_size = cvReadInt( find_param( fs, "max_log_img_size" ), max_log_img_size );
    
    min_log_point_count = cvtest::clipInt( min_log_point_count, 1, 10 );
    max_log_point_count = cvtest::clipInt( max_log_point_count, 1, 10 );
    if( min_log_point_count > max_log_point_count )
        CV_SWAP( min_log_point_count, max_log_point_count, t );

    min_log_img_size = cvtest::clipInt( min_log_img_size, 1, 10 );
    max_log_img_size = cvtest::clipInt( max_log_img_size, 1, 10 );
    if( min_log_img_size > max_log_img_size )
        CV_SWAP( min_log_img_size, max_log_img_size, t );

    return 0;
}


int CV_SubdivTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = cvtest::BaseTest::prepare_test_case( test_case_idx );
    if( code < 0 )
        return code;
    
    clear();

    point_count = cvRound(exp((cvtest::randReal(rng)*
        (max_log_point_count - min_log_point_count) + min_log_point_count)*CV_LOG2));
    img_size.width = cvRound(exp((cvtest::randReal(rng)*
        (max_log_img_size - min_log_img_size) + min_log_img_size)*CV_LOG2));
    img_size.height = cvRound(exp((cvtest::randReal(rng)*
        (max_log_img_size - min_log_img_size) + min_log_img_size)*CV_LOG2));

    storage = cvCreateMemStorage( 1 << 10 );
    return 1;
}


void CV_SubdivTest::run_func()
{
}


static inline double sqdist( CvPoint2D32f pt1, CvPoint2D32f pt2 )
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    
    return dx*dx + dy*dy;
}


static int
subdiv2DCheck( CvSubdiv2D* subdiv )
{
    int i, j, total = subdiv->edges->total;
    CV_Assert( subdiv != 0 );
    
    for( i = 0; i < total; i++ )
    {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)cvGetSetElem(subdiv->edges,i);
        
        if( edge && CV_IS_SET_ELEM( edge ))
        {
            for( j = 0; j < 4; j++ )
            {
                CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge + j;
                CvSubdiv2DEdge o_next = cvSubdiv2DNextEdge(e);
                CvSubdiv2DEdge o_prev = cvSubdiv2DGetEdge(e, CV_PREV_AROUND_ORG );
                CvSubdiv2DEdge d_prev = cvSubdiv2DGetEdge(e, CV_PREV_AROUND_DST );
                CvSubdiv2DEdge d_next = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_DST );
                
                // check points
                if( cvSubdiv2DEdgeOrg(e) != cvSubdiv2DEdgeOrg(o_next))
                    return 0;
                if( cvSubdiv2DEdgeOrg(e) != cvSubdiv2DEdgeOrg(o_prev))
                    return 0;
                if( cvSubdiv2DEdgeDst(e) != cvSubdiv2DEdgeDst(d_next))
                    return 0;
                if( cvSubdiv2DEdgeDst(e) != cvSubdiv2DEdgeDst(d_prev))
                    return 0;
                if( j % 2 == 0 )
                {
                    if( cvSubdiv2DEdgeDst(o_next) != cvSubdiv2DEdgeOrg(d_prev))
                        return 0;
                    if( cvSubdiv2DEdgeDst(o_prev) != cvSubdiv2DEdgeOrg(d_next))
                        return 0;
                    if( cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(
                                    e,CV_NEXT_AROUND_LEFT),CV_NEXT_AROUND_LEFT),CV_NEXT_AROUND_LEFT) != e )
                        return 0;
                    if( cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(
                                    e,CV_NEXT_AROUND_RIGHT),CV_NEXT_AROUND_RIGHT),CV_NEXT_AROUND_RIGHT) != e)
                        return 0;
                }
            }
        }
    }
    
    return 1;
}


// the whole testing is done here, run_func() is not utilized in this test
int CV_SubdivTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    RNG& rng = ts->get_rng();
    int j, k, real_count = point_count;
    double xrange = img_size.width*(1 - FLT_EPSILON);
    double yrange = img_size.height*(1 - FLT_EPSILON);
    
    subdiv = cvCreateSubdivDelaunay2D(
        cvRect( 0, 0, img_size.width, img_size.height ), storage );
    
    CvSeq* seq = cvCreateSeq( 0, sizeof(*seq), sizeof(CvPoint2D32f), storage );
    CvSeqWriter writer;
    cvStartAppendToSeq( seq, &writer );

    // insert random points
    for( j = 0; j < point_count; j++ )
    {
        CvPoint2D32f pt;
        CvSubdiv2DPoint* point;

        pt.x = (float)(cvtest::randReal(rng)*xrange);
        pt.y = (float)(cvtest::randReal(rng)*yrange);

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
                ts->printf( cvtest::TS::LOG, "The point #%d: (%.1f,%.1f) is said to coinside with a subdivision vertex, "
                    "however it could be found in a sequence of inserted points\n", j, pt.x, pt.y );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
            real_count--;
        }

        point = cvSubdivDelaunay2DInsert( subdiv, pt );
        if( point->pt.x != pt.x || point->pt.y != pt.y )
        {
            ts->printf( cvtest::TS::LOG, "The point #%d: (%.1f,%.1f) has been incorrectly added\n", j, pt.x, pt.y );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }

        if( (j + 1) % 10 == 0 || j == point_count - 1 )
        {
            if( !subdiv2DCheck( subdiv ))
            {
                ts->printf( cvtest::TS::LOG, "Subdivision consistency check failed after inserting the point #%d\n", j );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
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

    if( !subdiv2DCheck( subdiv ))
    {
        ts->printf( cvtest::TS::LOG, "The subdivision failed consistency check after building the Voronoi tesselation\n" );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
        goto _exit_;
    }

    for( j = 0; j < MAX((point_count - 5)/10 + 5, 10); j++ )
    {
        CvPoint2D32f pt;
        double minDistance;

        pt.x = (float)(cvtest::randReal(rng)*xrange);
        pt.y = (float)(cvtest::randReal(rng)*yrange);

        CvSubdiv2DPoint* point = cvFindNearestPoint2D( subdiv, pt );
        CvSeqReader reader;

        if( !point )
        {
            ts->printf( cvtest::TS::LOG, "There is no nearest point (?!) for the point (%.1f, %.1f) in the subdivision\n",
                pt.x, pt.y );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }

        cvStartReadSeq( seq, &reader );
        minDistance = sqdist( pt, point->pt );

        for( k = 0; k < seq->total; k++ )
        {
            CvPoint2D32f ptt;
            CV_READ_SEQ_ELEM( ptt, reader );

            double distance = sqdist( pt, ptt );
            if( minDistance > distance && sqdist(ptt, point->pt) > FLT_EPSILON*1000 )
            {
                ts->printf( cvtest::TS::LOG, "The triangulation vertex (%.3f,%.3f) was said to be nearest to (%.3f,%.3f),\n"
                    "whereas another vertex (%.3f,%.3f) is closer\n",
                    point->pt.x, point->pt.y, pt.x, pt.y, ptt.x, ptt.y );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
                goto _exit_;
            }
        }
    }

_exit_:
    if( code < 0 )
        ts->set_failed_test_info( code );

    return code;
}

TEST(Imgproc_Subdiv, correctness) { CV_SubdivTest test; test.safe_run(); }

/* End of file. */

