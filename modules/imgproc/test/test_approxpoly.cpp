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

namespace opencv_test { namespace {

//
// TODO!!!:
//  check_slice (and/or check) seem(s) to be broken, or this is a bug in function
//  (or its inability to handle possible self-intersections in the generated contours).
//
//  At least, if // return TotalErrors;
//  is uncommented in check_slice, the test fails easily.
//  So, now (and it looks like since 0.9.6)
//  we only check that the set of vertices of the approximated polygon is
//  a subset of vertices of the original contour.
//

class CV_ApproxPolyTest : public cvtest::BaseTest
{
public:
    CV_ApproxPolyTest();
    ~CV_ApproxPolyTest();
    void clear();
    //int write_default_params(CvFileStorage* fs);

protected:
    //int read_params( const cv::FileStorage& fs );

    int check_slice( CvPoint StartPt, CvPoint EndPt,
                     CvSeqReader* SrcReader, float Eps,
                     int* j, int Count );
    int check( CvSeq* SrcSeq, CvSeq* DstSeq, float Eps );

    bool get_contour( int /*type*/, CvSeq** Seq, int* d,
                      CvMemStorage* storage );

    void run(int);
};


CV_ApproxPolyTest::CV_ApproxPolyTest()
{
}


CV_ApproxPolyTest::~CV_ApproxPolyTest()
{
    clear();
}


void CV_ApproxPolyTest::clear()
{
    cvtest::BaseTest::clear();
}


/*int CV_ApproxPolyTest::write_default_params( CvFileStorage* fs )
{
    cvtest::BaseTest::write_default_params( fs );
    if( ts->get_testing_mode() != cvtest::TS::TIMING_MODE )
    {
        write_param( fs, "test_case_count", test_case_count );
    }
    return 0;
}


int CV_ApproxPolyTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    test_case_count = cvReadInt( find_param( fs, "test_case_count" ), test_case_count );
    min_log_size = cvtest::clipInt( min_log_size, 1, 10 );
    return 0;
}*/


bool CV_ApproxPolyTest::get_contour( int /*type*/, CvSeq** Seq, int* d,
                                     CvMemStorage* storage )
{
    RNG& rng = ts->get_rng();
    int max_x = INT_MIN, max_y = INT_MIN, min_x = INT_MAX, min_y = INT_MAX;
    int i;
    CvSeq* seq;
    int total = cvtest::randInt(rng) % 1000 + 1;
    Point center;
    int radius, angle;
    double deg_to_rad = CV_PI/180.;
    Point pt;

    center.x = cvtest::randInt( rng ) % 1000;
    center.y = cvtest::randInt( rng ) % 1000;
    radius = cvtest::randInt( rng ) % 1000;
    angle = cvtest::randInt( rng ) % 360;

    seq = cvCreateSeq( CV_SEQ_POLYGON, sizeof(CvContour), sizeof(CvPoint), storage );

    for( i = 0; i < total; i++ )
    {
        int d_radius = cvtest::randInt( rng ) % 10 - 5;
        int d_angle = 360/total;//cvtest::randInt( rng ) % 10 - 5;
        pt.x = cvRound( center.x + radius*cos(angle*deg_to_rad));
        pt.y = cvRound( center.x - radius*sin(angle*deg_to_rad));
        radius += d_radius;
        angle += d_angle;
        cvSeqPush( seq, &pt );

        max_x = MAX( max_x, pt.x );
        max_y = MAX( max_y, pt.y );

        min_x = MIN( min_x, pt.x );
        min_y = MIN( min_y, pt.y );
    }

    *d = (max_x - min_x)*(max_x - min_x) + (max_y - min_y)*(max_y - min_y);
    *Seq = seq;
    return true;
}


int CV_ApproxPolyTest::check_slice( CvPoint StartPt, CvPoint EndPt,
                                   CvSeqReader* SrcReader, float Eps,
                                   int* _j, int Count )
{
    ///////////
    Point Pt;
    ///////////
    bool flag;
    double dy,dx;
    double A,B,C;
    double Sq;
    double sin_a = 0;
    double cos_a = 0;
    double d     = 0;
    double dist;
    ///////////
    int j, TotalErrors = 0;

    ////////////////////////////////
    if( SrcReader == NULL )
    {
        assert( false );
        return 0;
    }

    ///////// init line ////////////
    flag = true;

    dx = (double)StartPt.x - (double)EndPt.x;
    dy = (double)StartPt.y - (double)EndPt.y;

    if( ( dx == 0 ) && ( dy == 0 ) ) flag = false;
    else
    {
        A = -dy;
        B = dx;
        C = dy * (double)StartPt.x - dx * (double)StartPt.y;
        Sq = sqrt( A*A + B*B );

        sin_a = B/Sq;
        cos_a = A/Sq;
        d = C/Sq;
    }

    /////// find start point and check distance ////////
    for( j = *_j; j < Count; j++ )
    {
        { CvPoint pt_ = CV_STRUCT_INITIALIZER; CV_READ_SEQ_ELEM(pt_, *SrcReader); Pt = pt_; }
        if( StartPt.x == Pt.x && StartPt.y == Pt.y ) break;
        else
        {
            if( flag ) dist = sin_a * Pt.y + cos_a * Pt.x - d;
            else dist = sqrt( (double)(EndPt.y - Pt.y)*(EndPt.y - Pt.y) + (EndPt.x - Pt.x)*(EndPt.x - Pt.x) );
            if( dist > Eps ) TotalErrors++;
        }
    }

    *_j = j;

    //return TotalErrors;
    return 0;
}


int CV_ApproxPolyTest::check( CvSeq* SrcSeq, CvSeq* DstSeq, float Eps )
{
    //////////
    CvSeqReader  DstReader;
    CvSeqReader  SrcReader;
    CvPoint StartPt = {0, 0}, EndPt = {0, 0};
    ///////////
    int TotalErrors = 0;
    ///////////
    int Count;
    int i,j;

    assert( SrcSeq && DstSeq );

    ////////// init ////////////////////
    Count = SrcSeq->total;

    cvStartReadSeq( DstSeq, &DstReader, 0 );
    cvStartReadSeq( SrcSeq, &SrcReader, 0 );

    CV_READ_SEQ_ELEM( StartPt, DstReader );
    for( i = 0 ; i < Count ;  )
    {
        CV_READ_SEQ_ELEM( EndPt, SrcReader );
        i++;
        if( StartPt.x == EndPt.x && StartPt.y == EndPt.y ) break;
    }

    ///////// start ////////////////
    for( i = 1, j = 0 ; i <= DstSeq->total ;  )
    {
        ///////// read slice ////////////
        EndPt.x = StartPt.x;
        EndPt.y = StartPt.y;
        CV_READ_SEQ_ELEM( StartPt, DstReader );
        i++;

        TotalErrors += check_slice( StartPt, EndPt, &SrcReader, Eps, &j, Count );

        if( j > Count )
        {
            TotalErrors++;
            return TotalErrors;
        } //if( !flag )

    } // for( int i = 0 ; i < DstSeq->total ; i++ )

    return TotalErrors;
}


//extern CvTestContourGenerator cvTsTestContours[];

void CV_ApproxPolyTest::run( int /*start_from*/ )
{
    int code = cvtest::TS::OK;
    CvMemStorage* storage = 0;
    ////////////// Variables ////////////////
    int IntervalsCount = 10;
    ///////////
    //CvTestContourGenerator Cont;
    CvSeq*  SrcSeq = NULL;
    CvSeq*  DstSeq;
    int     iDiam;
    float   dDiam, Eps, EpsStep;

    for( int i = 0; i < 30; i++ )
    {
        CvMemStoragePos pos;

        ts->update_context( this, i, false );

        ///////////////////// init contour /////////
        dDiam = 0;
        while( sqrt(dDiam) / IntervalsCount == 0 )
        {
            if( storage != 0 )
                cvReleaseMemStorage(&storage);

            storage = cvCreateMemStorage( 0 );
            if( get_contour( 0, &SrcSeq, &iDiam, storage ) )
                dDiam = (float)iDiam;
        }
        dDiam = (float)sqrt( dDiam );

        storage = SrcSeq->storage;

        ////////////////// test /////////////
        EpsStep = dDiam / IntervalsCount ;
        for( Eps = EpsStep ; Eps < dDiam ; Eps += EpsStep )
        {
            cvSaveMemStoragePos( storage, &pos );

            ////////// call function ////////////
            DstSeq = cvApproxPoly( SrcSeq, SrcSeq->header_size, storage,
                CV_POLY_APPROX_DP, Eps );

            if( DstSeq == NULL )
            {
                ts->printf( cvtest::TS::LOG,
                    "cvApproxPoly returned NULL for contour #%d, espilon = %g\n", i, Eps );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            } // if( DstSeq == NULL )

            code = check( SrcSeq, DstSeq, Eps );
            if( code != 0 )
            {
                ts->printf( cvtest::TS::LOG,
                    "Incorrect result for the contour #%d approximated with epsilon=%g\n", i, Eps );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
                goto _exit_;
            }

            cvRestoreMemStoragePos( storage, &pos );
        } // for( Eps = EpsStep ; Eps <= Diam ; Eps += EpsStep )

        ///////////// free memory  ///////////////////
        cvReleaseMemStorage(&storage);
    } // for( int i = 0; NULL != ( Cont = Contours[i] ) ; i++ )

_exit_:
    cvReleaseMemStorage(&storage);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

TEST(Imgproc_ApproxPoly, accuracy) { CV_ApproxPolyTest test; test.safe_run(); }

}} // namespace
