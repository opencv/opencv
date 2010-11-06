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

/*static int
cvTsPointConvexPolygon( CvPoint2D32f pt, CvPoint2D32f* v, int n )
{
    CvPoint2D32f v0 = v[n-1];
    int i, sign = 0;

    for( i = 0; i < n; i++ )
    {
        CvPoint2D32f v1 = v[i];
        float dx = pt.x - v0.x, dy = pt.y - v0.y;
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y;
        double t = (double)dx*dy1 - (double)dx1*dy;
        if( fabs(t) > DBL_EPSILON )
        {
            if( t*sign < 0 )
                break;
            if( sign == 0 )
                sign = t < 0 ? -1 : 1;
        }
        else if( fabs(dx) + fabs(dy) < DBL_EPSILON )
            return i+1;
        v0 = v1;
    }

    return i < n ? -1 : 0;
}*/

CV_INLINE double
cvTsDist( CvPoint2D32f a, CvPoint2D32f b )
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx*dx + dy*dy);
}

CV_INLINE double
cvTsPtLineDist( CvPoint2D32f pt, CvPoint2D32f a, CvPoint2D32f b )
{
    double d0 = cvTsDist( pt, a ), d1;
    double dd = cvTsDist( a, b );
    if( dd < FLT_EPSILON )
        return d0;
    d1 = cvTsDist( pt, b );
    dd = fabs((double)(pt.x - a.x)*(b.y - a.y) - (double)(pt.y - a.y)*(b.x - a.x))/dd;
    d0 = MIN( d0, d1 );
    return MIN( d0, dd );
}

static double
cvTsPointPolygonTest( CvPoint2D32f pt, const CvPoint2D32f* vv, int n, int* _idx=0, int* _on_edge=0 )
{
    int i;
    CvPoint2D32f v = vv[n-1], v0;
    double min_dist_num = FLT_MAX, min_dist_denom = 1;
    int min_dist_idx = -1, min_on_edge = 0;
    int counter = 0;
    double result;

    for( i = 0; i < n; i++ )
    {
        double dx, dy, dx1, dy1, dx2, dy2, dist_num, dist_denom = 1;
        int on_edge = 0, idx = i;

        v0 = v; v = vv[i];
        dx = v.x - v0.x; dy = v.y - v0.y;
        dx1 = pt.x - v0.x; dy1 = pt.y - v0.y;
        dx2 = pt.x - v.x; dy2 = pt.y - v.y;

        if( dx2*dx + dy2*dy >= 0 )
            dist_num = dx2*dx2 + dy2*dy2;
        else if( dx1*dx + dy1*dy <= 0 )
        {
            dist_num = dx1*dx1 + dy1*dy1;
            idx = i - 1;
            if( idx < 0 ) idx = n-1;
        }
        else
        {
            dist_num = (dy1*dx - dx1*dy);
            dist_num *= dist_num;
            dist_denom = dx*dx + dy*dy;
            on_edge = 1;
        }

        if( dist_num*min_dist_denom < min_dist_num*dist_denom )
        {
            min_dist_num = dist_num;
            min_dist_denom = dist_denom;
            min_dist_idx = idx;
            min_on_edge = on_edge;
            if( min_dist_num == 0 )
                break;
        }

        if( (v0.y <= pt.y && v.y <= pt.y) ||
            (v0.y > pt.y && v.y > pt.y) ||
            (v0.x < pt.x && v.x < pt.x) )
            continue;

        dist_num = dy1*dx - dx1*dy;
        if( dy < 0 )
            dist_num = -dist_num;
        counter += dist_num > 0;
    }

    result = sqrt(min_dist_num/min_dist_denom);
    if( counter % 2 == 0 )
        result = -result;

    if( _idx )
        *_idx = min_dist_idx;
    if( _on_edge )
        *_on_edge = min_on_edge;

    return result;
}


/****************************************************************************************\
*                              Base class for shape descriptor tests                     *
\****************************************************************************************/

class CV_BaseShapeDescrTest : public CvTest
{
public:
    CV_BaseShapeDescrTest( const char* test_name, const char* test_funcs );
    virtual ~CV_BaseShapeDescrTest();
    void clear();
    int write_default_params(CvFileStorage* fs);

protected:
    int read_params( CvFileStorage* fs );
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    virtual void generate_point_set( void* points );
    virtual void extract_points();

    int min_log_size;
    int max_log_size;
    int dims;
    bool enable_flt_points;

    CvMemStorage* storage;
    CvSeq* points1;
    CvMat* points2;
    void* points;
    void* result;
    double low_high_range;
    CvScalar low, high;
    
    bool test_cpp;
};


CV_BaseShapeDescrTest::CV_BaseShapeDescrTest( const char* test_name, const char* test_funcs ):
    CvTest( test_name, test_funcs )
{
    points1 = 0;
    points2 = 0;
    points = 0;
    storage = 0;
    test_case_count = 500;
    min_log_size = 0;
    max_log_size = 10;
    low = high = cvScalarAll(0);
    low_high_range = 50;
    dims = 2;
    enable_flt_points = true;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    
    test_cpp = false;
}


CV_BaseShapeDescrTest::~CV_BaseShapeDescrTest()
{
    clear();
}


void CV_BaseShapeDescrTest::clear()
{
    CvTest::clear();
    cvReleaseMemStorage( &storage );
    cvReleaseMat( &points2 );
    points1 = 0;
    points = 0;
}


int CV_BaseShapeDescrTest::write_default_params( CvFileStorage* fs )
{
    CvTest::write_default_params( fs );
    if( ts->get_testing_mode() != CvTS::TIMING_MODE )
    {
        write_param( fs, "test_case_count", test_case_count );
        write_param( fs, "min_log_size", min_log_size );
        write_param( fs, "max_log_size", max_log_size );
    }
    return 0;
}


int CV_BaseShapeDescrTest::read_params( CvFileStorage* fs )
{
    int code = CvTest::read_params( fs );
    if( code < 0 )
        return code;

    test_case_count = cvReadInt( find_param( fs, "struct_count" ), test_case_count );
    min_log_size = cvReadInt( find_param( fs, "min_log_size" ), min_log_size );
    max_log_size = cvReadInt( find_param( fs, "max_log_size" ), max_log_size );

    min_log_size = cvTsClipInt( min_log_size, 0, 8 );
    max_log_size = cvTsClipInt( max_log_size, 0, 10 );
    if( min_log_size > max_log_size )
    {
        int t;
        CV_SWAP( min_log_size, max_log_size, t );
    }

    return 0;
}


void CV_BaseShapeDescrTest::generate_point_set( void* points )
{
    CvRNG* rng = ts->get_rng();
    int i, k, n, total, point_type;
    CvSeqReader reader;
    uchar* data = 0;
    double a[4], b[4];

    for( k = 0; k < 4; k++ )
    {
        a[k] = high.val[k] - low.val[k];
        b[k] = low.val[k];
    }
    memset( &reader, 0, sizeof(reader) );

    if( CV_IS_SEQ(points) )
    {
        CvSeq* ptseq = (CvSeq*)points;
        total = ptseq->total;
        point_type = CV_SEQ_ELTYPE(ptseq);
        cvStartReadSeq( ptseq, &reader );
    }
    else
    {
        CvMat* ptm = (CvMat*)points;
        assert( CV_IS_MAT(ptm) && CV_IS_MAT_CONT(ptm->type) );
        total = ptm->rows + ptm->cols - 1;
        point_type = CV_MAT_TYPE(ptm->type);
        data = ptm->data.ptr;
    }

    n = CV_MAT_CN(point_type);
    point_type = CV_MAT_DEPTH(point_type);

    assert( (point_type == CV_32S || point_type == CV_32F) && n <= 4 );

    for( i = 0; i < total; i++ )
    {
        int* pi;
        float* pf;
        if( reader.ptr )
        {
            pi = (int*)reader.ptr;
            pf = (float*)reader.ptr;
            CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        }
        else
        {
            pi = (int*)data + i*n;
            pf = (float*)data + i*n;
        }
        if( point_type == CV_32S )
            for( k = 0; k < n; k++ )
                pi[k] = cvRound(cvTsRandReal(rng)*a[k] + b[k]);
        else
            for( k = 0; k < n; k++ )
                pf[k] = (float)(cvTsRandReal(rng)*a[k] + b[k]);
    }
}


int CV_BaseShapeDescrTest::prepare_test_case( int test_case_idx )
{
    int size;
    int use_storage = 0;
    int point_type;
    int i;
    CvRNG* rng = ts->get_rng();

    CvTest::prepare_test_case( test_case_idx );

    clear();
    size = cvRound( exp((cvTsRandReal(rng) * (max_log_size - min_log_size) + min_log_size)*CV_LOG2) );
    use_storage = cvTsRandInt(rng) % 2;
    point_type = CV_MAKETYPE(cvTsRandInt(rng) %
        (enable_flt_points ? 2 : 1) ? CV_32F : CV_32S, dims);

    if( use_storage )
    {
        storage = cvCreateMemStorage( (cvTsRandInt(rng)%10 + 1)*1024 );
        points1 = cvCreateSeq( point_type, sizeof(CvSeq), CV_ELEM_SIZE(point_type), storage );
        cvSeqPushMulti( points1, 0, size );
        points = points1;
    }
    else
    {
        int rows = 1, cols = size;
        if( cvTsRandInt(rng) % 2 )
            rows = size, cols = 1;

        points2 = cvCreateMat( rows, cols, point_type );
        points = points2;
    }

    for( i = 0; i < 4; i++ )
    {
        low.val[i] = (cvTsRandReal(rng)-0.5)*low_high_range*2;
        high.val[i] = (cvTsRandReal(rng)-0.5)*low_high_range*2;
        if( low.val[i] > high.val[i] )
        {
            double t;
            CV_SWAP( low.val[i], high.val[i], t );
        }
        if( high.val[i] < low.val[i] + 1 )
            high.val[i] += 1;
    }

    generate_point_set( points );
    
    test_cpp = (cvTsRandInt(rng) & 16) == 0;
    return 1;
}


void CV_BaseShapeDescrTest::extract_points()
{
    if( points1 )
    {
        points2 = cvCreateMat( 1, points1->total, CV_SEQ_ELTYPE(points1) );
        cvCvtSeqToArray( points1, points2->data.ptr );
    }

    if( CV_MAT_DEPTH(points2->type) != CV_32F && enable_flt_points )
    {
        CvMat tmp = cvMat( points2->rows, points2->cols,
            (points2->type & ~CV_MAT_DEPTH_MASK) | CV_32F, points2->data.ptr );
        cvConvert( points2, &tmp );
    }
}


void CV_BaseShapeDescrTest::run_func(void)
{
}


int CV_BaseShapeDescrTest::validate_test_results( int /*test_case_idx*/ )
{
    extract_points();
    return 0;
}


CV_BaseShapeDescrTest shape_basetest( "shape", "" );


/****************************************************************************************\
*                                     Convex Hull Test                                   *
\****************************************************************************************/

class CV_ConvHullTest : public CV_BaseShapeDescrTest
{
public:
    CV_ConvHullTest();
    virtual ~CV_ConvHullTest();
    void clear();

protected:
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );

    CvSeq* hull1;
    CvMat* hull2;
    void* hull_storage;
    int orientation;
    int return_points;
};


CV_ConvHullTest::CV_ConvHullTest():
    CV_BaseShapeDescrTest( "shape-convhull", "cvConvexHull2" )
{
    hull1 = 0;
    hull2 = 0;
    hull_storage = 0;
    orientation = return_points = 0;
}


CV_ConvHullTest::~CV_ConvHullTest()
{
    clear();
}


void CV_ConvHullTest::clear()
{
    CV_BaseShapeDescrTest::clear();
    cvReleaseMat( &hull2 );
    hull1 = 0;
    hull_storage = 0;
}


int CV_ConvHullTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::prepare_test_case( test_case_idx );
    int use_storage_for_hull = 0;
    CvRNG* rng = ts->get_rng();

    if( code <= 0 )
        return code;

    orientation = cvTsRandInt(rng) % 2 ? CV_CLOCKWISE : CV_COUNTER_CLOCKWISE;
    return_points = cvTsRandInt(rng) % 2;

    use_storage_for_hull = (cvTsRandInt(rng) % 2) && !test_cpp;
    if( use_storage_for_hull )
    {
        if( !storage )
            storage = cvCreateMemStorage( (cvTsRandInt(rng)%10 + 1)*1024 );
        hull_storage = storage;
    }
    else
    {
        int rows, cols;
        int sz = points1 ? points1->total : points2->cols + points2->rows - 1;
        int point_type = points1 ? CV_SEQ_ELTYPE(points1) : CV_MAT_TYPE(points2->type);

        if( cvTsRandInt(rng) % 2 )
            rows = sz, cols = 1;
        else
            rows = 1, cols = sz;

        hull2 = cvCreateMat( rows, cols, return_points ? point_type : CV_32SC1 );
        hull_storage = hull2;
    }

    return code;
}


void CV_ConvHullTest::run_func()
{
    if(!test_cpp)
        hull1 = cvConvexHull2( points, hull_storage, orientation, return_points );
    else
    {
        cv::Mat _points = cv::cvarrToMat(points);
        bool clockwise = orientation == CV_CLOCKWISE;
        size_t n = 0;
        if( !return_points )
        {
            std::vector<int> _hull;
            cv::convexHull(_points, _hull, clockwise);
            n = _hull.size();
            memcpy(hull2->data.ptr, &_hull[0], n*sizeof(_hull[0]));
        }
        else if(_points.type() == CV_32SC2)
        {
            std::vector<cv::Point> _hull;
            cv::convexHull(_points, _hull, clockwise);
            n = _hull.size();
            memcpy(hull2->data.ptr, &_hull[0], n*sizeof(_hull[0]));
        }
        else if(_points.type() == CV_32FC2)
        {
            std::vector<cv::Point2f> _hull;
            cv::convexHull(_points, _hull, clockwise);
            n = _hull.size();
            memcpy(hull2->data.ptr, &_hull[0], n*sizeof(_hull[0]));
        }
        if(hull2->rows > hull2->cols)
            hull2->rows = (int)n;
        else
            hull2->cols = (int)n;
    }
}


int CV_ConvHullTest::validate_test_results( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    CvMat* hull = 0;
    CvMat* mask = 0;
    int i, point_count, hull_count;
    CvPoint2D32f *p, *h;
    CvSeq header, hheader, *ptseq, *hseq;
    CvSeqBlock block, hblock;

    if( points1 )
        ptseq = points1;
    else
        ptseq = cvMakeSeqHeaderForArray( CV_MAT_TYPE(points2->type),
            sizeof(CvSeq), CV_ELEM_SIZE(points2->type), points2->data.ptr,
            points2->rows + points2->cols - 1, &header, &block );
    point_count = ptseq->total;
    p = (CvPoint2D32f*)(points2->data.ptr);

    if( hull1 )
        hseq = hull1;
    else
        hseq = cvMakeSeqHeaderForArray( CV_MAT_TYPE(hull2->type),
            sizeof(CvSeq), CV_ELEM_SIZE(hull2->type), hull2->data.ptr,
            hull2->rows + hull2->cols - 1, &hheader, &hblock );
    hull_count = hseq->total;
    hull = cvCreateMat( 1, hull_count, CV_32FC2 );
    mask = cvCreateMat( 1, hull_count, CV_8UC1 );
    cvZero( mask );
    h = (CvPoint2D32f*)(hull->data.ptr);

    // extract convex hull points
    if( return_points )
    {
        cvCvtSeqToArray( hseq, hull->data.ptr );
        if( CV_SEQ_ELTYPE(hseq) != CV_32FC2 )
        {
            CvMat tmp = cvMat( hull->rows, hull->cols, CV_32SC2, hull->data.ptr );
            cvConvert( &tmp, hull );
        }
    }
    else
    {
        CvSeqReader reader;
        cvStartReadSeq( hseq, &reader );

        for( i = 0; i < hull_count; i++ )
        {
            schar* ptr = reader.ptr;
            int idx;
            CV_NEXT_SEQ_ELEM( hseq->elem_size, reader );

            if( hull1 )
                idx = cvSeqElemIdx( ptseq, *(uchar**)ptr );
            else
                idx = *(int*)ptr;

            if( idx < 0 || idx >= point_count )
            {
                ts->printf( CvTS::LOG, "Invalid convex hull point #%d\n", i );
                code = CvTS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
            h[i] = p[idx];
        }
    }

    // check that the convex hull is a convex polygon
    if( hull_count >= 3 )
    {
        CvPoint2D32f pt0 = h[hull_count-1];
        for( i = 0; i < hull_count; i++ )
        {
            int j = i+1;
            CvPoint2D32f pt1 = h[i], pt2 = h[j < hull_count ? j : 0];
            float dx0 = pt1.x - pt0.x, dy0 = pt1.y - pt0.y;
            float dx1 = pt2.x - pt1.x, dy1 = pt2.y - pt1.y;
            double t = (double)dx0*dy1 - (double)dx1*dy0;
            if( (t < 0) ^ (orientation != CV_COUNTER_CLOCKWISE) )
            {
                ts->printf( CvTS::LOG, "The convex hull is not convex or has a wrong orientation (vtx %d)\n", i );
                code = CvTS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
            pt0 = pt1;
        }
    }

    // check that all the points are inside the hull or on the hull edge
    // and at least hull_point points are at the hull vertices
    for( i = 0; i < point_count; i++ )
    {
        int idx = 0, on_edge = 0;
        double result = cvTsPointPolygonTest( p[i], h, hull_count, &idx, &on_edge );

        if( result < 0 )
        {
            ts->printf( CvTS::LOG, "The point #%d is outside of the convex hull\n", i );
            code = CvTS::FAIL_BAD_ACCURACY;
            goto _exit_;
        }

        if( result < FLT_EPSILON && !on_edge )
            mask->data.ptr[idx] = (uchar)1;
    }

    if( cvNorm( mask, 0, CV_L1 ) != hull_count )
    {
        ts->printf( CvTS::LOG, "Not every convex hull vertex coincides with some input point\n" );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    cvReleaseMat( &hull );
    cvReleaseMat( &mask );
    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_ConvHullTest shape_convhull_test;


/****************************************************************************************\
*                                     MinAreaRect Test                                   *
\****************************************************************************************/

class CV_MinAreaRectTest : public CV_BaseShapeDescrTest
{
public:
    CV_MinAreaRectTest();

protected:
    void run_func(void);
    int validate_test_results( int test_case_idx );

    CvBox2D box;
    CvPoint2D32f box_pt[4];
};


CV_MinAreaRectTest::CV_MinAreaRectTest():
    CV_BaseShapeDescrTest( "shape-minarearect", "cvMinAreaRect2, cvBoxPoints" )
{
}


void CV_MinAreaRectTest::run_func()
{
    if(!test_cpp)
    {
        box = cvMinAreaRect2( points, storage );
        cvBoxPoints( box, box_pt );
    }
    else
    {
        cv::RotatedRect r = cv::minAreaRect(cv::cvarrToMat(points));
        box = (CvBox2D)r;
        r.points((cv::Point2f*)box_pt);
    }
}


int CV_MinAreaRectTest::validate_test_results( int test_case_idx )
{
    double eps = 1e-1;
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    int i, j, point_count = points2->rows + points2->cols - 1;
    CvPoint2D32f *p = (CvPoint2D32f*)(points2->data.ptr);
    int mask[] = {0,0,0,0};

    // check that the bounding box is a rotated rectangle:
    //  1. diagonals should be equal
    //  2. they must intersect in their middle points
    {
        double d0 = cvTsDist( box_pt[0], box_pt[2] );
        double d1 = cvTsDist( box_pt[1], box_pt[3] );

        double x0 = (box_pt[0].x + box_pt[2].x)*0.5;
        double y0 = (box_pt[0].y + box_pt[2].y)*0.5;
        double x1 = (box_pt[1].x + box_pt[3].x)*0.5;
        double y1 = (box_pt[1].y + box_pt[3].y)*0.5;

        if( fabs(d0 - d1) + fabs(x0 - x1) + fabs(y0 - y1) > eps*MAX(d0,d1) )
        {
            ts->printf( CvTS::LOG, "The bounding box is not a rectangle\n" );
            code = CvTS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }
    }

#if 0
    {
    int n = 4;
    double a = 8, c = 8, b = 100, d = 150;
    CvPoint bp[4], *bpp = bp;
    cvNamedWindow( "test", 1 );
    IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );
    cvZero(img);
    for( i = 0; i < point_count; i++ )
        cvCircle(img,cvPoint(cvRound(p[i].x*a+b),cvRound(p[i].y*c+d)), 3, CV_RGB(0,255,0), -1 );
    for( i = 0; i < n; i++ )
        bp[i] = cvPoint(cvRound(box_pt[i].x*a+b),cvRound(box_pt[i].y*c+d));
    cvPolyLine( img, &bpp, &n, 1, 1, CV_RGB(255,255,0), 1, CV_AA, 0 );
    cvShowImage( "test", img );
    cvWaitKey();
    cvReleaseImage(&img);
    }
#endif

    // check that the box includes all the points
    // and there is at least one point at (or very close to) every box side
    for( i = 0; i < point_count; i++ )
    {
        int idx = 0, on_edge = 0;
        double result = cvTsPointPolygonTest( p[i], box_pt, 4, &idx, &on_edge );
        if( result < -eps )
        {
            ts->printf( CvTS::LOG, "The point #%d is outside of the box\n", i );
            code = CvTS::FAIL_BAD_ACCURACY;
            goto _exit_;
        }

        if( result < eps )
        {
            for( j = 0; j < 4; j++ )
            {
                double d = cvTsPtLineDist( p[i], box_pt[(j-1)&3], box_pt[j] );
                if( d < eps )
                    mask[j] = (uchar)1;
            }
        }
    }

    if( mask[0] + mask[1] + mask[2] + mask[3] != 4 )
    {
        ts->printf( CvTS::LOG, "Not every box side has a point nearby\n" );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_MinAreaRectTest shape_minarearect_test;


/****************************************************************************************\
*                                     MinEnclosingCircle Test                            *
\****************************************************************************************/

class CV_MinCircleTest : public CV_BaseShapeDescrTest
{
public:
    CV_MinCircleTest();

protected:
    void run_func(void);
    int validate_test_results( int test_case_idx );

    CvPoint2D32f center;
    float radius;
};


CV_MinCircleTest::CV_MinCircleTest():
    CV_BaseShapeDescrTest( "shape-mincircle", "cvMinEnclosingCircle" )
{
}


void CV_MinCircleTest::run_func()
{
    if(!test_cpp)
        cvMinEnclosingCircle( points, &center, &radius );
    else
        cv::minEnclosingCircle(cv::cvarrToMat(points), (cv::Point2f&)center, radius);
}


int CV_MinCircleTest::validate_test_results( int test_case_idx )
{
    double eps = 1.03;
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    int i, j = 0, point_count = points2->rows + points2->cols - 1;
    CvPoint2D32f *p = (CvPoint2D32f*)(points2->data.ptr);
    CvPoint2D32f v[3];

#if 0
    {
    double a = 2, b = 200, d = 400;
    cvNamedWindow( "test", 1 );
    IplImage* img = cvCreateImage( cvSize(500,500), 8, 3 );
    cvZero(img);
    for( i = 0; i < point_count; i++ )
        cvCircle(img,cvPoint(cvRound(p[i].x*a+b),cvRound(p[i].y*a+d)), 3, CV_RGB(0,255,0), -1 );
    cvCircle( img, cvPoint(cvRound(center.x*a+b),cvRound(center.y*a+d)),
              cvRound(radius*a), CV_RGB(255,255,0), 1 );
    cvShowImage( "test", img );
    cvWaitKey();
    cvReleaseImage(&img);
    }
#endif

    // check that the circle contains all the points inside and
    // remember at most 3 points that are close to the boundary
    for( i = 0; i < point_count; i++ )
    {
        double d = cvTsDist( p[i], center );
        if( d > radius )
        {
            ts->printf( CvTS::LOG, "The point #%d is outside of the circle\n", i );
            code = CvTS::FAIL_BAD_ACCURACY;
            goto _exit_;
        }

        if( radius - d < eps*radius && j < 3 )
            v[j++] = p[i];
    }

    if( point_count >= 2 && (j < 2 || (j == 2 && cvTsDist(v[0],v[1]) < (radius-1)*2/eps)) )
    {
        ts->printf( CvTS::LOG,
            "There should be at at least 3 points near the circle boundary or 2 points on the diameter\n" );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_MinCircleTest shape_mincircle_test;


/****************************************************************************************\
*                                   Perimeter Test                                     *
\****************************************************************************************/

class CV_PerimeterTest : public CV_BaseShapeDescrTest
{
public:
    CV_PerimeterTest();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    CvSlice slice;
    int is_closed;
    double result;
};


CV_PerimeterTest::CV_PerimeterTest():
    CV_BaseShapeDescrTest( "shape-perimeter", "cvArcLength" )
{
}


int CV_PerimeterTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::prepare_test_case( test_case_idx );
    CvRNG* rng = ts->get_rng();
    int total;

    if( code < 0 )
        return code;

    is_closed = cvTsRandInt(rng) % 2;

    if( points1 )
    {
        points1->flags |= CV_SEQ_KIND_CURVE;
        if( is_closed )
            points1->flags |= CV_SEQ_FLAG_CLOSED;
        total = points1->total;
    }
    else
        total = points2->cols + points2->rows - 1;

    if( (cvTsRandInt(rng) % 3) && !test_cpp )
    {
        slice.start_index = cvTsRandInt(rng) % total;
        slice.end_index = cvTsRandInt(rng) % total;
    }
    else
        slice = CV_WHOLE_SEQ;

    return 1;
}


void CV_PerimeterTest::run_func()
{
    if(!test_cpp)
        result = cvArcLength( points, slice, points1 ? -1 : is_closed );
    else
        result = cv::arcLength(cv::cvarrToMat(points),
            !points1 ? is_closed != 0 : (points1->flags & CV_SEQ_FLAG_CLOSED) != 0);
}


int CV_PerimeterTest::validate_test_results( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    int i, len = slice.end_index - slice.start_index, total = points2->cols + points2->rows - 1;
    double result0 = 0;
    CvPoint2D32f prev_pt, pt, *ptr;

    if( len < 0 )
        len += total;

    len = MIN( len, total );
    len -= !is_closed && len == total;

    ptr = (CvPoint2D32f*)points2->data.fl;
    prev_pt = ptr[slice.start_index % total];

    for( i = 1; i <= len; i++ )
    {
        pt = ptr[(i + slice.start_index) % total];
        double dx = pt.x - prev_pt.x, dy = pt.y - prev_pt.y;
        result0 += sqrt(dx*dx + dy*dy);
        prev_pt = pt;
    }

    if( cvIsNaN(result) || cvIsInf(result) )
    {
        ts->printf( CvTS::LOG, "cvArcLength() returned invalid value (%g)\n", result );
        code = CvTS::FAIL_INVALID_OUTPUT;
    }
    else if( fabs(result - result0) > FLT_EPSILON*100*result0 )
    {
        ts->printf( CvTS::LOG, "The function returned %g, while the correct result is %g\n", result, result0 );
        code = CvTS::FAIL_BAD_ACCURACY;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_PerimeterTest shape_perimeter_test;



/****************************************************************************************\
*                                   FitEllipse Test                                      *
\****************************************************************************************/

class CV_FitEllipseTest : public CV_BaseShapeDescrTest
{
public:
    CV_FitEllipseTest();

protected:
    int prepare_test_case( int test_case_idx );
    void generate_point_set( void* points );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    CvBox2D box0, box;
    double min_ellipse_size, max_noise;
};


CV_FitEllipseTest::CV_FitEllipseTest():
    CV_BaseShapeDescrTest( "shape-fit-ellipse", "cvFitEllipse" )
{
    min_log_size = 5; // for robust ellipse fitting a dozen of points is needed at least
    max_log_size = 10;
    min_ellipse_size = 10;
    max_noise = 0.05;
}


void CV_FitEllipseTest::generate_point_set( void* points )
{
    CvRNG* rng = ts->get_rng();
    int i, total, point_type;
    CvSeqReader reader;
    uchar* data = 0;
    double a, b;

    box0.center.x = (float)((low.val[0] + high.val[0])*0.5);
    box0.center.y = (float)((low.val[1] + high.val[1])*0.5);
    box0.size.width = (float)(MAX(high.val[0] - low.val[0], min_ellipse_size)*2);
    box0.size.height = (float)(MAX(high.val[1] - low.val[1], min_ellipse_size)*2);
    box0.angle = (float)(cvTsRandReal(rng)*180);
    a = cos(box0.angle*CV_PI/180.);
    b = sin(box0.angle*CV_PI/180.);

    if( box0.size.width > box0.size.height )
    {
        float t;
        CV_SWAP( box0.size.width, box0.size.height, t );
    }
    memset( &reader, 0, sizeof(reader) );

    if( CV_IS_SEQ(points) )
    {
        CvSeq* ptseq = (CvSeq*)points;
        total = ptseq->total;
        point_type = CV_SEQ_ELTYPE(ptseq);
        cvStartReadSeq( ptseq, &reader );
    }
    else
    {
        CvMat* ptm = (CvMat*)points;
        assert( CV_IS_MAT(ptm) && CV_IS_MAT_CONT(ptm->type) );
        total = ptm->rows + ptm->cols - 1;
        point_type = CV_MAT_TYPE(ptm->type);
        data = ptm->data.ptr;
    }

    assert( point_type == CV_32SC2 || point_type == CV_32FC2 );

    for( i = 0; i < total; i++ )
    {
        CvPoint* pp;
        CvPoint2D32f p;
        double angle = cvTsRandReal(rng)*CV_PI*2;
        double x = box0.size.height*0.5*(cos(angle) + (cvTsRandReal(rng)-0.5)*2*max_noise);
        double y = box0.size.width*0.5*(sin(angle) + (cvTsRandReal(rng)-0.5)*2*max_noise);
        p.x = (float)(box0.center.x + a*x + b*y);
        p.y = (float)(box0.center.y - b*x + a*y);

        if( reader.ptr )
        {
            pp = (CvPoint*)reader.ptr;
            CV_NEXT_SEQ_ELEM( sizeof(*pp), reader );
        }
        else
            pp = ((CvPoint*)data) + i;
        if( point_type == CV_32SC2 )
        {
            pp->x = cvRound(p.x);
            pp->y = cvRound(p.y);
        }
        else
            *(CvPoint2D32f*)pp = p;
    }
}


int CV_FitEllipseTest::prepare_test_case( int test_case_idx )
{
    min_log_size = MAX(min_log_size,4);
    max_log_size = MAX(min_log_size,max_log_size);
    return CV_BaseShapeDescrTest::prepare_test_case( test_case_idx );
}


void CV_FitEllipseTest::run_func()
{
    if(!test_cpp)
        box = cvFitEllipse2( points );
    else
        box = (CvBox2D)cv::fitEllipse(cv::cvarrToMat(points));
}


int CV_FitEllipseTest::validate_test_results( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    double diff_angle;

    if( cvIsNaN(box.center.x) || cvIsInf(box.center.x) ||
        cvIsNaN(box.center.y) || cvIsInf(box.center.y) ||
        cvIsNaN(box.size.width) || cvIsInf(box.size.width) ||
        cvIsNaN(box.size.height) || cvIsInf(box.size.height) ||
        cvIsNaN(box.angle) || cvIsInf(box.angle) )
    {
        ts->printf( CvTS::LOG, "Some of the computed ellipse parameters are invalid (x=%g,y=%g,w=%g,h=%g,angle=%g)\n",
            box.center.x, box.center.y, box.size.width, box.size.height, box.angle );
        code = CvTS::FAIL_INVALID_OUTPUT;
        goto _exit_;
    }

    box.angle = (float)(90-box.angle);
    if( box.angle < 0 )
        box.angle += 360;
    if( box.angle > 360 )
        box.angle -= 360;

    if( fabs(box.center.x - box0.center.x) > 3 ||
        fabs(box.center.y - box0.center.y) > 3 ||
        fabs(box.size.width - box0.size.width) > 0.1*fabs(box0.size.width) ||
        fabs(box.size.height - box0.size.height) > 0.1*fabs(box0.size.height) )
    {
        ts->printf( CvTS::LOG, "The computed ellipse center and/or size are incorrect:\n\t"
            "(x=%.1f,y=%.1f,w=%.1f,h=%.1f), while it should be (x=%.1f,y=%.1f,w=%.1f,h=%.1f)\n",
            box.center.x, box.center.y, box.size.width, box.size.height,
            box0.center.x, box0.center.y, box0.size.width, box0.size.height );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    diff_angle = fabs(box0.angle - box.angle);
    diff_angle = MIN( diff_angle, fabs(diff_angle - 360));
    diff_angle = MIN( diff_angle, fabs(diff_angle - 180));

    if( box0.size.height >= 1.3*box0.size.width && diff_angle > 30 )
    {
        ts->printf( CvTS::LOG, "Incorrect ellipse angle (=%1.f, should be %1.f)\n",
            box.angle, box0.angle );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

#if 0
    cvNamedWindow( "test", 0 );
    IplImage* img = cvCreateImage( cvSize(cvRound(low_high_range*4),
        cvRound(low_high_range*4)), 8, 3 );
    cvZero( img );

    box.center.x += (float)low_high_range*2;
    box.center.y += (float)low_high_range*2;
    cvEllipseBox( img, box, CV_RGB(255,0,0), 3, 8 );

    for( int i = 0; i < points2->rows + points2->cols - 1; i++ )
    {
        CvPoint pt;
        pt.x = cvRound(points2->data.fl[i*2] + low_high_range*2);
        pt.y = cvRound(points2->data.fl[i*2+1] + low_high_range*2);
        cvCircle( img, pt, 1, CV_RGB(255,255,255), -1, 8 );
    }

    cvShowImage( "test", img );
    cvReleaseImage( &img );
    cvWaitKey(0);
#endif

    if( code < 0 )
    {
        ts->set_failed_test_info( code );
    }
    return code;
}


CV_FitEllipseTest fit_ellipse_test;



/****************************************************************************************\
*                                   FitLine Test                                         *
\****************************************************************************************/

class CV_FitLineTest : public CV_BaseShapeDescrTest
{
public:
    CV_FitLineTest();

protected:
    int prepare_test_case( int test_case_idx );
    void generate_point_set( void* points );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    double max_noise;
    float line[6], line0[6];
    int dist_type;
    double reps, aeps;
};


CV_FitLineTest::CV_FitLineTest():
    CV_BaseShapeDescrTest( "shape-fit-line", "cvFitLine" )
{
    min_log_size = 5; // for robust ellipse fitting a dozen of points is needed at least
    max_log_size = 10;
    max_noise = 0.05;
}


void CV_FitLineTest::generate_point_set( void* points )
{
    CvRNG* rng = ts->get_rng();
    int i, k, n, total, point_type;
    CvSeqReader reader;
    uchar* data = 0;
    double s = 0;

    n = dims;
    for( k = 0; k < n; k++ )
    {
        line0[k+n] = (float)((low.val[k] + high.val[k])*0.5);
        line0[k] = (float)(high.val[k] - low.val[k]);
        if( cvTsRandInt(rng) % 2 )
            line0[k] = -line0[k];
        s += (double)line0[k]*line0[k];
    }

    s = 1./sqrt(s);
    for( k = 0; k < n; k++ )
        line0[k] = (float)(line0[k]*s);

    memset( &reader, 0, sizeof(reader) );

    if( CV_IS_SEQ(points) )
    {
        CvSeq* ptseq = (CvSeq*)points;
        total = ptseq->total;
        point_type = CV_MAT_DEPTH(CV_SEQ_ELTYPE(ptseq));
        cvStartReadSeq( ptseq, &reader );
    }
    else
    {
        CvMat* ptm = (CvMat*)points;
        assert( CV_IS_MAT(ptm) && CV_IS_MAT_CONT(ptm->type) );
        total = ptm->rows + ptm->cols - 1;
        point_type = CV_MAT_DEPTH(CV_MAT_TYPE(ptm->type));
        data = ptm->data.ptr;
    }

    for( i = 0; i < total; i++ )
    {
        int* pi;
        float* pf;
        float p[4], t;
        if( reader.ptr )
        {
            pi = (int*)reader.ptr;
            pf = (float*)reader.ptr;
            CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        }
        else
        {
            pi = (int*)data + i*n;
            pf = (float*)data + i*n;
        }

        t = (float)((cvTsRandReal(rng)-0.5)*low_high_range*2);

        for( k = 0; k < n; k++ )
            p[k] = (float)((cvTsRandReal(rng)-0.5)*max_noise*2 + t*line0[k] + line0[k+n]);

        if( point_type == CV_32S )
            for( k = 0; k < n; k++ )
                pi[k] = cvRound(p[k]);
        else
            for( k = 0; k < n; k++ )
                pf[k] = p[k];
    }
}


int CV_FitLineTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    dims = cvTsRandInt(rng) % 2 + 2;
    min_log_size = MAX(min_log_size,5);
    max_log_size = MAX(min_log_size,max_log_size);
    int code = CV_BaseShapeDescrTest::prepare_test_case( test_case_idx );
    dist_type = cvTsRandInt(rng) % 6 + 1;
    dist_type += dist_type == CV_DIST_C;
    reps = 0.1; aeps = 0.01;
    return code;
}


void CV_FitLineTest::run_func()
{
    if(!test_cpp)
        cvFitLine( points, dist_type, 0, reps, aeps, line );
    else if(dims == 2)
        cv::fitLine(cv::cvarrToMat(points), (cv::Vec4f&)line[0], dist_type, 0, reps, aeps);
    else
        cv::fitLine(cv::cvarrToMat(points), (cv::Vec6f&)line[0], dist_type, 0, reps, aeps);
}


int CV_FitLineTest::validate_test_results( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    int k, max_k = 0;
    double vec_diff = 0, t;

    for( k = 0; k < dims*2; k++ )
    {
        if( cvIsNaN(line[k]) || cvIsInf(line[k]) )
        {
            ts->printf( CvTS::LOG, "Some of the computed line parameters are invalid (line[%d]=%g)\n",
                k, line[k] );
            code = CvTS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }
    }

    if( fabs(line0[1]) > fabs(line0[0]) )
        max_k = 1;
    if( fabs(line0[dims-1]) > fabs(line0[max_k]) )
        max_k = dims-1;
    if( line0[max_k] < 0 )
        for( k = 0; k < dims; k++ )
            line0[k] = -line0[k];
    if( line[max_k] < 0 )
        for( k = 0; k < dims; k++ )
            line[k] = -line[k];

    for( k = 0; k < dims; k++ )
    {
        double dt = line[k] - line0[k];
        vec_diff += dt*dt;
    }

    if( sqrt(vec_diff) > 0.05 )
    {
        if( dims == 2 )
            ts->printf( CvTS::LOG,
                "The computed line vector (%.2f,%.2f) is different from the actual (%.2f,%.2f)\n",
                line[0], line[1], line0[0], line0[1] );
        else
            ts->printf( CvTS::LOG,
                "The computed line vector (%.2f,%.2f,%.2f) is different from the actual (%.2f,%.2f,%.2f)\n",
                line[0], line[1], line[2], line0[0], line0[1], line0[2] );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    t = (line[max_k+dims] - line0[max_k+dims])/line0[max_k];
    for( k = 0; k < dims; k++ )
    {
        double p = line0[k+dims] + t*line0[k] - line[k+dims];
        vec_diff += p*p;
    }

    if( sqrt(vec_diff) > 1*MAX(fabs(t),1) )
    {
        if( dims == 2 )
            ts->printf( CvTS::LOG,
                "The computed line point (%.2f,%.2f) is too far from the actual line\n",
                line[2]+line0[2], line[3]+line0[3] );
        else
            ts->printf( CvTS::LOG,
                "The computed line point (%.2f,%.2f,%.2f) is too far from the actual line\n",
                line[3]+line0[3], line[4]+line0[4], line[5]+line0[5] );
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    if( code < 0 )
    {
        ts->set_failed_test_info( code );
    }
    return code;
}


CV_FitLineTest fit_line_test;


/****************************************************************************************\
*                                   ContourMoments Test                                  *
\****************************************************************************************/


static void
cvTsGenerateTousledBlob( CvPoint2D32f center, CvSize2D32f axes,
    double max_r_scale, double angle, CvArr* points, CvRNG* rng )
{
    int i, total, point_type;
    uchar* data = 0;
    CvSeqReader reader;
    memset( &reader, 0, sizeof(reader) );

    if( CV_IS_SEQ(points) )
    {
        CvSeq* ptseq = (CvSeq*)points;
        total = ptseq->total;
        point_type = CV_SEQ_ELTYPE(ptseq);
        cvStartReadSeq( ptseq, &reader );
    }
    else
    {
        CvMat* ptm = (CvMat*)points;
        assert( CV_IS_MAT(ptm) && CV_IS_MAT_CONT(ptm->type) );
        total = ptm->rows + ptm->cols - 1;
        point_type = CV_MAT_TYPE(ptm->type);
        data = ptm->data.ptr;
    }

    assert( point_type == CV_32SC2 || point_type == CV_32FC2 );

    for( i = 0; i < total; i++ )
    {
        CvPoint* pp;
        CvPoint2D32f p;

        double phi0 = 2*CV_PI*i/total;
        double phi = CV_PI*angle/180.;
        double t = cvTsRandReal(rng)*max_r_scale + (1 - max_r_scale);
        double ta = axes.height*t;
        double tb = axes.width*t;
        double c0 = cos(phi0)*ta, s0 = sin(phi0)*tb;
        double c = cos(phi), s = sin(phi);
        p.x = (float)(c0*c - s0*s + center.x);
        p.y = (float)(c0*s + s0*c + center.y);

        if( reader.ptr )
        {
            pp = (CvPoint*)reader.ptr;
            CV_NEXT_SEQ_ELEM( sizeof(*pp), reader );
        }
        else
            pp = ((CvPoint*)data) + i;

        if( point_type == CV_32SC2 )
        {
            pp->x = cvRound(p.x);
            pp->y = cvRound(p.y);
        }
        else
            *(CvPoint2D32f*)pp = p;
    }
}


class CV_ContourMomentsTest : public CV_BaseShapeDescrTest
{
public:
    CV_ContourMomentsTest();

protected:
    int prepare_test_case( int test_case_idx );
    void generate_point_set( void* points );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    CvMoments moments0, moments;
    double area0, area;
    CvSize2D32f axes;
    CvPoint2D32f center;
    int max_max_r_scale;
    double max_r_scale, angle;
    CvSize img_size;
};


CV_ContourMomentsTest::CV_ContourMomentsTest():
    CV_BaseShapeDescrTest( "shape-moments", "cvMoments, cvContourArea" )
{
    min_log_size = 3;
    max_log_size = 8;
    max_max_r_scale = 15;
    low_high_range = 200;
    enable_flt_points = false;
}


void CV_ContourMomentsTest::generate_point_set( void* points )
{
    CvRNG* rng = ts->get_rng();
    float max_sz;

    axes.width = (float)((cvTsRandReal(rng)*0.9 + 0.1)*low_high_range);
    axes.height = (float)((cvTsRandReal(rng)*0.9 + 0.1)*low_high_range);
    max_sz = MAX(axes.width, axes.height);

    img_size.width = img_size.height = cvRound(low_high_range*2.2);

    center.x = (float)(img_size.width*0.5 + (cvTsRandReal(rng)-0.5)*(img_size.width - max_sz*2)*0.8);
    center.y = (float)(img_size.height*0.5 + (cvTsRandReal(rng)-0.5)*(img_size.height - max_sz*2)*0.8);

    assert( 0 < center.x - max_sz && center.x + max_sz < img_size.width &&
        0 < center.y - max_sz && center.y + max_sz < img_size.height );

    max_r_scale = cvTsRandReal(rng)*max_max_r_scale*0.01;
    angle = cvTsRandReal(rng)*360;

    cvTsGenerateTousledBlob( center, axes, max_r_scale, angle, points, rng );

    if( points1 )
        points1->flags = CV_SEQ_MAGIC_VAL + CV_SEQ_POLYGON;
}


int CV_ContourMomentsTest::prepare_test_case( int test_case_idx )
{
    min_log_size = MAX(min_log_size,3);
    max_log_size = MIN(max_log_size,8);
    max_log_size = MAX(min_log_size,max_log_size);
    int code = CV_BaseShapeDescrTest::prepare_test_case( test_case_idx );
    return code;
}


void CV_ContourMomentsTest::run_func()
{
    if(!test_cpp)
    {
        cvMoments( points, &moments );
        area = cvContourArea( points );
    }
    else
    {
        moments = (CvMoments)cv::moments(cv::cvarrToMat(points));
        area = cv::contourArea(cv::cvarrToMat(points));
    }
}


int CV_ContourMomentsTest::validate_test_results( int test_case_idx )
{
    int code = CV_BaseShapeDescrTest::validate_test_results( test_case_idx );
    int i, n = (int)(sizeof(moments)/sizeof(moments.inv_sqrt_m00));
    CvMat* img = cvCreateMat( img_size.height, img_size.width, CV_8UC1 );
    CvPoint* pt = (CvPoint*)points2->data.i;
    int count = points2->cols + points2->rows - 1;
    double max_v0 = 0;

    cvZero(img);
    cvFillPoly( img, &pt, &count, 1, cvScalarAll(1));
    cvMoments( img, &moments0 );

    for( i = 0; i < n; i++ )
    {
        double t = fabs((&moments0.m00)[i]);
        max_v0 = MAX(max_v0, t);
    }

    for( i = 0; i <= n; i++ )
    {
        double v = i < n ? (&moments.m00)[i] : area;
        double v0 = i < n ? (&moments0.m00)[i] : moments0.m00;

        if( cvIsNaN(v) || cvIsInf(v) )
        {
            ts->printf( CvTS::LOG,
                "The contour %s is invalid (=%g)\n", i < n ? "moment" : "area", v );
            code = CvTS::FAIL_INVALID_OUTPUT;
            break;
        }

        if( fabs(v - v0) > 0.1*max_v0 )
        {
            ts->printf( CvTS::LOG,
                "The computed contour %s is %g, while it should be %g\n",
                i < n ? "moment" : "area", v, v0 );
            code = CvTS::FAIL_BAD_ACCURACY;
            break;
        }
    }

    if( code < 0 )
    {
#if 0
        cvCmpS( img, 0, img, CV_CMP_GT );
        cvNamedWindow( "test", 1 );
        cvShowImage( "test", img );
        cvWaitKey();
#endif
        ts->set_failed_test_info( code );
    }

    cvReleaseMat( &img );
    return code;
}


CV_ContourMomentsTest contour_moments_test;


/* End of file. */

