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

class CV_ImgWarpBaseTest : public cvtest::ArrayTest
{
public:
    CV_ImgWarpBaseTest( bool warp_matrix );

protected:
    int read_params( CvFileStorage* fs );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

    int interpolation;
    int max_interpolation;
    double spatial_scale_zoom, spatial_scale_decimate;
};


CV_ImgWarpBaseTest::CV_ImgWarpBaseTest( bool warp_matrix )
{
    test_array[INPUT].push_back(NULL);
    if( warp_matrix )
        test_array[INPUT].push_back(NULL);
    test_array[INPUT_OUTPUT].push_back(NULL);
    test_array[REF_INPUT_OUTPUT].push_back(NULL);
    max_interpolation = 5;
    interpolation = 0;
    element_wise_relative_error = false;
    spatial_scale_zoom = 0.01;
    spatial_scale_decimate = 0.005;
}


int CV_ImgWarpBaseTest::read_params( CvFileStorage* fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    return code;
}


void CV_ImgWarpBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( CV_MAT_DEPTH(type) == CV_32F )
    {
        low = Scalar::all(-10.);
        high = Scalar::all(10);
    }
}


void CV_ImgWarpBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 3;
    int cn = cvtest::randInt(rng) % 3 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    cn += cn == 2;

    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    if( test_array[INPUT].size() > 1 )
        types[INPUT][1] = cvtest::randInt(rng) & 1 ? CV_32FC1 : CV_64FC1;

    interpolation = cvtest::randInt(rng) % max_interpolation;
}


void CV_ImgWarpBaseTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT || j != 0 )
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
}

int CV_ImgWarpBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    Mat& img = test_mat[INPUT][0];
    int i, j, cols = img.cols;
    int type = img.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    double scale = depth == CV_16U ? 1000. : 255.*0.5;
    double space_scale = spatial_scale_decimate;
    vector<float> buffer(img.cols*cn);

    if( code <= 0 )
        return code;

    if( test_mat[INPUT_OUTPUT][0].cols >= img.cols &&
        test_mat[INPUT_OUTPUT][0].rows >= img.rows )
        space_scale = spatial_scale_zoom;

    for( i = 0; i < img.rows; i++ )
    {
        uchar* ptr = img.ptr(i);
        switch( cn )
        {
        case 1:
            for( j = 0; j < cols; j++ )
                buffer[j] = (float)((sin((i+1)*space_scale)*sin((j+1)*space_scale)+1.)*scale);
            break;
        case 2:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*2] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*2+1] = (float)((sin((i+j)*space_scale)+1.)*scale);
            }
            break;
        case 3:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*3] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*3+1] = (float)((sin(j*space_scale)+1.)*scale);
                buffer[j*3+2] = (float)((sin((i+j)*space_scale)+1.)*scale);
            }
            break;
        case 4:
            for( j = 0; j < cols; j++ )
            {
                buffer[j*4] = (float)((sin((i+1)*space_scale)+1.)*scale);
                buffer[j*4+1] = (float)((sin(j*space_scale)+1.)*scale);
                buffer[j*4+2] = (float)((sin((i+j)*space_scale)+1.)*scale);
                buffer[j*4+3] = (float)((sin((i-j)*space_scale)+1.)*scale);
            }
            break;
        default:
            assert(0);
        }

        /*switch( depth )
        {
        case CV_8U:
            for( j = 0; j < cols*cn; j++ )
                ptr[j] = (uchar)cvRound(buffer[j]);
            break;
        case CV_16U:
            for( j = 0; j < cols*cn; j++ )
                ((ushort*)ptr)[j] = (ushort)cvRound(buffer[j]);
            break;
        case CV_32F:
            for( j = 0; j < cols*cn; j++ )
                ((float*)ptr)[j] = (float)buffer[j];
            break;
        default:
            assert(0);
        }*/
        cv::Mat src(1, cols*cn, CV_32F, &buffer[0]);
        cv::Mat dst(1, cols*cn, depth, ptr);
        src.convertTo(dst, dst.type());
    }

    return code;
}


/////////////////////////

class CV_ResizeTest : public CV_ImgWarpBaseTest
{
public:
    CV_ResizeTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_ResizeTest::CV_ResizeTest() : CV_ImgWarpBaseTest( false )
{
}


void CV_ResizeTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz;

    sz.width = (cvtest::randInt(rng) % sizes[INPUT][0].width) + 1;
    sz.height = (cvtest::randInt(rng) % sizes[INPUT][0].height) + 1;

    if( cvtest::randInt(rng) & 1 )
    {
        int xfactor = cvtest::randInt(rng) % 10 + 1;
        int yfactor = cvtest::randInt(rng) % 10 + 1;

        if( cvtest::randInt(rng) & 1 )
            yfactor = xfactor;

        sz.width = sizes[INPUT][0].width / xfactor;
        sz.width = MAX(sz.width,1);
        sz.height = sizes[INPUT][0].height / yfactor;
        sz.height = MAX(sz.height,1);
        sizes[INPUT][0].width = sz.width * xfactor;
        sizes[INPUT][0].height = sz.height * yfactor;
    }

    if( cvtest::randInt(rng) & 1 )
        sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = sz;
    else
    {
        sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = sizes[INPUT][0];
        sizes[INPUT][0] = sz;
    }
    if( interpolation == 4 &&
       (MIN(sizes[INPUT][0].width,sizes[INPUT_OUTPUT][0].width) < 4 ||
        MIN(sizes[INPUT][0].height,sizes[INPUT_OUTPUT][0].height) < 4))
        interpolation = 2;
}


void CV_ResizeTest::run_func()
{
    cvResize( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], interpolation );
}


double CV_ResizeTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 1e-1;
}


void CV_ResizeTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat _src = test_mat[INPUT][0], _dst = test_mat[REF_INPUT_OUTPUT][0];
    CvMat *src = &_src, *dst = &_dst;
    int i, j, k;
    CvMat* x_idx = cvCreateMat( 1, dst->cols, CV_32SC1 );
    CvMat* y_idx = cvCreateMat( 1, dst->rows, CV_32SC1 );
    int* x_tab = x_idx->data.i;
    int elem_size = CV_ELEM_SIZE(src->type);
    int drows = dst->rows, dcols = dst->cols;

    if( interpolation == CV_INTER_NN )
    {
        for( j = 0; j < dcols; j++ )
        {
            int t = (j*src->cols*2 + MIN(src->cols,dcols) - 1)/(dcols*2);
            t -= t >= src->cols;
            x_idx->data.i[j] = t*elem_size;
        }

        for( j = 0; j < drows; j++ )
        {
            int t = (j*src->rows*2 + MIN(src->rows,drows) - 1)/(drows*2);
            t -= t >= src->rows;
            y_idx->data.i[j] = t;
        }
    }
    else
    {
        double scale_x = (double)src->cols/dcols;
        double scale_y = (double)src->rows/drows;

        for( j = 0; j < dcols; j++ )
        {
            double f = ((j+0.5)*scale_x - 0.5);
            i = cvRound(f);
            x_idx->data.i[j] = (i < 0 ? 0 : i >= src->cols ? src->cols - 1 : i)*elem_size;
        }

        for( j = 0; j < drows; j++ )
        {
            double f = ((j+0.5)*scale_y - 0.5);
            i = cvRound(f);
            y_idx->data.i[j] = i < 0 ? 0 : i >= src->rows ? src->rows - 1 : i;
        }
    }

    for( i = 0; i < drows; i++ )
    {
        uchar* dptr = dst->data.ptr + dst->step*i;
        const uchar* sptr0 = src->data.ptr + src->step*y_idx->data.i[i];

        for( j = 0; j < dcols; j++, dptr += elem_size )
        {
            const uchar* sptr = sptr0 + x_tab[j];
            for( k = 0; k < elem_size; k++ )
                dptr[k] = sptr[k];
        }
    }

    cvReleaseMat( &x_idx );
    cvReleaseMat( &y_idx );
}


/////////////////////////

static void test_remap( const Mat& src, Mat& dst, const Mat& mapx, const Mat& mapy,
                        Mat* mask=0, int interpolation=CV_INTER_LINEAR )
{
    int x, y, k;
    int drows = dst.rows, dcols = dst.cols;
    int srows = src.rows, scols = src.cols;
    uchar* sptr0 = src.data;
    int depth = src.depth(), cn = src.channels();
    int elem_size = (int)src.elemSize();
    int step = (int)(src.step / CV_ELEM_SIZE(depth));
    int delta;

    if( interpolation != CV_INTER_CUBIC )
    {
        delta = 0;
        scols -= 1; srows -= 1;
    }
    else
    {
        delta = 1;
        scols = MAX(scols - 3, 0);
        srows = MAX(srows - 3, 0);
    }

    int scols1 = MAX(scols - 2, 0);
    int srows1 = MAX(srows - 2, 0);

    if( mask )
        *mask = Scalar::all(0);

    for( y = 0; y < drows; y++ )
    {
        uchar* dptr = dst.ptr(y);
        const float* mx = mapx.ptr<float>(y);
        const float* my = mapy.ptr<float>(y);
        uchar* m = mask ? mask->ptr(y) : 0;

        for( x = 0; x < dcols; x++, dptr += elem_size )
        {
            float xs = mx[x];
            float ys = my[x];
            int ixs = cvFloor(xs);
            int iys = cvFloor(ys);

            if( (unsigned)(ixs - delta - 1) >= (unsigned)scols1 ||
                (unsigned)(iys - delta - 1) >= (unsigned)srows1 )
            {
                if( m )
                    m[x] = 1;
                if( (unsigned)(ixs - delta) >= (unsigned)scols ||
                    (unsigned)(iys - delta) >= (unsigned)srows )
                    continue;
            }

            xs -= ixs;
            ys -= iys;

            switch( depth )
            {
            case CV_8U:
                {
                const uchar* sptr = sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    dptr[k] = (uchar)cvRound(v00);
                }
                }
                break;
            case CV_16U:
                {
                const ushort* sptr = (const ushort*)sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    ((ushort*)dptr)[k] = (ushort)cvRound(v00);
                }
                }
                break;
            case CV_32F:
                {
                const float* sptr = (const float*)sptr0 + iys*step + ixs*cn;
                for( k = 0; k < cn; k++ )
                {
                    float v00 = sptr[k];
                    float v01 = sptr[cn + k];
                    float v10 = sptr[step + k];
                    float v11 = sptr[step + cn + k];

                    v00 = v00 + xs*(v01 - v00);
                    v10 = v10 + xs*(v11 - v10);
                    v00 = v00 + ys*(v10 - v00);
                    ((float*)dptr)[k] = (float)v00;
                }
                }
                break;
            default:
                assert(0);
            }
        }
    }
}

/////////////////////////

class CV_WarpAffineTest : public CV_ImgWarpBaseTest
{
public:
    CV_WarpAffineTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_WarpAffineTest::CV_WarpAffineTest() : CV_ImgWarpBaseTest( true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_WarpAffineTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0];
    // run for the second time to get output of a different size
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sz;
    sizes[INPUT][1] = cvSize( 3, 2 );
}


void CV_WarpAffineTest::run_func()
{
    CvMat mtx = test_mat[INPUT][1];
    cvWarpAffine( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], &mtx, interpolation );
}


double CV_WarpAffineTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_WarpAffineTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const Mat& src = test_mat[INPUT][0];
    const Mat& dst = test_mat[INPUT_OUTPUT][0];
    Mat& mat = test_mat[INPUT][1];
    CvPoint2D32f center;
    double scale, angle;

    if( code <= 0 )
        return code;

    double buffer[6];
    Mat tmp( 2, 3, mat.type(), buffer );

    center.x = (float)((cvtest::randReal(rng)*1.2 - 0.1)*src.cols);
    center.y = (float)((cvtest::randReal(rng)*1.2 - 0.1)*src.rows);
    angle = cvtest::randReal(rng)*360;
    scale = ((double)dst.rows/src.rows + (double)dst.cols/src.cols)*0.5;
    getRotationMatrix2D(center, angle, scale).convertTo(mat, mat.depth());
    rng.fill( tmp, CV_RAND_NORMAL, Scalar::all(1.), Scalar::all(0.01) );
    cv::max(tmp, 0.9, tmp);
    cv::min(tmp, 1.1, tmp);
    cv::multiply(tmp, mat, mat, 1.);

    return code;
}


void CV_WarpAffineTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mapx(dst.size(), CV_32F), mapy(dst.size(), CV_32F);
    double m[6];
    Mat srcAb, dstAb( 2, 3, CV_64FC1, m );

    //cvInvert( &tM, &M, CV_LU );
    // [R|t] -> [R^-1 | -(R^-1)*t]
    test_mat[INPUT][1].convertTo( srcAb, CV_64F );
    Mat A = srcAb.colRange(0, 2);
    Mat b = srcAb.col(2);
    Mat invA = dstAb.colRange(0, 2);
    Mat invAb = dstAb.col(2);
    cv::invert(A, invA, CV_SVD);
    cv::gemm(invA, b, -1, Mat(), 0, invAb);

    for( int y = 0; y < dst.rows; y++ )
        for( int x = 0; x < dst.cols; x++ )
        {
            mapx.at<float>(y, x) = (float)(x*m[0] + y*m[1] + m[2]);
            mapy.at<float>(y, x) = (float)(x*m[3] + y*m[4] + m[5]);
        }

    Mat mask( dst.size(), CV_8U );
    test_remap( src, dst, mapx, mapy, &mask );
    dst.setTo(Scalar::all(0), mask);
    dst0.setTo(Scalar::all(0), mask);
}


/////////////////////////

class CV_WarpPerspectiveTest : public CV_ImgWarpBaseTest
{
public:
    CV_WarpPerspectiveTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_WarpPerspectiveTest::CV_WarpPerspectiveTest() : CV_ImgWarpBaseTest( true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_WarpPerspectiveTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0];
    // run for the second time to get output of a different size
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sz;
    sizes[INPUT][1] = cvSize( 3, 3 );
}


void CV_WarpPerspectiveTest::run_func()
{
    CvMat mtx = test_mat[INPUT][1];
    cvWarpPerspective( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], &mtx, interpolation );
}


double CV_WarpPerspectiveTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_WarpPerspectiveTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const CvMat& src = test_mat[INPUT][0];
    const CvMat& dst = test_mat[INPUT_OUTPUT][0];
    Mat& mat = test_mat[INPUT][1];
    Point2f s[4], d[4];
    int i;

    if( code <= 0 )
        return code;

    s[0] = Point2f(0,0);
    d[0] = Point2f(0,0);
    s[1] = Point2f(src.cols-1.f,0);
    d[1] = Point2f(dst.cols-1.f,0);
    s[2] = Point2f(src.cols-1.f,src.rows-1.f);
    d[2] = Point2f(dst.cols-1.f,dst.rows-1.f);
    s[3] = Point2f(0,src.rows-1.f);
    d[3] = Point2f(0,dst.rows-1.f);

    float bufer[16];
    Mat tmp( 1, 16, CV_32FC1, bufer );

    rng.fill( tmp, CV_RAND_NORMAL, Scalar::all(0.), Scalar::all(0.1) );

    for( i = 0; i < 4; i++ )
    {
        s[i].x += bufer[i*4]*src.cols/2;
        s[i].y += bufer[i*4+1]*src.rows/2;
        d[i].x += bufer[i*4+2]*dst.cols/2;
        d[i].y += bufer[i*4+3]*dst.rows/2;
    }

    cv::getPerspectiveTransform( s, d ).convertTo( mat, mat.depth() );
    return code;
}


void CV_WarpPerspectiveTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mapx(dst.size(), CV_32F), mapy(dst.size(), CV_32F);
    double m[9];
    Mat srcM, dstM(3, 3, CV_64F, m);

    //cvInvert( &tM, &M, CV_LU );
    // [R|t] -> [R^-1 | -(R^-1)*t]
    test_mat[INPUT][1].convertTo( srcM, CV_64F );
    cv::invert(srcM, dstM, CV_SVD);

    for( int y = 0; y < dst.rows; y++ )
    {
        for( int x = 0; x < dst.cols; x++ )
        {
            double xs = x*m[0] + y*m[1] + m[2];
            double ys = x*m[3] + y*m[4] + m[5];
            double ds = x*m[6] + y*m[7] + m[8];

            ds = ds ? 1./ds : 0;
            xs *= ds;
            ys *= ds;

            mapx.at<float>(y, x) = (float)xs;
            mapy.at<float>(y, x) = (float)ys;
        }
    }

    Mat mask( dst.size(), CV_8U );
    test_remap( src, dst, mapx, mapy, &mask );
    dst.setTo(Scalar::all(0), mask);
    dst0.setTo(Scalar::all(0), mask);
}


/////////////////////////

class CV_RemapTest : public CV_ImgWarpBaseTest
{
public:
    CV_RemapTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );
};


CV_RemapTest::CV_RemapTest() : CV_ImgWarpBaseTest( false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_RemapTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[INPUT][1] = types[INPUT][2] = CV_32FC1;
    interpolation = CV_INTER_LINEAR;
}


void CV_RemapTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTest::fill_array( test_case_idx, i, j, arr );
}


void CV_RemapTest::run_func()
{
    cvRemap( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
             test_array[INPUT][1], test_array[INPUT][2], interpolation );
}


double CV_RemapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_RemapTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const Mat& src = test_mat[INPUT][0];
    double a[9] = {0,0,0,0,0,0,0,0,1}, k[4];
    Mat _a( 3, 3, CV_64F, a );
    Mat _k( 4, 1, CV_64F, k );
    double sz = MAX(src.rows, src.cols);

    if( code <= 0 )
        return code;

    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    a[2] = (src.cols - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (src.rows - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];
    k[0] = cvtest::randReal(rng)*0.06 - 0.03;
    k[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( k[0]*k[1] > 0 )
        k[1] = -k[1];
    k[2] = cvtest::randReal(rng)*0.004 - 0.002;
    k[3] = cvtest::randReal(rng)*0.004 - 0.002;

    cvtest::initUndistortMap( _a, _k, test_mat[INPUT][1].size(), test_mat[INPUT][1], test_mat[INPUT][2] );
    return code;
}


void CV_RemapTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mask( dst.size(), CV_8U );
    test_remap(test_mat[INPUT][0], dst, test_mat[INPUT][1],
               test_mat[INPUT][2], &mask, interpolation );
    dst.setTo(Scalar::all(0), mask);
    dst0.setTo(Scalar::all(0), mask);
}


////////////////////////////// undistort /////////////////////////////////

class CV_UndistortTest : public CV_ImgWarpBaseTest
{
public:
    CV_UndistortTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

private:
    bool useCPlus;
    cv::Mat input0;
    cv::Mat input1;
    cv::Mat input2;
    cv::Mat input_new_cam;
    cv::Mat input_output;

    bool zero_new_cam;
    bool zero_distortion;
};


CV_UndistortTest::CV_UndistortTest() : CV_ImgWarpBaseTest( false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_UndistortTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int type = types[INPUT][0];
    type = CV_MAKETYPE( CV_8U, CV_MAT_CN(type) );
    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = type;
    types[INPUT][1] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][2] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][1] = cvSize(3,3);
    sizes[INPUT][2] = cvtest::randInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);
    types[INPUT][3] =  types[INPUT][1];
    sizes[INPUT][3] = sizes[INPUT][1];
    interpolation = CV_INTER_LINEAR;
}


void CV_UndistortTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTest::fill_array( test_case_idx, i, j, arr );
}


void CV_UndistortTest::run_func()
{
    if (!useCPlus)
    {
        CvMat a = test_mat[INPUT][1], k = test_mat[INPUT][2];
        cvUndistort2( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], &a, &k);
    }
    else
    {
        if (zero_distortion)
        {
            cv::undistort(input0,input_output,input1,cv::Mat());
        }
        else
        {
            cv::undistort(input0,input_output,input1,input2);
        }
    }
}


double CV_UndistortTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_UndistortTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );

    const Mat& src = test_mat[INPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double new_cam[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(src.rows, src.cols);

    Mat& _new_cam0 = test_mat[INPUT][3];
    Mat _new_cam(test_mat[INPUT][3].rows,test_mat[INPUT][3].cols,CV_64F,new_cam);
    Mat& _a0 = test_mat[INPUT][1];
    Mat _a(3,3,CV_64F,a);
    Mat& _k0 = test_mat[INPUT][2];
    Mat _k(_k0.rows,_k0.cols, CV_MAKETYPE(CV_64F,_k0.channels()),k);

    if( code <= 0 )
        return code;

    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    a[2] = (src.cols - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (src.rows - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];
    k[0] = cvtest::randReal(rng)*0.06 - 0.03;
    k[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( k[0]*k[1] > 0 )
        k[1] = -k[1];
    if( cvtest::randInt(rng)%4 != 0 )
    {
        k[2] = cvtest::randReal(rng)*0.004 - 0.002;
        k[3] = cvtest::randReal(rng)*0.004 - 0.002;
    }
    else
        k[2] = k[3] = 0;

    new_cam[0] = a[0] + (cvtest::randReal(rng) - (double)0.5)*0.2*a[0]; //10%
    new_cam[4] = a[4] + (cvtest::randReal(rng) - (double)0.5)*0.2*a[4]; //10%
    new_cam[2] = a[2] + (cvtest::randReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].rows; //15%
    new_cam[5] = a[5] + (cvtest::randReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].cols; //15%

    _a.convertTo(_a0, _a0.depth());

    zero_distortion = (cvtest::randInt(rng)%2) == 0 ? false : true;
    _k.convertTo(_k0, _k0.depth());

    zero_new_cam = (cvtest::randInt(rng)%2) == 0 ? false : true;
    _new_cam.convertTo(_new_cam0, _new_cam0.depth());

    //Testing C++ code
    useCPlus = ((cvtest::randInt(rng) % 2)!=0);
    if (useCPlus)
    {
        input0 = test_mat[INPUT][0];
        input1 = test_mat[INPUT][1];
        input2 = test_mat[INPUT][2];
        input_new_cam = test_mat[INPUT][3];
    }

    return code;
}


void CV_UndistortTest::prepare_to_validation( int /*test_case_idx*/ )
{
    if (useCPlus)
    {
        Mat& output = test_mat[INPUT_OUTPUT][0];
        input_output.convertTo(output, output.type());
    }
    Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mapx, mapy;
    cvtest::initUndistortMap( test_mat[INPUT][1], test_mat[INPUT][2], dst.size(), mapx, mapy );
    Mat mask( dst.size(), CV_8U );
    test_remap( src, dst, mapx, mapy, &mask, interpolation );
    dst.setTo(Scalar::all(0), mask);
    dst0.setTo(Scalar::all(0), mask);
}


class CV_UndistortMapTest : public cvtest::ArrayTest
{
public:
    CV_UndistortMapTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

private:
    bool dualChannel;
};


CV_UndistortMapTest::CV_UndistortMapTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    element_wise_relative_error = false;
}


void CV_UndistortMapTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;

    CvSize sz = sizes[OUTPUT][0];
    types[INPUT][0] = types[INPUT][1] = depth;
    dualChannel = cvtest::randInt(rng)%2 == 0;
    types[OUTPUT][0] = types[OUTPUT][1] =
        types[REF_OUTPUT][0] = types[REF_OUTPUT][1] = dualChannel ? CV_32FC2 : CV_32F;
    sizes[INPUT][0] = cvSize(3,3);
    sizes[INPUT][1] = cvtest::randInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);

    sz.width = MAX(sz.width,16);
    sz.height = MAX(sz.height,16);
    sizes[OUTPUT][0] = sizes[OUTPUT][1] =
        sizes[REF_OUTPUT][0] = sizes[REF_OUTPUT][1] = sz;
}


void CV_UndistortMapTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
}


void CV_UndistortMapTest::run_func()
{
    CvMat a = test_mat[INPUT][0], k = test_mat[INPUT][1];

    if (!dualChannel )
        cvInitUndistortMap( &a, &k, test_array[OUTPUT][0], test_array[OUTPUT][1] );
    else
        cvInitUndistortMap( &a, &k, test_array[OUTPUT][0], 0 );
}


double CV_UndistortMapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-3;
}


int CV_UndistortMapTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    const Mat& mapx = test_mat[OUTPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(mapx.rows, mapx.cols);
    Mat& _a0 = test_mat[INPUT][0], &_k0 = test_mat[INPUT][1];
    Mat _a(3,3,CV_64F,a);
    Mat _k(_k0.rows,_k0.cols, CV_MAKETYPE(CV_64F,_k0.channels()),k);

    if( code <= 0 )
        return code;

    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    a[2] = (mapx.cols - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (mapx.rows - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];
    k[0] = cvtest::randReal(rng)*0.06 - 0.03;
    k[1] = cvtest::randReal(rng)*0.06 - 0.03;
    if( k[0]*k[1] > 0 )
        k[1] = -k[1];
    k[2] = cvtest::randReal(rng)*0.004 - 0.002;
    k[3] = cvtest::randReal(rng)*0.004 - 0.002;

    _a.convertTo(_a0, _a0.depth());
    _k.convertTo(_k0, _k0.depth());

    if (dualChannel)
    {
        test_mat[REF_OUTPUT][1] = Scalar::all(0);
        test_mat[OUTPUT][1] = Scalar::all(0);
    }

    return code;
}


void CV_UndistortMapTest::prepare_to_validation( int )
{
    Mat mapx, mapy;
    cvtest::initUndistortMap( test_mat[INPUT][0], test_mat[INPUT][1], test_mat[REF_OUTPUT][0].size(), mapx, mapy );
    if( !dualChannel )
    {
        mapx.copyTo(test_mat[REF_OUTPUT][0]);
        mapy.copyTo(test_mat[REF_OUTPUT][1]);
    }
    else
    {
        Mat p[2] = {mapx, mapy};
        cv::merge(p, 2, test_mat[REF_OUTPUT][0]);
    }
}

////////////////////////////// GetRectSubPix /////////////////////////////////

static void
test_getQuadrangeSubPix( const Mat& src, Mat& dst, double* a )
{
    int sstep = (int)(src.step / sizeof(float));
    int scols = src.cols, srows = src.rows;

    CV_Assert( src.depth() == CV_32F && src.type() == dst.type() );

    int cn = dst.channels();

    for( int y = 0; y < dst.rows; y++ )
        for( int x = 0; x < dst.cols; x++ )
        {
            float* d = dst.ptr<float>(y) + x*cn;
            float sx = (float)(a[0]*x + a[1]*y + a[2]);
            float sy = (float)(a[3]*x + a[4]*y + a[5]);
            int ix = cvFloor(sx), iy = cvFloor(sy);
            int dx = cn, dy = sstep;
            const float* s;
            sx -= ix; sy -= iy;

            if( (unsigned)ix >= (unsigned)(scols-1) )
                ix = ix < 0 ? 0 : scols - 1, sx = 0, dx = 0;
            if( (unsigned)iy >= (unsigned)(srows-1) )
                iy = iy < 0 ? 0 : srows - 1, sy = 0, dy = 0;

            s = src.ptr<float>(iy) + ix*cn;
            for( int k = 0; k < cn; k++, s++ )
            {
                float t0 = s[0] + sx*(s[dx] - s[0]);
                float t1 = s[dy] + sx*(s[dy + dx] - s[dy]);
                d[k] = t0 + sy*(t1 - t0);
            }
        }
}


class CV_GetRectSubPixTest : public CV_ImgWarpBaseTest
{
public:
    CV_GetRectSubPixTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );

    CvPoint2D32f center;
    bool test_cpp;
};


CV_GetRectSubPixTest::CV_GetRectSubPixTest() : CV_ImgWarpBaseTest( false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
    test_cpp = false;
}


void CV_GetRectSubPixTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int src_depth = cvtest::randInt(rng) % 2, dst_depth;
    int cn = cvtest::randInt(rng) % 2 ? 3 : 1;
    CvSize src_size, dst_size;

    dst_depth = src_depth = src_depth == 0 ? CV_8U : CV_32F;
    if( src_depth < CV_32F && cvtest::randInt(rng) % 2 )
        dst_depth = CV_32F;

    types[INPUT][0] = CV_MAKETYPE(src_depth,cn);
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(dst_depth,cn);

    src_size = sizes[INPUT][0];
    dst_size.width = cvRound(sqrt(cvtest::randReal(rng)*src_size.width) + 1);
    dst_size.height = cvRound(sqrt(cvtest::randReal(rng)*src_size.height) + 1);
    dst_size.width = MIN(dst_size.width,src_size.width);
    dst_size.height = MIN(dst_size.width,src_size.height);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = dst_size;

    center.x = (float)(cvtest::randReal(rng)*src_size.width);
    center.y = (float)(cvtest::randReal(rng)*src_size.height);
    interpolation = CV_INTER_LINEAR;

    test_cpp = (cvtest::randInt(rng) & 256) == 0;
}


void CV_GetRectSubPixTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTest::fill_array( test_case_idx, i, j, arr );
}


void CV_GetRectSubPixTest::run_func()
{
    if(!test_cpp)
        cvGetRectSubPix( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], center );
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[INPUT_OUTPUT][0]);
        cv::getRectSubPix( cv::cvarrToMat(test_array[INPUT][0]), _out.size(), center, _out, _out.type());
    }
}


double CV_GetRectSubPixTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int in_depth = test_mat[INPUT][0].depth();
    int out_depth = test_mat[INPUT_OUTPUT][0].depth();

    return in_depth >= CV_32F ? 1e-3 : out_depth >= CV_32F ? 1e-2 : 1;
}


int CV_GetRectSubPixTest::prepare_test_case( int test_case_idx )
{
    return CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
}


void CV_GetRectSubPixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src0 = test_mat[INPUT][0];
    Mat& dst0 = test_mat[REF_INPUT_OUTPUT][0];
    Mat src = src0, dst = dst0;
    int ftype = CV_MAKETYPE(CV_32F,src0.channels());
    double a[] = { 1, 0, center.x - dst.cols*0.5 + 0.5,
                   0, 1, center.y - dst.rows*0.5 + 0.5 };
    if( src.depth() != CV_32F )
        src0.convertTo(src, CV_32F);

    if( dst.depth() != CV_32F )
        dst.create(dst0.size(), ftype);

    test_getQuadrangeSubPix( src, dst, a );

    if( dst.data != dst0.data )
        dst.convertTo(dst0, dst0.depth());
}


class CV_GetQuadSubPixTest : public CV_ImgWarpBaseTest
{
public:
    CV_GetQuadSubPixTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_GetQuadSubPixTest::CV_GetQuadSubPixTest() : CV_ImgWarpBaseTest( true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
}


void CV_GetQuadSubPixTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    int min_size = 4;
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0], dsz;
    RNG& rng = ts->get_rng();
    int msz, src_depth = cvtest::randInt(rng) % 2, dst_depth;
    int cn = cvtest::randInt(rng) % 2 ? 3 : 1;

    dst_depth = src_depth = src_depth == 0 ? CV_8U : CV_32F;
    if( src_depth < CV_32F && cvtest::randInt(rng) % 2 )
        dst_depth = CV_32F;

    types[INPUT][0] = CV_MAKETYPE(src_depth,cn);
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(dst_depth,cn);

    sz.width = MAX(sz.width,min_size);
    sz.height = MAX(sz.height,min_size);
    sizes[INPUT][0] = sz;
    msz = MIN( sz.width, sz.height );

    dsz.width = cvRound(sqrt(cvtest::randReal(rng)*msz) + 1);
    dsz.height = cvRound(sqrt(cvtest::randReal(rng)*msz) + 1);
    dsz.width = MIN(dsz.width,msz);
    dsz.height = MIN(dsz.width,msz);
    dsz.width = MAX(dsz.width,min_size);
    dsz.height = MAX(dsz.height,min_size);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = dsz;
    sizes[INPUT][1] = cvSize( 3, 2 );
}


void CV_GetQuadSubPixTest::run_func()
{
    CvMat mtx = test_mat[INPUT][1];
    cvGetQuadrangleSubPix( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], &mtx );
}


double CV_GetQuadSubPixTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int in_depth = test_mat[INPUT][0].depth();
    //int out_depth = test_mat[INPUT_OUTPUT][0].depth();

    return in_depth >= CV_32F ? 1e-2 : 4;
}


int CV_GetQuadSubPixTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const Mat& src = test_mat[INPUT][0];
    Mat& mat = test_mat[INPUT][1];
    CvPoint2D32f center;
    double scale, angle;

    if( code <= 0 )
        return code;

    double a[6];
    Mat A( 2, 3, CV_64FC1, a );

    center.x = (float)((cvtest::randReal(rng)*1.2 - 0.1)*src.cols);
    center.y = (float)((cvtest::randReal(rng)*1.2 - 0.1)*src.rows);
    angle = cvtest::randReal(rng)*360;
    scale = cvtest::randReal(rng)*0.2 + 0.9;

    // y = Ax + b -> x = A^-1(y - b) = A^-1*y - A^-1*b
    scale = 1./scale;
    angle = angle*(CV_PI/180.);
    a[0] = a[4] = cos(angle)*scale;
    a[1] = sin(angle)*scale;
    a[3] = -a[1];
    a[2] = center.x - a[0]*center.x - a[1]*center.y;
    a[5] = center.y - a[3]*center.x - a[4]*center.y;
    A.convertTo( mat, mat.depth() );

    return code;
}


void CV_GetQuadSubPixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src0 = test_mat[INPUT][0];
    Mat& dst0 = test_mat[REF_INPUT_OUTPUT][0];
    Mat src = src0, dst = dst0;
    int ftype = CV_MAKETYPE(CV_32F,src0.channels());
    double a[6], dx = (dst0.cols - 1)*0.5, dy = (dst0.rows - 1)*0.5;
    Mat A( 2, 3, CV_64F, a );

    if( src.depth() != CV_32F )
        src0.convertTo(src, CV_32F);

    if( dst.depth() != CV_32F )
        dst.create(dst0.size(), ftype);

    test_mat[INPUT][1].convertTo( A, CV_64F );
    a[2] -= a[0]*dx + a[1]*dy;
    a[5] -= a[3]*dx + a[4]*dy;
    test_getQuadrangeSubPix( src, dst, a );

    if( dst.data != dst0.data )
        dst.convertTo(dst0, dst0.depth());
}

TEST(Imgproc_cvWarpAffine, regression)
{
    IplImage* src = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);
    IplImage* dst = cvCreateImage(cvSize(100, 100), IPL_DEPTH_8U, 1);

    float m[6];
    CvMat M = cvMat( 2, 3, CV_32F, m );
    int w = src->width;
    int h = src->height;
    cv2DRotationMatrix(cvPoint2D32f(w*0.5f, h*0.5f), 45.0, 1.0, &M);
    cvWarpAffine(src, dst, &M);
}

TEST(Imgproc_fitLine_vector_3d, regression)
{
    std::vector<Point3f> points_vector;

    Point3f p21(4,4,4);
    Point3f p22(8,8,8);

    points_vector.push_back(p21);
    points_vector.push_back(p22);

    std::vector<float> line;

    cv::fitLine(points_vector, line, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line.size(), (size_t)6);

}

TEST(Imgproc_fitLine_vector_2d, regression)
{
    std::vector<Point2f> points_vector;

    Point2f p21(4,4);
    Point2f p22(8,8);
    Point2f p23(16,16);

    points_vector.push_back(p21);
    points_vector.push_back(p22);
    points_vector.push_back(p23);

    std::vector<float> line;

    cv::fitLine(points_vector, line, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_2dC2, regression)
{
    cv::Mat mat1 = Mat::zeros(3, 1, CV_32SC2);
    std::vector<float> line1;

    cv::fitLine(mat1, line1, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line1.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_2dC1, regression)
{
    cv::Matx<int, 3, 2> mat2;
    std::vector<float> line2;

    cv::fitLine(mat2, line2, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line2.size(), (size_t)4);
}

TEST(Imgproc_fitLine_Mat_3dC3, regression)
{
    cv::Mat mat1 = Mat::zeros(2, 1, CV_32SC3);
    std::vector<float> line1;

    cv::fitLine(mat1, line1, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line1.size(), (size_t)6);
}

TEST(Imgproc_fitLine_Mat_3dC1, regression)
{
    cv::Mat mat2 = Mat::zeros(2, 3, CV_32SC1);
    std::vector<float> line2;

    cv::fitLine(mat2, line2, CV_DIST_L2, 0 ,0 ,0);

    ASSERT_EQ(line2.size(), (size_t)6);
}

TEST(Imgproc_resize_area, regression)
{
    static ushort input_data[16 * 16] = {
         90,  94,  80,   3, 231,   2, 186, 245, 188, 165,  10,  19, 201, 169,   8, 228,
         86,   5, 203, 120, 136, 185,  24,  94,  81, 150, 163, 137,  88, 105, 132, 132,
        236,  48, 250, 218,  19,  52,  54, 221, 159, 112,  45,  11, 152, 153, 112, 134,
         78, 133, 136,  83,  65,  76,  82, 250,   9, 235, 148,  26, 236, 179, 200,  50,
         99,  51, 103, 142, 201,  65, 176,  33,  49, 226, 177, 109,  46,  21,  67, 130,
         54, 125, 107, 154, 145,  51, 199, 189, 161, 142, 231, 240, 139, 162, 240,  22,
        231,  86,  79, 106,  92,  47, 146, 156,  36, 207,  71,  33,   2, 244, 221,  71,
         44, 127,  71, 177,  75, 126,  68, 119, 200, 129, 191, 251,   6, 236, 247,  6,
        133, 175,  56, 239, 147, 221, 243, 154, 242,  82, 106,  99,  77, 158,  60, 229,
          2,  42,  24, 174,  27, 198,  14, 204, 246, 251, 141,  31, 114, 163,  29, 147,
        121,  53,  74,  31, 147, 189,  42,  98, 202,  17, 228, 123, 209,  40,  77,  49,
        112, 203,  30,  12, 205,  25,  19, 106, 145, 185, 163, 201, 237, 223, 247,  38,
         33, 105, 243, 117,  92, 179, 204, 248, 160,  90,  73, 126,   2,  41, 213, 204,
          6, 124, 195, 201, 230, 187, 210, 167,  48,  79, 123, 159, 145, 218, 105, 209,
        240, 152, 136, 235, 235, 164, 157,  9,  152,  38,  27, 209, 120,  77, 238, 196,
        240, 233,  10, 241,  90,  67,  12, 79,    0,  43,  58,  27,  83, 199, 190, 182};

    static ushort expected_data[5 * 5] = {
        120, 100, 151, 101, 130,
        106, 115, 141, 130, 127,
         91, 136, 170, 114, 140,
        104, 122, 131, 147, 133,
        161, 163,  70, 107, 182
    };

    cv::Mat src(16, 16, CV_16UC1, input_data);
    cv::Mat expected(5, 5, CV_16UC1, expected_data);
    cv::Mat actual(expected.size(), expected.type());

    cv::resize(src, actual, cv::Size(), 0.3, 0.3, INTER_AREA);

    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.size(), expected.size());

    Mat diff;
    absdiff(actual, expected, diff);

    Mat one_channel_diff = diff; //.reshape(1);

    float elem_diff = 1.0f;
    Size dsize = actual.size();
    bool next = true;
    for (int dy = 0; dy < dsize.height && next; ++dy)
    {
        ushort* eD = expected.ptr<ushort>(dy);
        ushort* aD = actual.ptr<ushort>(dy);

        for (int dx = 0; dx < dsize.width && next; ++dx)
            if (fabs(static_cast<float>(aD[dx] - eD[dx])) > elem_diff)
            {
                cvtest::TS::ptr()->printf(cvtest::TS::SUMMARY, "Inf norm: %f\n", static_cast<float>(norm(actual, expected, NORM_INF)));
                cvtest::TS::ptr()->printf(cvtest::TS::SUMMARY, "Error in : (%d, %d)\n", dx, dy);

                const int radius = 3;
                int rmin = MAX(dy - radius, 0), rmax = MIN(dy + radius, dsize.height);
                int cmin = MAX(dx - radius, 0), cmax = MIN(dx + radius, dsize.width);

                std::cout << "Abs diff:" << std::endl << diff << std::endl;
                std::cout << "actual result:\n" << actual(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;
                std::cout << "expected result:\n" << expected(Range(rmin, rmax), Range(cmin, cmax)) << std::endl;

                next = false;
            }
    }

    ASSERT_EQ(cvtest::norm(one_channel_diff, cv::NORM_INF), 0);
}


//////////////////////////////////////////////////////////////////////////

TEST(Imgproc_Resize, accuracy) { CV_ResizeTest test; test.safe_run(); }
TEST(Imgproc_WarpAffine, accuracy) { CV_WarpAffineTest test; test.safe_run(); }
TEST(Imgproc_WarpPerspective, accuracy) { CV_WarpPerspectiveTest test; test.safe_run(); }
TEST(Imgproc_Remap, accuracy) { CV_RemapTest test; test.safe_run(); }
TEST(Imgproc_Undistort, accuracy) { CV_UndistortTest test; test.safe_run(); }
TEST(Imgproc_InitUndistortMap, accuracy) { CV_UndistortMapTest test; test.safe_run(); }
TEST(Imgproc_GetRectSubPix, accuracy) { CV_GetRectSubPixTest test; test.safe_run(); }
TEST(Imgproc_GetQuadSubPix, accuracy) { CV_GetQuadSubPixTest test; test.safe_run(); }

/* End of file. */
