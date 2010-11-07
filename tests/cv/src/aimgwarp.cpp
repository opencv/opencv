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

static const int imgwarp_depths[] = { CV_8U, CV_16U, CV_32F, -1 };
static const int imgwarp_channels[] = { 1, 3, 4, -1 };
static const CvSize imgwarp_sizes[] = {{320, 240}, {1024,768}, {-1,-1}};

static const double imgwarp_resize_coeffs[] = { 0.5, 0.333, 2, 2.9 };
static const char* imgwarp_resize_methods[] = { "nearest", "linear", "cubic", "area", 0 };
static const char* imgwarp_resize_param_names[] = { "method", "coeff", "size", "channels", "depth", 0 };

static const double imgwarp_affine_rotate_scale[][4] = { {0.5,0.5,30.,1.4}, {0.5,0.5,-130,0.4}, {-1,-1,-1,-1} };
static const char* imgwarp_affine_param_names[] = { "rotate_scale", "size", "channels", "depth", 0 };

static const double imgwarp_perspective_shift_vtx[][8] = { {0.03,0.01,0.04,0.02,0.01,0.01,0.01,0.02}, {-1} };
static const char* imgwarp_perspective_param_names[] = { "shift_vtx", "size", "channels", "depth", 0 };

class CV_ImgWarpBaseTestImpl : public CvArrTest
{
public:
    CV_ImgWarpBaseTestImpl( const char* test_name, const char* test_funcs, bool warp_matrix );

protected:
    int read_params( CvFileStorage* fs );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );

    int interpolation;
    int max_interpolation;
    double spatial_scale_zoom, spatial_scale_decimate;
};


CV_ImgWarpBaseTestImpl::CV_ImgWarpBaseTestImpl( const char* test_name, const char* test_funcs, bool warp_matrix )
    : CvArrTest( test_name, test_funcs, "" )
{
    test_array[INPUT].push(NULL);
    if( warp_matrix )
        test_array[INPUT].push(NULL);
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    max_interpolation = 5;
    interpolation = 0;
    element_wise_relative_error = false;
    spatial_scale_zoom = 0.01;
    spatial_scale_decimate = 0.005;

    size_list = whole_size_list = imgwarp_sizes;
    depth_list = imgwarp_depths;
    cn_list = imgwarp_channels;
    default_timing_param_names = 0;
}


int CV_ImgWarpBaseTestImpl::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    return code;
}


void CV_ImgWarpBaseTestImpl::get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high )
{
    CvArrTest::get_minmax_bounds( i, j, type, low, high );
    if( CV_MAT_DEPTH(type) == CV_32F )
    {
        *low = cvScalarAll(-10.);
        *high = cvScalarAll(10);
    }
}


void CV_ImgWarpBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 3;
    int cn = cvTsRandInt(rng) % 3 + 1;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    cn += cn == 2;

    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    if( test_array[INPUT].size() > 1 )
        types[INPUT][1] = cvTsRandInt(rng) & 1 ? CV_32FC1 : CV_64FC1;

    interpolation = cvTsRandInt(rng) % max_interpolation;
}


void CV_ImgWarpBaseTestImpl::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i != INPUT || j != 0 )
        CvArrTest::fill_array( test_case_idx, i, j, arr );
}

int CV_ImgWarpBaseTestImpl::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    CvMat* img = &test_mat[INPUT][0];
    int i, j, cols = img->cols;
    int type = CV_MAT_TYPE(img->type), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    double scale = depth == CV_16U ? 1000. : 255.*0.5;
    double space_scale = spatial_scale_decimate;
    float* buffer;

    if( code <= 0 )
        return code;

    if( test_mat[INPUT_OUTPUT][0].cols >= img->cols &&
        test_mat[INPUT_OUTPUT][0].rows >= img->rows )
        space_scale = spatial_scale_zoom;

    buffer = (float*)cvAlloc( img->cols*cn*sizeof(buffer[0]) );
    
    for( i = 0; i < img->rows; i++ )
    {
        uchar* ptr = img->data.ptr + i*img->step;
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
        cv::Mat src(1, cols*cn, CV_32F, buffer);
        cv::Mat dst(1, cols*cn, depth, ptr);
        src.convertTo(dst, dst.type());        
    }

    cvFree( &buffer );

    return code;
}

CV_ImgWarpBaseTestImpl imgwarp_base( "warp", "", false );


class CV_ImgWarpBaseTest : public CV_ImgWarpBaseTestImpl
{
public:
    CV_ImgWarpBaseTest( const char* test_name, const char* test_funcs, bool warp_matrix );
};


CV_ImgWarpBaseTest::CV_ImgWarpBaseTest( const char* test_name, const char* test_funcs, bool warp_matrix )
    : CV_ImgWarpBaseTestImpl( test_name, test_funcs, warp_matrix )
{
    size_list = whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


/////////////////////////

class CV_ResizeTest : public CV_ImgWarpBaseTest
{
public:
    CV_ResizeTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
};


CV_ResizeTest::CV_ResizeTest()
    : CV_ImgWarpBaseTest( "warp-resize", "cvResize", false )
{
    default_timing_param_names = imgwarp_resize_param_names;
}


int CV_ResizeTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        start_write_param( fs );
        write_real_list( fs, "coeff", imgwarp_resize_coeffs, CV_DIM(imgwarp_resize_coeffs) );
        write_string_list( fs, "method", imgwarp_resize_methods );
    }

    return code;
}


void CV_ResizeTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz;

    sz.width = (cvTsRandInt(rng) % sizes[INPUT][0].width) + 1;
    sz.height = (cvTsRandInt(rng) % sizes[INPUT][0].height) + 1;

    if( cvTsRandInt(rng) & 1 )
    {
        int xfactor = cvTsRandInt(rng) % 10 + 1;
        int yfactor = cvTsRandInt(rng) % 10 + 1;

        if( cvTsRandInt(rng) & 1 )
            yfactor = xfactor;

        sz.width = sizes[INPUT][0].width / xfactor;
        sz.width = MAX(sz.width,1);
        sz.height = sizes[INPUT][0].height / yfactor;
        sz.height = MAX(sz.height,1);
        sizes[INPUT][0].width = sz.width * xfactor;
        sizes[INPUT][0].height = sz.height * yfactor;
    }

    if( cvTsRandInt(rng) & 1 )
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


void CV_ResizeTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );
    const char* method_str = cvReadString( find_timing_param( "method" ), "linear" );
    double coeff = cvReadReal( find_timing_param( "coeff" ), 1. );
    CvSize size = sizes[INPUT][0];

    size.width = cvRound(size.width*coeff);
    size.height = cvRound(size.height*coeff);
    sizes[INPUT_OUTPUT][0] = whole_sizes[INPUT_OUTPUT][0] = size;

    interpolation = strcmp( method_str, "nearest" ) == 0 ? CV_INTER_NN :
                    strcmp( method_str, "linear" ) == 0 ? CV_INTER_LINEAR :
                    strcmp( method_str, "cubic" ) == 0 ? CV_INTER_CUBIC : CV_INTER_AREA;
}


void CV_ResizeTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "coeff=%.3f,", cvReadReal( find_timing_param( "coeff" ), 1. ) );
    ptr += strlen(ptr);
    sprintf( ptr, "method=%s,", cvReadString( find_timing_param( "method" ), "linear" ) );
    ptr += strlen(ptr);
    params_left -= 2;

    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_ResizeTest::run_func()
{
    cvResize( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], interpolation );
}


double CV_ResizeTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 1e-1;
}


void CV_ResizeTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
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

CV_ResizeTest warp_resize_test;


/////////////////////////

void cvTsRemap( const CvMat* src, CvMat* dst,
                const CvMat* mapx, const CvMat* mapy,
                CvMat* mask, int interpolation=CV_INTER_LINEAR )
{
    int x, y, k;
    int drows = dst->rows, dcols = dst->cols;
    int srows = src->rows, scols = src->cols;
    uchar* sptr0 = src->data.ptr;
    int depth = CV_MAT_DEPTH(src->type), cn = CV_MAT_CN(src->type);
    int elem_size = CV_ELEM_SIZE(src->type);
    int step = src->step / CV_ELEM_SIZE(depth);
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
        cvTsZero(mask);

    for( y = 0; y < drows; y++ )
    {
        uchar* dptr = dst->data.ptr + dst->step*y;
        const float* mx = (const float*)(mapx->data.ptr + mapx->step*y);
        const float* my = (const float*)(mapy->data.ptr + mapy->step*y);
        uchar* m = mask ? mask->data.ptr + mask->step*y : 0;

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
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
};


CV_WarpAffineTest::CV_WarpAffineTest()
    : CV_ImgWarpBaseTest( "warp-affine", "cvWarpAffine", true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    
    spatial_scale_decimate = spatial_scale_zoom;

    default_timing_param_names = imgwarp_affine_param_names;
}


int CV_WarpAffineTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "rotate_scale", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; imgwarp_affine_rotate_scale[i][0] >= 0; i++ )
        {
            cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteRawData( fs, imgwarp_affine_rotate_scale[i], 4, "d" );
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_WarpAffineTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0];
    // run for the second time to get output of a different size
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sz;
    sizes[INPUT][1] = cvSize( 3, 2 );
    sizes[TEMP][0] = sizes[TEMP][1] = sizes[INPUT_OUTPUT][0];
    types[TEMP][0] = types[TEMP][1] = CV_32FC1;
}


void CV_WarpAffineTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );

    sizes[INPUT][1] = whole_sizes[INPUT][1] = cvSize(3,2);
    sizes[TEMP][0] = whole_sizes[TEMP][0] =
        sizes[TEMP][1] = whole_sizes[TEMP][1] = cvSize(0,0);
    types[INPUT][1] = CV_64FC1;

    interpolation = CV_INTER_LINEAR;
}


void CV_WarpAffineTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    double coeffs[4];
    const CvFileNode* node = find_timing_param( "rotate_scale" );
    assert( node && CV_NODE_IS_SEQ(node->tag) );
    cvReadRawData( ts->get_file_storage(), node, coeffs, "4d" );
    
    sprintf( ptr, "fx=%.2f,fy=%.2f,angle=%.1fdeg,scale=%.1f,", coeffs[0], coeffs[1], coeffs[2], coeffs[3] );
    ptr += strlen(ptr);
    params_left -= 4;

    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_WarpAffineTest::run_func()
{
    cvWarpAffine( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
                  &test_mat[INPUT][1], interpolation );
}


double CV_WarpAffineTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_WarpAffineTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const CvMat* src = &test_mat[INPUT][0];
    const CvMat* dst = &test_mat[INPUT_OUTPUT][0]; 
    CvMat* mat = &test_mat[INPUT][1];
    CvPoint2D32f center;
    double scale, angle;

    if( code <= 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        double buf[6];
        CvMat tmp = cvMat( 2, 3, mat->type, buf );

        center.x = (float)((cvTsRandReal(rng)*1.2 - 0.1)*src->cols);
        center.y = (float)((cvTsRandReal(rng)*1.2 - 0.1)*src->rows);
        angle = cvTsRandReal(rng)*360;
        scale = ((double)dst->rows/src->rows + (double)dst->cols/src->cols)*0.5;
        cv2DRotationMatrix( center, angle, scale, mat );
        cvRandArr( rng, &tmp, CV_RAND_NORMAL, cvScalarAll(1.), cvScalarAll(0.01) );
        cvMaxS( &tmp, 0.9, &tmp );
        cvMinS( &tmp, 1.1, &tmp );
        cvMul( &tmp, mat, mat, 1. );
    }
    else
    {
        double coeffs[4];
        const CvFileNode* node = find_timing_param( "rotate_scale" );

        assert( node && CV_NODE_IS_SEQ(node->tag) );
        cvReadRawData( ts->get_file_storage(), node, coeffs, "4d" );

        center.x = (float)(coeffs[0]*src->cols);
        center.y = (float)(coeffs[1]*src->rows);
        angle = coeffs[2];
        scale = coeffs[3];
        cv2DRotationMatrix( center, angle, scale, mat );
    }

    return code;
}


void CV_WarpAffineTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* dst0 = &test_mat[INPUT_OUTPUT][0];
    CvMat* mapx = &test_mat[TEMP][0];
    CvMat* mapy = &test_mat[TEMP][1];
    int x, y;
    double m[6], tm[6];
    CvMat srcAb = cvMat(2, 3, CV_64FC1, tm ), A, b, invA, invAb, dstAb = cvMat( 2, 3, CV_64FC1, m );

    //cvInvert( &tM, &M, CV_LU );
    // [R|t] -> [R^-1 | -(R^-1)*t]
    cvTsConvert( &test_mat[INPUT][1], &srcAb );
    cvGetCols( &srcAb, &A, 0, 2 );
    cvGetCol( &srcAb, &b, 2 );
    cvGetCols( &dstAb, &invA, 0, 2 );
    cvGetCol( &dstAb, &invAb, 2 );
    cvInvert( &A, &invA, CV_SVD );
    cvGEMM( &invA, &b, -1, 0, 0, &invAb );

    for( y = 0; y < dst->rows; y++ )
    {
        float* mx = (float*)(mapx->data.ptr + y*mapx->step);
        float* my = (float*)(mapy->data.ptr + y*mapy->step);

        for( x = 0; x < dst->cols; x++ )
        {
            mx[x] = (float)(x*m[0] + y*m[1] + m[2]);
            my[x] = (float)(x*m[3] + y*m[4] + m[5]);
        }
    }

    CvMat* mask = cvCreateMat( dst->rows, dst->cols, CV_8U );
    cvTsRemap( src, dst, mapx, mapy, mask );
    cvTsZero( dst, mask );
    cvTsZero( dst0, mask );
    cvReleaseMat( &mask );
}


CV_WarpAffineTest warp_affine_test;



/////////////////////////

class CV_WarpPerspectiveTest : public CV_ImgWarpBaseTest
{
public:
    CV_WarpPerspectiveTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
};


CV_WarpPerspectiveTest::CV_WarpPerspectiveTest()
    : CV_ImgWarpBaseTest( "warp-perspective", "cvWarpPerspective", true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
    default_timing_param_names = imgwarp_perspective_param_names;
}


int CV_WarpPerspectiveTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "shift_vtx", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; imgwarp_perspective_shift_vtx[i][0] >= 0; i++ )
        {
            cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteRawData( fs, imgwarp_perspective_shift_vtx[i], 8, "d" );
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_WarpPerspectiveTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0];
    // run for the second time to get output of a different size
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sz;
    sizes[INPUT][1] = cvSize( 3, 3 );

    sizes[TEMP][0] = sizes[TEMP][1] = sizes[INPUT_OUTPUT][0];
    types[TEMP][0] = types[TEMP][1] = CV_32FC1;
}


void CV_WarpPerspectiveTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );

    sizes[INPUT][1] = whole_sizes[INPUT][1] = cvSize(3,3);
    sizes[TEMP][0] = whole_sizes[TEMP][0] =
        sizes[TEMP][1] = whole_sizes[TEMP][1] = cvSize(0,0);
    types[INPUT][1] = CV_64FC1;

    interpolation = CV_INTER_LINEAR;
}


void CV_WarpPerspectiveTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_WarpPerspectiveTest::run_func()
{
    cvWarpPerspective( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
                       &test_mat[INPUT][1], interpolation );
}


double CV_WarpPerspectiveTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_WarpPerspectiveTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const CvMat* src = &test_mat[INPUT][0];
    const CvMat* dst = &test_mat[INPUT_OUTPUT][0]; 
    CvMat* mat = &test_mat[INPUT][1];
    CvPoint2D32f s[4], d[4];
    int i;

    if( code <= 0 )
        return code;

    s[0] = cvPoint2D32f(0,0);
    d[0] = cvPoint2D32f(0,0);
    s[1] = cvPoint2D32f(src->cols-1,0);
    d[1] = cvPoint2D32f(dst->cols-1,0);
    s[2] = cvPoint2D32f(src->cols-1,src->rows-1);
    d[2] = cvPoint2D32f(dst->cols-1,dst->rows-1);
    s[3] = cvPoint2D32f(0,src->rows-1);
    d[3] = cvPoint2D32f(0,dst->rows-1);

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        float buf[16];
        CvMat tmp = cvMat( 1, 16, CV_32FC1, buf );

        cvRandArr( rng, &tmp, CV_RAND_NORMAL, cvScalarAll(0.), cvScalarAll(0.1) );

        for( i = 0; i < 4; i++ )
        {
            s[i].x += buf[i*4]*src->cols/2;
            s[i].y += buf[i*4+1]*src->rows/2;
            d[i].x += buf[i*4+2]*dst->cols/2;
            d[i].y += buf[i*4+3]*dst->rows/2;
        }
    }
    else
    {
        double coeffs[8];
        const CvFileNode* node = find_timing_param( "shift_vtx" );

        assert( node && CV_NODE_IS_SEQ(node->tag) );
        cvReadRawData( ts->get_file_storage(), node, coeffs, "8d" );

        for( i = 0; i < 4; i++ )
        {
            d[i].x += (float)(coeffs[i*2]*src->cols*(i == 0 || i == 3 ? 1 : -1));
            d[i].y += (float)(coeffs[i*2+1]*src->rows*(i == 0 || i == 1 ? 1 : -1));
        }
    }

    cvWarpPerspectiveQMatrix( s, d, mat );
    return code;
}


void CV_WarpPerspectiveTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* dst0 = &test_mat[INPUT_OUTPUT][0];
    CvMat* mapx = &test_mat[TEMP][0];
    CvMat* mapy = &test_mat[TEMP][1];
    int x, y;
    double m[9], tm[9];
    CvMat srcM = cvMat(3, 3, CV_64FC1, tm ), dstM = cvMat( 3, 3, CV_64FC1, m );

    //cvInvert( &tM, &M, CV_LU );
    // [R|t] -> [R^-1 | -(R^-1)*t]
    cvTsConvert( &test_mat[INPUT][1], &srcM );
    cvInvert( &srcM, &dstM, CV_SVD );

    for( y = 0; y < dst->rows; y++ )
    {
        float* mx = (float*)(mapx->data.ptr + y*mapx->step);
        float* my = (float*)(mapy->data.ptr + y*mapy->step);

        for( x = 0; x < dst->cols; x++ )
        {
            double xs = x*m[0] + y*m[1] + m[2];
            double ys = x*m[3] + y*m[4] + m[5];
            double ds = x*m[6] + y*m[7] + m[8];
            
            ds = ds ? 1./ds : 0;
            xs *= ds;
            ys *= ds;
            
            mx[x] = (float)xs;
            my[x] = (float)ys;
        }
    }

    CvMat* mask = cvCreateMat( dst->rows, dst->cols, CV_8U );
    cvTsRemap( src, dst, mapx, mapy, mask );
    cvTsZero( dst, mask );
    cvTsZero( dst0, mask );
    cvReleaseMat( &mask );
}


CV_WarpPerspectiveTest warp_perspective_test;



/////////////////////////

void cvTsInitUndistortMap( const CvMat* _a0, const CvMat* _k0, CvMat* _mapx, CvMat* _mapy )
{
	CvMat* mapx = cvCreateMat(_mapx->rows,_mapx->cols,CV_32F);
	CvMat* mapy = cvCreateMat(_mapx->rows,_mapx->cols,CV_32F);

    int u, v;
    double a[9], k[5]={0,0,0,0,0};
    CvMat _a = cvMat(3, 3, CV_64F, a);
    CvMat _k = cvMat(_k0->rows,_k0->cols,
        CV_MAKETYPE(CV_64F,CV_MAT_CN(_k0->type)),k);
    double fx, fy, cx, cy, ifx, ify, cxn, cyn;
    
    cvTsConvert( _a0, &_a );
    cvTsConvert( _k0, &_k );
    fx = a[0]; fy = a[4]; cx = a[2]; cy = a[5];
    ifx = 1./fx; ify = 1./fy;
    cxn = cx;//(mapy->cols - 1)*0.5;
    cyn = cy;//(mapy->rows - 1)*0.5;

    for( v = 0; v < mapy->rows; v++ )
    {
        float* mx = (float*)(mapx->data.ptr + v*mapx->step);
		float* my = (float*)(mapy->data.ptr + v*mapy->step);
        
        for( u = 0; u < mapy->cols; u++ )
        {
            double x = (u - cxn)*ifx;
            double y = (v - cyn)*ify;
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2;
            double cdist = 1 + (k[0] + (k[1] + k[4]*r2)*r2)*r2;
            double x1 = x*cdist + k[2]*2*x*y + k[3]*(r2 + 2*x2);
            double y1 = y*cdist + k[3]*2*x*y + k[2]*(r2 + 2*y2);
           
			my[u] = (float)(y1*fy + cy);
			mx[u] = (float)(x1*fx + cx);
        }
    }

	if (_mapy)
	{
		cvCopy(mapy,_mapy);
		cvCopy(mapx,_mapx);
	}
	else
	{
		for (int i=0;i<mapx->rows;i++)
		{
			float* _mx = (float*)(_mapx->data.ptr + _mapx->step*i);
			float* _my = (float*)(_mapx->data.ptr + _mapx->step*i);
			for (int j=0;j<mapx->cols;j++)
			{
				_mx[2*j] = mapx->data.fl[j+i*mapx->cols];
				_my[2*j+1] = mapy->data.fl[j+i*mapy->cols];
			}
		}
	}
	cvReleaseMat(&mapx);
	cvReleaseMat(&mapy);
}


static double remap_undistort_params[] = { 0.5, 0.5, 0.5, 0.5, 0.01, -0.01, 0.001, -0.001 };

class CV_RemapTest : public CV_ImgWarpBaseTest
{
public:
    CV_RemapTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
};


CV_RemapTest::CV_RemapTest()
    : CV_ImgWarpBaseTest( "warp-remap", "cvRemap", false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
    //default_timing_param_names = imgwarp_perspective_param_names;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_RemapTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "params", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; i < 8; i++ )
            cvWriteReal( fs, 0, remap_undistort_params[i] );
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_RemapTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[INPUT][1] = types[INPUT][2] = CV_32FC1;
    interpolation = CV_INTER_LINEAR;
}


void CV_RemapTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTestImpl::fill_array( test_case_idx, i, j, arr );
}

void CV_RemapTest::get_timing_test_array_types_and_sizes( int test_case_idx,
        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );
    types[INPUT][1] = types[INPUT][2] = CV_32FC1;
    interpolation = CV_INTER_LINEAR;
}


void CV_RemapTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_RemapTest::run_func()
{
    cvRemap( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
             test_array[INPUT][1], test_array[INPUT][2], interpolation );
}


double CV_RemapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_RemapTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const CvMat* src = &test_mat[INPUT][0];
    double a[9] = {0,0,0,0,0,0,0,0,1}, k[4];
    CvMat _a = cvMat( 3, 3, CV_64F, a );
    CvMat _k = cvMat( 4, 1, CV_64F, k );
    double sz = MAX(src->rows, src->cols);

    if( code <= 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
        a[2] = (src->cols - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[5] = (src->rows - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
        a[4] = aspect_ratio*a[0];
        k[0] = cvTsRandReal(rng)*0.06 - 0.03;
        k[1] = cvTsRandReal(rng)*0.06 - 0.03;
        if( k[0]*k[1] > 0 )
            k[1] = -k[1];
        k[2] = cvTsRandReal(rng)*0.004 - 0.002;
        k[3] = cvTsRandReal(rng)*0.004 - 0.002;
    }
    else
    {
        int i;
        a[2] = (src->cols - 1)*remap_undistort_params[0];
        a[5] = (src->rows - 1)*remap_undistort_params[1];
        a[0] = sz/remap_undistort_params[2];
        a[4] = sz/remap_undistort_params[3];
        for( i = 0; i < 4; i++ )
            k[i] = remap_undistort_params[i+4];
    }

    cvTsInitUndistortMap( &_a, &_k, &test_mat[INPUT][1], &test_mat[INPUT][2] );
    return code;
}


void CV_RemapTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* dst0 = &test_mat[INPUT_OUTPUT][0];
    CvMat* mask = cvCreateMat( dst->rows, dst->cols, CV_8U );
    cvTsRemap( &test_mat[INPUT][0], dst,
               &test_mat[INPUT][1], &test_mat[INPUT][2],
               mask, interpolation );
    cvTsZero( dst, mask );
    cvTsZero( dst0, mask );
    cvReleaseMat( &mask );
}


CV_RemapTest remap_test;


////////////////////////////// undistort /////////////////////////////////

class CV_UndistortTest : public CV_ImgWarpBaseTest
{
public:
    CV_UndistortTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );

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


CV_UndistortTest::CV_UndistortTest()
    : CV_ImgWarpBaseTest( "warp-undistort", "cvUndistort2", false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
	test_array[INPUT].push(NULL);

    spatial_scale_decimate = spatial_scale_zoom;
    //default_timing_param_names = imgwarp_perspective_param_names;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_UndistortTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "params", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; i < 8; i++ )
            cvWriteReal( fs, 0, remap_undistort_params[i] );
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_UndistortTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int type = types[INPUT][0];
    type = CV_MAKETYPE( CV_8U, CV_MAT_CN(type) ); 
    types[INPUT][0] = types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = type;
    types[INPUT][1] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
    types[INPUT][2] = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][1] = cvSize(3,3);
    sizes[INPUT][2] = cvTsRandInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);
	types[INPUT][3] =  types[INPUT][1];
	sizes[INPUT][3] = sizes[INPUT][1];
    interpolation = CV_INTER_LINEAR;
}


void CV_UndistortTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTestImpl::fill_array( test_case_idx, i, j, arr );
}

void CV_UndistortTest::get_timing_test_array_types_and_sizes( int test_case_idx,
        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );
    types[INPUT][1] = types[INPUT][2] = CV_32FC1;
    interpolation = CV_INTER_LINEAR;
}


void CV_UndistortTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_UndistortTest::run_func()
{
	if (!useCPlus)
	{
		cvUndistort2( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
                 &test_mat[INPUT][1], &test_mat[INPUT][2] );
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
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 1024 : 5e-2;
}


int CV_UndistortTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );

    const CvMat* src = &test_mat[INPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(src->rows, src->cols);

	double new_cam[9] = {0,0,0,0,0,0,0,0,1};
	CvMat _new_cam = cvMat(test_mat[INPUT][3].rows,test_mat[INPUT][3].cols,CV_64F,new_cam);
	CvMat* _new_cam0 = &test_mat[INPUT][3];

    CvMat* _a0 = &test_mat[INPUT][1], *_k0 = &test_mat[INPUT][2];
    CvMat _a = cvMat(3,3,CV_64F,a);
    CvMat _k = cvMat(_k0->rows,_k0->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(_k0->type)),k);

    if( code <= 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
        a[2] = (src->cols - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[5] = (src->rows - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
        a[4] = aspect_ratio*a[0];
        k[0] = cvTsRandReal(rng)*0.06 - 0.03;
        k[1] = cvTsRandReal(rng)*0.06 - 0.03;
        if( k[0]*k[1] > 0 )
            k[1] = -k[1];
        if( cvTsRandInt(rng)%4 != 0 )
        {
            k[2] = cvTsRandReal(rng)*0.004 - 0.002;
            k[3] = cvTsRandReal(rng)*0.004 - 0.002;
        }
        else
            k[2] = k[3] = 0;

		new_cam[0] = a[0] + (cvTsRandReal(rng) - (double)0.5)*0.2*a[0]; //10%
		new_cam[4] = a[4] + (cvTsRandReal(rng) - (double)0.5)*0.2*a[4]; //10%
		new_cam[2] = a[2] + (cvTsRandReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].rows; //15%
		new_cam[5] = a[5] + (cvTsRandReal(rng) - (double)0.5)*0.3*test_mat[INPUT][0].cols; //15%
    }
    else
    {
        int i;
        a[2] = (src->cols - 1)*remap_undistort_params[0];
        a[5] = (src->rows - 1)*remap_undistort_params[1];
        a[0] = sz/remap_undistort_params[2];
        a[4] = sz/remap_undistort_params[3];
        for( i = 0; i < 4; i++ )
            k[i] = remap_undistort_params[i+4];
    }

    cvTsConvert( &_a, _a0 );


	zero_distortion = (cvRandInt(rng)%2) == 0 ? false : true;
	cvTsConvert( &_k, _k0 );

	zero_new_cam = (cvRandInt(rng)%2) == 0 ? false : true;
	cvTsConvert( &_new_cam, _new_cam0 );
    

	//Testing C++ code
	useCPlus = ((cvTsRandInt(rng) % 2)!=0);
	if (useCPlus)
	{
		input0 = &test_mat[INPUT][0];
		input1 = &test_mat[INPUT][1];
		input2 = &test_mat[INPUT][2];
		input_new_cam = &test_mat[INPUT][3];
	}

    return code;
}


void CV_UndistortTest::prepare_to_validation( int /*test_case_idx*/ )
{
	if (useCPlus)
	{
		CvMat result = input_output;
		CvMat* test_input_output = &test_mat[INPUT_OUTPUT][0];
		cvTsConvert(&result,test_input_output);
	}
    CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* dst0 = &test_mat[INPUT_OUTPUT][0];
    CvMat* mapx = cvCreateMat( dst->rows, dst->cols, CV_32FC1 );
    CvMat* mapy = cvCreateMat( dst->rows, dst->cols, CV_32FC1 );
    cvTsInitUndistortMap( &test_mat[INPUT][1], &test_mat[INPUT][2],
                          mapx, mapy );
    CvMat* mask = cvCreateMat( dst->rows, dst->cols, CV_8U );
    cvTsRemap( src, dst, mapx, mapy, mask, interpolation );
    cvTsZero( dst, mask );
    cvTsZero( dst0, mask );

    cvReleaseMat( &mapx );
    cvReleaseMat( &mapy );
    cvReleaseMat( &mask );
}


CV_UndistortTest undistort_test;



class CV_UndistortMapTest : public CvArrTest
{
public:
    CV_UndistortMapTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
private:
	bool dualChannel;
};


CV_UndistortMapTest::CV_UndistortMapTest()
    : CvArrTest( "warp-undistort-map", "cvInitUndistortMap", "" )
{
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);

    element_wise_relative_error = false;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_UndistortMapTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "params", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; i < 8; i++ )
            cvWriteReal( fs, 0, remap_undistort_params[i] );
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_UndistortMapTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvTsRandInt(rng)%2 ? CV_64F : CV_32F;



    CvSize sz = sizes[OUTPUT][0];
    types[INPUT][0] = types[INPUT][1] = depth;
	dualChannel = cvTsRandInt(rng)%2 == 0;
    types[OUTPUT][0] = types[OUTPUT][1] = 
        types[REF_OUTPUT][0] = types[REF_OUTPUT][1] = dualChannel ? CV_32FC2 : CV_32F;
    sizes[INPUT][0] = cvSize(3,3);
    sizes[INPUT][1] = cvTsRandInt(rng)%2 ? cvSize(4,1) : cvSize(1,4);

    sz.width = MAX(sz.width,16);
    sz.height = MAX(sz.height,16);
    sizes[OUTPUT][0] = sizes[OUTPUT][1] =
        sizes[REF_OUTPUT][0] = sizes[REF_OUTPUT][1] = sz;
}


void CV_UndistortMapTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i != INPUT )
        CvArrTest::fill_array( test_case_idx, i, j, arr );
}

void CV_UndistortMapTest::get_timing_test_array_types_and_sizes( int test_case_idx,
        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
}


void CV_UndistortMapTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_UndistortMapTest::run_func()
{
	if (!dualChannel )
		cvInitUndistortMap( &test_mat[INPUT][0], &test_mat[INPUT][1],
                        test_array[OUTPUT][0], test_array[OUTPUT][1] );
	else
		cvInitUndistortMap( &test_mat[INPUT][0], &test_mat[INPUT][1],
                        test_array[OUTPUT][0], 0 );

}


double CV_UndistortMapTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-3;
}


int CV_UndistortMapTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CvArrTest::prepare_test_case( test_case_idx );
    const CvMat* mapx = &test_mat[OUTPUT][0];
    double k[4], a[9] = {0,0,0,0,0,0,0,0,1};
    double sz = MAX(mapx->rows, mapx->cols);
    CvMat* _a0 = &test_mat[INPUT][0], *_k0 = &test_mat[INPUT][1];
    CvMat _a = cvMat(3,3,CV_64F,a);
    CvMat _k = cvMat(_k0->rows,_k0->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(_k0->type)),k);

    if( code <= 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        double aspect_ratio = cvTsRandReal(rng)*0.6 + 0.7;
        a[2] = (mapx->cols - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[5] = (mapx->rows - 1)*0.5 + cvTsRandReal(rng)*10 - 5;
        a[0] = sz/(0.9 - cvTsRandReal(rng)*0.6);
        a[4] = aspect_ratio*a[0];
        k[0] = cvTsRandReal(rng)*0.06 - 0.03;
        k[1] = cvTsRandReal(rng)*0.06 - 0.03;
        if( k[0]*k[1] > 0 )
            k[1] = -k[1];
        k[2] = cvTsRandReal(rng)*0.004 - 0.002;
        k[3] = cvTsRandReal(rng)*0.004 - 0.002;
    }
    else
    {
        int i;
        a[2] = (mapx->cols - 1)*remap_undistort_params[0];
        a[5] = (mapx->rows - 1)*remap_undistort_params[1];
        a[0] = sz/remap_undistort_params[2];
        a[4] = sz/remap_undistort_params[3];
        for( i = 0; i < 4; i++ )
            k[i] = remap_undistort_params[i+4];
    }

    cvTsConvert( &_a, _a0 );
	cvTsConvert( &_k, _k0 );

	if (dualChannel)
	{
		cvZero(&test_mat[REF_OUTPUT][1]);
		cvZero(&test_mat[OUTPUT][1]);
	}

    return code;
}


void CV_UndistortMapTest::prepare_to_validation( int )
{
	if (!dualChannel )
		cvTsInitUndistortMap( &test_mat[INPUT][0], &test_mat[INPUT][1],
                          &test_mat[REF_OUTPUT][0], &test_mat[REF_OUTPUT][1] );
	else
		cvTsInitUndistortMap( &test_mat[INPUT][0], &test_mat[INPUT][1],
                          &test_mat[REF_OUTPUT][0], 0 );
}


CV_UndistortMapTest undistortmap_test;



////////////////////////////// GetRectSubPix /////////////////////////////////

static const CvSize rectsubpix_sizes[] = {{11, 11}, {21,21}, {41,41},{-1,-1}};

static void
cvTsGetQuadrangeSubPix( const CvMat* src, CvMat* dst, double* a )
{
    int y, x, k, cn;
    int sstep = src->step / sizeof(float);
    int scols = src->cols, srows = src->rows;
    
    assert( CV_MAT_DEPTH(src->type) == CV_32F &&
            CV_ARE_TYPES_EQ(src, dst));

    cn = CV_MAT_CN(dst->type);

    for( y = 0; y < dst->rows; y++ )
        for( x = 0; x < dst->cols; x++ )
        {
            float* d = (float*)(dst->data.ptr + y*dst->step) + x*cn;
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

            s = src->data.fl + sstep*iy + ix*cn;
            for( k = 0; k < cn; k++, s++ )
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
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );
    void fill_array( int test_case_idx, int i, int j, CvMat* arr );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    CvPoint2D32f center;
    bool test_cpp;
};


CV_GetRectSubPixTest::CV_GetRectSubPixTest()
    : CV_ImgWarpBaseTest( "warp-subpix-rect", "cvGetRectSubPix", false )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
    //default_timing_param_names = imgwarp_perspective_param_names;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
    test_cpp = false;
}


int CV_GetRectSubPixTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "rect_size", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; rectsubpix_sizes[i].width > 0; i++ )
        {
            cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteInt( fs, 0, rectsubpix_sizes[i].width );
            cvWriteInt( fs, 0, rectsubpix_sizes[i].height );
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_GetRectSubPixTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int src_depth = cvTsRandInt(rng) % 2, dst_depth;
    int cn = cvTsRandInt(rng) % 2 ? 3 : 1;
    CvSize src_size, dst_size;
    
    dst_depth = src_depth = src_depth == 0 ? CV_8U : CV_32F;
    if( src_depth < CV_32F && cvTsRandInt(rng) % 2 )
        dst_depth = CV_32F;
    
    types[INPUT][0] = CV_MAKETYPE(src_depth,cn);
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(dst_depth,cn);

    src_size = sizes[INPUT][0];
    dst_size.width = cvRound(sqrt(cvTsRandReal(rng)*src_size.width) + 1);
    dst_size.height = cvRound(sqrt(cvTsRandReal(rng)*src_size.height) + 1);
    dst_size.width = MIN(dst_size.width,src_size.width);
    dst_size.height = MIN(dst_size.width,src_size.height);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = dst_size;
    
    center.x = (float)(cvTsRandReal(rng)*src_size.width);
    center.y = (float)(cvTsRandReal(rng)*src_size.height);
    interpolation = CV_INTER_LINEAR;
    
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


void CV_GetRectSubPixTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTestImpl::fill_array( test_case_idx, i, j, arr );
}

void CV_GetRectSubPixTest::get_timing_test_array_types_and_sizes( int test_case_idx,
        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );
    interpolation = CV_INTER_LINEAR;
}


void CV_GetRectSubPixTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
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
    int in_depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    int out_depth = CV_MAT_DEPTH(test_mat[INPUT_OUTPUT][0].type);

    return in_depth >= CV_32F ? 1e-3 : out_depth >= CV_32F ? 1e-2 : 1;
}


int CV_GetRectSubPixTest::prepare_test_case( int test_case_idx )
{
    return CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
}


void CV_GetRectSubPixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src0 = &test_mat[INPUT][0];
    CvMat* dst0 = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* src = src0, *dst = dst0;
    int ftype = CV_MAKETYPE(CV_32F,CV_MAT_CN(src0->type));
    double a[] = { 1, 0, center.x - dst->cols*0.5 + 0.5,
                   0, 1, center.y - dst->rows*0.5 + 0.5 };
    if( CV_MAT_DEPTH(src->type) != CV_32F )
    {
        src = cvCreateMat( src0->rows, src0->cols, ftype );
        cvTsConvert( src0, src );
    }

    if( CV_MAT_DEPTH(dst->type) != CV_32F )
        dst = cvCreateMat( dst0->rows, dst0->cols, ftype );

    cvTsGetQuadrangeSubPix( src, dst, a );

    if( dst != dst0 )
    {
        cvTsConvert( dst, dst0 );
        cvReleaseMat( &dst );
    }
    if( src != src0 )
        cvReleaseMat( &src );
}


CV_GetRectSubPixTest subpix_rect_test;


class CV_GetQuadSubPixTest : public CV_ImgWarpBaseTest
{
public:
    CV_GetQuadSubPixTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    double get_success_error_level( int test_case_idx, int i, int j );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
};


CV_GetQuadSubPixTest::CV_GetQuadSubPixTest()
    : CV_ImgWarpBaseTest( "warp-subpix-quad", "cvGetQuadSubPix", true )
{
    //spatial_scale_zoom = spatial_scale_decimate;
    spatial_scale_decimate = spatial_scale_zoom;
    //default_timing_param_names = imgwarp_affine_param_names;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
}


int CV_GetQuadSubPixTest::write_default_params( CvFileStorage* fs )
{
    int code = CV_ImgWarpBaseTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "rotate_scale", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; imgwarp_affine_rotate_scale[i][0] >= 0; i++ )
        {
            cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteRawData( fs, imgwarp_affine_rotate_scale[i], 4, "d" );
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);
    }

    return code;
}


void CV_GetQuadSubPixTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    int min_size = 4;
    CV_ImgWarpBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize sz = sizes[INPUT][0], dsz;
    CvRNG* rng = ts->get_rng();
    int msz, src_depth = cvTsRandInt(rng) % 2, dst_depth;
    int cn = cvTsRandInt(rng) % 2 ? 3 : 1;
    
    dst_depth = src_depth = src_depth == 0 ? CV_8U : CV_32F;
    if( src_depth < CV_32F && cvTsRandInt(rng) % 2 )
        dst_depth = CV_32F;
    
    types[INPUT][0] = CV_MAKETYPE(src_depth,cn);
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(dst_depth,cn);

    sz.width = MAX(sz.width,min_size);
    sz.height = MAX(sz.height,min_size);
    sizes[INPUT][0] = sz;
    msz = MIN( sz.width, sz.height );

    dsz.width = cvRound(sqrt(cvTsRandReal(rng)*msz) + 1);
    dsz.height = cvRound(sqrt(cvTsRandReal(rng)*msz) + 1);
    dsz.width = MIN(dsz.width,msz);
    dsz.height = MIN(dsz.width,msz);
    dsz.width = MAX(dsz.width,min_size);
    dsz.height = MAX(dsz.height,min_size);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = dsz;
    sizes[INPUT][1] = cvSize( 3, 2 );
}


void CV_GetQuadSubPixTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ImgWarpBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                               whole_sizes, are_images );

    sizes[INPUT][1] = whole_sizes[INPUT][1] = cvSize(3,2);
    sizes[TEMP][0] = whole_sizes[TEMP][0] =
        sizes[TEMP][1] = whole_sizes[TEMP][1] = cvSize(0,0);
    types[INPUT][1] = CV_64FC1;

    interpolation = CV_INTER_LINEAR;
}


void CV_GetQuadSubPixTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    double coeffs[4];
    const CvFileNode* node = find_timing_param( "rotate_scale" );
    assert( node && CV_NODE_IS_SEQ(node->tag) );
    cvReadRawData( ts->get_file_storage(), node, coeffs, "4d" );
    
    sprintf( ptr, "fx=%.2f,fy=%.2f,angle=%.1fdeg,scale=%.1f,", coeffs[0], coeffs[1], coeffs[2], coeffs[3] );
    ptr += strlen(ptr);
    params_left -= 4;

    CV_ImgWarpBaseTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_GetQuadSubPixTest::run_func()
{
    cvGetQuadrangleSubPix( test_array[INPUT][0],
        test_array[INPUT_OUTPUT][0], &test_mat[INPUT][1] );
}


double CV_GetQuadSubPixTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int in_depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    //int out_depth = CV_MAT_DEPTH(test_mat[INPUT_OUTPUT][0].type);

    return in_depth >= CV_32F ? 1e-2 : 4;
}


int CV_GetQuadSubPixTest::prepare_test_case( int test_case_idx )
{
    CvRNG* rng = ts->get_rng();
    int code = CV_ImgWarpBaseTest::prepare_test_case( test_case_idx );
    const CvMat* src = &test_mat[INPUT][0];
    CvMat* mat = &test_mat[INPUT][1];
    CvPoint2D32f center;
    double scale, angle;

    if( code <= 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        double a[6];
        CvMat A = cvMat( 2, 3, CV_64FC1, a );

        center.x = (float)((cvTsRandReal(rng)*1.2 - 0.1)*src->cols);
        center.y = (float)((cvTsRandReal(rng)*1.2 - 0.1)*src->rows);
        angle = cvTsRandReal(rng)*360;
        scale = cvTsRandReal(rng)*0.2 + 0.9;
        
        // y = Ax + b -> x = A^-1(y - b) = A^-1*y - A^-1*b
        scale = 1./scale;
        angle = angle*(CV_PI/180.);
        a[0] = a[4] = cos(angle)*scale;
        a[1] = sin(angle)*scale;
        a[3] = -a[1];
        a[2] = center.x - a[0]*center.x - a[1]*center.y;
        a[5] = center.y - a[3]*center.x - a[4]*center.y;
        cvTsConvert( &A, mat );
    }

    return code;
}


void CV_GetQuadSubPixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* src0 = &test_mat[INPUT][0];
    CvMat* dst0 = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* src = src0, *dst = dst0;
    int ftype = CV_MAKETYPE(CV_32F,CV_MAT_CN(src0->type));
    double a[6], dx = (dst0->cols - 1)*0.5, dy = (dst0->rows - 1)*0.5;
    CvMat A = cvMat( 2, 3, CV_64F, a );

    if( CV_MAT_DEPTH(src->type) != CV_32F )
    {
        src = cvCreateMat( src0->rows, src0->cols, ftype );
        cvTsConvert( src0, src );
    }

    if( CV_MAT_DEPTH(dst->type) != CV_32F )
        dst = cvCreateMat( dst0->rows, dst0->cols, ftype );

    cvTsConvert( &test_mat[INPUT][1], &A );
    a[2] -= a[0]*dx + a[1]*dy;
    a[5] -= a[3]*dx + a[4]*dy;
    cvTsGetQuadrangeSubPix( src, dst, a );

    if( dst != dst0 )
    {
        cvTsConvert( dst, dst0 );
        cvReleaseMat( &dst );
    }

    if( src != src0 )
        cvReleaseMat( &src );
}


CV_GetQuadSubPixTest warp_subpix_quad_test;

/* End of file. */
