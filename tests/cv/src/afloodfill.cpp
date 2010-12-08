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

static const char* floodfill_param_names[] = { "size", "channels", "depth", "dist_type", "labels", 0 };
static const CvSize floodfill_sizes[] = {{320, 240}, {720,480}, {-1,-1}};
static const CvSize floodfill_whole_sizes[] = {{320, 240}, {720,480}, {-1,-1}};
//static const char* floodfill_types[] = { "fixed_level", "fixed_range", "floating_range", 0 };
static const int floodfill_depths[] = { CV_8U, CV_32F, -1 };
static const int floodfill_channels[] = { 1, 3, -1 };

class CV_FloodFillTest : public CvArrTest
{
public:
    CV_FloodFillTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    void fill_array( int test_case_idx, int i, int j, CvMat* arr );
    
    /*int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );*/
    CvPoint seed_pt;
    CvScalar new_val;
    CvScalar l_diff, u_diff;
    int connectivity;
    bool use_mask, mask_only;
    int range_type;
    int new_mask_val;
    bool test_cpp;
};


CV_FloodFillTest::CV_FloodFillTest()
    : CvArrTest( "floodfill", "cvFloodFill", "" )
{
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    optional_mask = false;
    element_wise_relative_error = true;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;

    default_timing_param_names = floodfill_param_names;
    depth_list = floodfill_depths;
    size_list = floodfill_sizes;
    whole_size_list = floodfill_whole_sizes;
    cn_list = floodfill_channels;
    test_cpp = false;
}


void CV_FloodFillTest::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth, cn;
    int i;
    double buf[8];
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    depth = cvTsRandInt(rng) % 2;
    depth = depth == 0 ? CV_8U : CV_32F;
    cn = cvTsRandInt(rng) & 1 ? 3 : 1;

    use_mask = (cvTsRandInt(rng) & 1) != 0;
    connectivity = (cvTsRandInt(rng) & 1) ? 4 : 8;
    mask_only = use_mask && (cvTsRandInt(rng) & 1) != 0;
    new_mask_val = cvTsRandInt(rng) & 255;
    range_type = cvTsRandInt(rng) % 3;

    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    types[INPUT_OUTPUT][1] = types[REF_INPUT_OUTPUT][1] = CV_8UC1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(9,1);

    if( !use_mask )
        sizes[INPUT_OUTPUT][1] = sizes[REF_INPUT_OUTPUT][1] = cvSize(0,0);
    else
    {
        CvSize sz = sizes[INPUT_OUTPUT][0];
        sizes[INPUT_OUTPUT][1] = sizes[REF_INPUT_OUTPUT][1] = cvSize(sz.width+2,sz.height+2);
    }
    
    seed_pt.x = cvTsRandInt(rng) % sizes[INPUT_OUTPUT][0].width;
    seed_pt.y = cvTsRandInt(rng) % sizes[INPUT_OUTPUT][0].height;

    if( range_type == 0 )
        l_diff = u_diff = cvScalarAll(0.);
    else
    {
        CvMat m = cvMat( 1, 8, CV_16S, buf );
        cvRandArr( rng, &m, CV_RAND_NORMAL, cvScalarAll(0), cvScalarAll(32) );
        for( i = 0; i < 4; i++ )
        {
            l_diff.val[i] = fabs(m.data.s[i]/16.);
            u_diff.val[i] = fabs(m.data.s[i+4]/16.);
        }
    }

    new_val = cvScalarAll(0.);
    for( i = 0; i < cn; i++ )
        new_val.val[i] = cvTsRandReal(rng)*255;
    
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


double CV_FloodFillTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    return i == OUTPUT ? FLT_EPSILON : j == 0 ? FLT_EPSILON : 0;
}


void CV_FloodFillTest::fill_array( int test_case_idx, int i, int j, CvMat* arr )
{
    CvRNG* rng = ts->get_rng();
    
    if( i != INPUT && i != INPUT_OUTPUT )
    {
        CvArrTest::fill_array( test_case_idx, i, j, arr );
        return;
    }
    
    if( j == 0 )
    {
        CvMat* tmp = arr;
        CvScalar m = cvScalarAll(128);
        CvScalar s = cvScalarAll(10);

        if( CV_MAT_DEPTH(arr->type) == CV_32FC1 )
            tmp = cvCreateMat( arr->rows, arr->cols, arr->type & CV_MAT_CN_MASK );

        if( range_type == 0 )
            s = cvScalarAll(2);

        cvRandArr( rng, tmp, CV_RAND_NORMAL, m, s );
        if( arr != tmp )
        {
            cvTsConvert( tmp, arr );
            cvReleaseMat( &tmp );
        }
    }
    else
    {
        CvScalar l = cvScalarAll(-2);
        CvScalar u = cvScalarAll(2);
        cvRandArr( rng, arr, CV_RAND_UNI, l, u );
        cvRectangle( arr, cvPoint(0,0), cvPoint(arr->cols-1,arr->rows-1), cvScalarAll(1), 1, 8, 0 );
    }
}


void CV_FloodFillTest::run_func()
{
    int flags = connectivity + (mask_only ? CV_FLOODFILL_MASK_ONLY : 0) +
        (range_type == 1 ? CV_FLOODFILL_FIXED_RANGE : 0) + (new_mask_val << 8);
    double* odata = test_mat[OUTPUT][0].data.db;
    
    if(!test_cpp)
    {
        CvConnectedComp comp;
        cvFloodFill( test_array[INPUT_OUTPUT][0], seed_pt, new_val, l_diff, u_diff, &comp,
                     flags, test_array[INPUT_OUTPUT][1] );
        odata[0] = comp.area;
        odata[1] = comp.rect.x;
        odata[2] = comp.rect.y;
        odata[3] = comp.rect.width;
        odata[4] = comp.rect.height;
        odata[5] = comp.value.val[0];
        odata[6] = comp.value.val[1];
        odata[7] = comp.value.val[2];
        odata[8] = comp.value.val[3];
    }
    else
    {
        cv::Mat img = cv::cvarrToMat(test_array[INPUT_OUTPUT][0]),
            mask = test_array[INPUT_OUTPUT][1] ? cv::cvarrToMat(test_array[INPUT_OUTPUT][1]) : cv::Mat();
        cv::Rect rect;
        int area;
        if( !mask.data )
            area = cv::floodFill( img, seed_pt, new_val, &rect, l_diff, u_diff, flags );
        else
            area = cv::floodFill( img, mask, seed_pt, new_val, &rect, l_diff, u_diff, flags );
        odata[0] = area;
        odata[1] = rect.x;
        odata[2] = rect.y;
        odata[3] = rect.width;
        odata[4] = rect.height;
        odata[5] = odata[6] = odata[7] = odata[8] = 0;
    }
}


typedef struct ff_offset_pair_t
{
    int mofs, iofs;
}
ff_offset_pair_t;

static void
cvTsFloodFill( CvMat* _img, CvPoint seed_pt, CvScalar new_val,
               CvScalar l_diff, CvScalar u_diff, CvMat* _mask,
               double* comp, int connectivity, int range_type,
               int new_mask_val, bool mask_only )
{
    CvMemStorage* st = cvCreateMemStorage();
    ff_offset_pair_t p0, p;
    CvSeq* seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(p0), st );
    CvMat* tmp = _img;
    CvMat* mask;
    CvRect r = cvRect( 0, 0, -1, -1 );
    int area = 0;
    int i, j;
    ushort* m;
    float* img;
    int mstep, step;
    int cn = CV_MAT_CN(_img->type);
    int mdelta[8], idelta[8], ncount;
    int cols = _img->cols, rows = _img->rows;
    int u0 = 0, u1 = 0, u2 = 0;
    double s0 = 0, s1 = 0, s2 = 0;
    
    if( CV_MAT_DEPTH(_img->type) == CV_8U )
    {
        tmp = cvCreateMat( rows, cols, CV_MAKETYPE(CV_32F,CV_MAT_CN(_img->type)) );
        cvTsConvert(_img, tmp);
    }

    mask = cvCreateMat( rows + 2, cols + 2, CV_16UC1 );

    if( _mask )
        cvTsConvert( _mask, mask );
    else
    {
        cvTsZero( mask );
        cvRectangle( mask, cvPoint(0,0), cvPoint(mask->cols-1,mask->rows-1), cvScalarAll(1.), 1, 8, 0 );
    }

    new_mask_val = (new_mask_val != 0 ? new_mask_val : 1) << 8;

    m = (ushort*)(mask->data.ptr + mask->step) + 1;
    mstep = mask->step / sizeof(m[0]);
    img = tmp->data.fl;
    step = tmp->step / sizeof(img[0]);

    p0.mofs = seed_pt.y*mstep + seed_pt.x;
    p0.iofs = seed_pt.y*step + seed_pt.x*cn;

    if( m[p0.mofs] )
        goto _exit_;

    cvSeqPush( seq, &p0 );
    m[p0.mofs] = (ushort)new_mask_val;

    if( connectivity == 4 )
    {
        ncount = 4;
        mdelta[0] = -mstep; idelta[0] = -step;
        mdelta[1] = -1; idelta[1] = -cn;
        mdelta[2] = 1; idelta[2] = cn;
        mdelta[3] = mstep; idelta[3] = step;
    }
    else
    {
        ncount = 8;
        mdelta[0] = -mstep-1; mdelta[1] = -mstep; mdelta[2] = -mstep+1;
        idelta[0] = -step-cn; idelta[1] = -step; idelta[2] = -step+cn;

        mdelta[3] = -1; mdelta[4] = 1;
        idelta[3] = -cn; idelta[4] = cn;

        mdelta[5] = mstep-1; mdelta[6] = mstep; mdelta[7] = mstep+1;
        idelta[5] = step-cn; idelta[6] = step; idelta[7] = step+cn;
    }

    if( cn == 1 )
    {
        float a0 = (float)-l_diff.val[0];
        float b0 = (float)u_diff.val[0];

        s0 = img[p0.iofs];

        if( range_type < 2 )
        {
            a0 += (float)s0; b0 += (float)s0;
        }

        while( seq->total )
        {
            cvSeqPop( seq, &p0 );
            float a = a0, b = b0;
            float* ptr = img + p0.iofs;
            ushort* mptr = m + p0.mofs;

            if( range_type == 2 )
                a += ptr[0], b += ptr[0];

            for( i = 0; i < ncount; i++ )
            {
                int md = mdelta[i], id = idelta[i];
                float v;
                if( !mptr[md] && a <= (v = ptr[id]) && v <= b )
                {
                    mptr[md] = (ushort)new_mask_val;
                    p.mofs = p0.mofs + md;
                    p.iofs = p0.iofs + id;
                    cvSeqPush( seq, &p );
                }
            }
        }
    }
    else
    {
        float a0 = (float)-l_diff.val[0];
        float a1 = (float)-l_diff.val[1];
        float a2 = (float)-l_diff.val[2];
        float b0 = (float)u_diff.val[0];
        float b1 = (float)u_diff.val[1];
        float b2 = (float)u_diff.val[2];

        s0 = img[p0.iofs];
        s1 = img[p0.iofs + 1];
        s2 = img[p0.iofs + 2];

        if( range_type < 2 )
        {
            a0 += (float)s0; b0 += (float)s0;
            a1 += (float)s1; b1 += (float)s1;
            a2 += (float)s2; b2 += (float)s2;
        }

        while( seq->total )
        {
            cvSeqPop( seq, &p0 );
            float _a0 = a0, _a1 = a1, _a2 = a2;
            float _b0 = b0, _b1 = b1, _b2 = b2;
            float* ptr = img + p0.iofs;
            ushort* mptr = m + p0.mofs;

            if( range_type == 2 )
            {
                _a0 += ptr[0]; _b0 += ptr[0];
                _a1 += ptr[1]; _b1 += ptr[1];
                _a2 += ptr[2]; _b2 += ptr[2];
            }

            for( i = 0; i < ncount; i++ )
            {
                int md = mdelta[i], id = idelta[i];
                float v;
                if( !mptr[md] &&
                    _a0 <= (v = ptr[id]) && v <= _b0 &&
                    _a1 <= (v = ptr[id+1]) && v <= _b1 &&
                    _a2 <= (v = ptr[id+2]) && v <= _b2 )
                {
                    mptr[md] = (ushort)new_mask_val;
                    p.mofs = p0.mofs + md;
                    p.iofs = p0.iofs + id;
                    cvSeqPush( seq, &p );
                }
            }
        }        
    }

    r.x = r.width = seed_pt.x;
    r.y = r.height = seed_pt.y;

    if( !mask_only )
    {
        s0 = new_val.val[0];
        s1 = new_val.val[1];
        s2 = new_val.val[2];

        if( tmp != _img )
        {
            u0 = cvRound(s0);
            u0 = CV_CAST_8U(u0);
            u1 = cvRound(s1);
            u1 = CV_CAST_8U(u1);
            u2 = cvRound(s2);
            u2 = CV_CAST_8U(u2);

            s0 = u0;
            s1 = u1;
            s2 = u2;
        }
    }
    else
        s0 = s1 = s2 = 0;

    new_mask_val >>= 8;

    for( i = 0; i < rows; i++ )
    {
        float* ptr = img + i*step;
        ushort* mptr = m + i*mstep;
        uchar* dmptr = _mask ? _mask->data.ptr + (i+1)*_mask->step + 1 : 0;
        uchar* dptr = tmp != _img ? _img->data.ptr + i*_img->step : 0;
        double area0 = area;

        for( j = 0; j < cols; j++ )
        {
            if( mptr[j] > 255 )
            {
                if( dmptr )
                    dmptr[j] = (uchar)new_mask_val;
                if( !mask_only )
                {
                    if( cn == 1 )
                    {
                        if( dptr )
                            dptr[j] = (uchar)u0;
                        else
                            ptr[j] = (float)s0;
                    }
                    else
                    {
                        if( dptr )
                        {
                            dptr[j*3] = (uchar)u0;
                            dptr[j*3+1] = (uchar)u1;
                            dptr[j*3+2] = (uchar)u2;
                        }
                        else
                        {
                            ptr[j*3] = (float)s0;
                            ptr[j*3+1] = (float)s1;
                            ptr[j*3+2] = (float)s2;
                        }
                    }
                }
                else
                {
                    if( cn == 1 )
                        s0 += ptr[j];
                    else
                    {
                        s0 += ptr[j*3];
                        s1 += ptr[j*3+1];
                        s2 += ptr[j*3+2];
                    }
                }

                area++;
                if( r.x > j )
                    r.x = j;
                if( r.width < j )
                    r.width = j;
            }
        }

        if( area != area0 )
        {
            if( r.y > i )
                r.y = i;
            if( r.height < i )
                r.height = i;
        }
    }

_exit_:
    cvReleaseMat( &mask );
    if( tmp != _img )
        cvReleaseMat( &tmp );

    comp[0] = area;
    comp[1] = r.x;
    comp[2] = r.y;
    comp[3] = r.width - r.x + 1;
    comp[4] = r.height - r.y + 1;
    if( mask_only )
    {
        double t = area ? 1./area : 0;
        s0 *= t;
        s1 *= t;
        s2 *= t;
    }
    comp[5] = s0;
    comp[6] = s1;
    comp[7] = s2;
    comp[8] = 0;
}


void CV_FloodFillTest::prepare_to_validation( int /*test_case_idx*/ )
{
    double* comp = test_mat[REF_OUTPUT][0].data.db;
    cvTsFloodFill( &test_mat[REF_INPUT_OUTPUT][0], seed_pt, new_val, l_diff, u_diff,
                   test_array[REF_INPUT_OUTPUT][1] ? &test_mat[REF_INPUT_OUTPUT][1] : 0,
                   comp, connectivity, range_type, new_mask_val, mask_only );
    if(test_cpp)
        comp[5] = comp[6] = comp[7] = comp[8] = 0;
}


/*int CV_FloodFillTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        start_write_param( fs );        
        write_string_list( fs, "dist_type", floodfill_types );
        write_int_list( fs, "labels", floodfill_labels, -1, -1 );
    }

    return code;
}


void CV_FloodFillTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    const char* distype_str = cvReadString( find_timing_param( "dist_type" ), "l2_5x5" );
    mask_size = strstr( distype_str, "3x3" ) ? 3 : 5;
    dist_type = distype_str[0] == 'c' ? CV_DIST_C : distype_str[1] == '1' ? CV_DIST_L1 : CV_DIST_L2;
    fill_labels = cvReadInt( find_timing_param( "labels" ), 0 );

    types[INPUT][0] = CV_8UC1;
    types[OUTPUT][0] = CV_32FC1;
    types[OUTPUT][1] = CV_32SC1;

    if( !fill_labels )
        sizes[OUTPUT][0] = whole_sizes[OUTPUT][0] = cvSize(0,0);
}


void CV_FloodFillTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", cvReadString( find_timing_param( "dist_type" ), "l2_5x5" ) );
    ptr += strlen(ptr);
    sprintf( ptr, "%s,", fill_labels ? "labels" : "no_labels" );
    ptr += strlen(ptr);
    params_left -= 2;

    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}*/


CV_FloodFillTest floodfill_test;

/* End of file. */
