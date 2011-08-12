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

#include "precomp.hpp"

namespace cvtest
{

static const int default_test_case_count = 500;
static const int default_max_log_array_size = 9;

ArrayTest::ArrayTest()
{
    test_case_count = default_test_case_count;

    iplimage_allowed = true;
    cvmat_allowed = true;
    optional_mask = false;
    min_log_array_size = 0;
    max_log_array_size = default_max_log_array_size;
    element_wise_relative_error = true;
    
    test_array.resize(MAX_ARR);
}


ArrayTest::~ArrayTest()
{
    clear();
}


void ArrayTest::clear()
{
    for( size_t i = 0; i < test_array.size(); i++ )
    {
        for( size_t j = 0; j < test_array[i].size(); j++ )
            cvRelease( &test_array[i][j] );
    }
    BaseTest::clear();
}


int ArrayTest::read_params( CvFileStorage* fs )
{
    int code = BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    min_log_array_size = cvReadInt( find_param( fs, "min_log_array_size" ), min_log_array_size );
    max_log_array_size = cvReadInt( find_param( fs, "max_log_array_size" ), max_log_array_size );
    test_case_count = cvReadInt( find_param( fs, "test_case_count" ), test_case_count );
    test_case_count = cvRound( test_case_count*ts->get_test_case_count_scale() );

    min_log_array_size = clipInt( min_log_array_size, 0, 20 );
    max_log_array_size = clipInt( max_log_array_size, min_log_array_size, 20 );
    test_case_count = clipInt( test_case_count, 0, 100000 );

    return code;
}


void ArrayTest::get_test_array_types_and_sizes( int /*test_case_idx*/, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    Size size;
    double val;
    size_t i, j;

    val = randReal(rng) * (max_log_array_size - min_log_array_size) + min_log_array_size;
    size.width = cvRound( exp(val*CV_LOG2) );
    val = randReal(rng) * (max_log_array_size - min_log_array_size) + min_log_array_size;
    size.height = cvRound( exp(val*CV_LOG2) );

    for( i = 0; i < test_array.size(); i++ )
    {
        size_t sizei = test_array[i].size();
        for( j = 0; j < sizei; j++ )
        {
            sizes[i][j] = size;
            types[i][j] = CV_8UC1;
        }
    }
}


static const int icvTsTypeToDepth[] =
{
    IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16U, IPL_DEPTH_16S,
    IPL_DEPTH_32S, IPL_DEPTH_32F, IPL_DEPTH_64F
};


int ArrayTest::prepare_test_case( int test_case_idx )
{
    int code = 1;
    size_t max_arr = test_array.size();
    vector<vector<Size> > sizes(max_arr);
    vector<vector<Size> > whole_sizes(max_arr);
    vector<vector<int> > types(max_arr);
    size_t i, j;
    RNG& rng = ts->get_rng();
    bool is_image = false;

    for( i = 0; i < max_arr; i++ )
    {
        size_t sizei = std::max(test_array[i].size(), (size_t)1);
        sizes[i].resize(sizei);
        types[i].resize(sizei);
        whole_sizes[i].resize(sizei);
    }

    get_test_array_types_and_sizes( test_case_idx, sizes, types );

    for( i = 0; i < max_arr; i++ )
    {
        size_t sizei = test_array[i].size();
        for( j = 0; j < sizei; j++ )
        {
            unsigned t = randInt(rng);
            bool create_mask = true, use_roi = false;
            CvSize size = sizes[i][j], whole_size = size;
            CvRect roi = {0,0,0,0};

            is_image = !cvmat_allowed ? true : iplimage_allowed ? (t & 1) != 0 : false;
            create_mask = (t & 6) == 0; // ~ each of 3 tests will use mask
            use_roi = (t & 8) != 0;
            if( use_roi )
            {
                whole_size.width += randInt(rng) % 10;
                whole_size.height += randInt(rng) % 10;
            }

            cvRelease( &test_array[i][j] );
            if( size.width > 0 && size.height > 0 &&
                types[i][j] >= 0 && (i != MASK || create_mask) )
            {
                if( use_roi )
                {
                    roi.width = size.width;
                    roi.height = size.height;
                    if( whole_size.width > size.width )
                        roi.x = randInt(rng) % (whole_size.width - size.width);

                    if( whole_size.height > size.height )
                        roi.y = randInt(rng) % (whole_size.height - size.height);
                }

                if( is_image )
                {
                    test_array[i][j] = cvCreateImage( whole_size,
                        icvTsTypeToDepth[CV_MAT_DEPTH(types[i][j])], CV_MAT_CN(types[i][j]) );
                    if( use_roi )
                        cvSetImageROI( (IplImage*)test_array[i][j], roi );
                }
                else
                {
                    test_array[i][j] = cvCreateMat( whole_size.height, whole_size.width, types[i][j] );
                    if( use_roi )
                    {
                        CvMat submat, *mat = (CvMat*)test_array[i][j];
                        cvGetSubRect( test_array[i][j], &submat, roi );
                        submat.refcount = mat->refcount;
                        *mat = submat;
                    }
                }
            }
        }
    }

    test_mat.resize(test_array.size());
    for( i = 0; i < max_arr; i++ )
    {
        size_t sizei = test_array[i].size();
        test_mat[i].resize(sizei);
        for( j = 0; j < sizei; j++ )
        {
            CvArr* arr = test_array[i][j];
            test_mat[i][j] = cv::cvarrToMat(arr);
            if( !test_mat[i][j].empty() )
                fill_array( test_case_idx, (int)i, (int)j, test_mat[i][j] );
        }
    }

    return code;
}


void ArrayTest::get_minmax_bounds( int i, int /*j*/, int type, Scalar& low, Scalar& high )
{
    double l, u;
    int depth = CV_MAT_DEPTH(type);

    if( i == MASK )
    {
        l = -2;
        u = 2;
    }
    else if( depth < CV_32S )
    {
        l = getMinVal(type);
        u = getMaxVal(type);
    }
    else
    {
        u = depth == CV_32S ? 1000000 : 1000.;
        l = -u;
    }

    low = Scalar::all(l);
    high = Scalar::all(u);
}


void ArrayTest::fill_array( int /*test_case_idx*/, int i, int j, Mat& arr )
{
    if( i == REF_INPUT_OUTPUT )
        cvtest::copy( test_mat[INPUT_OUTPUT][j], arr, Mat() );
    else if( i == INPUT || i == INPUT_OUTPUT || i == MASK )
    {
        Scalar low, high;

        get_minmax_bounds( i, j, arr.type(), low, high );
        randUni( ts->get_rng(), arr, low, high );
    }
}


double ArrayTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int elem_depth = CV_MAT_DEPTH(cvGetElemType(test_array[i][j]));
    assert( i == OUTPUT || i == INPUT_OUTPUT );
    return elem_depth < CV_32F ? 0 : elem_depth == CV_32F ? FLT_EPSILON*100: DBL_EPSILON*5000;
}


void ArrayTest::prepare_to_validation( int /*test_case_idx*/ )
{
    assert(0);
}


int ArrayTest::validate_test_results( int test_case_idx )
{
    static const char* arr_names[] = { "input", "input/output", "output",
                                       "ref input/output", "ref output",
                                       "temporary", "mask" };
    size_t i, j;
    prepare_to_validation( test_case_idx );

    for( i = 0; i < 2; i++ )
    {
        int i0 = i == 0 ? OUTPUT : INPUT_OUTPUT;
        int i1 = i == 0 ? REF_OUTPUT : REF_INPUT_OUTPUT;
        size_t sizei = test_array[i0].size();

        assert( sizei == test_array[i1].size() );
        for( j = 0; j < sizei; j++ )
        {
            double err_level;
            vector<int> idx;
            double max_diff = 0;
            int code;
            char msg[100];

            if( !test_array[i1][j] )
                continue;

            err_level = get_success_error_level( test_case_idx, i0, (int)j );
            code = cmpEps( test_mat[i0][j], test_mat[i1][j], &max_diff, err_level, &idx, element_wise_relative_error );

            switch( code )
            {
            case -1:
                sprintf( msg, "Too big difference (=%g)", max_diff );
                code = TS::FAIL_BAD_ACCURACY;
                break;
            case -2:
                strcpy( msg, "Invalid output" );
                code = TS::FAIL_INVALID_OUTPUT;
                break;
            case -3:
                strcpy( msg, "Invalid output in the reference array" );
                code = TS::FAIL_INVALID_OUTPUT;
                break;
            default:
                continue;
            }
            string idxstr = vec2str(", ", &idx[0], idx.size());
            
            ts->printf( TS::LOG, "%s in %s array %d at (%s)", msg, arr_names[i0], j, idxstr.c_str() );
            
            for( i0 = 0; i0 < (int)test_array.size(); i0++ )
            {
                size_t sizei0 = test_array[i0].size();
                if( i0 == REF_INPUT_OUTPUT || i0 == OUTPUT || i0 == TEMP )
                    continue;
                for( i1 = 0; i1 < (int)sizei0; i1++ )
                {
                    const Mat& arr = test_mat[i0][i1];
                    if( !arr.empty() )
                    {
                        string sizestr = vec2str(", ", &arr.size[0], arr.dims);
                        ts->printf( TS::LOG, "%s array %d type=%sC%d, size=(%s)\n",
                                    arr_names[i0], i1, getTypeName(arr.depth()),
                                    arr.channels(), sizestr.c_str() );
                    }
                }
            }
            ts->set_failed_test_info( code );
            return code;
        }
    }

    return 0;
}
    
}

/* End of file. */
