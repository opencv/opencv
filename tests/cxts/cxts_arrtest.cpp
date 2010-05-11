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

#include "_cxts.h"

static const int default_test_case_count = 500;
static const int default_max_log_array_size = 9;

CvArrTest::CvArrTest( const char* _test_name, const char* _test_funcs, const char* _test_descr ) :
    CvTest( _test_name, _test_funcs, _test_descr )
{
    test_case_count = default_test_case_count;

    iplimage_allowed = true;
    cvmat_allowed = true;
    optional_mask = false;
    min_log_array_size = 0;
    max_log_array_size = default_max_log_array_size;
    element_wise_relative_error = true;

    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;

    max_arr = MAX_ARR;
    test_array = new CvTestPtrVec[max_arr];
    max_hdr = 0;
    hdr = 0;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE + CvTS::TIMING_MODE;
}


CvArrTest::~CvArrTest()
{
    clear();
    delete[] test_array;
    test_array = 0;
}


int CvArrTest::write_default_params( CvFileStorage* fs )
{
    int code = CvTest::write_default_params( fs );
    if( code < 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        write_param( fs, "test_case_count", test_case_count );
        write_param( fs, "min_log_array_size", min_log_array_size );
        write_param( fs, "max_log_array_size", max_log_array_size );
    }
    else if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        int i;

        start_write_param( fs ); // make sure we have written the entry header containing the test name
        if( size_list )
        {
            cvStartWriteStruct( fs, "size", CV_NODE_SEQ+CV_NODE_FLOW );
            for( i = 0; size_list[i].width >= 0; i++ )
            {
                cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
                cvWriteInt( fs, 0, size_list[i].width );
                cvWriteInt( fs, 0, size_list[i].height );
                if( whole_size_list &&
                    (whole_size_list[i].width > size_list[i].width ||
                    whole_size_list[i].height > size_list[i].height) )
                {
                    cvWriteInt( fs, 0, whole_size_list[i].width );
                    cvWriteInt( fs, 0, whole_size_list[i].height );
                }
                cvEndWriteStruct( fs );
            }
            cvEndWriteStruct(fs);
        }

        if( depth_list )
        {
            cvStartWriteStruct( fs, "depth", CV_NODE_SEQ+CV_NODE_FLOW );
            for( i = 0; depth_list[i] >= 0; i++ )
                cvWriteString( fs, 0, cvTsGetTypeName(depth_list[i]) );
            cvEndWriteStruct(fs);
        }

        write_int_list( fs, "channels", cn_list, -1, -1 );

        if( optional_mask )
        {
            static const int use_mask[] = { 0, 1 };
            write_int_list( fs, "use_mask", use_mask, 2 );
        }
    }
    return 0;
}


void CvArrTest::clear()
{
    if( test_array )
    {
        int i, j, n;

        for( i = 0; i < max_arr; i++ )
        {
            n = test_array[i].size();
            for( j = 0; j < n; j++ )
                cvRelease( &test_array[i][j] );
        }
    }
    delete[] hdr;
    hdr = 0;
    max_hdr = 0;
    CvTest::clear();
}


int CvArrTest::read_params( CvFileStorage* fs )
{
    int code = CvTest::read_params( fs );
    if( code < 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        min_log_array_size = cvReadInt( find_param( fs, "min_log_array_size" ), min_log_array_size );
        max_log_array_size = cvReadInt( find_param( fs, "max_log_array_size" ), max_log_array_size );
        test_case_count = cvReadInt( find_param( fs, "test_case_count" ), test_case_count );
        test_case_count = cvRound( test_case_count*ts->get_test_case_count_scale() );

        min_log_array_size = cvTsClipInt( min_log_array_size, 0, 20 );
        max_log_array_size = cvTsClipInt( max_log_array_size, min_log_array_size, 20 );
        test_case_count = cvTsClipInt( test_case_count, 0, 100000 );
    }

    return code;
}


void CvArrTest::get_test_array_types_and_sizes( int /*test_case_idx*/, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CvSize size;
    double val;
    int i, j;

    val = cvRandReal(rng) * (max_log_array_size - min_log_array_size) + min_log_array_size;
    size.width = cvRound( exp(val*CV_LOG2) );
    val = cvRandReal(rng) * (max_log_array_size - min_log_array_size) + min_log_array_size;
    size.height = cvRound( exp(val*CV_LOG2) );

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        for( j = 0; j < count; j++ )
        {
            sizes[i][j] = size;
            types[i][j] = CV_8UC1;
        }
    }
}


void CvArrTest::get_timing_test_array_types_and_sizes( int /*test_case_idx*/, CvSize** sizes, int** types,
                                                       CvSize** whole_sizes, bool *are_images )
{
    const CvFileNode* size_node = find_timing_param( "size" );
    const CvFileNode* depth_node = find_timing_param( "depth" );
    const CvFileNode* channels_node = find_timing_param( "channels" );
    int i, j;
    int depth = 0, channels = 1;
    CvSize size = {1,1}, whole_size = size;

    if( size_node && CV_NODE_IS_SEQ(size_node->tag) )
    {
        CvSeq* seq = size_node->data.seq;
        size.width = cvReadInt((const CvFileNode*)cvGetSeqElem(seq,0), 1);
        size.height = cvReadInt((const CvFileNode*)cvGetSeqElem(seq,1), 1);
        whole_size = size;
        if( seq->total > 2 )
        {
            whole_size.width = cvReadInt((const CvFileNode*)cvGetSeqElem(seq,2), 1);
            whole_size.height = cvReadInt((const CvFileNode*)cvGetSeqElem(seq,3), 1);
            whole_size.width = MAX( whole_size.width, size.width );
            whole_size.height = MAX( whole_size.height, size.height );
        }
    }

    if( depth_node && CV_NODE_IS_STRING(depth_node->tag) )
    {
        depth = cvTsTypeByName( depth_node->data.str.ptr );
        if( depth < 0 || depth > CV_64F )
            depth = 0;
    }

    if( channels_node && CV_NODE_IS_INT(channels_node->tag) )
    {
        channels = cvReadInt( channels_node, 1 );
        if( channels < 0 || channels > CV_CN_MAX )
            channels = 1;
    }

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        for( j = 0; j < count; j++ )
        {
            sizes[i][j] = size;
            whole_sizes[i][j] = whole_size;
            if( i != MASK )
                types[i][j] = CV_MAKETYPE(depth,channels);
            else
                types[i][j] = CV_8UC1;
            if( i == REF_OUTPUT || i == REF_INPUT_OUTPUT )
                sizes[i][j] = cvSize(0,0);
        }
    }

    if( are_images )
        *are_images = false; // by default CvMat is used in performance tests
}


void CvArrTest::print_timing_params( int /*test_case_idx*/, char* ptr, int params_left )
{
    int i;
    for( i = 0; i < params_left; i++ )
    {
        sprintf( ptr, "-," );
        ptr += 2;
    }
}


void CvArrTest::print_time( int test_case_idx, double time_clocks, double time_cpu_clocks )
{
    int in_type = -1, out_type = -1;
    CvSize size = { -1, -1 };
    const CvFileNode* size_node = find_timing_param( "size" );
    char str[1024], *ptr = str;
    int len;
    bool have_mask;
    double cpe;

    if( size_node )
    {
        if( !CV_NODE_IS_SEQ(size_node->tag) )
        {
            size.width = cvReadInt(size_node,-1);
            size.height = 1;
        }
        else
        {
            size.width = cvReadInt((const CvFileNode*)cvGetSeqElem(size_node->data.seq,0),-1);
            size.height = cvReadInt((const CvFileNode*)cvGetSeqElem(size_node->data.seq,1),-1);
        }
    }

    if( test_array[INPUT].size() )
    {
        in_type = CV_MAT_TYPE(test_mat[INPUT][0].type);
        if( size.width == -1 )
            size = cvGetMatSize(&test_mat[INPUT][0]);
    }

    if( test_array[OUTPUT].size() )
    {
        out_type = CV_MAT_TYPE(test_mat[OUTPUT][0].type);
        if( in_type < 0 )
            in_type = out_type;
        if( size.width == -1 )
            size = cvGetMatSize(&test_mat[OUTPUT][0]);
    }

    if( out_type < 0 && test_array[INPUT_OUTPUT].size() )
    {
        out_type = CV_MAT_TYPE(test_mat[INPUT_OUTPUT][0].type);
        if( in_type < 0 )
            in_type = out_type;
        if( size.width == -1 )
            size = cvGetMatSize(&test_mat[INPUT_OUTPUT][0]);
    }

    have_mask = test_array[MASK].size() > 0 && test_array[MASK][0] != 0;

    if( in_type < 0 && out_type < 0 )
        return;

    if( out_type < 0 )
        out_type = in_type;

    ptr = strchr( (char*)tested_functions, ',' );
    if( ptr )
    {
        len = (int)(ptr - tested_functions);
        strncpy( str, tested_functions, len );
    }
    else
    {
        len = (int)strlen( tested_functions );
        strcpy( str, tested_functions );
    }
    ptr = str + len;
    *ptr = '\0';
    if( have_mask )
    {
        sprintf( ptr, "(Mask)" );
        ptr += strlen(ptr);
    }
    *ptr++ = ',';
    sprintf( ptr, "%s", cvTsGetTypeName(in_type) );
    ptr += strlen(ptr);

    if( CV_MAT_DEPTH(out_type) != CV_MAT_DEPTH(in_type) )
    {
        sprintf( ptr, "%s", cvTsGetTypeName(out_type) );
        ptr += strlen(ptr);
    }
    *ptr++ = ',';

    sprintf( ptr, "C%d", CV_MAT_CN(in_type) );
    ptr += strlen(ptr);

    if( CV_MAT_CN(out_type) != CV_MAT_CN(in_type) )
    {
        sprintf( ptr, "C%d", CV_MAT_CN(out_type) );
        ptr += strlen(ptr);
    }
    *ptr++ = ',';

    sprintf( ptr, "%dx%d,", size.width, size.height );
    ptr += strlen(ptr);

    print_timing_params( test_case_idx, ptr );
    ptr += strlen(ptr);
    cpe = time_cpu_clocks / ((double)size.width * size.height);
    if( cpe >= 100 )
        sprintf( ptr, "%.0f,", cpe );
    else
        sprintf( ptr, "%.1f,", cpe );
    ptr += strlen(ptr);
    sprintf( ptr, "%g", time_clocks*1e6/cv::getTickFrequency() );

    ts->printf( CvTS::CSV, "%s\n", str );
}


static const int icvTsTypeToDepth[] =
{
    IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16U, IPL_DEPTH_16S,
    IPL_DEPTH_32S, IPL_DEPTH_32F, IPL_DEPTH_64F
};


int CvArrTest::prepare_test_case( int test_case_idx )
{
    int code = 1;
    CvSize** sizes = (CvSize**)malloc( max_arr*sizeof(sizes[0]) );
    CvSize** whole_sizes = (CvSize**)malloc( max_arr*sizeof(whole_sizes[0]) );
    int** types = (int**)malloc( max_arr*sizeof(types[0]) );
    int i, j, total = 0;
    CvRNG* rng = ts->get_rng();
    bool is_image = false;
    bool is_timing_test = false;

    CV_FUNCNAME( "CvArrTest::prepare_test_case" );

    __BEGIN__;

    is_timing_test = ts->get_testing_mode() == CvTS::TIMING_MODE;

    if( is_timing_test )
    {
        if( !get_next_timing_param_tuple() )
        {
            code = -1;
            EXIT;
        }
    }

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        count = MAX(count, 1);
        sizes[i] = (CvSize*)malloc( count*sizeof(sizes[i][0]) );
        types[i] = (int*)malloc( count*sizeof(types[i][0]) );
        whole_sizes[i] = (CvSize*)malloc( count*sizeof(whole_sizes[i][0]) );
    }

    if( !is_timing_test )
        get_test_array_types_and_sizes( test_case_idx, sizes, types );
    else
    {
        get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                               whole_sizes, &is_image );
    }

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        total += count;
        for( j = 0; j < count; j++ )
        {
            unsigned t = cvRandInt(rng);
            bool create_mask = true, use_roi = false;
            CvSize size = sizes[i][j], whole_size = size;
            CvRect roi = {0,0,0,0};

            if( !is_timing_test )
            {
                is_image = !cvmat_allowed ? true : iplimage_allowed ? (t & 1) != 0 : false;
                create_mask = (t & 6) == 0; // ~ each of 3 tests will use mask
                use_roi = (t & 8) != 0;
                if( use_roi )
                {
                    whole_size.width += cvRandInt(rng) % 10;
                    whole_size.height += cvRandInt(rng) % 10;
                }
            }
            else
            {
                whole_size = whole_sizes[i][j];
                use_roi = whole_size.width != size.width || whole_size.height != size.height;
                create_mask = cvReadInt(find_timing_param( "use_mask" ),0) != 0;
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
                    {
                        if( !is_timing_test )
                            roi.x = cvRandInt(rng) % (whole_size.width - size.width);
                        else
                            roi.x = 1;
                    }

                    if( whole_size.height > size.height )
                    {
                        if( !is_timing_test )
                            roi.y = cvRandInt(rng) % (whole_size.height - size.height);
                        else
                            roi.y = 1;
                    }
                }

                if( is_image )
                {
                    CV_CALL( test_array[i][j] = cvCreateImage( whole_size,
                        icvTsTypeToDepth[CV_MAT_DEPTH(types[i][j])],
                        CV_MAT_CN(types[i][j]) ));
                    if( use_roi )
                        cvSetImageROI( (IplImage*)test_array[i][j], roi );
                }
                else
                {
                    CV_CALL( test_array[i][j] = cvCreateMat( whole_size.height,
                                                whole_size.width, types[i][j] ));
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

    if( total > max_hdr )
    {
        delete hdr;
        max_hdr = total;
        hdr = new CvMat[max_hdr];
    }

    total = 0;
    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        test_mat[i] = count > 0 ? hdr + total : 0;
        for( j = 0; j < count; j++ )
        {
            CvArr* arr = test_array[i][j];
            CvMat* mat = &test_mat[i][j];
            if( !arr )
                memset( mat, 0, sizeof(*mat) );
            else if( CV_IS_MAT( arr ))
            {
                *mat = *(CvMat*)arr;
                mat->refcount = 0;
            }
            else
                cvGetMat( arr, mat, 0, 0 );
            if( mat->data.ptr )
                fill_array( test_case_idx, i, j, mat );
        }
        total += count;
    }

    __END__;

    for( i = 0; i < max_arr; i++ )
    {
        if( sizes )
            free( sizes[i] );
        if( whole_sizes )
            free( whole_sizes[i] );
        if( types )
            free( types[i] );
    }

    free( sizes );
    free( whole_sizes );
    free( types );

    return code;
}


void CvArrTest::get_minmax_bounds( int i, int /*j*/, int type, CvScalar* low, CvScalar* high )
{
    double l, u;

    if( i == MASK )
    {
        l = -2;
        u = 2;
    }
    else
    {
        l = cvTsMinVal(type);
        u = cvTsMaxVal(type);
    }

    *low = cvScalarAll(l);
    *high = cvScalarAll(u);
}


void CvArrTest::fill_array( int /*test_case_idx*/, int i, int j, CvMat* arr )
{
    if( i == REF_INPUT_OUTPUT )
        cvTsCopy( &test_mat[INPUT_OUTPUT][j], arr, 0 );
    else if( i == INPUT || i == INPUT_OUTPUT || i == MASK )
    {
        int type = cvGetElemType( arr );
        CvScalar low, high;

        get_minmax_bounds( i, j, type, &low, &high );
        cvTsRandUni( ts->get_rng(), arr, low, high );
    }
}


double CvArrTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int elem_depth = CV_MAT_DEPTH(cvGetElemType(test_array[i][j]));
    assert( i == OUTPUT || i == INPUT_OUTPUT );
    return elem_depth < CV_32F ? 0 : elem_depth == CV_32F ? FLT_EPSILON*100: DBL_EPSILON*5000;
}


void CvArrTest::prepare_to_validation( int /*test_case_idx*/ )
{
    assert(0);
}


int CvArrTest::validate_test_results( int test_case_idx )
{
    static const char* arr_names[] = { "input", "input/output", "output",
                                       "ref input/output", "ref output",
                                       "temporary", "mask" };
    int i, j;
    prepare_to_validation( test_case_idx );

    for( i = 0; i < 2; i++ )
    {
        int i0 = i == 0 ? OUTPUT : INPUT_OUTPUT;
        int i1 = i == 0 ? REF_OUTPUT : REF_INPUT_OUTPUT;
        int count = test_array[i0].size();

        assert( count == test_array[i1].size() );
        for( j = 0; j < count; j++ )
        {
            double err_level;
            CvPoint idx = {0,0};
            double max_diff = 0;
            int code;
            char msg[100];

            if( !test_array[i1][j] )
                continue;

            err_level = get_success_error_level( test_case_idx, i0, j );
            code = cvTsCmpEps( &test_mat[i0][j], &test_mat[i1][j], &max_diff, err_level, &idx,
                               element_wise_relative_error );

            switch( code )
            {
            case -1:
                sprintf( msg, "Too big difference (=%g)", max_diff );
                code = CvTS::FAIL_BAD_ACCURACY;
                break;
            case -2:
                strcpy( msg, "Invalid output" );
                code = CvTS::FAIL_INVALID_OUTPUT;
                break;
            case -3:
                strcpy( msg, "Invalid output in the reference array" );
                code = CvTS::FAIL_INVALID_OUTPUT;
                break;
            default:
                continue;
            }
            ts->printf( CvTS::LOG, "%s in %s array %d at (%d,%d)\n", msg,
                        arr_names[i0], j, idx.x, idx.y );
            for( i0 = 0; i0 < max_arr; i0++ )
            {
                int count = test_array[i0].size();
                if( i0 == REF_INPUT_OUTPUT || i0 == OUTPUT || i0 == TEMP )
                    continue;
                for( i1 = 0; i1 < count; i1++ )
                {
                    CvArr* arr = test_array[i0][i1];
                    if( arr )
                    {
                        CvSize size = cvGetSize(arr);
                        int type = cvGetElemType(arr);
                        ts->printf( CvTS::LOG, "%s array %d type=%sC%d, size=(%d,%d)\n",
                                    arr_names[i0], i1, cvTsGetTypeName(type),
                                    CV_MAT_CN(type), size.width, size.height );
                    }
                }
            }
            ts->set_failed_test_info( code );
            return code;
        }
    }

    return 0;
}

/* End of file. */
