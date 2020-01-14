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

class CV_BaseHistTest : public cvtest::BaseTest
{
public:
    enum { MAX_HIST = 12 };

    CV_BaseHistTest();
    ~CV_BaseHistTest();
    void clear();

protected:
    int read_params( const cv::FileStorage& fs );
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    virtual void init_hist( int test_case_idx, int i );

    virtual void get_hist_params( int test_case_idx );
    virtual float** get_hist_ranges( int test_case_idx );

    int max_log_size;
    int max_cdims;
    int cdims;
    int dims[CV_MAX_DIM];
    int total_size;
    int hist_type;
    int hist_count;
    int uniform;
    int gen_random_hist;
    double gen_hist_max_val, gen_hist_sparse_nz_ratio;

    int init_ranges;
    int img_type;
    int img_max_log_size;
    double low, high, range_delta;
    Size img_size;

    vector<CvHistogram*> hist;
    vector<float> _ranges;
    vector<float*> ranges;
    bool test_cpp;
};


CV_BaseHistTest::CV_BaseHistTest()
{
    test_case_count = 100;
    max_log_size = 20;
    img_max_log_size = 8;
    max_cdims = 6;
    hist_count = 1;
    init_ranges = 0;
    gen_random_hist = 0;
    gen_hist_max_val = 100;

    test_cpp = false;
}


CV_BaseHistTest::~CV_BaseHistTest()
{
    clear();
}


void CV_BaseHistTest::clear()
{
    cvtest::BaseTest::clear();
    for( size_t i = 0; i < hist.size(); i++ )
        cvReleaseHist( &hist[i] );
}


int CV_BaseHistTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    read( find_param( fs, "struct_count" ), test_case_count, test_case_count );
    read( find_param( fs, "max_log_size" ), max_log_size, max_log_size );
    max_log_size = cvtest::clipInt( max_log_size, 1, 20 );
    read( find_param( fs, "max_log_array_size" ), img_max_log_size, img_max_log_size );
    img_max_log_size = cvtest::clipInt( img_max_log_size, 1, 9 );

    read( find_param( fs, "max_cdims" ), max_cdims, max_cdims );
    max_cdims = cvtest::clipInt( max_cdims, 1, 6 );

    return 0;
}


void CV_BaseHistTest::get_hist_params( int /*test_case_idx*/ )
{
    RNG& rng = ts->get_rng();
    int i, max_dim_size, max_ni_dim_size = 31;
    double hist_size;

    cdims = cvtest::randInt(rng) % max_cdims + 1;
    hist_size = exp(cvtest::randReal(rng)*max_log_size*CV_LOG2);
    max_dim_size = cvRound(pow(hist_size,1./cdims));
    total_size = 1;
    uniform = cvtest::randInt(rng) % 2;
    hist_type = cvtest::randInt(rng) % 2 ? CV_HIST_SPARSE : CV_HIST_ARRAY;

    for( i = 0; i < cdims; i++ )
    {
        dims[i] = cvtest::randInt(rng) % (max_dim_size + 2) + 2;
        if( !uniform )
            dims[i] = MIN(dims[i], max_ni_dim_size);
        total_size *= dims[i];
    }

    img_type = cvtest::randInt(rng) % 2 ? CV_32F : CV_8U;
    img_size.width = cvRound( exp(cvtest::randReal(rng) * img_max_log_size * CV_LOG2) );
    img_size.height = cvRound( exp(cvtest::randReal(rng) * img_max_log_size * CV_LOG2) );

    if( img_type < CV_32F )
    {
        low = cvtest::getMinVal(img_type);
        high = cvtest::getMaxVal(img_type);
    }
    else
    {
        high = 1000;
        low = -high;
    }

    range_delta = (cvtest::randInt(rng) % 2)*(high-low)*0.05;
}


float** CV_BaseHistTest::get_hist_ranges( int /*test_case_idx*/ )
{
    double _low = low + range_delta, _high = high - range_delta;

    if( !init_ranges )
        return 0;

    ranges.resize(cdims);

    if( uniform )
    {
        _ranges.resize(cdims*2);
        for( int i = 0; i < cdims; i++ )
        {
            _ranges[i*2] = (float)_low;
            _ranges[i*2+1] = (float)_high;
            ranges[i] = &_ranges[i*2];
        }
    }
    else
    {
        int i, dims_sum = 0, ofs = 0;
        for( i = 0; i < cdims; i++ )
            dims_sum += dims[i] + 1;
        _ranges.resize(dims_sum);

        for( i = 0; i < cdims; i++ )
        {
            int j, n = dims[i];
            // generate logarithmic scale
            double delta, q, val;
            for( j = 0; j < 10; j++ )
            {
                q = 1. + (j+1)*0.1;
                if( (pow(q,(double)n)-1)/(q-1.) >= _high-_low )
                    break;
            }

            if( j == 0 )
            {
                delta = (_high-_low)/n;
                q = 1.;
            }
            else
            {
                q = 1 + j*0.1;
                delta = cvFloor((_high-_low)*(q-1)/(pow(q,(double)n) - 1));
                delta = MAX(delta, 1.);
            }
            val = _low;

            for( j = 0; j <= n; j++ )
            {
                _ranges[j+ofs] = (float)MIN(val,_high);
                val += delta;
                delta *= q;
            }
            ranges[i] = &_ranges[ofs];
            ofs += n + 1;
        }
    }

    return &ranges[0];
}


void CV_BaseHistTest::init_hist( int /*test_case_idx*/, int hist_i )
{
    if( gen_random_hist )
    {
        RNG& rng = ts->get_rng();

        if( hist_type == CV_HIST_ARRAY )
        {
            Mat h = cvarrToMat(hist[hist_i]->bins);
            cvtest::randUni(rng, h, Scalar::all(0), Scalar::all(gen_hist_max_val) );
        }
        else
        {
            CvArr* arr = hist[hist_i]->bins;
            int i, j, totalSize = 1, nz_count;
            int idx[CV_MAX_DIM];
            for( i = 0; i < cdims; i++ )
                totalSize *= dims[i];

            nz_count = cvtest::randInt(rng) % MAX( totalSize/4, 100 );
            nz_count = MIN( nz_count, totalSize );

            // a zero number of non-zero elements should be allowed
            for( i = 0; i < nz_count; i++ )
            {
                for( j = 0; j < cdims; j++ )
                    idx[j] = cvtest::randInt(rng) % dims[j];
                cvSetRealND(arr, idx, cvtest::randReal(rng)*gen_hist_max_val);
            }
        }
    }
}


int CV_BaseHistTest::prepare_test_case( int test_case_idx )
{
    int i;
    float** r;

    clear();

    cvtest::BaseTest::prepare_test_case( test_case_idx );
    get_hist_params( test_case_idx );
    r = get_hist_ranges( test_case_idx );
    hist.resize(hist_count);

    for( i = 0; i < hist_count; i++ )
    {
        hist[i] = cvCreateHist( cdims, dims, hist_type, r, uniform );
        init_hist( test_case_idx, i );
    }
    test_cpp = (cvtest::randInt(ts->get_rng()) % 2) != 0;

    return 1;
}


void CV_BaseHistTest::run_func(void)
{
}


int CV_BaseHistTest::validate_test_results( int /*test_case_idx*/ )
{
    return 0;
}


////////////// testing operation for reading/writing individual histogram bins //////////////

class CV_QueryHistTest : public CV_BaseHistTest
{
public:
    CV_QueryHistTest();
    ~CV_QueryHistTest();
    void clear();

protected:
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void init_hist( int test_case_idx, int i );

    Mat indices;
    Mat values;
    Mat values0;
};


CV_QueryHistTest::CV_QueryHistTest()
{
    hist_count = 1;
}


CV_QueryHistTest::~CV_QueryHistTest()
{
    clear();
}


void CV_QueryHistTest::clear()
{
    CV_BaseHistTest::clear();
}


void CV_QueryHistTest::init_hist( int /*test_case_idx*/, int i )
{
    if( hist_type == CV_HIST_ARRAY )
        cvZero( hist[i]->bins );
}


int CV_QueryHistTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        int i, j, iters;
        float default_value = 0.f;
        RNG& rng = ts->get_rng();
        int* idx;

        iters = (cvtest::randInt(rng) % MAX(total_size/10,100)) + 1;
        iters = MIN( iters, total_size*9/10 + 1 );

        indices = Mat(1, iters*cdims, CV_32S);
        values  = Mat(1, iters, CV_32F );
        values0 = Mat( 1, iters, CV_32F );
        idx = indices.ptr<int>();

        //printf( "total_size = %d, cdims = %d, iters = %d\n", total_size, cdims, iters );

        Mat bit_mask(1, (total_size + 7)/8, CV_8U, Scalar(0));

        #define GET_BIT(n) (bit_mask.data[(n)/8] &  (1 << ((n)&7)))
        #define SET_BIT(n)  bit_mask.data[(n)/8] |= (1 << ((n)&7))

        // set random histogram bins' values to the linear indices of the bins
        for( i = 0; i < iters; i++ )
        {
            int lin_idx = 0;
            for( j = 0; j < cdims; j++ )
            {
                int t = cvtest::randInt(rng) % dims[j];
                idx[i*cdims + j] = t;
                lin_idx = lin_idx*dims[j] + t;
            }

            if( cvtest::randInt(rng) % 8 || GET_BIT(lin_idx) )
            {
                values0.at<float>(i) = (float)(lin_idx+1);
                SET_BIT(lin_idx);
            }
            else
                // some histogram bins will not be initialized intentionally,
                // they should be equal to the default value
                values0.at<float>(i) = default_value;
        }

        // do the second pass to make values0 consistent with bit_mask
        for( i = 0; i < iters; i++ )
        {
            int lin_idx = 0;
            for( j = 0; j < cdims; j++ )
                lin_idx = lin_idx*dims[j] + idx[i*cdims + j];

            if( GET_BIT(lin_idx) )
                values0.at<float>(i) = (float)(lin_idx+1);
        }
    }

    return code;
}


void CV_QueryHistTest::run_func(void)
{
    int i, iters = values.cols;
    CvArr* h = hist[0]->bins;
    const int* idx = indices.ptr<int>();
    float* val = values.ptr<float>();
    float default_value = 0.f;

    // stage 1: write bins
    if( cdims == 1 )
        for( i = 0; i < iters; i++ )
        {
            float v0 = values0.at<float>(i);
            if( fabs(v0 - default_value) < FLT_EPSILON )
                continue;
            if( !(i % 2) )
            {
                if( !(i % 4) )
                    cvSetReal1D( h, idx[i], v0 );
                else
                    *(float*)cvPtr1D( h, idx[i] ) = v0;
            }
            else
                cvSetRealND( h, idx+i, v0 );
        }
    else if( cdims == 2 )
        for( i = 0; i < iters; i++ )
        {
            float v0 = values0.at<float>(i);
            if( fabs(v0 - default_value) < FLT_EPSILON )
                continue;
            if( !(i % 2) )
            {
                if( !(i % 4) )
                    cvSetReal2D( h, idx[i*2], idx[i*2+1], v0 );
                else
                    *(float*)cvPtr2D( h, idx[i*2], idx[i*2+1] ) = v0;
            }
            else
                cvSetRealND( h, idx+i*2, v0 );
        }
    else if( cdims == 3 )
        for( i = 0; i < iters; i++ )
        {
            float v0 = values0.at<float>(i);
            if( fabs(v0 - default_value) < FLT_EPSILON )
                continue;
            if( !(i % 2) )
            {
                if( !(i % 4) )
                    cvSetReal3D( h, idx[i*3], idx[i*3+1], idx[i*3+2], v0 );
                else
                    *(float*)cvPtr3D( h, idx[i*3], idx[i*3+1], idx[i*3+2] ) = v0;
            }
            else
                cvSetRealND( h, idx+i*3, v0 );
        }
    else
        for( i = 0; i < iters; i++ )
        {
            float v0 = values0.at<float>(i);
            if( fabs(v0 - default_value) < FLT_EPSILON )
                continue;
            if( !(i % 2) )
                cvSetRealND( h, idx+i*cdims, v0 );
            else
                *(float*)cvPtrND( h, idx+i*cdims ) = v0;
        }

    // stage 2: read bins
    if( cdims == 1 )
        for( i = 0; i < iters; i++ )
        {
            if( !(i % 2) )
                val[i] = *(float*)cvPtr1D( h, idx[i] );
            else
                val[i] = (float)cvGetReal1D( h, idx[i] );
        }
    else if( cdims == 2 )
        for( i = 0; i < iters; i++ )
        {
            if( !(i % 2) )
                val[i] = *(float*)cvPtr2D( h, idx[i*2], idx[i*2+1] );
            else
                val[i] = (float)cvGetReal2D( h, idx[i*2], idx[i*2+1] );
        }
    else if( cdims == 3 )
        for( i = 0; i < iters; i++ )
        {
            if( !(i % 2) )
                val[i] = *(float*)cvPtr3D( h, idx[i*3], idx[i*3+1], idx[i*3+2] );
            else
                val[i] = (float)cvGetReal3D( h, idx[i*3], idx[i*3+1], idx[i*3+2] );
        }
    else
        for( i = 0; i < iters; i++ )
        {
            if( !(i % 2) )
                val[i] = *(float*)cvPtrND( h, idx+i*cdims );
            else
                val[i] = (float)cvGetRealND( h, idx+i*cdims );
        }
}


int CV_QueryHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    int i, j, iters = values.cols;

    for( i = 0; i < iters; i++ )
    {
        float v = values.at<float>(i), v0 = values0.at<float>(i);

        if( cvIsNaN(v) || cvIsInf(v) )
        {
            ts->printf( cvtest::TS::LOG, "The bin #%d has invalid value\n", i );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
        }
        else if( fabs(v - v0) > FLT_EPSILON )
        {
            ts->printf( cvtest::TS::LOG, "The bin #%d = %g, while it should be %g\n", i, v, v0 );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
        }

        if( code < 0 )
        {
            ts->printf( cvtest::TS::LOG, "The bin index = (" );
            for( j = 0; j < cdims; j++ )
                ts->printf( cvtest::TS::LOG, "%d%s", indices.at<int>(i*cdims + j),
                                        j < cdims-1 ? ", " : ")\n" );
            break;
        }
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


////////////// cvGetMinMaxHistValue //////////////

class CV_MinMaxHistTest : public CV_BaseHistTest
{
public:
    CV_MinMaxHistTest();

protected:
    void run_func(void);
    void init_hist(int, int);
    int validate_test_results( int test_case_idx );
    int min_idx[CV_MAX_DIM], max_idx[CV_MAX_DIM];
    float min_val, max_val;
    int min_idx0[CV_MAX_DIM], max_idx0[CV_MAX_DIM];
    float min_val0, max_val0;
};



CV_MinMaxHistTest::CV_MinMaxHistTest()
{
    hist_count = 1;
    gen_random_hist = 1;
}


void CV_MinMaxHistTest::init_hist(int test_case_idx, int hist_i)
{
    int i, eq = 1;
    RNG& rng = ts->get_rng();
    CV_BaseHistTest::init_hist( test_case_idx, hist_i );

    for(;;)
    {
        for( i = 0; i < cdims; i++ )
        {
            min_idx0[i] = cvtest::randInt(rng) % dims[i];
            max_idx0[i] = cvtest::randInt(rng) % dims[i];
            eq &= min_idx0[i] == max_idx0[i];
        }
        if( !eq || total_size == 1 )
            break;
    }

    min_val0 = (float)(-cvtest::randReal(rng)*10 - FLT_EPSILON);
    max_val0 = (float)(cvtest::randReal(rng)*10 + FLT_EPSILON + gen_hist_max_val);

    if( total_size == 1 )
        min_val0 = max_val0;

    cvSetRealND( hist[0]->bins, min_idx0, min_val0 );
    cvSetRealND( hist[0]->bins, max_idx0, max_val0 );
}


void CV_MinMaxHistTest::run_func(void)
{
    if( hist_type != CV_HIST_ARRAY && test_cpp )
    {
        cv::SparseMat h;
        ((CvSparseMat*)hist[0]->bins)->copyToSparseMat(h);
        double _min_val = 0, _max_val = 0;
        cv::minMaxLoc(h, &_min_val, &_max_val, min_idx, max_idx );
        min_val = (float)_min_val;
        max_val = (float)_max_val;
    }
    else
        cvGetMinMaxHistValue( hist[0], &min_val, &max_val, min_idx, max_idx );
}


int CV_MinMaxHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;

    if( cvIsNaN(min_val) || cvIsInf(min_val) ||
        cvIsNaN(max_val) || cvIsInf(max_val) )
    {
        ts->printf( cvtest::TS::LOG,
            "The extrema histogram bin values are invalid (min = %g, max = %g)\n", min_val, max_val );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if( fabs(min_val - min_val0) > FLT_EPSILON ||
             fabs(max_val - max_val0) > FLT_EPSILON )
    {
        ts->printf( cvtest::TS::LOG,
            "The extrema histogram bin values are incorrect: (min = %g, should be = %g), (max = %g, should be = %g)\n",
            min_val, min_val0, max_val, max_val0 );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }
    else
    {
        int i;
        for( i = 0; i < cdims; i++ )
        {
            if( min_idx[i] != min_idx0[i] || max_idx[i] != max_idx0[i] )
            {
                ts->printf( cvtest::TS::LOG,
                    "The %d-th coordinates of extrema histogram bin values are incorrect: "
                    "(min = %d, should be = %d), (max = %d, should be = %d)\n",
                    i, min_idx[i], min_idx0[i], max_idx[i], max_idx0[i] );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
            }
        }
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


////////////// cvNormalizeHist //////////////

class CV_NormHistTest : public CV_BaseHistTest
{
public:
    CV_NormHistTest();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    double factor;
};



CV_NormHistTest::CV_NormHistTest()
{
    hist_count = 1;
    gen_random_hist = 1;
    factor = 0;
}


int CV_NormHistTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        factor = cvtest::randReal(rng)*10 + 0.1;
        if( hist_type == CV_HIST_SPARSE &&
            ((CvSparseMat*)hist[0]->bins)->heap->active_count == 0 )
            factor = 0;
    }

    return code;
}


void CV_NormHistTest::run_func(void)
{
    if( hist_type != CV_HIST_ARRAY && test_cpp )
    {
        cv::SparseMat h;
        ((CvSparseMat*)hist[0]->bins)->copyToSparseMat(h);
        cv::normalize(h, h, factor, CV_L1);
        cvReleaseSparseMat((CvSparseMat**)&hist[0]->bins);
        hist[0]->bins = cvCreateSparseMat(h);
    }
    else
        cvNormalizeHist( hist[0], factor );
}


int CV_NormHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    double sum = 0;

    if( hist_type == CV_HIST_ARRAY )
    {
        int i;
        const float* ptr = (float*)cvPtr1D( hist[0]->bins, 0 );

        for( i = 0; i < total_size; i++ )
            sum += ptr[i];
    }
    else
    {
        CvSparseMat* sparse = (CvSparseMat*)hist[0]->bins;
        CvSparseMatIterator iterator;
        CvSparseNode *node;

        for( node = cvInitSparseMatIterator( sparse, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            sum += *(float*)CV_NODE_VAL(sparse,node);
        }
    }

    if( cvIsNaN(sum) || cvIsInf(sum) )
    {
        ts->printf( cvtest::TS::LOG,
            "The normalized histogram has invalid sum =%g\n", sum );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }
    else if( fabs(sum - factor) > FLT_EPSILON*10*fabs(factor) )
    {
        ts->printf( cvtest::TS::LOG,
            "The normalized histogram has incorrect sum =%g, while it should be =%g\n", sum, factor );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


////////////// cvThreshHist //////////////

class CV_ThreshHistTest : public CV_BaseHistTest
{
public:
    CV_ThreshHistTest();
    ~CV_ThreshHistTest();
    void clear();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    Mat indices;
    Mat values;
    int orig_nz_count;

    double threshold;
};



CV_ThreshHistTest::CV_ThreshHistTest() : threshold(0)
{
    hist_count = 1;
    gen_random_hist = 1;
}


CV_ThreshHistTest::~CV_ThreshHistTest()
{
    clear();
}


void CV_ThreshHistTest::clear()
{
    CV_BaseHistTest::clear();
}


int CV_ThreshHistTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        threshold = cvtest::randReal(rng)*gen_hist_max_val;

        if( hist_type == CV_HIST_ARRAY )
        {
            orig_nz_count = total_size;

            values = Mat( 1, total_size, CV_32F );
            indices = Mat();
            memcpy( values.ptr<float>(), cvPtr1D( hist[0]->bins, 0 ), total_size*sizeof(float) );
        }
        else
        {
            CvSparseMat* sparse = (CvSparseMat*)hist[0]->bins;
            CvSparseMatIterator iterator;
            CvSparseNode* node;
            int i, k;

            orig_nz_count = sparse->heap->active_count;

            values  = Mat( 1, orig_nz_count+1, CV_32F );
            indices = Mat( 1, (orig_nz_count+1)*cdims, CV_32S );

            for( node = cvInitSparseMatIterator( sparse, &iterator ), i = 0;
                 node != 0; node = cvGetNextSparseNode( &iterator ), i++ )
            {
                 const int* idx = CV_NODE_IDX(sparse,node);

                 OPENCV_ASSERT( i < orig_nz_count, "CV_ThreshHistTest::prepare_test_case", "Buffer overflow" );

                 values.at<float>(i) = *(float*)CV_NODE_VAL(sparse,node);
                 for( k = 0; k < cdims; k++ )
                     indices.at<int>(i*cdims + k) = idx[k];
            }

            OPENCV_ASSERT( i == orig_nz_count, "Unmatched buffer size",
                "CV_ThreshHistTest::prepare_test_case" );
        }
    }

    return code;
}


void CV_ThreshHistTest::run_func(void)
{
    cvThreshHist( hist[0], threshold );
}


int CV_ThreshHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    int i;
    float* ptr0 = values.ptr<float>();
    float* ptr = 0;
    CvSparseMat* sparse = 0;

    if( hist_type == CV_HIST_ARRAY )
        ptr = (float*)cvPtr1D( hist[0]->bins, 0 );
    else
        sparse = (CvSparseMat*)hist[0]->bins;

    if( code > 0 )
    {
        for( i = 0; i < orig_nz_count; i++ )
        {
            float v0 = ptr0[i], v;

            if( hist_type == CV_HIST_ARRAY )
                v = ptr[i];
            else
            {
                v = (float)cvGetRealND( sparse, indices.ptr<int>() + i*cdims );
                cvClearND( sparse, indices.ptr<int>() + i*cdims );
            }

            if( v0 <= threshold ) v0 = 0.f;
            if( cvIsNaN(v) || cvIsInf(v) )
            {
                ts->printf( cvtest::TS::LOG, "The %d-th bin is invalid (=%g)\n", i, v );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                break;
            }
            else if( fabs(v0 - v) > FLT_EPSILON*10*fabs(v0) )
            {
                ts->printf( cvtest::TS::LOG, "The %d-th bin is incorrect (=%g, should be =%g)\n", i, v, v0 );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
                break;
            }
        }
    }

    if( code > 0 && hist_type == CV_HIST_SPARSE )
    {
        if( sparse->heap->active_count > 0 )
        {
            ts->printf( cvtest::TS::LOG,
                "There some extra histogram bins in the sparse histogram after the thresholding\n" );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
        }
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


////////////// cvCompareHist //////////////

class CV_CompareHistTest : public CV_BaseHistTest
{
public:
    enum { MAX_METHOD = 6 };

    CV_CompareHistTest();
protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    double result[MAX_METHOD+1];
};



CV_CompareHistTest::CV_CompareHistTest()
{
    hist_count = 2;
    gen_random_hist = 1;
}


int CV_CompareHistTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    return code;
}


void CV_CompareHistTest::run_func(void)
{
    int k;
    if( hist_type != CV_HIST_ARRAY && test_cpp )
    {
        cv::SparseMat h0, h1;
        ((CvSparseMat*)hist[0]->bins)->copyToSparseMat(h0);
        ((CvSparseMat*)hist[1]->bins)->copyToSparseMat(h1);
        for( k = 0; k < MAX_METHOD; k++ )
            result[k] = cv::compareHist(h0, h1, k);
    }
    else
        for( k = 0; k < MAX_METHOD; k++ )
            result[k] = cvCompareHist( hist[0], hist[1], k );
}


int CV_CompareHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    int i;
    double result0[MAX_METHOD+1];
    double s0 = 0, s1 = 0, sq0 = 0, sq1 = 0, t;

    for( i = 0; i < MAX_METHOD; i++ )
        result0[i] = 0;

    if( hist_type == CV_HIST_ARRAY )
    {
        float* ptr0 = (float*)cvPtr1D( hist[0]->bins, 0 );
        float* ptr1 = (float*)cvPtr1D( hist[1]->bins, 0 );

        for( i = 0; i < total_size; i++ )
        {
            double v0 = ptr0[i], v1 = ptr1[i];
            result0[CV_COMP_CORREL] += v0*v1;
            result0[CV_COMP_INTERSECT] += MIN(v0,v1);
            if( fabs(v0) > DBL_EPSILON )
                result0[CV_COMP_CHISQR] += (v0 - v1)*(v0 - v1)/v0;
            if( fabs(v0 + v1) > DBL_EPSILON )
                result0[CV_COMP_CHISQR_ALT] += (v0 - v1)*(v0 - v1)/(v0 + v1);
            s0 += v0;
            s1 += v1;
            sq0 += v0*v0;
            sq1 += v1*v1;
            result0[CV_COMP_BHATTACHARYYA] += sqrt(v0*v1);
            {
            if( fabs(v0) <= DBL_EPSILON  )
                continue;
            if( fabs(v1) <= DBL_EPSILON )
                v1 = 1e-10;
            result0[CV_COMP_KL_DIV] += v0 * std::log( v0 / v1 );
            }
        }
    }
    else
    {
        CvSparseMat* sparse0 = (CvSparseMat*)hist[0]->bins;
        CvSparseMat* sparse1 = (CvSparseMat*)hist[1]->bins;
        CvSparseMatIterator iterator;
        CvSparseNode* node;

        for( node = cvInitSparseMatIterator( sparse0, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ) )
        {
            const int* idx = CV_NODE_IDX(sparse0, node);
            double v0 = *(float*)CV_NODE_VAL(sparse0, node);
            double v1 = (float)cvGetRealND(sparse1, idx);

            result0[CV_COMP_CORREL] += v0*v1;
            result0[CV_COMP_INTERSECT] += MIN(v0,v1);
            if( fabs(v0) > DBL_EPSILON )
                result0[CV_COMP_CHISQR] += (v0 - v1)*(v0 - v1)/v0;
            if( fabs(v0 + v1) > DBL_EPSILON )
                result0[CV_COMP_CHISQR_ALT] += (v0 - v1)*(v0 - v1)/(v0 + v1);
            s0 += v0;
            sq0 += v0*v0;
            result0[CV_COMP_BHATTACHARYYA] += sqrt(v0*v1);
            {
            if (v0 <= DBL_EPSILON)
                continue;
            if (!v1)
                v1 = 1e-10;
            result0[CV_COMP_KL_DIV] += v0 * std::log( v0 / v1 );
            }
        }

        for( node = cvInitSparseMatIterator( sparse1, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ) )
        {
            double v1 = *(float*)CV_NODE_VAL(sparse1, node);
            s1 += v1;
            sq1 += v1*v1;
        }
    }

    result0[CV_COMP_CHISQR_ALT] *= 2;

    t = (sq0 - s0*s0/total_size)*(sq1 - s1*s1/total_size);
    result0[CV_COMP_CORREL] = fabs(t) > DBL_EPSILON ?
        (result0[CV_COMP_CORREL] - s0*s1/total_size)/sqrt(t) : 1;

    s1 *= s0;
    s0 = result0[CV_COMP_BHATTACHARYYA];
    s0 = 1. - s0*(s1 > FLT_EPSILON ? 1./sqrt(s1) : 1.);
    result0[CV_COMP_BHATTACHARYYA] = sqrt(MAX(s0,0.));

    for( i = 0; i < MAX_METHOD; i++ )
    {
        double v = result[i], v0 = result0[i];
        const char* method_name =
            i == CV_COMP_CHISQR ? "Chi-Square" :
            i == CV_COMP_CHISQR_ALT ? "Alternative Chi-Square" :
            i == CV_COMP_CORREL ? "Correlation" :
            i == CV_COMP_INTERSECT ? "Intersection" :
            i == CV_COMP_BHATTACHARYYA ? "Bhattacharyya" :
            i == CV_COMP_KL_DIV ? "Kullback-Leibler" : "Unknown";

        if( cvIsNaN(v) || cvIsInf(v) )
        {
            ts->printf( cvtest::TS::LOG, "The comparison result using the method #%d (%s) is invalid (=%g)\n",
                i, method_name, v );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
            break;
        }
        else if( fabs(v0 - v) > FLT_EPSILON*14*MAX(fabs(v0),0.1) )
        {
            ts->printf( cvtest::TS::LOG, "The comparison result using the method #%d (%s)\n\tis inaccurate (=%g, should be =%g)\n",
                i, method_name, v, v0 );
            code = cvtest::TS::FAIL_BAD_ACCURACY;
            break;
        }
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


////////////// cvCalcHist //////////////

class CV_CalcHistTest : public CV_BaseHistTest
{
public:
    CV_CalcHistTest();
    ~CV_CalcHistTest();
    void clear();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    vector<Mat> images;
    vector<int> channels;
};



CV_CalcHistTest::CV_CalcHistTest() : images(CV_MAX_DIM+1), channels(CV_MAX_DIM+1)
{
    hist_count = 2;
    gen_random_hist = 0;
    init_ranges = 1;
}


CV_CalcHistTest::~CV_CalcHistTest()
{
    clear();
}


void CV_CalcHistTest::clear()
{
    CV_BaseHistTest::clear();
}


int CV_CalcHistTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        int i;

        for( i = 0; i <= CV_MAX_DIM; i++ )
        {
            if( i < cdims )
            {
                int nch = 1; //cvtest::randInt(rng) % 3 + 1;
                images[i] = Mat(img_size, CV_MAKETYPE(img_type, nch));
                channels[i] = cvtest::randInt(rng) % nch;
                cvtest::randUni( rng, images[i], Scalar::all(low), Scalar::all(high) );
            }
            else if( i == CV_MAX_DIM )
            {
                if( cvtest::randInt(rng) % 2 )
                {
                    // create mask
                    images[i] = Mat(img_size, CV_8U);

                    // make ~25% pixels in the mask non-zero
                    cvtest::randUni( rng, images[i], Scalar::all(-2), Scalar::all(2) );
                }
                else
                    images[i] = Mat();
            }
        }
    }

    return code;
}


void CV_CalcHistTest::run_func(void)
{
    int size[CV_MAX_DIM];
    int hdims = cvGetDims( hist[0]->bins, size);
    bool huniform = CV_IS_UNIFORM_HIST(hist[0]);

    const float* uranges[CV_MAX_DIM] = {0};
    const float** hranges = 0;

    if( hist[0]->type & CV_HIST_RANGES_FLAG )
    {
        hranges = (const float**)hist[0]->thresh2;
        if( huniform )
        {
            for(int i = 0; i < hdims; i++ )
                uranges[i] = &hist[0]->thresh[i][0];
            hranges = uranges;
        }
    }

    std::vector<cv::Mat> imagesv(cdims);
    copy(images.begin(), images.begin() + cdims, imagesv.begin());

    Mat mask = images[CV_MAX_DIM];
    if( !CV_IS_SPARSE_HIST(hist[0]) )
    {
        cv::Mat H = cv::cvarrToMat(hist[0]->bins);
        if(huniform)
        {
            vector<int> emptyChannels;
            vector<int> hSize(hdims);
            for(int i = 0; i < hdims; i++)
                hSize[i] = size[i];
            vector<float> vranges;
            if(hranges)
            {
                vranges.resize(hdims*2);
                for(int i = 0; i < hdims; i++ )
                {
                    vranges[i*2  ] = hist[0]->thresh[i][0];
                    vranges[i*2+1] = hist[0]->thresh[i][1];
                }
            }
            cv::calcHist(imagesv, emptyChannels, mask, H, hSize, vranges);
        }
        else
        {
            cv::calcHist( &imagesv[0], (int)imagesv.size(), 0, mask,
                          H, cvGetDims(hist[0]->bins), H.size, hranges, huniform );
        }
    }
    else
    {
        CvSparseMat* sparsemat = (CvSparseMat*)hist[0]->bins;

        cvZero( hist[0]->bins );

        cv::SparseMat sH;
        sparsemat->copyToSparseMat(sH);

        cv::calcHist( &imagesv[0], (int)imagesv.size(), 0, mask, sH, sH.dims(),
                      sH.dims() > 0 ? sH.hdr->size : 0, hranges, huniform, false);

        cv::SparseMatConstIterator it = sH.begin();
        int nz = (int)sH.nzcount();
        for(int i = 0; i < nz; i++, ++it )
        {
            CV_Assert(it.ptr != NULL);
            *(float*)cvPtrND(sparsemat, it.node()->idx, 0, -2) = *(const float*)it.ptr;
        }
    }
}


static void
cvTsCalcHist( const vector<Mat>& images, CvHistogram* hist, Mat mask, const vector<int>& channels )
{
    int x, y, k;
    union
    {
        const float* fl;
        const uchar* ptr;
    }
    plane[CV_MAX_DIM];
    int nch[CV_MAX_DIM];
    int dims[CV_MAX_DIM];
    int uniform = CV_IS_UNIFORM_HIST(hist);

    int cdims = cvGetDims( hist->bins, dims );
    cvZero( hist->bins );

    Size img_size = images[0].size();
    int img_depth = images[0].depth();
    for( k = 0; k < cdims; k++ )
    {
        nch[k] = images[k].channels();
    }

    for( y = 0; y < img_size.height; y++ )
    {
        const uchar* mptr = mask.empty() ? 0 : mask.ptr<uchar>(y);

        if( img_depth == CV_8U )
            for( k = 0; k < cdims; k++ )
                plane[k].ptr = images[k].ptr<uchar>(y) + channels[k];
        else
            for( k = 0; k < cdims; k++ )
                plane[k].fl  = images[k].ptr<float>(y) + channels[k];

        for( x = 0; x < img_size.width; x++ )
        {
            float val[CV_MAX_DIM];
            int idx[CV_MAX_DIM];

            if( mptr && !mptr[x] )
                continue;
            if( img_depth == CV_8U )
                for( k = 0; k < cdims; k++ )
                    val[k] = plane[k].ptr[x*nch[k]];
            else
                for( k = 0; k < cdims; k++ )
                    val[k] = plane[k].fl[x*nch[k]];

            idx[cdims-1] = -1;

            if( uniform )
            {
                for( k = 0; k < cdims; k++ )
                {
                    double v = val[k], lo = hist->thresh[k][0], hi = hist->thresh[k][1];
                    if (v < lo || v >= hi)
                        break;
                    double idx_ = (v - lo)*dims[k]/(hi - lo);
                    idx[k] = cvFloor(idx_);
                    if (idx[k] < 0)
                    {
                        idx[k] = 0;
                    }
                    if (idx[k] >= dims[k])
                    {
                        idx[k] = dims[k] - 1;
                    }
                }
            }
            else
            {
                for( k = 0; k < cdims; k++ )
                {
                    float v = val[k];
                    float* t = hist->thresh2[k];
                    int j, n = dims[k];

                    for( j = 0; j <= n; j++ )
                        if( v < t[j] )
                            break;
                    if( j <= 0 || j > n )
                        break;
                    idx[k] = j-1;
                }
            }

            if( k < cdims )
                continue;

            (*(float*)cvPtrND( hist->bins, idx ))++;
        }
    }
}


int CV_CalcHistTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    double diff;
    cvTsCalcHist( images, hist[1], images[CV_MAX_DIM], channels );
    diff = cvCompareHist( hist[0], hist[1], CV_COMP_CHISQR );
    if( diff > DBL_EPSILON )
    {
        ts->printf( cvtest::TS::LOG, "The histogram does not match to the reference one\n" );
        code = cvtest::TS::FAIL_BAD_ACCURACY;

    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_CalcHistTest hist_calc_test;



////////////// cvCalcBackProject //////////////

class CV_CalcBackProjectTest : public CV_BaseHistTest
{
public:
    CV_CalcBackProjectTest();
    ~CV_CalcBackProjectTest();
    void clear();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    vector<Mat> images;
    vector<int> channels;
};



CV_CalcBackProjectTest::CV_CalcBackProjectTest() : images(CV_MAX_DIM+3), channels(CV_MAX_DIM+3)
{
    hist_count = 1;
    gen_random_hist = 0;
    init_ranges = 1;
}


CV_CalcBackProjectTest::~CV_CalcBackProjectTest()
{
    clear();
}


void CV_CalcBackProjectTest::clear()
{
    CV_BaseHistTest::clear();
}


int CV_CalcBackProjectTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        int i, j, n, img_len = img_size.area();

        for( i = 0; i < CV_MAX_DIM + 3; i++ )
        {
            if( i < cdims )
            {
                int nch = 1; //cvtest::randInt(rng) % 3 + 1;
                images[i] = Mat(img_size, CV_MAKETYPE(img_type, nch));
                channels[i] = cvtest::randInt(rng) % nch;

                cvtest::randUni( rng, images[i], Scalar::all(low), Scalar::all(high) );
            }
            else if( i == CV_MAX_DIM )
            {
                if(cvtest::randInt(rng) % 2 )
                {
                    // create mask
                    images[i] = Mat(img_size, CV_8U);
                    // make ~25% pixels in the mask non-zero
                    cvtest::randUni( rng, images[i], Scalar::all(-2), Scalar::all(2) );
                }
                else
                {
                    images[i] = Mat();
                }
            }
            else if( i > CV_MAX_DIM )
            {
                images[i] = Mat(img_size, images[0].type());
            }
        }

        cvTsCalcHist( images, hist[0], images[CV_MAX_DIM], channels );

        // now modify the images a bit to add some zeros go to the backprojection
        n = cvtest::randInt(rng) % (img_len/20+1);
        for( i = 0; i < cdims; i++ )
        {
            uchar* data = images[i].data;
            for( j = 0; j < n; j++ )
            {
                int idx = cvtest::randInt(rng) % img_len;
                double val = cvtest::randReal(rng)*(high - low) + low;

                if( img_type == CV_8U )
                    ((uchar*)data)[idx] = (uchar)cvRound(val);
                else
                    ((float*)data)[idx] = (float)val;
            }
        }
    }

    return code;
}


void CV_CalcBackProjectTest::run_func(void)
{
    int size[CV_MAX_DIM];
    int hdims = cvGetDims( hist[0]->bins, size );

    bool huniform = CV_IS_UNIFORM_HIST(hist[0]);
    const float* uranges[CV_MAX_DIM] = {0};
    const float** hranges = 0;

    if( hist[0]->type & CV_HIST_RANGES_FLAG )
    {
        hranges = (const float**)hist[0]->thresh2;
        if( huniform )
        {
            for(int i = 0; i < hdims; i++ )
                uranges[i] = &hist[0]->thresh[i][0];
            hranges = uranges;
        }
    }

    std::vector<cv::Mat> imagesv(hdims);
    copy(images.begin(), images.begin() + hdims, imagesv.begin());

    cv::Mat dst = images[CV_MAX_DIM+1];

    CV_Assert( dst.size() == imagesv[0].size() && dst.depth() == imagesv[0].depth() );

    if( !CV_IS_SPARSE_HIST(hist[0]) )
    {
        cv::Mat H = cv::cvarrToMat(hist[0]->bins);
        if(huniform)
        {
            vector<int> emptyChannels;
            vector<float> vranges;
            if(hranges)
            {
                vranges.resize(hdims*2);
                for(int i = 0; i < hdims; i++ )
                {
                    vranges[i*2  ] = hist[0]->thresh[i][0];
                    vranges[i*2+1] = hist[0]->thresh[i][1];
                }
            }
            cv::calcBackProject(imagesv, emptyChannels, H, dst, vranges, 1);
        }
        else
        {
            cv::calcBackProject( &imagesv[0], (int)imagesv.size(),
                                 0, H, dst, hranges, 1, false );
        }
    }
    else
    {
        cv::SparseMat sH;
        ((const CvSparseMat*)hist[0]->bins)->copyToSparseMat(sH);
        cv::calcBackProject( &imagesv[0], (int)imagesv.size(),
                             0, sH, dst, hranges, 1, huniform );
    }
}


static void
cvTsCalcBackProject( const vector<Mat>& images, Mat dst, CvHistogram* hist, const vector<int>& channels )
{
    int x, y, k, cdims;
    union
    {
        const float* fl;
        const uchar* ptr;
    }
    plane[CV_MAX_DIM];
    int nch[CV_MAX_DIM];
    int dims[CV_MAX_DIM];
    int uniform = CV_IS_UNIFORM_HIST(hist);
    Size img_size = images[0].size();
    int img_depth = images[0].depth();

    cdims = cvGetDims( hist->bins, dims );

    for( k = 0; k < cdims; k++ )
        nch[k] = images[k].channels();

    for( y = 0; y < img_size.height; y++ )
    {
        if( img_depth == CV_8U )
            for( k = 0; k < cdims; k++ )
                plane[k].ptr = images[k].ptr<uchar>(y) + channels[k];
        else
            for( k = 0; k < cdims; k++ )
                plane[k].fl = images[k].ptr<float>(y) + channels[k];

        for( x = 0; x < img_size.width; x++ )
        {
            float val[CV_MAX_DIM];
            float bin_val = 0;
            int idx[CV_MAX_DIM];

            if( img_depth == CV_8U )
                for( k = 0; k < cdims; k++ )
                    val[k] = plane[k].ptr[x*nch[k]];
            else
                for( k = 0; k < cdims; k++ )
                    val[k] = plane[k].fl[x*nch[k]];
            idx[cdims-1] = -1;

            if( uniform )
            {
                for( k = 0; k < cdims; k++ )
                {
                    double v = val[k], lo = hist->thresh[k][0], hi = hist->thresh[k][1];
                    idx[k] = cvFloor((v - lo)*dims[k]/(hi - lo));
                    if( idx[k] < 0 || idx[k] >= dims[k] )
                        break;
                }
            }
            else
            {
                for( k = 0; k < cdims; k++ )
                {
                    float v = val[k];
                    float* t = hist->thresh2[k];
                    int j, n = dims[k];

                    for( j = 0; j <= n; j++ )
                        if( v < t[j] )
                            break;
                    if( j <= 0 || j > n )
                        break;
                    idx[k] = j-1;
                }
            }

            if( k == cdims )
                bin_val = (float)cvGetRealND( hist->bins, idx );

            if( img_depth == CV_8U )
            {
                int t = cvRound(bin_val);
                dst.at<uchar>(y, x) = saturate_cast<uchar>(t);
            }
            else
                dst.at<float>(y, x) = bin_val;
        }
    }
}


int CV_CalcBackProjectTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;

    cvTsCalcBackProject( images, images[CV_MAX_DIM+2], hist[0], channels );
    Mat a = images[CV_MAX_DIM+1], b = images[CV_MAX_DIM+2];
    double threshold = a.depth() == CV_8U ? 2 : FLT_EPSILON;
    code = cvtest::cmpEps2( ts, a, b, threshold, true, "Back project image" );

    if( code < 0 )
        ts->set_failed_test_info( code );

    return code;
}


////////////// cvCalcBackProjectPatch //////////////

class CV_CalcBackProjectPatchTest : public CV_BaseHistTest
{
public:
    CV_CalcBackProjectPatchTest();
    ~CV_CalcBackProjectPatchTest();
    void clear();

protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    vector<Mat> images;
    vector<int> channels;

    Size patch_size;
    double factor;
    int method;
};


CV_CalcBackProjectPatchTest::CV_CalcBackProjectPatchTest() :
    images(CV_MAX_DIM+2), channels(CV_MAX_DIM+2)
{
    hist_count = 1;
    gen_random_hist = 0;
    init_ranges = 1;
    img_max_log_size = 6;
}


CV_CalcBackProjectPatchTest::~CV_CalcBackProjectPatchTest()
{
    clear();
}


void CV_CalcBackProjectPatchTest::clear()
{
    CV_BaseHistTest::clear();
}


int CV_CalcBackProjectPatchTest::prepare_test_case( int test_case_idx )
{
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        int i, j, n, img_len = img_size.area();

        patch_size.width = cvtest::randInt(rng) % img_size.width + 1;
        patch_size.height = cvtest::randInt(rng) % img_size.height + 1;
        patch_size.width = MIN( patch_size.width, 30 );
        patch_size.height = MIN( patch_size.height, 30 );

        factor = 1.;
        method = cvtest::randInt(rng) % CV_CompareHistTest::MAX_METHOD;

        for( i = 0; i < CV_MAX_DIM + 2; i++ )
        {
            if( i < cdims )
            {
                int nch = 1; //cvtest::randInt(rng) % 3 + 1;
                images[i] = Mat(img_size, CV_MAKETYPE(img_type, nch));
                channels[i] = cvtest::randInt(rng) % nch;
                cvtest::randUni( rng, images[i], Scalar::all(low), Scalar::all(high) );
            }
            else if( i >= CV_MAX_DIM )
            {
                images[i] = Mat(img_size - patch_size + Size(1, 1), CV_32F);
            }
        }

        cvTsCalcHist( images, hist[0], Mat(), channels );
        cvNormalizeHist( hist[0], factor );

        // now modify the images a bit
        n = cvtest::randInt(rng) % (img_len/10+1);
        for( i = 0; i < cdims; i++ )
        {
            uchar* data = images[i].data;
            for( j = 0; j < n; j++ )
            {
                int idx = cvtest::randInt(rng) % img_len;
                double val = cvtest::randReal(rng)*(high - low) + low;

                if( img_type == CV_8U )
                    ((uchar*)data)[idx] = (uchar)cvRound(val);
                else
                    ((float*)data)[idx] = (float)val;
            }
        }
    }

    return code;
}


void CV_CalcBackProjectPatchTest::run_func(void)
{
    CvMat dst = cvMat(images[CV_MAX_DIM]);
    vector<CvMat >  img(cdims);
    vector<CvMat*> pimg(cdims);
    for(int i = 0; i < cdims; i++)
    {
        img[i] = cvMat(images[i]);
        pimg[i] = &img[i];
    }
    cvCalcArrBackProjectPatch( (CvArr**)&pimg[0], &dst, cvSize(patch_size), hist[0], method, factor );
}


static void
cvTsCalcBackProjectPatch( const vector<Mat>& images, Mat dst, Size patch_size,
                          CvHistogram* hist, int method,
                          double factor, const vector<int>& channels )
{
    CvHistogram* model = 0;

    int x, y;
    Size size = dst.size();

    cvCopyHist( hist, &model );
    cvNormalizeHist( hist, factor );

    vector<Mat> img(images.size());
    for( y = 0; y < size.height; y++ )
    {
        for( x = 0; x < size.width; x++ )
        {
            double result;

            Rect roi(Point(x, y), patch_size);
            for(size_t i = 0; i < img.size(); i++)
                img[i] = images[i](roi);

            cvTsCalcHist( img, model, Mat(), channels );
            cvNormalizeHist( model, factor );
            result = cvCompareHist( model, hist, method );
            dst.at<float>(y, x) = (float)result;
        }
    }

    cvReleaseHist( &model );
}


int CV_CalcBackProjectPatchTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    double err_level = 5e-3;

    Mat dst = images[CV_MAX_DIM+1];
    vector<Mat> imagesv(cdims);
    for(int i = 0; i < cdims; i++)
        imagesv[i] = images[i];
    cvTsCalcBackProjectPatch( imagesv, dst, patch_size, hist[0],
                              method, factor, channels );

    Mat a = images[CV_MAX_DIM], b = images[CV_MAX_DIM+1];
    code = cvtest::cmpEps2( ts, a, b, err_level, true, "BackProjectPatch result" );

    if( code < 0 )
        ts->set_failed_test_info( code );

    return code;
}


////////////// cvCalcBayesianProb //////////////

class CV_BayesianProbTest : public CV_BaseHistTest
{
public:
    enum { MAX_METHOD = 4 };

    CV_BayesianProbTest();
protected:
    int prepare_test_case( int test_case_idx );
    void run_func(void);
    int validate_test_results( int test_case_idx );
    void init_hist( int test_case_idx, int i );
    void get_hist_params( int test_case_idx );
};



CV_BayesianProbTest::CV_BayesianProbTest()
{
    hist_count = CV_MAX_DIM;
    gen_random_hist = 1;
}


void CV_BayesianProbTest::get_hist_params( int test_case_idx )
{
    CV_BaseHistTest::get_hist_params( test_case_idx );
    hist_type = CV_HIST_ARRAY;
}


void CV_BayesianProbTest::init_hist( int test_case_idx, int hist_i )
{
    if( hist_i < hist_count/2 )
        CV_BaseHistTest::init_hist( test_case_idx, hist_i );
}


int CV_BayesianProbTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();

    hist_count = (cvtest::randInt(rng) % (MAX_HIST/2-1) + 2)*2;
    hist_count = MIN( hist_count, MAX_HIST );
    int code = CV_BaseHistTest::prepare_test_case( test_case_idx );

    return code;
}


void CV_BayesianProbTest::run_func(void)
{
    cvCalcBayesianProb( &hist[0], hist_count/2, &hist[hist_count/2] );
}


int CV_BayesianProbTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    int i, j, n = hist_count/2;
    double s[CV_MAX_DIM];
    const double err_level = 1e-5;

    for( i = 0; i < total_size; i++ )
    {
        double sum = 0;
        for( j = 0; j < n; j++ )
        {
            double v = hist[j]->mat.data.fl[i];
            sum += v;
            s[j] = v;
        }
        sum = sum > DBL_EPSILON ? 1./sum : 0;

        for( j = 0; j < n; j++ )
        {
            double v0 = s[j]*sum;
            double v = hist[j+n]->mat.data.fl[i];

            if( cvIsNaN(v) || cvIsInf(v) )
            {
                ts->printf( cvtest::TS::LOG,
                    "The element #%d in the destination histogram #%d is invalid (=%g)\n",
                    i, j, v );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                break;
            }
            else if( fabs(v0 - v) > err_level*fabs(v0) )
            {
                ts->printf( cvtest::TS::LOG,
                    "The element #%d in the destination histogram #%d is inaccurate (=%g, should be =%g)\n",
                    i, j, v, v0 );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
                break;
            }
        }
        if( j < n )
            break;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_Hist_Calc, accuracy) { CV_CalcHistTest test; test.safe_run(); }
TEST(Imgproc_Hist_Query, accuracy) { CV_QueryHistTest test; test.safe_run(); }

TEST(Imgproc_Hist_Compare, accuracy) { CV_CompareHistTest test; test.safe_run(); }
TEST(Imgproc_Hist_Threshold, accuracy) { CV_ThreshHistTest test; test.safe_run(); }
TEST(Imgproc_Hist_Normalize, accuracy) { CV_NormHistTest test; test.safe_run(); }
TEST(Imgproc_Hist_MinMaxVal, accuracy) { CV_MinMaxHistTest test; test.safe_run(); }

TEST(Imgproc_Hist_CalcBackProject, accuracy) { CV_CalcBackProjectTest test; test.safe_run(); }
TEST(Imgproc_Hist_CalcBackProjectPatch, accuracy) { CV_CalcBackProjectPatchTest test; test.safe_run(); }
TEST(Imgproc_Hist_BayesianProb, accuracy) { CV_BayesianProbTest test; test.safe_run(); }

TEST(Imgproc_Hist_Calc, calcHist_regression_11544)
{
    cv::Mat1w m = cv::Mat1w::zeros(10, 10);
    int n_images = 1;
    int channels[] = { 0 };
    cv::Mat mask;
    cv::MatND hist1, hist2;
    cv::MatND hist1_opt, hist2_opt;
    int dims = 1;
    int hist_size[] = { 1000 };
    float range1[] = { 0, 900 };
    float range2[] = { 0, 1000 };
    const float* ranges1[] = { range1 };
    const float* ranges2[] = { range2 };

    setUseOptimized(false);
    cv::calcHist(&m, n_images, channels, mask, hist1, dims, hist_size, ranges1);
    cv::calcHist(&m, n_images, channels, mask, hist2, dims, hist_size, ranges2);

    setUseOptimized(true);
    cv::calcHist(&m, n_images, channels, mask, hist1_opt, dims, hist_size, ranges1);
    cv::calcHist(&m, n_images, channels, mask, hist2_opt, dims, hist_size, ranges2);

    for(int i = 0; i < 1000; i++)
    {
        EXPECT_EQ(hist1.at<float>(i, 0), hist1_opt.at<float>(i, 0)) << i;
        EXPECT_EQ(hist2.at<float>(i, 0), hist2_opt.at<float>(i, 0)) << i;
    }
}

TEST(Imgproc_Hist_Calc, badarg)
{
    const int channels[] = {0};
    float range1[] = {0, 10};
    float range2[] = {10, 20};
    const float * ranges[] = {range1, range2};
    Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    Mat imgInt = cv::Mat::zeros(10, 10, CV_32SC1);
    Mat hist;
    const int hist_size[] = { 100 };
    // base run
    EXPECT_NO_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, hist_size, ranges, true));
    // bad parameters
    EXPECT_THROW(cv::calcHist(NULL, 1, channels, noArray(), hist, 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 0, channels, noArray(), hist, 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, NULL, noArray(), hist, 2, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), noArray(), 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, -1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, NULL, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&imgInt, 1, channels, noArray(), hist, 1, hist_size, NULL, true), cv::Exception);
    // special case
    EXPECT_NO_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, hist_size, NULL, true));

    Mat backProj;
    // base run
    EXPECT_NO_THROW(cv::calcBackProject(&img, 1, channels, hist, backProj, ranges, 1, true));
    // bad parameters
    EXPECT_THROW(cv::calcBackProject(NULL, 1, channels, hist, backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 0, channels, hist, backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 1, channels, noArray(), backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 1, channels, hist, noArray(), ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&imgInt, 1, channels, hist, backProj, NULL, 1, true), cv::Exception);
    // special case
    EXPECT_NO_THROW(cv::calcBackProject(&img, 1, channels, hist, backProj, NULL, 1, true));
}


}} // namespace
/* End Of File */
