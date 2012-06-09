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

class CV_FilterBaseTest : public cvtest::ArrayTest
{
public:
    CV_FilterBaseTest( bool _fp_kernel );

protected:
    int prepare_test_case( int test_case_idx );
    int read_params( CvFileStorage* fs );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    CvSize aperture_size;
    CvPoint anchor;
    int max_aperture_size;
    bool fp_kernel;
    bool inplace;
    int border;
};


CV_FilterBaseTest::CV_FilterBaseTest( bool _fp_kernel ) : fp_kernel(_fp_kernel)
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    max_aperture_size = 13;
    inplace = false;
    aperture_size = cvSize(0,0);
    anchor = cvPoint(0,0);
    element_wise_relative_error = false;
}


int CV_FilterBaseTest::read_params( CvFileStorage* fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    if( code < 0 )
        return code;

    max_aperture_size = cvReadInt( find_param( fs, "max_aperture_size" ), max_aperture_size );
    max_aperture_size = cvtest::clipInt( max_aperture_size, 1, 100 );

    return code;
}


void CV_FilterBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT )
    {
        if( j == 1 )
        {
            if( fp_kernel )
            {
                RNG& rng = ts->get_rng();
                double val = exp( cvtest::randReal(rng)*10 - 4 );
                low = Scalar::all(-val);
                high = Scalar::all(val);
            }
            else
            {
                low = Scalar::all(0);
                high = Scalar::all(2);
            }
        }
        else if( CV_MAT_DEPTH(type) == CV_32F )
        {
            low = Scalar::all(-10.);
            high = Scalar::all(10.);
        }
    }
}


void CV_FilterBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                        vector<vector<Size> >& sizes,
                                                        vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % CV_32F;
    int cn = cvtest::randInt(rng) % 3 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth += depth == CV_8S;
    cn += cn == 2;

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);

    aperture_size.width = cvtest::randInt(rng) % max_aperture_size + 1;
    aperture_size.height = cvtest::randInt(rng) % max_aperture_size + 1;
    anchor.x = cvtest::randInt(rng) % aperture_size.width;
    anchor.y = cvtest::randInt(rng) % aperture_size.height;

    types[INPUT][1] = fp_kernel ? CV_32FC1 : CV_8UC1;
    sizes[INPUT][1] = aperture_size;

    inplace = cvtest::randInt(rng) % 2 != 0;
    border = BORDER_REPLICATE;
}


int CV_FilterBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        if( inplace && test_mat[INPUT][0].type() == test_mat[OUTPUT][0].type())
            cvtest::copy( test_mat[INPUT][0], test_mat[OUTPUT][0] );
        else
            inplace = false;
    }
    return code;
}


/////////////////////////

class CV_MorphologyBaseTest : public CV_FilterBaseTest
{
public:
    CV_MorphologyBaseTest();

protected:
    void prepare_to_validation( int test_case_idx );
    int prepare_test_case( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int optype, optype_min, optype_max;
    int shape;
    IplConvKernel* element;
};


CV_MorphologyBaseTest::CV_MorphologyBaseTest() : CV_FilterBaseTest( false )
{
    shape = -1;
    element = 0;
    optype = optype_min = optype_max = -1;
}


void CV_MorphologyBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_FilterBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng) % 4;
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : depth == 2 ? CV_16S : CV_32F;
    int cn = CV_MAT_CN(types[INPUT][0]);

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    shape = cvtest::randInt(rng) % 4;
    if( shape >= 3 )
        shape = CV_SHAPE_CUSTOM;
    else
        sizes[INPUT][1] = cvSize(0,0);
    optype = cvtest::randInt(rng) % (optype_max - optype_min + 1) + optype_min;
}


double CV_MorphologyBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return test_mat[INPUT][0].depth() < CV_32F ||
        (optype == CV_MOP_ERODE || optype == CV_MOP_DILATE ||
        optype == CV_MOP_OPEN || optype == CV_MOP_CLOSE) ? 0 : 1e-5;
}


int CV_MorphologyBaseTest::prepare_test_case( int test_case_idx )
{
    int code = CV_FilterBaseTest::prepare_test_case( test_case_idx );
    vector<int> eldata;

    if( code <= 0 )
        return code;

    if( shape == CV_SHAPE_CUSTOM )
    {
        eldata.resize(aperture_size.width*aperture_size.height);
        uchar* src = test_mat[INPUT][1].data;
        int srcstep = (int)test_mat[INPUT][1].step;
        int i, j, nonzero = 0;

        for( i = 0; i < aperture_size.height; i++ )
        {
            for( j = 0; j < aperture_size.width; j++ )
            {
                eldata[i*aperture_size.width + j] = src[i*srcstep + j];
                nonzero += src[i*srcstep + j] != 0;
            }
        }

        if( nonzero == 0 )
            eldata[anchor.y*aperture_size.width + anchor.x] = 1;
    }

    cvReleaseStructuringElement( &element );
    element = cvCreateStructuringElementEx( aperture_size.width, aperture_size.height,
                                           anchor.x, anchor.y, shape, eldata.empty() ? 0 : &eldata[0] );
    return code;
}


void CV_MorphologyBaseTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0], &dst = test_mat[REF_OUTPUT][0];
    Mat _ielement(element->nRows, element->nCols, CV_32S, element->values);
    Mat _element;
    _ielement.convertTo(_element, CV_8U);
    Point _anchor(element->anchorX, element->anchorY);
    int _border = BORDER_REPLICATE;

    if( optype == CV_MOP_ERODE )
    {
        cvtest::erode( src, dst, _element, _anchor, _border );
    }
    else if( optype == CV_MOP_DILATE )
    {
        cvtest::dilate( src, dst, _element, _anchor, _border );
    }
    else
    {
        Mat temp;
        if( optype == CV_MOP_OPEN )
        {
            cvtest::erode( src, temp, _element, _anchor, _border );
            cvtest::dilate( temp, dst, _element, _anchor, _border );
        }
        else if( optype == CV_MOP_CLOSE )
        {
            cvtest::dilate( src, temp, _element, _anchor, _border );
            cvtest::erode( temp, dst, _element, _anchor, _border );
        }
        else if( optype == CV_MOP_GRADIENT )
        {
            cvtest::erode( src, temp, _element, _anchor, _border );
            cvtest::dilate( src, dst, _element, _anchor, _border );
            cvtest::add( dst, 1, temp, -1, Scalar::all(0), dst, dst.type() );
        }
        else if( optype == CV_MOP_TOPHAT )
        {
            cvtest::erode( src, temp, _element, _anchor, _border );
            cvtest::dilate( temp, dst, _element, _anchor, _border );
            cvtest::add( src, 1, dst, -1, Scalar::all(0), dst, dst.type() );
        }
        else if( optype == CV_MOP_BLACKHAT )
        {
            cvtest::dilate( src, temp, _element, _anchor, _border );
            cvtest::erode( temp, dst, _element, _anchor, _border );
            cvtest::add( dst, 1, src, -1, Scalar::all(0), dst, dst.type() );
        }
        else
            CV_Error( CV_StsBadArg, "Unknown operation" );
    }

    cvReleaseStructuringElement( &element );
}


/////////////// erode ///////////////

class CV_ErodeTest : public CV_MorphologyBaseTest
{
public:
    CV_ErodeTest();
protected:
    void run_func();
};


CV_ErodeTest::CV_ErodeTest()
{
    optype_min = optype_max = CV_MOP_ERODE;
}


void CV_ErodeTest::run_func()
{
    cvErode( inplace ? test_array[OUTPUT][0] : test_array[INPUT][0],
             test_array[OUTPUT][0], element, 1 );
}


/////////////// dilate ///////////////

class CV_DilateTest : public CV_MorphologyBaseTest
{
public:
    CV_DilateTest();
protected:
    void run_func();
};


CV_DilateTest::CV_DilateTest()
{
    optype_min = optype_max = CV_MOP_DILATE;
}


void CV_DilateTest::run_func()
{
    cvDilate( inplace ? test_array[OUTPUT][0] : test_array[INPUT][0],
             test_array[OUTPUT][0], element, 1 );
}

/////////////// morphEx ///////////////

class CV_MorphExTest : public CV_MorphologyBaseTest
{
public:
    CV_MorphExTest();
protected:
    void run_func();
};


CV_MorphExTest::CV_MorphExTest()
{
    optype_min = CV_MOP_ERODE;
    optype_max = CV_MOP_BLACKHAT;
}


void CV_MorphExTest::run_func()
{
    cvMorphologyEx( test_array[inplace ? OUTPUT : INPUT][0],
             test_array[OUTPUT][0], 0, element, optype, 1 );
}

/////////////// generic filter ///////////////

class CV_FilterTest : public CV_FilterBaseTest
{
public:
    CV_FilterTest();

protected:
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_FilterTest::CV_FilterTest() : CV_FilterBaseTest( true )
{
}


void CV_FilterTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_FilterBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng)%3;
    int cn = CV_MAT_CN(types[INPUT][0]);
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);
}


double CV_FilterTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 2 : depth <= CV_32S ? 32 :
           depth == CV_32F ? 1e-4 : 1e-10;
}


void CV_FilterTest::run_func()
{
    CvMat kernel = test_mat[INPUT][1];
    cvFilter2D( test_array[inplace ? OUTPUT : INPUT][0],
                test_array[OUTPUT][0], &kernel, anchor );
}


void CV_FilterTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].type(),
                      test_mat[INPUT][1], anchor, 0, BORDER_REPLICATE );
}


////////////////////////

class CV_DerivBaseTest : public CV_FilterBaseTest
{
public:
    CV_DerivBaseTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int _aperture_size;
};


CV_DerivBaseTest::CV_DerivBaseTest() : CV_FilterBaseTest( true )
{
    max_aperture_size = 7;
}


void CV_DerivBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_FilterBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng) % 2;
    depth = depth == 0 ? CV_8U : CV_32F;
    types[INPUT][0] = CV_MAKETYPE(depth,1);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth==CV_8U?CV_16S:CV_32F,1);
    _aperture_size = (cvtest::randInt(rng)%5)*2 - 1;
    sizes[INPUT][1] = aperture_size = cvSize(_aperture_size, _aperture_size);
}


double CV_DerivBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 2 : 5e-4;
}


/////////////// sobel ///////////////

class CV_SobelTest : public CV_DerivBaseTest
{
public:
    CV_SobelTest();

protected:
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx,
        vector<vector<Size> >& sizes, vector<vector<int> >& types );
    int dx, dy, origin;
};


CV_SobelTest::CV_SobelTest() {}


void CV_SobelTest::get_test_array_types_and_sizes( int test_case_idx,
                                                   vector<vector<Size> >& sizes,
                                                   vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_DerivBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int max_d = _aperture_size > 0 ? 2 : 1;
    origin = cvtest::randInt(rng) % 2;
    dx = cvtest::randInt(rng) % (max_d + 1);
    dy = cvtest::randInt(rng) % (max_d + 1 - dx);
    if( dx == 0 && dy == 0 )
        dx = 1;
    if( cvtest::randInt(rng) % 2 )
    {
        int t;
        CV_SWAP( dx, dy, t );
    }

    if( _aperture_size < 0 )
        aperture_size = cvSize(3, 3);
    else if( _aperture_size == 1 )
    {
        if( dx == 0 )
            aperture_size = cvSize(1, 3);
        else if( dy == 0 )
            aperture_size = cvSize(3, 1);
        else
        {
            _aperture_size = 3;
            aperture_size = cvSize(3, 3);
        }
    }
    else
        aperture_size = cvSize(_aperture_size, _aperture_size);

    sizes[INPUT][1] = aperture_size;
    anchor.x = aperture_size.width / 2;
    anchor.y = aperture_size.height / 2;
}


void CV_SobelTest::run_func()
{
    cvSobel( test_array[inplace ? OUTPUT : INPUT][0],
             test_array[OUTPUT][0], dx, dy, _aperture_size );
    /*cv::Sobel( test_mat[inplace ? OUTPUT : INPUT][0],
               test_mat[OUTPUT][0], test_mat[OUTPUT][0].depth(),
               dx, dy, _aperture_size, 1, 0, border );*/
}


void CV_SobelTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat kernel = cvtest::calcSobelKernel2D( dx, dy, _aperture_size, 0 );
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].depth(),
                      kernel, anchor, 0, BORDER_REPLICATE);
}


/////////////// laplace ///////////////

class CV_LaplaceTest : public CV_DerivBaseTest
{
public:
    CV_LaplaceTest();

protected:
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
};


CV_LaplaceTest::CV_LaplaceTest()
{
}


void CV_LaplaceTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_DerivBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    if( _aperture_size <= 1 )
    {
        if( _aperture_size < 0 )
            _aperture_size = 1;
        aperture_size = cvSize(3, 3);
    }
    else
        aperture_size = cvSize(_aperture_size, _aperture_size);

    sizes[INPUT][1] = aperture_size;
    anchor.x = aperture_size.width / 2;
    anchor.y = aperture_size.height / 2;
}


void CV_LaplaceTest::run_func()
{
    cvLaplace( test_array[inplace ? OUTPUT : INPUT][0],
               test_array[OUTPUT][0], _aperture_size );
}


int CV_LaplaceTest::prepare_test_case( int test_case_idx )
{
    int code = CV_DerivBaseTest::prepare_test_case( test_case_idx );
    return _aperture_size < 0 ? 0 : code;
}


void CV_LaplaceTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat kernel = cvtest::calcLaplaceKernel2D( _aperture_size );
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].depth(),
                      kernel, anchor, 0, BORDER_REPLICATE );
}


////////////////////////////////////////////////////////////

class CV_SmoothBaseTest : public CV_FilterBaseTest
{
public:
    CV_SmoothBaseTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    const char* smooth_type;
};


CV_SmoothBaseTest::CV_SmoothBaseTest() : CV_FilterBaseTest( true )
{
    smooth_type = "";
}


void CV_SmoothBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_FilterBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng) % 2;
    int cn = CV_MAT_CN(types[INPUT][0]);
    depth = depth == 0 ? CV_8U : CV_32F;
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    anchor.x = cvtest::randInt(rng)%(max_aperture_size/2+1);
    anchor.y = cvtest::randInt(rng)%(max_aperture_size/2+1);
    aperture_size.width = anchor.x*2 + 1;
    aperture_size.height = anchor.y*2 + 1;
    sizes[INPUT][1] = aperture_size;
}


double CV_SmoothBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 1 : 1e-5;
}


/////////////// blur ///////////////

class CV_BlurTest : public CV_SmoothBaseTest
{
public:
    CV_BlurTest();

protected:
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    bool normalize;
};


CV_BlurTest::CV_BlurTest()
{
}


void CV_BlurTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_SmoothBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    normalize = cvtest::randInt(rng) % 2 != 0;
    if( !normalize )
    {
        int depth = CV_MAT_DEPTH(types[INPUT][0]);
        types[INPUT][0] = CV_MAKETYPE(depth, 1);
        types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth==CV_8U?CV_16S:CV_32F,1);
    }
}


void CV_BlurTest::run_func()
{
    cvSmooth( inplace ? test_array[OUTPUT][0] : test_array[INPUT][0],
              test_array[OUTPUT][0], normalize ? CV_BLUR : CV_BLUR_NO_SCALE,
              aperture_size.width, aperture_size.height );
}


int CV_BlurTest::prepare_test_case( int test_case_idx )
{
    int code = CV_SmoothBaseTest::prepare_test_case( test_case_idx );
    return code > 0 && !normalize && test_mat[INPUT][0].channels() > 1 ? 0 : code;
}


void CV_BlurTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat kernel(aperture_size, CV_64F);
    kernel.setTo(Scalar::all(normalize ? 1./(aperture_size.width*aperture_size.height) : 1.));
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].depth(),
                      kernel, anchor, 0, BORDER_REPLICATE );
}


/////////////// gaussian ///////////////

class CV_GaussianBlurTest : public CV_SmoothBaseTest
{
public:
    CV_GaussianBlurTest();

protected:
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ );
    double sigma;
    int param1, param2;
};


CV_GaussianBlurTest::CV_GaussianBlurTest() : CV_SmoothBaseTest()
{
    sigma = 0.;
    smooth_type = "Gaussian";
}


double CV_GaussianBlurTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 8 : 1e-5;
}


void CV_GaussianBlurTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int kernel_case = cvtest::randInt(rng) % 2;
    CV_SmoothBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    anchor = cvPoint(aperture_size.width/2,aperture_size.height/2);

    sigma = exp(cvtest::randReal(rng)*5-2);
    param1 = aperture_size.width;
    param2 = aperture_size.height;

    if( kernel_case == 0 )
        sigma = 0.;
}

void CV_GaussianBlurTest::run_func()
{
    cvSmooth( test_array[inplace ? OUTPUT : INPUT][0],
              test_array[OUTPUT][0], CV_GAUSSIAN,
              param1, param2, sigma, sigma );
}


// !!! Copied from cvSmooth, if the code is changed in cvSmooth,
// make sure to update this one too.
#define SMALL_GAUSSIAN_SIZE 7
static void
calcGaussianKernel( int n, double sigma, vector<float>& kernel )
{
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125}
    };

    kernel.resize(n);
    if( n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 )
    {
        assert( n%2 == 1 );
        memcpy( &kernel[0], small_gaussian_tab[n>>1], n*sizeof(kernel[0]));
    }
    else
    {
        double sigmaX = sigma > 0 ? sigma : (n/2 - 1)*0.3 + 0.8;
        double scale2X = -0.5/(sigmaX*sigmaX);
        double sum = 1.;
        int i;
        sum = kernel[n/2] = 1.f;

        for( i = 1; i <= n/2; i++ )
        {
            kernel[n/2+i] = kernel[n/2-i] = (float)exp(scale2X*i*i);
            sum += kernel[n/2+i]*2;
        }

        sum = 1./sum;
        for( i = 0; i <= n/2; i++ )
            kernel[n/2+i] = kernel[n/2-i] = (float)(kernel[n/2+i]*sum);
    }
}


static Mat calcGaussianKernel2D( Size ksize, double sigma )
{
    vector<float> kx, ky;
    Mat kernel(ksize, CV_32F);

    calcGaussianKernel( kernel.cols, sigma, kx );
    calcGaussianKernel( kernel.rows, sigma, ky );

    for( int i = 0; i < kernel.rows; i++ )
        for( int j = 0; j < kernel.cols; j++ )
            kernel.at<float>(i, j) = kx[j]*ky[i];
    return kernel;
}


void CV_GaussianBlurTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat kernel = calcGaussianKernel2D( aperture_size, sigma );
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].depth(),
                      kernel, anchor, 0, border & ~BORDER_ISOLATED );
}


/////////////// median ///////////////

class CV_MedianBlurTest : public CV_SmoothBaseTest
{
public:
    CV_MedianBlurTest();

protected:
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
};


CV_MedianBlurTest::CV_MedianBlurTest()
{
    smooth_type = "Median";
}


void CV_MedianBlurTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_SmoothBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = CV_8U;
    int cn = CV_MAT_CN(types[INPUT][0]);
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    types[INPUT][1] = CV_MAKETYPE(depth,1);

    aperture_size.height = aperture_size.width;
    anchor.x = anchor.y = aperture_size.width / 2;
    sizes[INPUT][1] = cvSize(aperture_size.width,aperture_size.height);

    sizes[OUTPUT][0] = sizes[INPUT][0];
    sizes[REF_OUTPUT][0] = sizes[INPUT][0];

    inplace = false;
    border = BORDER_REPLICATE | BORDER_ISOLATED;
}


double CV_MedianBlurTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_MedianBlurTest::run_func()
{
    cvSmooth( test_array[INPUT][0], test_array[OUTPUT][0],
              CV_MEDIAN, aperture_size.width );
}


struct median_pair
{
    int col;
    int val;
    median_pair() {};
    median_pair( int _col, int _val ) : col(_col), val(_val) {};
};


static void test_medianFilter( const Mat& src, Mat& dst, int m )
{
    int i, j, k, l, m2 = m*m, n;
    vector<int> col_buf(m+1);
    vector<median_pair> _buf0(m*m+1), _buf1(m*m+1);
    median_pair *buf0 = &_buf0[0], *buf1 = &_buf1[0];
    int step = (int)(src.step/src.elemSize());

    assert( src.rows == dst.rows + m - 1 && src.cols == dst.cols + m - 1 &&
            src.type() == dst.type() && src.type() == CV_8UC1 );

    for( i = 0; i < dst.rows; i++ )
    {
        uchar* dst1 = dst.ptr<uchar>(i);
        for( k = 0; k < m; k++ )
        {
            const uchar* src1 = src.ptr<uchar>(i+k);
            for( j = 0; j < m-1; j++ )
                *buf0++ = median_pair(j, src1[j]);
        }

        n = m2 - m;
        buf0 -= n;
        for( k = n-1; k > 0; k-- )
        {
            int f = 0;
            for( j = 0; j < k; j++ )
            {
                if( buf0[j].val > buf0[j+1].val )
                {
                    median_pair t;
                    CV_SWAP( buf0[j], buf0[j+1], t );
                    f = 1;
                }
            }
            if( !f )
                break;
        }

        for( j = 0; j < dst.cols; j++ )
        {
            int ins_col = j + m - 1;
            int del_col = j - 1;
            const uchar* src1 = src.ptr<uchar>(i) + ins_col;
            for( k = 0; k < m; k++, src1 += step )
            {
                col_buf[k] = src1[0];
                for( l = k-1; l >= 0; l-- )
                {
                    int t;
                    if( col_buf[l] < col_buf[l+1] )
                        break;
                    CV_SWAP( col_buf[l], col_buf[l+1], t );
                }
            }

            col_buf[m] = INT_MAX;

            for( k = 0, l = 0; k < n; )
            {
                if( buf0[k].col == del_col )
                    k++;
                else if( buf0[k].val < col_buf[l] )
                    *buf1++ = buf0[k++];
                else
                {
                    assert( col_buf[l] < INT_MAX );
                    *buf1++ = median_pair(ins_col,col_buf[l++]);
                }
            }

            for( ; l < m; l++ )
                *buf1++ = median_pair(ins_col,col_buf[l]);

            if( del_col < 0 )
                n += m;
            buf1 -= n;
            assert( n == m2 );
            dst1[j] = (uchar)buf1[n/2].val;
            median_pair* tbuf;
            CV_SWAP( buf0, buf1, tbuf );
        }
    }
}


void CV_MedianBlurTest::prepare_to_validation( int /*test_case_idx*/ )
{
    // CV_SmoothBaseTest::prepare_to_validation( test_case_idx );
    const Mat& src0 = test_mat[INPUT][0];
    Mat& dst0 = test_mat[REF_OUTPUT][0];
    int i, cn = src0.channels();
    int m = aperture_size.width;
    Mat src(src0.rows + m - 1, src0.cols + m - 1, src0.depth());
    Mat dst;
    if( cn == 1 )
        dst = dst0;
    else
        dst.create(src0.size(), src0.depth());

    for( i = 0; i < cn; i++ )
    {
        Mat ptr = src0;
        if( cn > 1 )
        {
            cvtest::extract( src0, dst, i );
            ptr = dst;
        }
        cvtest::copyMakeBorder( ptr, src, m/2, m/2, m/2, m/2, border & ~BORDER_ISOLATED );
        test_medianFilter( src, dst, m );
        if( cn > 1 )
            cvtest::insert( dst, dst0, i );
    }
}


/////////////// pyramid tests ///////////////

class CV_PyramidBaseTest : public CV_FilterBaseTest
{
public:
    CV_PyramidBaseTest( bool downsample );

protected:
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    bool downsample;
    Mat kernel;
};


CV_PyramidBaseTest::CV_PyramidBaseTest( bool _downsample ) : CV_FilterBaseTest(true)
{
    static float kdata[] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    downsample = _downsample;
    Mat kernel1d(1, 5, CV_32F, kdata);
    kernel = (kernel1d.t()*kernel1d)*((downsample ? 1 : 4)/256.);
}


double CV_PyramidBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth < CV_32F ? 1 : 1e-5;
}


void CV_PyramidBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                         vector<vector<Size> >& sizes,
                                                         vector<vector<int> >& types )
{
    const int channels[] = {1, 3, 4};
    const int depthes[] = {CV_8U, CV_16S, CV_16U, CV_32F};

    RNG& rng = ts->get_rng();
    CvSize sz;
    CV_FilterBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    int depth = depthes[cvtest::randInt(rng) % (sizeof(depthes)/sizeof(depthes[0]))];
    int cn = channels[cvtest::randInt(rng) % (sizeof(channels)/sizeof(channels[0]))];

    aperture_size = cvSize(5,5);
    anchor = cvPoint(aperture_size.width/2, aperture_size.height/2);

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);

    sz.width = MAX( sizes[INPUT][0].width/2, 1 );
    sz.height = MAX( sizes[INPUT][0].height/2, 1 );

    if( downsample )
    {
        sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sz;
        sz.width *= 2;
        sz.height *= 2;
        sizes[INPUT][0] = sz;
    }
    else
    {
        sizes[INPUT][0] = sz;
        sz.width *= 2;
        sz.height *= 2;
        sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sz;
    }

    sizes[INPUT][1] = aperture_size;
    inplace = false;
}


/////// pyrdown ////////

class CV_PyramidDownTest : public CV_PyramidBaseTest
{
public:
    CV_PyramidDownTest();

protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_PyramidDownTest::CV_PyramidDownTest() : CV_PyramidBaseTest( true )
{
}


void CV_PyramidDownTest::run_func()
{
    cvPyrDown( test_array[INPUT][0], test_array[OUTPUT][0], CV_GAUSSIAN_5x5 );
}


void CV_PyramidDownTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0], &dst = test_mat[REF_OUTPUT][0];
    Mat temp;
    cvtest::filter2D(src, temp, src.depth(),
                     kernel, Point(kernel.cols/2, kernel.rows/2),
                     0, BORDER_REFLECT_101);
    
    size_t elem_size = temp.elemSize();
    size_t ncols = dst.cols*elem_size;
    
    for( int i = 0; i < dst.rows; i++ )
    {
        const uchar* src_row = temp.ptr(i*2);
        uchar* dst_row = dst.ptr(i);
        
        for( size_t j = 0; j < ncols; j += elem_size )
        {
            for( size_t k = 0; k < elem_size; k++ )
                dst_row[j+k] = src_row[j*2+k];
        }
    }
}


/////// pyrup ////////

class CV_PyramidUpTest : public CV_PyramidBaseTest
{
public:
    CV_PyramidUpTest();

protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_PyramidUpTest::CV_PyramidUpTest() : CV_PyramidBaseTest( false )
{
}


void CV_PyramidUpTest::run_func()
{
    cvPyrUp( test_array[INPUT][0], test_array[OUTPUT][0], CV_GAUSSIAN_5x5 );
}


void CV_PyramidUpTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0], &dst = test_mat[REF_OUTPUT][0];
    Mat temp(dst.size(), dst.type());
    
    size_t elem_size = src.elemSize();
    size_t ncols = src.cols*elem_size;
    
    for( int i = 0; i < src.rows; i++ )
    {
        const uchar* src_row = src.ptr(i);
        uchar* dst_row = temp.ptr(i*2);
        
        if( i*2 + 1 < temp.rows )
            memset( temp.ptr(i*2+1), 0, temp.cols*elem_size );
        for( size_t j = 0; j < ncols; j += elem_size )
        {
            for( size_t k = 0; k < elem_size; k++ )
            {
                dst_row[j*2+k] = src_row[j+k];
                dst_row[j*2+k+elem_size] = 0;
            }
        }
    }
    
    cvtest::filter2D(temp, dst, dst.depth(),
                     kernel, Point(kernel.cols/2, kernel.rows/2),
                     0, BORDER_REFLECT_101);
}


//////////////////////// feature selection //////////////////////////

class CV_FeatureSelBaseTest : public cvtest::ArrayTest
{
public:
    CV_FeatureSelBaseTest( int width_factor );

protected:
    int read_params( CvFileStorage* fs );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int aperture_size, block_size;
    int max_aperture_size;
    int max_block_size;
    int width_factor;
};


CV_FeatureSelBaseTest::CV_FeatureSelBaseTest( int _width_factor )
{
    max_aperture_size = 7;
    max_block_size = 21;
    // 1 input, 1 output, temp arrays are allocated in the reference functions
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    element_wise_relative_error = false;
    width_factor = _width_factor;
}


int CV_FeatureSelBaseTest::read_params( CvFileStorage* fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    max_aperture_size = cvReadInt( find_param( fs, "max_aperture_size" ), max_aperture_size );
    max_aperture_size = cvtest::clipInt( max_aperture_size, 1, 9 );
    max_block_size = cvReadInt( find_param( fs, "max_block_size" ), max_block_size );
    max_block_size = cvtest::clipInt( max_aperture_size, 1, 100 );

    return code;
}


double CV_FeatureSelBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 3e-2 : depth == CV_32F ? 1e-3 : 1e-10;
}


void CV_FeatureSelBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT && CV_MAT_DEPTH(type) == CV_32F )
    {
        low = Scalar::all(-10.);
        high = Scalar::all(10.);
    }
}


void CV_FeatureSelBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int depth = cvtest::randInt(rng) % 2, asz;

    depth = depth == 0 ? CV_8U : CV_32F;
    types[INPUT][0] = depth;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_32FC1;

    aperture_size = (cvtest::randInt(rng) % (max_aperture_size+2) - 1) | 1;
    if( aperture_size == 1 )
        aperture_size = 3;
    if( depth == CV_8U )
        aperture_size = MIN( aperture_size, 5 );
    block_size = (cvtest::randInt(rng) % max_block_size + 1) | 1;
    if( block_size <= 3 )
        block_size = 3;
    asz = aperture_size > 0 ? aperture_size : 3;

    sizes[INPUT][0].width = MAX( sizes[INPUT][0].width, asz + block_size );
    sizes[INPUT][0].height = MAX( sizes[INPUT][0].height, asz + block_size );
    sizes[OUTPUT][0].height = sizes[REF_OUTPUT][0].height = sizes[INPUT][0].height;
    sizes[OUTPUT][0].width = sizes[REF_OUTPUT][0].width = sizes[INPUT][0].width*width_factor;
}


static void
test_cornerEigenValsVecs( const Mat& src, Mat& eigenv, Mat& ocv_eigenv,
                          int block_size, int _aperture_size, int mode )
{
    int i, j;
    int aperture_size = _aperture_size < 0 ? 3 : _aperture_size;
    Point anchor( aperture_size/2, aperture_size/2 );

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );
    CV_Assert( eigenv.type() == CV_32FC1 );
    CV_Assert( src.rows == eigenv.rows &&
              ((mode > 0 && src.cols == eigenv.cols) ||
              (mode == 0 && src.cols*6 == eigenv.cols)) );

    int type = src.type();
    int ftype = CV_32FC1;
    double kernel_scale = type != ftype ? 1./255 : 1;

    Mat dx2, dy2, dxdy(src.size(), CV_32F), kernel;

    kernel = cvtest::calcSobelKernel2D( 1, 0, _aperture_size );
    cvtest::filter2D( src, dx2, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE );
    kernel = cvtest::calcSobelKernel2D( 0, 1, _aperture_size );
    cvtest::filter2D( src, dy2, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE );

    double denom = (1 << (aperture_size-1))*block_size;
    denom = denom * denom;
    if( _aperture_size < 0 )
        denom *= 4;
    denom = 1./denom;

    for( i = 0; i < src.rows; i++ )
    {
        float* dxdyp = dxdy.ptr<float>(i);
        float* dx2p = dx2.ptr<float>(i);
        float* dy2p = dy2.ptr<float>(i);

        for( j = 0; j < src.cols; j++ )
        {
            double xval = dx2p[j], yval = dy2p[j];
            dxdyp[j] = (float)(xval*yval*denom);
            dx2p[j] = (float)(xval*xval*denom);
            dy2p[j] = (float)(yval*yval*denom);
        }
    }

    kernel = Mat::ones(block_size, block_size, CV_32F);
    anchor = Point(block_size/2, block_size/2);

    cvtest::filter2D( dx2, dx2, ftype, kernel, anchor, 0, BORDER_REPLICATE );
    cvtest::filter2D( dy2, dy2, ftype, kernel, anchor, 0, BORDER_REPLICATE );
    cvtest::filter2D( dxdy, dxdy, ftype, kernel, anchor, 0, BORDER_REPLICATE );

    if( mode == 0 )
    {
        for( i = 0; i < src.rows; i++ )
        {
            float* eigenvp = eigenv.ptr<float>(i);
            float* ocv_eigenvp = ocv_eigenv.ptr<float>(i);
            const float* dxdyp = dxdy.ptr<float>(i);
            const float* dx2p = dx2.ptr<float>(i);
            const float* dy2p = dy2.ptr<float>(i);

            for( j = 0; j < src.cols; j++ )
            {
                double a = dx2p[j], b = dxdyp[j], c = dy2p[j];
                double d = sqrt((a-c)*(a-c) + 4*b*b);
                double l1 = 0.5*(a + c + d);
                double l2 = 0.5*(a + c - d);
                double x1, y1, x2, y2, s;

                if( fabs(a - l1) + fabs(b) >= 1e-3 )
                    x1 = b, y1 = l1 - a;
                else
                    x1 = l1 - c, y1 = b;
                s = 1./(sqrt(x1*x1+y1*y1)+DBL_EPSILON);
                x1 *= s; y1 *= s;

                if( fabs(a - l2) + fabs(b) >= 1e-3 )
                    x2 = b, y2 = l2 - a;
                else
                    x2 = l2 - c, y2 = b;
                s = 1./(sqrt(x2*x2+y2*y2)+DBL_EPSILON);
                x2 *= s; y2 *= s;

                /* the orientation of eigen vectors might be inversed relative to OpenCV function,
                   which is normal */
                if( (fabs(x1) >= fabs(y1) && ocv_eigenvp[j*6+2]*x1 < 0) ||
                    (fabs(x1) < fabs(y1) && ocv_eigenvp[j*6+3]*y1 < 0) )
                    x1 = -x1, y1 = -y1;

                if( (fabs(x2) >= fabs(y2) && ocv_eigenvp[j*6+4]*x2 < 0) ||
                    (fabs(x2) < fabs(y2) && ocv_eigenvp[j*6+5]*y2 < 0) )
                    x2 = -x2, y2 = -y2;

                eigenvp[j*6] = (float)l1;
                eigenvp[j*6+1] = (float)l2;
                eigenvp[j*6+2] = (float)x1;
                eigenvp[j*6+3] = (float)y1;
                eigenvp[j*6+4] = (float)x2;
                eigenvp[j*6+5] = (float)y2;
            }
        }
    }
    else if( mode == 1 )
    {
        for( i = 0; i < src.rows; i++ )
        {
            float* eigenvp = eigenv.ptr<float>(i);
            const float* dxdyp = dxdy.ptr<float>(i);
            const float* dx2p = dx2.ptr<float>(i);
            const float* dy2p = dy2.ptr<float>(i);

            for( j = 0; j < src.cols; j++ )
            {
                double a = dx2p[j], b = dxdyp[j], c = dy2p[j];
                double d = sqrt((a-c)*(a-c) + 4*b*b);
                eigenvp[j] = (float)(0.5*(a + c - d));
            }
        }
    }
}


// min eigenval
class CV_MinEigenValTest : public CV_FeatureSelBaseTest
{
public:
    CV_MinEigenValTest();

protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_MinEigenValTest::CV_MinEigenValTest() : CV_FeatureSelBaseTest( 1 )
{
}


void CV_MinEigenValTest::run_func()
{
    cvCornerMinEigenVal( test_array[INPUT][0], test_array[OUTPUT][0],
                         block_size, aperture_size );
}


void CV_MinEigenValTest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_cornerEigenValsVecs( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                    test_mat[OUTPUT][0], block_size, aperture_size, 1 );
}


// eigenval's & vec's
class CV_EigenValVecTest : public CV_FeatureSelBaseTest
{
public:
    CV_EigenValVecTest();

protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_EigenValVecTest::CV_EigenValVecTest() : CV_FeatureSelBaseTest( 6 )
{
}


void CV_EigenValVecTest::run_func()
{
    cvCornerEigenValsAndVecs( test_array[INPUT][0], test_array[OUTPUT][0],
                              block_size, aperture_size );
}


void CV_EigenValVecTest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_cornerEigenValsVecs( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                    test_mat[OUTPUT][0], block_size, aperture_size, 0 );
}


// precornerdetect
class CV_PreCornerDetectTest : public CV_FeatureSelBaseTest
{
public:
    CV_PreCornerDetectTest();

protected:
    void run_func();
    void prepare_to_validation( int );
    int prepare_test_case( int );
};


CV_PreCornerDetectTest::CV_PreCornerDetectTest() : CV_FeatureSelBaseTest( 1 )
{
}


void CV_PreCornerDetectTest::run_func()
{
    cvPreCornerDetect( test_array[INPUT][0], test_array[OUTPUT][0], aperture_size );
}


int CV_PreCornerDetectTest::prepare_test_case( int test_case_idx )
{
    int code = CV_FeatureSelBaseTest::prepare_test_case( test_case_idx );
    if( aperture_size < 0 )
        aperture_size = 3;
    return code;
}


void CV_PreCornerDetectTest::prepare_to_validation( int /*test_case_idx*/ )
{
    /*cvTsCornerEigenValsVecs( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                             block_size, aperture_size, 0 );*/
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];

    int type = src.type(), ftype = CV_32FC1;
    Point anchor(aperture_size/2, aperture_size/2);

    double kernel_scale = type != ftype ? 1./255 : 1.;

    Mat dx, dy, d2x, d2y, dxy, kernel;
    
    kernel = cvtest::calcSobelKernel2D(1, 0, aperture_size);
    cvtest::filter2D(src, dx, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE);
    kernel = cvtest::calcSobelKernel2D(2, 0, aperture_size);
    cvtest::filter2D(src, d2x, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE);
    kernel = cvtest::calcSobelKernel2D(0, 1, aperture_size);
    cvtest::filter2D(src, dy, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE);
    kernel = cvtest::calcSobelKernel2D(0, 2, aperture_size);
    cvtest::filter2D(src, d2y, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE);
    kernel = cvtest::calcSobelKernel2D(1, 1, aperture_size);
    cvtest::filter2D(src, dxy, ftype, kernel*kernel_scale, anchor, 0, BORDER_REPLICATE);

    double denom = 1 << (aperture_size-1);
    denom = denom * denom * denom;
    denom = 1./denom;

    for( int i = 0; i < src.rows; i++ )
    {
        const float* _dx = dx.ptr<float>(i);
        const float* _dy = dy.ptr<float>(i);
        const float* _d2x = d2x.ptr<float>(i);
        const float* _d2y = d2y.ptr<float>(i);
        const float* _dxy = dxy.ptr<float>(i);
        float* corner = dst.ptr<float>(i);

        for( int j = 0; j < src.cols; j++ )
        {
            double x = _dx[j];
            double y = _dy[j];

            corner[j] = (float)(denom*(x*x*_d2y[j] + y*y*_d2x[j] - 2*x*y*_dxy[j]));
        }
    }
}


///////// integral /////////

class CV_IntegralTest : public cvtest::ArrayTest
{
public:
    CV_IntegralTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int prepare_test_case( int test_case_idx );
};


CV_IntegralTest::CV_IntegralTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    element_wise_relative_error = true;
}


void CV_IntegralTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    int depth = CV_MAT_DEPTH(type);
    if( depth == CV_32F )
    {
        low = Scalar::all(-10.);
        high = Scalar::all(10.);
    }
}


void CV_IntegralTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 2, sum_depth;
    int cn = cvtest::randInt(rng) % 3 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    Size sum_size;

    depth = depth == 0 ? CV_8U : CV_32F;
    cn += cn == 2;
    int b = (cvtest::randInt(rng) & 1) != 0;
    sum_depth = depth == CV_8U && b ? CV_32S : b ? CV_32F : CV_64F;

    types[INPUT][0] = CV_MAKETYPE(depth,cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] =
        types[OUTPUT][2] = types[REF_OUTPUT][2] = CV_MAKETYPE(sum_depth, cn);
    types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(CV_64F, cn);

    sum_size.width = sizes[INPUT][0].width + 1;
    sum_size.height = sizes[INPUT][0].height + 1;

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sum_size;
    sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] =
        sizes[OUTPUT][2] = sizes[REF_OUTPUT][2] = Size(0,0);

    if( cvtest::randInt(rng) % 3 > 0 )
    {
        sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] = sum_size;
        if( cvtest::randInt(rng) % 2 > 0 )
            sizes[REF_OUTPUT][2] = sizes[OUTPUT][2] = sum_size;
    }
}


double CV_IntegralTest::get_success_error_level( int, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_32S ? 0 : depth == CV_64F ? FLT_EPSILON : 5e-3;
}


int CV_IntegralTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    return code > 0 && ((test_array[OUTPUT][2] && test_mat[OUTPUT][2].channels() > 1) ||
        test_mat[OUTPUT][0].depth() < test_mat[INPUT][0].depth()) ? 0 : code;
}


void CV_IntegralTest::run_func()
{
    cvIntegral( test_array[INPUT][0], test_array[OUTPUT][0],
                test_array[OUTPUT][1], test_array[OUTPUT][2] );
}


static void test_integral( const Mat& img, Mat* sum, Mat* sqsum, Mat* tilted )
{
    CV_Assert( img.depth() == CV_32F );
    
    sum->create(img.rows+1, img.cols+1, CV_64F);
    if( sqsum )
        sqsum->create(img.rows+1, img.cols+1, CV_64F);
    if( tilted )
        tilted->create(img.rows+1, img.cols+1, CV_64F);
    
    const float* data = img.ptr<float>();
    double* sdata = sum->ptr<double>();
    double* sqdata = sqsum ? sqsum->ptr<double>() : 0;
    double* tdata = tilted ? tilted->ptr<double>() : 0;
    int step = (int)(img.step/sizeof(data[0]));
    int sstep = (int)(sum->step/sizeof(sdata[0]));
    int sqstep = sqsum ? (int)(sqsum->step/sizeof(sqdata[0])) : 0;
    int tstep = tilted ? (int)(tilted->step/sizeof(tdata[0])) : 0;
    Size size = img.size();

    memset( sdata, 0, (size.width+1)*sizeof(sdata[0]) );
    if( sqsum )
        memset( sqdata, 0, (size.width+1)*sizeof(sqdata[0]) );
    if( tilted )
        memset( tdata, 0, (size.width+1)*sizeof(tdata[0]) );

    for( ; size.height--; data += step )
    {
        double s = 0, sq = 0;
        int x;
        sdata += sstep;
        sqdata += sqstep;
        tdata += tstep;

        for( x = 0; x <= size.width; x++ )
        {
            double t = x > 0 ? data[x-1] : 0, ts = t;
            s += t;
            sq += t*t;

            sdata[x] = s + sdata[x - sstep];
            if( sqdata )
                sqdata[x] = sq + sqdata[x - sqstep];

            if( !tdata )
                continue;

            if( x == 0 )
                ts += tdata[-tstep+1];
            else
            {
                ts += tdata[x-tstep-1];
                if( data > img.ptr<float>() )
                {
                    ts += data[x-step-1];
                    if( x < size.width )
                        ts += tdata[x-tstep+1] - tdata[x-tstep*2];
                }
            }

            tdata[x] = ts;
        }
    }
}


void CV_IntegralTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& src = test_mat[INPUT][0];
    int cn = src.channels();

    Mat* sum0 = &test_mat[REF_OUTPUT][0];
    Mat* sqsum0 = test_array[REF_OUTPUT][1] ? &test_mat[REF_OUTPUT][1] : 0;
    Mat* tsum0 = test_array[REF_OUTPUT][2] ? &test_mat[REF_OUTPUT][2] : 0;

    Mat plane, srcf, psum, psqsum, ptsum, psum2, psqsum2, ptsum2;
    if( cn == 1 )
    {
        plane = src;
        psum2 = *sum0;
        psqsum2 = sqsum0 ? *sqsum0 : Mat();
        ptsum2 = tsum0 ? *tsum0 : Mat();
    }
    
    for( int i = 0; i < cn; i++ )
    {
        if( cn > 1 )
            cvtest::extract(src, plane, i);
        plane.convertTo(srcf, CV_32F);
        
        test_integral( srcf, &psum, sqsum0 ? &psqsum : 0, tsum0 ? &ptsum : 0 );
        psum.convertTo(psum2, sum0->depth());
        if( sqsum0 )
            psqsum.convertTo(psqsum2, sqsum0->depth());
        if( tsum0 )
            ptsum.convertTo(ptsum2, tsum0->depth());
        
        if( cn > 1 )
        {
            cvtest::insert(psum2, *sum0, i);
            if( sqsum0 )
                cvtest::insert(psqsum2, *sqsum0, i);
            if( tsum0 )
                cvtest::insert(ptsum2, *tsum0, i);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_Erode, accuracy) { CV_ErodeTest test; test.safe_run(); }
TEST(Imgproc_Dilate, accuracy) { CV_DilateTest test; test.safe_run(); }
TEST(Imgproc_MorphologyEx, accuracy) { CV_MorphExTest test; test.safe_run(); }
TEST(Imgproc_Filter2D, accuracy) { CV_FilterTest test; test.safe_run(); }
TEST(Imgproc_Sobel, accuracy) { CV_SobelTest test; test.safe_run(); }
TEST(Imgproc_Laplace, accuracy) { CV_LaplaceTest test; test.safe_run(); }
TEST(Imgproc_Blur, accuracy) { CV_BlurTest test; test.safe_run(); }
TEST(Imgproc_GaussianBlur, accuracy) { CV_GaussianBlurTest test; test.safe_run(); }
TEST(Imgproc_MedianBlur, accuracy) { CV_MedianBlurTest test; test.safe_run(); }
TEST(Imgproc_PyramidDown, accuracy) { CV_PyramidDownTest test; test.safe_run(); }
TEST(Imgproc_PyramidUp, accuracy) { CV_PyramidUpTest test; test.safe_run(); }
TEST(Imgproc_MinEigenVal, accuracy) { CV_MinEigenValTest test; test.safe_run(); }
TEST(Imgproc_EigenValsVecs, accuracy) { CV_EigenValVecTest test; test.safe_run(); }
TEST(Imgproc_PreCornerDetect, accuracy) { CV_PreCornerDetectTest test; test.safe_run(); }
TEST(Imgproc_Integral, accuracy) { CV_IntegralTest test; test.safe_run(); }

//////////////////////////////////////////////////////////////////////////////////

class CV_FilterSupportedFormatsTest : public cvtest::BaseTest
{
public:
    CV_FilterSupportedFormatsTest() {}
    ~CV_FilterSupportedFormatsTest() {}   
protected:
    void run(int)
    {
        const int depths[][2] = 
        {
            {CV_8U, CV_8U},
            {CV_8U, CV_16U},
            {CV_8U, CV_16S},
            {CV_8U, CV_32F},
            {CV_8U, CV_64F},
            {CV_16U, CV_16U},
            {CV_16U, CV_32F},
            {CV_16U, CV_64F},
            {CV_16S, CV_16S},
            {CV_16S, CV_32F},
            {CV_16S, CV_64F},
            {CV_32F, CV_32F},
            {CV_64F, CV_64F},
            {-1, -1}
        };
        
        int i = 0;
        volatile int fidx = -1;
        try
        {
            // use some "odd" size to do yet another smoke
            // testing of the non-SIMD loop tails
            Size sz(163, 117);
            Mat small_kernel(5, 5, CV_32F), big_kernel(21, 21, CV_32F);
            Mat kernelX(11, 1, CV_32F), kernelY(7, 1, CV_32F);
            Mat symkernelX(11, 1, CV_32F), symkernelY(7, 1, CV_32F);
            randu(small_kernel, -10, 10);
            randu(big_kernel, -1, 1);
            randu(kernelX, -1, 1);
            randu(kernelY, -1, 1);
            flip(kernelX, symkernelX, 0);
            symkernelX += kernelX;
            flip(kernelY, symkernelY, 0);
            symkernelY += kernelY;
            
            Mat elem_ellipse = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
            Mat elem_rect = getStructuringElement(MORPH_RECT, Size(7, 7));
            
            for( i = 0; depths[i][0] >= 0; i++ )
            {
                int sdepth = depths[i][0];
                int ddepth = depths[i][1];
                Mat src(sz, CV_MAKETYPE(sdepth, 5)), dst;
                randu(src, 0, 100);
                // non-separable filtering with a small kernel
                fidx = 0;
                filter2D(src, dst, ddepth, small_kernel);
                fidx++;
                filter2D(src, dst, ddepth, big_kernel);
                fidx++;
                sepFilter2D(src, dst, ddepth, kernelX, kernelY);
                fidx++;
                sepFilter2D(src, dst, ddepth, symkernelX, symkernelY);
                fidx++;
                Sobel(src, dst, ddepth, 2, 0, 5);
                fidx++;
                Scharr(src, dst, ddepth, 0, 1);
                if( sdepth != ddepth )
                    continue;
                fidx++;
                GaussianBlur(src, dst, Size(5, 5), 1.2, 1.2);
                fidx++;
                blur(src, dst, Size(11, 11));
                fidx++;
                morphologyEx(src, dst, MORPH_GRADIENT, elem_ellipse);
                fidx++;
                morphologyEx(src, dst, MORPH_GRADIENT, elem_rect);
            }
        }
        catch(...)
        {
            ts->printf(cvtest::TS::LOG, "Combination of depths %d => %d in %s is not supported (yet it should be)",
                       depths[i][0], depths[i][1],
                       fidx == 0 ? "filter2D (small kernel)" :
                       fidx == 1 ? "filter2D (large kernel)" :
                       fidx == 2 ? "sepFilter2D" :
                       fidx == 3 ? "sepFilter2D (symmetrical/asymmetrical kernel)" :
                       fidx == 4 ? "Sobel" :
                       fidx == 5 ? "Scharr" :
                       fidx == 6 ? "GaussianBlur" :
                       fidx == 7 ? "blur" :
                       fidx == 8 || fidx == 9 ? "morphologyEx" :
                       "unknown???");
                       
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Imgproc_Filtering, supportedFormats) { CV_FilterSupportedFormatsTest test; test.safe_run(); }

