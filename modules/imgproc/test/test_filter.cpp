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

class CV_FilterBaseTest : public cvtest::ArrayTest
{
public:
    CV_FilterBaseTest( bool _fp_kernel );

protected:
    int prepare_test_case( int test_case_idx ) CV_OVERRIDE;
    int read_params( const cv::FileStorage& fs ) CV_OVERRIDE;
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types ) CV_OVERRIDE;
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high ) CV_OVERRIDE;
    Size aperture_size;
    Point anchor;
    int max_aperture_size;
    bool fp_kernel;
    bool inplace;
    int border;

    void dump_test_case(int test_case_idx, std::ostream* out) CV_OVERRIDE
    {
        ArrayTest::dump_test_case(test_case_idx, out);
        *out << "border=" << border << std::endl;
    }

};


CV_FilterBaseTest::CV_FilterBaseTest( bool _fp_kernel ) : fp_kernel(_fp_kernel)
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    max_aperture_size = 13;
    inplace = false;
    aperture_size = Size(0,0);
    anchor = Point(0,0);
    element_wise_relative_error = false;
}


int CV_FilterBaseTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    if( code < 0 )
        return code;

    read( find_param( fs, "max_aperture_size" ), max_aperture_size, max_aperture_size );
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
        else if( CV_MAT_DEPTH(type) == CV_16U )
        {
            low = Scalar::all(0.);
            high = Scalar::all(40000.);
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
        const uchar* src = test_mat[INPUT][1].ptr();
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
    int depth = cvtest::randInt(rng)%4;
    int cn = CV_MAT_CN(types[INPUT][0]);
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : depth == 2 ? CV_16S : CV_32F;
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
    CvMat kernel = cvMat(test_mat[INPUT][1]);
    cvFilter2D( test_array[inplace ? OUTPUT : INPUT][0],
                test_array[OUTPUT][0], &kernel, cvPoint(anchor));
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
    int depth = cvtest::randInt(rng) % 4;
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : depth == 2 ? CV_16S : CV_32F;
    types[INPUT][0] = CV_MAKETYPE(depth,1);
    int sameDepth = cvtest::randInt(rng) % 2;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = sameDepth ? depth : CV_MAKETYPE(depth==CV_8U?CV_16S:CV_32F,1);
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
    cv::Sobel( test_mat[inplace ? OUTPUT : INPUT][0],
               test_mat[OUTPUT][0], test_mat[OUTPUT][0].depth(),
               dx, dy, _aperture_size, 1, 0, border );
}


void CV_SobelTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat kernel = cvtest::calcSobelKernel2D( dx, dy, _aperture_size, 0 );
    cvtest::filter2D( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], test_mat[REF_OUTPUT][0].depth(),
                      kernel, anchor, 0, BORDER_REPLICATE);
}


/////////////// spatialGradient ///////////////

class CV_SpatialGradientTest : public CV_DerivBaseTest
{
public:
    CV_SpatialGradientTest();

protected:
    void prepare_to_validation( int test_case_idx );
    void run_func();
    void get_test_array_types_and_sizes( int test_case_idx,
        vector<vector<Size> >& sizes, vector<vector<int> >& types );
    int ksize;
};

CV_SpatialGradientTest::CV_SpatialGradientTest() {
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    inplace = false;
}


void CV_SpatialGradientTest::get_test_array_types_and_sizes( int test_case_idx,
                                                             vector<vector<Size> >& sizes,
                                                             vector<vector<int> >& types )
{
    CV_DerivBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] = sizes[OUTPUT][0];

    // Inputs are only CV_8UC1 for now
    types[INPUT][0] = CV_8UC1;

    // Outputs are only CV_16SC1 for now
    types[OUTPUT][0] = types[OUTPUT][1] = types[REF_OUTPUT][0]
                     = types[REF_OUTPUT][1] = CV_16SC1;

    ksize = 3;
    border = BORDER_DEFAULT; // TODO: Add BORDER_REPLICATE
}


void CV_SpatialGradientTest::run_func()
{
    spatialGradient( test_mat[INPUT][0], test_mat[OUTPUT][0],
                     test_mat[OUTPUT][1], ksize, border );
}

void CV_SpatialGradientTest::prepare_to_validation( int /*test_case_idx*/ )
{
    int dx, dy;

    dx = 1; dy = 0;
    Sobel( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], CV_16SC1, dx, dy, ksize,
           1, 0, border );

    dx = 0; dy = 1;
    Sobel( test_mat[INPUT][0], test_mat[REF_OUTPUT][1], CV_16SC1, dx, dy, ksize,
           1, 0, border );
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
    cv::Laplacian( test_mat[inplace ? OUTPUT : INPUT][0],
                   test_mat[OUTPUT][0],test_mat[OUTPUT][0].depth(),
                   _aperture_size, 1, 0, cv::BORDER_REPLICATE );
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
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types ) CV_OVERRIDE;
    double get_success_error_level( int test_case_idx, int i, int j ) CV_OVERRIDE;
    const char* smooth_type;

    void dump_test_case(int test_case_idx, std::ostream* out) CV_OVERRIDE
    {
        CV_FilterBaseTest::dump_test_case(test_case_idx, out);
        *out << "smooth_type=" << smooth_type << std::endl;
    }
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
    return depth < CV_32F ? 1 : 1e-5;
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
    int depth = cvtest::randInt(rng) % 4;
    int cn = (cvtest::randInt(rng) % 4) + 1;
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : depth == 2 ? CV_16S : CV_32F;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0] = CV_MAKETYPE(depth, cn);
    normalize = cvtest::randInt(rng) % 2 != 0;
    if( !normalize )
    {
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
    void prepare_to_validation( int test_case_idx ) CV_OVERRIDE;
    void run_func() CV_OVERRIDE;
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types ) CV_OVERRIDE;
    double get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ ) CV_OVERRIDE;
    double sigma;
    int param1, param2;

    void dump_test_case(int test_case_idx, std::ostream* out) CV_OVERRIDE
    {
        CV_SmoothBaseTest::dump_test_case(test_case_idx, out);
        *out << "kernel=(" << param1 << ", " << param2 << ") sigma=" << sigma << std::endl;
    }
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
#define SMALL_GAUSSIAN_SIZE 9
static void
calcGaussianKernel( int n, double sigma, vector<float>& kernel )
{
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125},
        {4.0 / 256, 13.0 / 256, 30.0 / 256, 51.0 / 256, 60.0 / 256, 51.0 / 256, 30.0 / 256, 13.0 / 256, 4.0 / 256}
    };

    kernel.resize(n);
    if( n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 )
    {
        CV_Assert(n%2 == 1);
        memcpy(&kernel[0], small_gaussian_tab[n / 2], n*sizeof(kernel[0]));
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
    median_pair() { }
    median_pair( int _col, int _val ) : col(_col), val(_val) { }
};


static void test_medianFilter( const Mat& src, Mat& dst, int m )
{
    int i, j, k, l, m2 = m*m, n;
    vector<int> col_buf(m+1);
    vector<median_pair> _buf0(m*m+1), _buf1(m*m+1);
    median_pair *buf0 = &_buf0[0], *buf1 = &_buf1[0];
    int step = (int)(src.step/src.elemSize());

    CV_Assert( src.rows == dst.rows + m - 1 && src.cols == dst.cols + m - 1 &&
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
                    CV_Assert( col_buf[l] < INT_MAX );
                    *buf1++ = median_pair(ins_col,col_buf[l++]);
                }
            }

            for( ; l < m; l++ )
                *buf1++ = median_pair(ins_col,col_buf[l]);

            if( del_col < 0 )
                n += m;
            buf1 -= n;
            CV_Assert( n == m2 );
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
    CvSize sz = {0, 0};
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
    int read_params( const FileStorage& fs );
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


int CV_FeatureSelBaseTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    read( find_param( fs, "max_aperture_size" ), max_aperture_size, max_aperture_size );
    max_aperture_size = cvtest::clipInt( max_aperture_size, 1, 9 );
    read( find_param( fs, "max_block_size" ), max_block_size, max_block_size );
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
    int cn = cvtest::randInt(rng) % 4 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    Size sum_size;

    depth = depth == 0 ? CV_8U : CV_32F;
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
TEST(Imgproc_SpatialGradient, accuracy) { CV_SpatialGradientTest test; test.safe_run(); }
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
                cv::filter2D(src, dst, ddepth, small_kernel);
                fidx++;
                cv::filter2D(src, dst, ddepth, big_kernel);
                fidx++;
                cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY);
                fidx++;
                cv::sepFilter2D(src, dst, ddepth, symkernelX, symkernelY);
                fidx++;
                cv::Sobel(src, dst, ddepth, 2, 0, 5);
                fidx++;
                cv::Scharr(src, dst, ddepth, 0, 1);
                if( sdepth != ddepth )
                    continue;
                fidx++;
                cv::GaussianBlur(src, dst, Size(5, 5), 1.2, 1.2);
                fidx++;
                cv::blur(src, dst, Size(11, 11));
                fidx++;
                cv::morphologyEx(src, dst, MORPH_GRADIENT, elem_ellipse);
                fidx++;
                cv::morphologyEx(src, dst, MORPH_GRADIENT, elem_rect);
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

TEST(Imgproc_Blur, borderTypes)
{
    Size kernelSize(3, 3);

    /// ksize > src_roi.size()
    Mat src(3, 3, CV_8UC1, cv::Scalar::all(255)), dst;
    Mat src_roi = src(Rect(1, 1, 1, 1));
    src_roi.setTo(cv::Scalar::all(0));

    // should work like !BORDER_ISOLATED
    blur(src_roi, dst, kernelSize, Point(-1, -1), BORDER_REPLICATE);
    EXPECT_EQ(227, dst.at<uchar>(0, 0));

    // should work like BORDER_ISOLATED
    cv::blur(src_roi, dst, kernelSize, Point(-1, -1), BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_EQ(0, dst.at<uchar>(0, 0));

    /// ksize <= src_roi.size()
    src = Mat(5, 5, CV_8UC1, cv::Scalar(255));
    src_roi = src(Rect(1, 1, 3, 3));
    src_roi.setTo(0);
    src.at<uchar>(2, 2) = 255;

    // should work like !BORDER_ISOLATED
    cv::blur(src_roi, dst, kernelSize, Point(-1, -1), BORDER_REPLICATE);
    Mat expected_dst =
            (Mat_<uchar>(3, 3) << 170, 113, 170, 113, 28, 113, 170, 113, 170);
    EXPECT_EQ(expected_dst.type(), dst.type());
    EXPECT_EQ(expected_dst.size(), dst.size());
    EXPECT_DOUBLE_EQ(0.0, cvtest::norm(expected_dst, dst, NORM_INF));
}

TEST(Imgproc_GaussianBlur, borderTypes)
{
    Size kernelSize(3, 3);

    Mat src_16(16, 16, CV_8UC1, cv::Scalar::all(42)), dst_16;
    Mat src_roi_16 = src_16(Rect(1, 1, 14, 14));
    src_roi_16.setTo(cv::Scalar::all(3));

    cv::GaussianBlur(src_roi_16, dst_16, kernelSize, 0, 0, BORDER_REPLICATE);

    EXPECT_EQ(20, dst_16.at<uchar>(0, 0));

    Mat src(3, 12, CV_8UC1, cv::Scalar::all(42)), dst;
    Mat src_roi = src(Rect(1, 1, 10, 1));
    src_roi.setTo(cv::Scalar::all(2));

    cv::GaussianBlur(src_roi, dst, kernelSize, 0, 0, BORDER_REPLICATE);

    EXPECT_EQ(27, dst.at<uchar>(0, 0));
}

TEST(Imgproc_Morphology, iterated)
{
    RNG& rng = theRNG();
    for( int iter = 0; iter < 30; iter++ )
    {
        int width = rng.uniform(5, 33);
        int height = rng.uniform(5, 33);
        int cn = rng.uniform(1, 5);
        int iterations = rng.uniform(1, 11);
        int op = rng.uniform(0, 2);
        Mat src(height, width, CV_8UC(cn)), dst0, dst1, dst2;

        randu(src, 0, 256);
        if( op == 0 )
            cv::dilate(src, dst0, Mat(), Point(-1,-1), iterations);
        else
            cv::erode(src, dst0, Mat(), Point(-1,-1), iterations);

        for( int i = 0; i < iterations; i++ )
            if( op == 0 )
                cv::dilate(i == 0 ? src : dst1, dst1, Mat(), Point(-1,-1), 1);
            else
                cv::erode(i == 0 ? src : dst1, dst1, Mat(), Point(-1,-1), 1);

        Mat kern = getStructuringElement(MORPH_RECT, Size(3,3));
        if( op == 0 )
            cv::dilate(src, dst2, kern, Point(-1,-1), iterations);
        else
            cv::erode(src, dst2, kern, Point(-1,-1), iterations);
        ASSERT_EQ(0.0, cvtest::norm(dst0, dst1, NORM_INF));
        ASSERT_EQ(0.0, cvtest::norm(dst0, dst2, NORM_INF));
    }
}

TEST(Imgproc_Sobel, borderTypes)
{
    int kernelSize = 3;

    /// ksize > src_roi.size()
    Mat src = (Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9), dst, expected_dst;
    Mat src_roi = src(Rect(1, 1, 1, 1));
    src_roi.setTo(cv::Scalar::all(0));

    // should work like !BORDER_ISOLATED, so the function MUST read values in full matrix
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE);
    EXPECT_EQ(8, dst.at<short>(0, 0));
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REFLECT);
    EXPECT_EQ(8, dst.at<short>(0, 0));

    // should work like BORDER_ISOLATED
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_EQ(0, dst.at<short>(0, 0));
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_EQ(0, dst.at<short>(0, 0));

    /// ksize <= src_roi.size()
    src = Mat(5, 5, CV_8UC1, cv::Scalar(5));
    src_roi = src(Rect(1, 1, 3, 3));
    src_roi.setTo(0);

    // should work like !BORDER_ISOLATED, so the function MUST read values in full matrix
    expected_dst =
        (Mat_<short>(3, 3) << -15, 0, 15, -20, 0, 20, -15, 0, 15);
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE);
    EXPECT_EQ(expected_dst.type(), dst.type());
    EXPECT_EQ(expected_dst.size(), dst.size());
    EXPECT_DOUBLE_EQ(0.0, cvtest::norm(expected_dst, dst, NORM_INF));
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REFLECT);
    EXPECT_EQ(expected_dst.type(), dst.type());
    EXPECT_EQ(expected_dst.size(), dst.size());
    EXPECT_DOUBLE_EQ(0.0, cvtest::norm(expected_dst, dst, NORM_INF));

    // should work like !BORDER_ISOLATED, so the function MUST read values in full matrix
    expected_dst = Mat::zeros(3, 3, CV_16SC1);
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_EQ(expected_dst.type(), dst.type());
    EXPECT_EQ(expected_dst.size(), dst.size());
    EXPECT_DOUBLE_EQ(0.0, cvtest::norm(expected_dst, dst, NORM_INF));
    cv::Sobel(src_roi, dst, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_EQ(expected_dst.type(), dst.type());
    EXPECT_EQ(expected_dst.size(), dst.size());
    EXPECT_DOUBLE_EQ(0.0, cvtest::norm(expected_dst, dst, NORM_INF));
}

TEST(Imgproc_MorphEx, hitmiss_regression_8957)
{
    Mat_<uchar> src(3, 3);
    src << 0, 255, 0,
           0,   0, 0,
           0, 255, 0;

    Mat_<uchar> kernel = src / 255;

    Mat dst;
    cv::morphologyEx(src, dst, MORPH_HITMISS, kernel);

    Mat ref = Mat::zeros(3, 3, CV_8U);
    ref.at<uchar>(1, 1) = 255;

    ASSERT_DOUBLE_EQ(cvtest::norm(dst, ref, NORM_INF), 0.);

    src.at<uchar>(1, 1) = 255;
    ref.at<uchar>(0, 1) = 255;
    ref.at<uchar>(2, 1) = 255;
    cv::morphologyEx(src, dst, MORPH_HITMISS, kernel);
    ASSERT_DOUBLE_EQ(cvtest::norm(dst, ref, NORM_INF), 0.);
}

TEST(Imgproc_MorphEx, hitmiss_zero_kernel)
{
    Mat_<uchar> src(3, 3);
    src << 0, 255, 0,
           0,   0, 0,
           0, 255, 0;

    Mat_<uchar> kernel = Mat_<uchar>::zeros(3, 3);

    Mat dst;
    cv::morphologyEx(src, dst, MORPH_HITMISS, kernel);

    ASSERT_DOUBLE_EQ(cvtest::norm(dst, src, NORM_INF), 0.);
}

TEST(Imgproc_Filter2D, dftFilter2d_regression_10683)
{
    uchar src_[24*24] = {
        0, 40, 0, 0, 255, 0, 0, 78, 131, 0, 196, 0, 255, 0, 0, 0, 0, 255, 70, 0, 255, 0, 0, 0,
        0, 0, 255, 204, 0, 0, 255, 93, 255, 0, 0, 255, 12, 0, 0, 0, 255, 121, 0, 255, 0, 0, 0, 255,
        0, 178, 0, 25, 67, 0, 165, 0, 255, 0, 0, 181, 151, 175, 0, 0, 32, 0, 0, 255, 165, 93, 0, 255,
        255, 255, 0, 0, 255, 126, 0, 0, 0, 0, 133, 29, 9, 0, 220, 255, 0, 142, 255, 255, 255, 0, 255, 0,
        255, 32, 255, 0, 13, 237, 0, 0, 0, 0, 0, 19, 90, 0, 0, 85, 122, 62, 95, 29, 255, 20, 0, 0,
        0, 0, 166, 41, 0, 48, 70, 0, 68, 0, 255, 0, 139, 7, 63, 144, 0, 204, 0, 0, 0, 98, 114, 255,
        105, 0, 0, 0, 0, 255, 91, 0, 73, 0, 255, 0, 0, 0, 255, 198, 21, 0, 0, 0, 255, 43, 153, 128,
        0, 98, 26, 0, 101, 0, 0, 0, 255, 0, 0, 0, 255, 77, 56, 0, 241, 0, 169, 132, 0, 255, 186, 255,
        255, 87, 0, 1, 0, 0, 10, 39, 120, 0, 23, 69, 207, 0, 0, 0, 0, 84, 0, 0, 0, 0, 255, 0,
        255, 0, 0, 136, 255, 77, 247, 0, 67, 0, 15, 255, 0, 143, 0, 243, 255, 0, 0, 238, 255, 0, 255, 8,
        42, 0, 0, 255, 29, 0, 0, 0, 255, 255, 255, 75, 0, 0, 0, 255, 0, 0, 255, 38, 197, 0, 255, 87,
        0, 123, 17, 0, 234, 0, 0, 149, 0, 0, 255, 16, 0, 0, 0, 255, 0, 255, 0, 38, 0, 114, 255, 76,
        0, 0, 8, 0, 255, 0, 0, 0, 220, 0, 11, 255, 0, 0, 55, 98, 0, 0, 0, 255, 0, 175, 255, 110,
        235, 0, 175, 0, 255, 227, 38, 206, 0, 0, 255, 246, 0, 0, 123, 183, 255, 0, 0, 255, 0, 156, 0, 54,
        0, 255, 0, 202, 0, 0, 0, 0, 157, 0, 255, 63, 0, 0, 0, 0, 0, 255, 132, 0, 255, 0, 0, 0,
        0, 0, 0, 255, 0, 0, 128, 126, 0, 243, 46, 7, 0, 211, 108, 166, 0, 0, 162, 227, 0, 204, 0, 51,
        255, 216, 0, 0, 43, 0, 255, 40, 188, 188, 255, 0, 0, 255, 34, 0, 0, 168, 0, 0, 0, 35, 0, 0,
        0, 80, 131, 255, 0, 255, 10, 0, 0, 0, 180, 255, 209, 255, 173, 34, 0, 66, 0, 49, 0, 255, 83, 0,
        0, 204, 0, 91, 0, 0, 0, 205, 84, 0, 0, 0, 92, 255, 91, 0, 126, 0, 185, 145, 0, 0, 9, 0,
        255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 216, 0, 187, 221, 0, 0, 141, 0, 0, 209, 0, 0, 255, 0,
        255, 0, 0, 154, 150, 0, 0, 0, 148, 0, 201, 255, 0, 255, 16, 0, 0, 160, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 255, 198, 255, 147, 131, 0, 255, 202, 0, 0, 0, 0, 255, 0,
        0, 0, 0, 164, 181, 0, 0, 0, 69, 255, 31, 0, 255, 195, 0, 0, 255, 164, 109, 0, 0, 202, 0, 206,
        0, 0, 61, 235, 33, 255, 77, 0, 0, 0, 0, 85, 0, 228, 0, 0, 0, 0, 255, 0, 0, 5, 255, 255
    };
    Mat_<uchar> src(24, 24, src_);
    Mat dst = Mat::zeros(src.size(), src.type());

    int sz = 12, size2 = sz * sz;
    Mat kernel = Mat::ones(sz, sz, CV_32F) / size2;

    uchar expected_[24*24] = {
        83, 83, 77, 80, 76, 76, 76, 75, 71, 67, 72, 71, 73, 70, 80, 83, 86, 84, 89, 88, 88, 96, 99, 98,
        83, 83, 77, 80, 76, 76, 76, 75, 71, 67, 72, 71, 73, 70, 80, 83, 86, 84, 89, 88, 88, 96, 99, 98,
        82, 82, 77, 80, 77, 75, 74, 75, 70, 68, 71, 72, 72, 72, 82, 84, 88, 88, 93, 92, 93, 100, 105, 104,
        76, 76, 72, 77, 73, 74, 73, 74, 69, 68, 71, 71, 73, 72, 82, 81, 86, 87, 92, 91, 92, 98, 103, 102,
        75, 75, 72, 77, 73, 72, 75, 76, 74, 71, 73, 75, 76, 72, 81, 80, 85, 87, 90, 89, 90, 97, 102, 97,
        74, 74, 71, 77, 72, 74, 77, 76, 74, 72, 74, 76, 77, 76, 84, 83, 85, 87, 90, 92, 93, 100, 102, 99,
        72, 72, 69, 71, 68, 73, 73, 73, 70, 69, 74, 72, 75, 75, 81, 82, 85, 87, 90, 94, 96, 103, 102, 101,
        71, 71, 68, 70, 68, 71, 73, 71, 69, 68, 74, 72, 73, 73, 81, 80, 84, 89, 91, 99, 102, 107, 106, 105,
        74, 74, 70, 69, 67, 73, 76, 72, 69, 70, 79, 75, 74, 75, 82, 83, 88, 91, 92, 100, 104, 108, 106, 105,
        75, 75, 71, 70, 67, 75, 76, 71, 67, 68, 75, 72, 72, 75, 81, 83, 87, 89, 89, 97, 102, 107, 103, 103,
        69, 69, 67, 67, 65, 72, 74, 71, 70, 70, 75, 74, 74, 75, 80, 80, 84, 85, 85, 92, 96, 100, 97, 97,
        67, 67, 67, 68, 67, 77, 79, 75, 74, 76, 81, 78, 81, 80, 84, 81, 84, 83, 83, 91, 94, 95, 93, 93,
        73, 73, 71, 73, 70, 80, 82, 79, 80, 83, 85, 82, 82, 82, 87, 84, 88, 87, 84, 91, 93, 94, 93, 92,
        72, 72, 74, 75, 71, 80, 81, 79, 80, 82, 82, 80, 82, 84, 88, 83, 87, 87, 83, 88, 88, 89, 90, 90,
        78, 78, 81, 80, 74, 84, 86, 82, 85, 86, 85, 81, 83, 83, 86, 84, 85, 84, 78, 85, 82, 83, 85, 84,
        81, 81, 84, 81, 75, 86, 90, 85, 89, 91, 89, 84, 86, 87, 90, 87, 89, 85, 78, 84, 79, 80, 81, 81,
        76, 76, 80, 79, 73, 86, 90, 87, 92, 95, 92, 87, 91, 92, 93, 87, 89, 84, 77, 81, 76, 74, 76, 76,
        77, 77, 80, 77, 72, 83, 86, 86, 93, 95, 91, 87, 92, 92, 93, 87, 90, 84, 79, 79, 75, 72, 75, 72,
        80, 80, 81, 79, 72, 82, 86, 86, 95, 97, 89, 87, 89, 89, 91, 85, 88, 84, 79, 80, 73, 69, 74, 73,
        82, 82, 82, 80, 74, 83, 86, 87, 98, 100, 90, 90, 93, 94, 94, 89, 90, 84, 82, 79, 71, 68, 72, 69,
        76, 76, 77, 76, 70, 81, 83, 88, 99, 102, 92, 91, 97, 97, 97, 90, 90, 86, 83, 81, 70, 67, 70, 68,
        75, 75, 76, 74, 69, 79, 84, 88, 102, 106, 95, 94, 99, 98, 98, 90, 89, 86, 82, 79, 67, 62, 65, 62,
        80, 80, 82, 78, 71, 82, 87, 90, 105, 108, 96, 94, 99, 98, 97, 88, 88, 85, 81, 79, 65, 61, 65, 60,
        77, 77, 80, 75, 66, 76, 81, 87, 102, 105, 92, 91, 95, 97, 96, 88, 89, 88, 84, 81, 67, 63, 68, 63
    };
    Mat_<uchar> expected(24, 24, expected_);

    for(int r = 0; r < src.rows / 3; ++r)
    {
        for(int c = 0; c < src.cols / 3; ++c)
        {
            cv::Rect region(c * 3, r * 3, 3, 3);
            Mat roi_i(src, region);
            Mat roi_o(dst, region);
            cv::filter2D(roi_i, roi_o, -1, kernel);
        }
    }

    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 2);
}

TEST(Imgproc_Filter2D, dftFilter2d_regression_13179)
{
    uchar src_[24*24] = {
        0, 40, 0, 0, 255, 0, 0, 78, 131, 0, 196, 0, 255, 0, 0, 0, 0, 255, 70, 0, 255, 0, 0, 0,
        0, 0, 255, 204, 0, 0, 255, 93, 255, 0, 0, 255, 12, 0, 0, 0, 255, 121, 0, 255, 0, 0, 0, 255,
        0, 178, 0, 25, 67, 0, 165, 0, 255, 0, 0, 181, 151, 175, 0, 0, 32, 0, 0, 255, 165, 93, 0, 255,
        255, 255, 0, 0, 255, 126, 0, 0, 0, 0, 133, 29, 9, 0, 220, 255, 0, 142, 255, 255, 255, 0, 255, 0,
        255, 32, 255, 0, 13, 237, 0, 0, 0, 0, 0, 19, 90, 0, 0, 85, 122, 62, 95, 29, 255, 20, 0, 0,
        0, 0, 166, 41, 0, 48, 70, 0, 68, 0, 255, 0, 139, 7, 63, 144, 0, 204, 0, 0, 0, 98, 114, 255,
        105, 0, 0, 0, 0, 255, 91, 0, 73, 0, 255, 0, 0, 0, 255, 198, 21, 0, 0, 0, 255, 43, 153, 128,
        0, 98, 26, 0, 101, 0, 0, 0, 255, 0, 0, 0, 255, 77, 56, 0, 241, 0, 169, 132, 0, 255, 186, 255,
        255, 87, 0, 1, 0, 0, 10, 39, 120, 0, 23, 69, 207, 0, 0, 0, 0, 84, 0, 0, 0, 0, 255, 0,
        255, 0, 0, 136, 255, 77, 247, 0, 67, 0, 15, 255, 0, 143, 0, 243, 255, 0, 0, 238, 255, 0, 255, 8,
        42, 0, 0, 255, 29, 0, 0, 0, 255, 255, 255, 75, 0, 0, 0, 255, 0, 0, 255, 38, 197, 0, 255, 87,
        0, 123, 17, 0, 234, 0, 0, 149, 0, 0, 255, 16, 0, 0, 0, 255, 0, 255, 0, 38, 0, 114, 255, 76,
        0, 0, 8, 0, 255, 0, 0, 0, 220, 0, 11, 255, 0, 0, 55, 98, 0, 0, 0, 255, 0, 175, 255, 110,
        235, 0, 175, 0, 255, 227, 38, 206, 0, 0, 255, 246, 0, 0, 123, 183, 255, 0, 0, 255, 0, 156, 0, 54,
        0, 255, 0, 202, 0, 0, 0, 0, 157, 0, 255, 63, 0, 0, 0, 0, 0, 255, 132, 0, 255, 0, 0, 0,
        0, 0, 0, 255, 0, 0, 128, 126, 0, 243, 46, 7, 0, 211, 108, 166, 0, 0, 162, 227, 0, 204, 0, 51,
        255, 216, 0, 0, 43, 0, 255, 40, 188, 188, 255, 0, 0, 255, 34, 0, 0, 168, 0, 0, 0, 35, 0, 0,
        0, 80, 131, 255, 0, 255, 10, 0, 0, 0, 180, 255, 209, 255, 173, 34, 0, 66, 0, 49, 0, 255, 83, 0,
        0, 204, 0, 91, 0, 0, 0, 205, 84, 0, 0, 0, 92, 255, 91, 0, 126, 0, 185, 145, 0, 0, 9, 0,
        255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 216, 0, 187, 221, 0, 0, 141, 0, 0, 209, 0, 0, 255, 0,
        255, 0, 0, 154, 150, 0, 0, 0, 148, 0, 201, 255, 0, 255, 16, 0, 0, 160, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 255, 198, 255, 147, 131, 0, 255, 202, 0, 0, 0, 0, 255, 0,
        0, 0, 0, 164, 181, 0, 0, 0, 69, 255, 31, 0, 255, 195, 0, 0, 255, 164, 109, 0, 0, 202, 0, 206,
        0, 0, 61, 235, 33, 255, 77, 0, 0, 0, 0, 85, 0, 228, 0, 0, 0, 0, 255, 0, 0, 5, 255, 255
    };
    cv::Mat_<uchar> src(24, 24, src_);

    uchar expected_[16*16] = {
         0,255,  0,  0,255,  0,  0,255,  0,  0,255,255,  0,255,  0,  0,
         0,255,  0,  0,255,  0,  0,255,  0,  0,255,255,  0,255,  0,  0,
         0,255,  0,  0,255,  0,  0,255, 70,  0,255,255,  0,255,  0,  0,
         0,234,138,  0,255,  0,  0,255,  8,  0,255,255,  0,255,  0,  0,
         0,  0,255,  0,255,228,  0,255,255,  0,255,255,  0,255,  0,  5,
         0,  0,255,  0,255,  0,  0,255,  0,  0,255,255,  0,255,  0,  0,
         0,253,  0,  0,255,  0,  0,255,  0,  0,255,255,  0,255,  0,  0,
         0,255,  0,  0,255,  0,  0,255,  0,  0,255, 93,  0,255,  0,255,
         0,255,  0,  0,255,  0,182,255,  0,  0,255,  0,  0,255,  0,  0,
         0,  0,253,  0,228,  0,255,255,  0,  0,255,  0,  0,  0,  0, 75,
         0,  0,255,  0,  0,  0,255,255,  0,255,206,  0,  1,162,  0,255,
         0,  0,255,  0,  0,  0,255,255,  0,255,255,  0,  0,255,  0,255,
         0,  0,255,  0,  0,  0,255,255,  0,255,255,  0,255,255,  0,255,
         0,  0,255,255,  0,  0,255,  0,  0,255,255,  0,255,168,  0,255,
         0,  0,255,255,  0,  0,255, 26,  0,255,255,  0,255,255,  0,255,
         0,  0,255,255,  0,  0,255,  0,  0,255,255,  0,255,255,  0,255,
    };
    cv::Mat_<uchar> expected(16, 16, expected_);

    cv::Mat kernel = cv::getGaborKernel(cv::Size(13, 13), 8, 0, 3, 0.25);

    cv::Mat roi(src, cv::Rect(0, 0, 16, 16));

    cv::Mat filtered(16, 16, roi.type());

    cv::filter2D(roi, filtered, -1, kernel);

    EXPECT_LE(cvtest::norm(filtered, expected, cv::NORM_INF), 2);
}

TEST(Imgproc_MedianBlur, hires_regression_13409)
{
    Mat src(2048, 2048, CV_8UC1), dst_hires, dst_ref;
    randu(src, 0, 256);

    medianBlur(src, dst_hires, 9);
    medianBlur(src(Rect(512, 512, 1024, 1024)), dst_ref, 9);

    ASSERT_EQ(0.0, cvtest::norm(dst_hires(Rect(516, 516, 1016, 1016)), dst_ref(Rect(4, 4, 1016, 1016)), NORM_INF));
}

TEST(Imgproc_Sobel, s16_regression_13506)
{
    Mat src = (Mat_<short>(8, 16) << 127, 138, 130, 102, 118,  97,  76,  84, 124,  90, 146,  63, 130,  87, 212,  85,
                                     164,   3,  51, 124, 151,  89, 154, 117,  36,  88, 116, 117, 180, 112, 147, 124,
                                      63,  50, 115, 103,  83, 148, 106,  79, 213, 106, 135,  53,  79, 106, 122, 112,
                                     218, 107,  81, 126,  78, 138,  85, 142, 151, 108, 104, 158, 155,  81, 112, 178,
                                     184,  96, 187, 148, 150, 112, 138, 162, 222, 146, 128,  49, 124,  46, 165, 104,
                                     119, 164,  77, 144, 186,  98, 106, 148, 155, 157, 160, 151, 156, 149,  43, 122,
                                     106, 155, 120, 132, 159, 115, 126, 188,  44,  79, 164, 201, 153,  97, 139, 133,
                                     133,  98, 111, 165,  66, 106, 131,  85, 176, 156,  67, 108, 142,  91,  74, 137);
    Mat ref = (Mat_<short>(8, 16) <<     0,    0,    0,    0,     0,    0,    0,     0,     0,     0,     0,    0,    0,     0,     0,     0,
                                     -1020, -796, -489, -469,  -247,  317,  760,  1429,  1983,  1384,   254, -459, -899, -1197, -1172, -1058,
                                      2552, 2340, 1617,  591,     9,   96,  722,  1985,  2746,  1916,   676,    9, -635, -1115,  -779,  -380,
                                      3546, 3349, 2838, 2206,  1388,  669,  938,  1880,  2252,  1785,  1083,  606,  180,  -298,  -464,  -418,
                                       816,  966, 1255, 1652,  1619,  924,  535,   288,     5,   601,  1581, 1870, 1520,   625,  -627, -1260,
                                      -782, -610, -395, -267,  -122,  -42, -317, -1378, -2293, -1451,   596, 1870, 1679,   763,   -69,  -394,
                                      -882, -681, -463, -818, -1167, -732, -463, -1042, -1604, -1592, -1047, -334, -104,  -117,   229,   512,
                                         0,    0,    0,    0,     0,    0,    0,     0,     0,     0,     0,    0,    0,     0,     0,     0);
    Mat dst;
    Sobel(src, dst, CV_16S, 0, 1, 5);
    ASSERT_EQ(0.0, cvtest::norm(dst, ref, NORM_INF));
}

TEST(Imgproc_Pyrdown, issue_12961)
{
    Mat src(9, 9, CV_8UC1, Scalar::all(0));
    Mat dst;
    cv::pyrDown(src, dst);
    ASSERT_EQ(0.0, cv::norm(dst));
}


// https://github.com/opencv/opencv/issues/16857
TEST(Imgproc, filter_empty_src_16857)
{
#define CV_TEST_EXPECT_EMPTY_THROW(statement) CV_TEST_EXPECT_EXCEPTION_MESSAGE(statement, ".empty()")

    Mat src, dst, dst2;

    CV_TEST_EXPECT_EMPTY_THROW(bilateralFilter(src, dst, 5, 50, 20));
    CV_TEST_EXPECT_EMPTY_THROW(blur(src, dst, Size(3, 3)));
    CV_TEST_EXPECT_EMPTY_THROW(boxFilter(src, dst, CV_8U, Size(3, 3)));
    CV_TEST_EXPECT_EMPTY_THROW(sqrBoxFilter(src, dst, CV_8U, Size(3, 3)));
    CV_TEST_EXPECT_EMPTY_THROW(medianBlur(src, dst, 3));
    CV_TEST_EXPECT_EMPTY_THROW(GaussianBlur(src, dst, Size(3, 3), 0));
    CV_TEST_EXPECT_EMPTY_THROW(cv::filter2D(src, dst, CV_8U, Mat_<float>::zeros(Size(3, 3))));
    CV_TEST_EXPECT_EMPTY_THROW(sepFilter2D(src, dst, CV_8U, Mat_<float>::zeros(Size(3, 1)), Mat_<float>::zeros(Size(1, 3))));
    CV_TEST_EXPECT_EMPTY_THROW(Sobel(src, dst, CV_8U, 1, 1));
    CV_TEST_EXPECT_EMPTY_THROW(spatialGradient(src, dst, dst2));
    CV_TEST_EXPECT_EMPTY_THROW(Scharr(src, dst, CV_8U, 1, 1));
    CV_TEST_EXPECT_EMPTY_THROW(Laplacian(src, dst, CV_8U));

    CV_TEST_EXPECT_EMPTY_THROW(cv::dilate(src, dst, Mat()));  // cvtest:: by default
    CV_TEST_EXPECT_EMPTY_THROW(cv::erode(src, dst, Mat()));  // cvtest:: by default
    CV_TEST_EXPECT_EMPTY_THROW(morphologyEx(src, dst, MORPH_OPEN, Mat()));

    //debug: CV_TEST_EXPECT_EMPTY_THROW(blur(Mat_<uchar>(Size(3,3)), dst, Size(3, 3)));

    EXPECT_TRUE(src.empty());
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(dst2.empty());
}

TEST(Imgproc_GaussianBlur, regression_11303)
{
    cv::Mat dst;
    int width = 2115;
    int height = 211;
    double sigma = 8.64421;
    cv::Mat src(cv::Size(width, height), CV_32F, 1);
    cv::GaussianBlur(src, dst, cv::Size(), sigma, sigma);
    EXPECT_LE(cv::norm(src, dst, NORM_L2), 1e-3);
}

TEST(Imgproc, morphologyEx_small_input_22893)
{
    char input_data[] = {1, 2, 3, 4};
    char gold_data[] = {2, 3, 4, 4};
    cv::Mat img(1, 4, CV_8UC1, input_data);
    cv::Mat gold(1, 4, CV_8UC1, gold_data);

    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(4,4));
    cv::Mat result;
    morphologyEx(img, result, cv::MORPH_DILATE, kernel);

    ASSERT_EQ(0, cvtest::norm(result, gold, NORM_INF));
}

}} // namespace
