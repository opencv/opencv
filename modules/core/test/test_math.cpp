//////////////////////////////////////////////////////////////////////////////////////////
/////////////////// tests for matrix operations and math functions ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "test_precomp.hpp"
#include <float.h>
#include <math.h>

using namespace cv;
using namespace std;

/// !!! NOTE !!! These tests happily avoid overflow cases & out-of-range arguments
/// so that output arrays contain neigher Inf's nor Nan's.
/// Handling such cases would require special modification of check function
/// (validate_test_results) => TBD.
/// Also, need some logarithmic-scale generation of input data. Right now it is done (in some tests)
/// by generating min/max boundaries for random data in logarimithic scale, but
/// within the same test case all the input array elements are of the same order.

class Core_MathTest : public cvtest::ArrayTest
{
public:
    typedef cvtest::ArrayTest Base;
    Core_MathTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types);
    double get_success_error_level( int /*test_case_idx*/, int i, int j );
    bool test_nd;
};


Core_MathTest::Core_MathTest()
{
    optional_mask = false;
    
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    
    test_nd = false;
}


double Core_MathTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    return test_mat[i][j].depth() == CV_32F ? FLT_EPSILON*128 : DBL_EPSILON*1024;
}


void Core_MathTest::get_test_array_types_and_sizes( int test_case_idx,
                                                     vector<vector<Size> >& sizes,
                                                     vector<vector<int> >& types)
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng)%2 + CV_32F;
    int cn = cvtest::randInt(rng) % 4 + 1, type = CV_MAKETYPE(depth, cn);
    size_t i, j;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvtest::randInt(rng)%3 == 0;
}


////////// pow /////////////

class Core_PowTest : public Core_MathTest
{
public:
    typedef Core_MathTest Base;
    Core_PowTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
    double power;
};


Core_PowTest::Core_PowTest()
{
    power = 0;
}


void Core_PowTest::get_test_array_types_and_sizes( int test_case_idx,
                                                    vector<vector<Size> >& sizes,
                                                    vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % (CV_64F+1);
    int cn = cvtest::randInt(rng) % 4 + 1;
    size_t i, j;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth += depth == CV_8S;
    
    if( depth < CV_32F || cvtest::randInt(rng)%8 == 0 )
        // integer power
        power = (int)(cvtest::randInt(rng)%21 - 10);
    else
    {
        i = cvtest::randInt(rng)%17;
        power = i == 16 ? 1./3 : i == 15 ? 0.5 : i == 14 ? -0.5 : cvtest::randReal(rng)*10 - 5;
    }
    
    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        int type = CV_MAKETYPE(depth, cn);
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvtest::randInt(rng)%3 == 0;
}


double Core_PowTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[i][j].depth();
    if( depth < CV_32F )
        return power == cvRound(power) && power >= 0 ? 0 : 1;
    else
        return Base::get_success_error_level( test_case_idx, i, j );
}


void Core_PowTest::get_minmax_bounds( int /*i*/, int /*j*/, int type, Scalar& low, Scalar& high )
{
    double l, u = cvtest::randInt(ts->get_rng())%1000 + 1;
    if( power > 0 )
    {
        double mval = cvtest::getMaxVal(type);
        double u1 = pow(mval,1./power)*2;
        u = MIN(u,u1);
    }
    
    l = power == cvRound(power) ? -u : FLT_EPSILON;
    low = Scalar::all(l);
    high = Scalar::all(u);
}


void Core_PowTest::run_func()
{
    if(!test_nd)
    {
        if( fabs(power-1./3) <= DBL_EPSILON && test_mat[INPUT][0].depth() == CV_32F )
        {
            Mat a = test_mat[INPUT][0], b = test_mat[OUTPUT][0];
            
            a = a.reshape(1);
            b = b.reshape(1);
            for( int i = 0; i < a.rows; i++ )
            {
                b.at<float>(i,0) = (float)fabs(cvCbrt(a.at<float>(i,0)));
                for( int j = 1; j < a.cols; j++ )
                    b.at<float>(i,j) = (float)fabs(cv::cubeRoot(a.at<float>(i,j)));
            }
        }
        else
            cvPow( test_array[INPUT][0], test_array[OUTPUT][0], power );
    }
    else
    {
        Mat& a = test_mat[INPUT][0];
        Mat& b = test_mat[OUTPUT][0];
        if(power == 0.5)
            cv::sqrt(a, b);
        else
            cv::pow(a, power, b);
    }
}


inline static int ipow( int a, int power )
{
    int b = 1;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


inline static double ipow( double a, int power )
{
    double b = 1.;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


void Core_PowTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& a = test_mat[INPUT][0];
    Mat& b = test_mat[REF_OUTPUT][0];
    
    int depth = a.depth();
    int ncols = a.cols*a.channels();
    int ipower = cvRound(power), apower = abs(ipower);
    int i, j;
    
    for( i = 0; i < a.rows; i++ )
    {
        const uchar* a_data = a.ptr(i);
        uchar* b_data = b.ptr(i);
        
        switch( depth )
        {
            case CV_8U:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((uchar*)a_data)[j];
                        ((uchar*)b_data)[j] = (uchar)(val <= 1 ? val :
                                                      val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((uchar*)a_data)[j];
                        val = ipow( val, ipower );
                        ((uchar*)b_data)[j] = saturate_cast<uchar>(val);
                    }
                break;
            case CV_8S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((char*)a_data)[j];
                        ((char*)b_data)[j] = (char)((val&~1)==0 ? val :
                                                    val ==-1 ? 1-2*(ipower&1) :
                                                    val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((char*)a_data)[j];
                        val = ipow( val, ipower );
                        ((char*)b_data)[j] = saturate_cast<schar>(val);
                    }
                break;
            case CV_16U:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((ushort*)a_data)[j];
                        ((ushort*)b_data)[j] = (ushort)((val&~1)==0 ? val :
                                                        val ==-1 ? 1-2*(ipower&1) :
                                                        val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((ushort*)a_data)[j];
                        val = ipow( val, ipower );
                        ((ushort*)b_data)[j] = saturate_cast<ushort>(val);
                    }
                break;
            case CV_16S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((short*)a_data)[j];
                        ((short*)b_data)[j] = (short)((val&~1)==0 ? val :
                                                      val ==-1 ? 1-2*(ipower&1) :
                                                      val == 2 && ipower == -1 ? 1 : 0);
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((short*)a_data)[j];
                        val = ipow( val, ipower );
                        ((short*)b_data)[j] = saturate_cast<short>(val);
                    }
                break;
            case CV_32S:
                if( ipower < 0 )
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((int*)a_data)[j];
                        ((int*)b_data)[j] = (val&~1)==0 ? val :
                        val ==-1 ? 1-2*(ipower&1) :
                        val == 2 && ipower == -1 ? 1 : 0;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        int val = ((int*)a_data)[j];
                        val = ipow( val, ipower );
                        ((int*)b_data)[j] = val;
                    }
                break;
            case CV_32F:
                if( power != ipower )
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((float*)a_data)[j];
                        val = pow( fabs(val), power );
                        ((float*)b_data)[j] = (float)val;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((float*)a_data)[j];
                        if( ipower < 0 )
                            val = 1./val;
                        val = ipow( val, apower );
                        ((float*)b_data)[j] = (float)val;
                    }
                break;
            case CV_64F:
                if( power != ipower )
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((double*)a_data)[j];
                        val = pow( fabs(val), power );
                        ((double*)b_data)[j] = (double)val;
                    }
                else
                    for( j = 0; j < ncols; j++ )
                    {
                        double val = ((double*)a_data)[j];
                        if( ipower < 0 )
                            val = 1./val;
                        val = ipow( val, apower );
                        ((double*)b_data)[j] = (double)val;
                    }
                break;
        }
    }
}



///////////////////////////////////////// matrix tests ////////////////////////////////////////////

class Core_MatrixTest : public cvtest::ArrayTest
{
public:
    typedef cvtest::ArrayTest Base;
    Core_MatrixTest( int in_count, int out_count,
                       bool allow_int, bool scalar_output, int max_cn );
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    bool allow_int;
    bool scalar_output;
    int max_cn;
};


Core_MatrixTest::Core_MatrixTest( int in_count, int out_count,
                                      bool _allow_int, bool _scalar_output, int _max_cn )
: allow_int(_allow_int), scalar_output(_scalar_output), max_cn(_max_cn)
{
    int i;
    for( i = 0; i < in_count; i++ )
        test_array[INPUT].push_back(NULL);
    
    for( i = 0; i < out_count; i++ )
    {
        test_array[OUTPUT].push_back(NULL);
        test_array[REF_OUTPUT].push_back(NULL);
    }
    
    element_wise_relative_error = false;
}


void Core_MatrixTest::get_test_array_types_and_sizes( int test_case_idx,
                                                       vector<vector<Size> >& sizes,
                                                       vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % (allow_int ? CV_64F+1 : 2);
    int cn = cvtest::randInt(rng) % max_cn + 1;
    size_t i, j;
    
    if( allow_int )
        depth += depth == CV_8S;
    else
        depth += CV_32F;
    
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    for( i = 0; i < test_array.size(); i++ )
    {
        size_t count = test_array[i].size();
        int flag = (i == OUTPUT || i == REF_OUTPUT) && scalar_output;
        int type = !flag ? CV_MAKETYPE(depth, cn) : CV_64FC1;
        
        for( j = 0; j < count; j++ )
        {
            types[i][j] = type;
            if( flag )
                sizes[i][j] = Size( 4, 1 );
        }
    }
}


double Core_MatrixTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int input_depth = test_mat[INPUT][0].depth();
    double input_precision = input_depth < CV_32F ? 0 : input_depth == CV_32F ? 5e-5 : 5e-10;
    double output_precision = Base::get_success_error_level( test_case_idx, i, j );
    return MAX(input_precision, output_precision);
}


///////////////// Trace /////////////////////

class Core_TraceTest : public Core_MatrixTest
{
public:
    Core_TraceTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_TraceTest::Core_TraceTest() : Core_MatrixTest( 1, 1, true, true, 4 )
{
}


void Core_TraceTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = cvTrace(test_array[INPUT][0]);
}


void Core_TraceTest::prepare_to_validation( int )
{
    Mat& mat = test_mat[INPUT][0];
    int count = MIN( mat.rows, mat.cols );
    Mat diag(count, 1, mat.type(), mat.data, mat.step + mat.elemSize());
    Scalar r = cvtest::mean(diag);
    r *= (double)count;
    
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = r;
}


///////// dotproduct //////////

class Core_DotProductTest : public Core_MatrixTest
{
public:
    Core_DotProductTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_DotProductTest::Core_DotProductTest() : Core_MatrixTest( 2, 1, true, true, 4 )
{
}


void Core_DotProductTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = Scalar(cvDotProduct( test_array[INPUT][0], test_array[INPUT][1] ));
}


void Core_DotProductTest::prepare_to_validation( int )
{
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = Scalar(cvtest::crossCorr( test_mat[INPUT][0], test_mat[INPUT][1] ));
}


///////// crossproduct //////////

class Core_CrossProductTest : public Core_MatrixTest
{
public:
    Core_CrossProductTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                        vector<vector<Size> >& sizes,
                                        vector<vector<int> >& types );    
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_CrossProductTest::Core_CrossProductTest() : Core_MatrixTest( 2, 1, false, false, 1 )
{
}


void Core_CrossProductTest::get_test_array_types_and_sizes( int,
                                                             vector<vector<Size> >& sizes,
                                                             vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 2 + CV_32F;
    int cn = cvtest::randInt(rng) & 1 ? 3 : 1, type = CV_MAKETYPE(depth, cn);
    CvSize sz;
    
    types[INPUT][0] = types[INPUT][1] = types[OUTPUT][0] = types[REF_OUTPUT][0] = type;
    
    if( cn == 3 )
        sz = Size(1,1);
    else if( cvtest::randInt(rng) & 1 )
        sz = Size(3,1);
    else
        sz = Size(1,3);
    
    sizes[INPUT][0] = sizes[INPUT][1] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sz;
}


void Core_CrossProductTest::run_func()
{
    cvCrossProduct( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0] );
}


void Core_CrossProductTest::prepare_to_validation( int )
{
    CvScalar a = {{0,0,0,0}}, b = {{0,0,0,0}}, c = {{0,0,0,0}};
    
    if( test_mat[INPUT][0].rows > 1 )
    {
        a.val[0] = cvGetReal2D( test_array[INPUT][0], 0, 0 );
        a.val[1] = cvGetReal2D( test_array[INPUT][0], 1, 0 );
        a.val[2] = cvGetReal2D( test_array[INPUT][0], 2, 0 );
        
        b.val[0] = cvGetReal2D( test_array[INPUT][1], 0, 0 );
        b.val[1] = cvGetReal2D( test_array[INPUT][1], 1, 0 );
        b.val[2] = cvGetReal2D( test_array[INPUT][1], 2, 0 );
    }
    else if( test_mat[INPUT][0].cols > 1 )
    {
        a.val[0] = cvGetReal1D( test_array[INPUT][0], 0 );
        a.val[1] = cvGetReal1D( test_array[INPUT][0], 1 );
        a.val[2] = cvGetReal1D( test_array[INPUT][0], 2 );
        
        b.val[0] = cvGetReal1D( test_array[INPUT][1], 0 );
        b.val[1] = cvGetReal1D( test_array[INPUT][1], 1 );
        b.val[2] = cvGetReal1D( test_array[INPUT][1], 2 );
    }
    else
    {
        a = cvGet1D( test_array[INPUT][0], 0 );
        b = cvGet1D( test_array[INPUT][1], 0 );
    }
    
    c.val[2] = a.val[0]*b.val[1] - a.val[1]*b.val[0];
    c.val[1] = -a.val[0]*b.val[2] + a.val[2]*b.val[0];
    c.val[0] = a.val[1]*b.val[2] - a.val[2]*b.val[1];
    
    if( test_mat[REF_OUTPUT][0].rows > 1 )
    {
        cvSetReal2D( test_array[REF_OUTPUT][0], 0, 0, c.val[0] );
        cvSetReal2D( test_array[REF_OUTPUT][0], 1, 0, c.val[1] );
        cvSetReal2D( test_array[REF_OUTPUT][0], 2, 0, c.val[2] );
    }
    else if( test_mat[REF_OUTPUT][0].cols > 1 )
    {
        cvSetReal1D( test_array[REF_OUTPUT][0], 0, c.val[0] );
        cvSetReal1D( test_array[REF_OUTPUT][0], 1, c.val[1] );
        cvSetReal1D( test_array[REF_OUTPUT][0], 2, c.val[2] );
    }
    else
    {
        cvSet1D( test_array[REF_OUTPUT][0], 0, c );
    }
}


///////////////// gemm /////////////////////

class Core_GEMMTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_GEMMTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int tabc_flag;
    double alpha, beta;
};

Core_GEMMTest::Core_GEMMTest() : Core_MatrixTest( 5, 1, false, false, 2 )
{
    test_case_count = 100;
    max_log_array_size = 10;
    alpha = beta = 0;
}


void Core_GEMMTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    Size sizeA;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizeA = sizes[INPUT][0];
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sizeA;
    sizes[INPUT][2] = sizes[INPUT][3] = Size(1,1);
    types[INPUT][2] = types[INPUT][3] &= ~CV_MAT_CN_MASK;
    
    tabc_flag = cvtest::randInt(rng) & 7;
    
    switch( tabc_flag & (CV_GEMM_A_T|CV_GEMM_B_T) )
    {
        case 0:
            sizes[INPUT][1].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].width = sizes[INPUT][1].width;
            break;
        case CV_GEMM_B_T:
            sizes[INPUT][1].width = sizes[INPUT][0].width;
            sizes[OUTPUT][0].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].width = sizes[INPUT][1].height;
            break;
        case CV_GEMM_A_T:
            sizes[INPUT][1].height = sizes[INPUT][0].height;
            sizes[OUTPUT][0].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].width = sizes[INPUT][1].width;
            break;
        case CV_GEMM_A_T | CV_GEMM_B_T:
            sizes[INPUT][1].width = sizes[INPUT][0].height;
            sizes[OUTPUT][0].height = sizes[INPUT][0].width;
            sizes[OUTPUT][0].width = sizes[INPUT][1].height;
            break;
    }
    
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
    
    if( cvtest::randInt(rng) & 1 )
        sizes[INPUT][4] = Size(0,0);
    else if( !(tabc_flag & CV_GEMM_C_T) )
        sizes[INPUT][4] = sizes[OUTPUT][0];
    else
    {
        sizes[INPUT][4].width = sizes[OUTPUT][0].height;
        sizes[INPUT][4].height = sizes[OUTPUT][0].width;
    }
}


int Core_GEMMTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        alpha = cvGetReal2D( test_array[INPUT][2], 0, 0 );
        beta = cvGetReal2D( test_array[INPUT][3], 0, 0 );
    }
    return code;
}


void Core_GEMMTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = Scalar::all(-10.);
    high = Scalar::all(10.);
}


void Core_GEMMTest::run_func()
{
    cvGEMM( test_array[INPUT][0], test_array[INPUT][1], alpha,
           test_array[INPUT][4], beta, test_array[OUTPUT][0], tabc_flag );
}


void Core_GEMMTest::prepare_to_validation( int )
{
    cvtest::gemm( test_mat[INPUT][0], test_mat[INPUT][1], alpha,
             test_array[INPUT][4] ? test_mat[INPUT][4] : Mat(),
             beta, test_mat[REF_OUTPUT][0], tabc_flag );
}


///////////////// multransposed /////////////////////

class Core_MulTransposedTest : public Core_MatrixTest
{
public:
    Core_MulTransposedTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int order;
};


Core_MulTransposedTest::Core_MulTransposedTest() : Core_MatrixTest( 2, 1, false, false, 1 )
{
    test_case_count = 100;
    order = 0;
    test_array[TEMP].push_back(NULL);
}


void Core_MulTransposedTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    int src_type = cvtest::randInt(rng) % 5;
    int dst_type = cvtest::randInt(rng) % 2;
    
    src_type = src_type == 0 ? CV_8U : src_type == 1 ? CV_16U : src_type == 2 ? CV_16S :
    src_type == 3 ? CV_32F : CV_64F;
    dst_type = dst_type == 0 ? CV_32F : CV_64F;
    dst_type = MAX( dst_type, src_type );
    
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    if( bits & 1 )
        sizes[INPUT][1] = Size(0,0);
    else
    {
        sizes[INPUT][1] = sizes[INPUT][0];
        if( bits & 2 )
            sizes[INPUT][1].height = 1;
        if( bits & 4 )
            sizes[INPUT][1].width = 1;
    }
    
    sizes[TEMP][0] = sizes[INPUT][0];
    types[INPUT][0] = src_type;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][1] = types[TEMP][0] = dst_type;
    
    order = (bits & 8) != 0;
    sizes[OUTPUT][0].width = sizes[OUTPUT][0].height = order == 0 ?
    sizes[INPUT][0].height : sizes[INPUT][0].width;
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


void Core_MulTransposedTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-10.);
    high = cvScalarAll(10.);
}


void Core_MulTransposedTest::run_func()
{
    cvMulTransposed( test_array[INPUT][0], test_array[OUTPUT][0],
                    order, test_array[INPUT][1] );
}


void Core_MulTransposedTest::prepare_to_validation( int )
{
    const Mat& src = test_mat[INPUT][0];
    Mat delta = test_mat[INPUT][1];
    Mat& temp = test_mat[TEMP][0];
    if( !delta.empty() )
    {
        if( delta.rows < src.rows || delta.cols < src.cols )
        {
            cv::repeat( delta, src.rows/delta.rows, src.cols/delta.cols, temp);
            delta = temp;
        }
        cvtest::add( src, 1, delta, -1, Scalar::all(0), temp, temp.type());
    }
    else
        src.convertTo(temp, temp.type());
    
    cvtest::gemm( temp, temp, 1., Mat(), 0, test_mat[REF_OUTPUT][0], order == 0 ? GEMM_2_T : GEMM_1_T );
}


///////////////// Transform /////////////////////

class Core_TransformTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_TransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    
    double scale;
    bool diagMtx;
};


Core_TransformTest::Core_TransformTest() : Core_MatrixTest( 3, 1, true, false, 4 )
{
}


void Core_TransformTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    int depth, dst_cn, mat_cols, mattype;
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    mat_cols = CV_MAT_CN(types[INPUT][0]);
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    dst_cn = cvtest::randInt(rng) % 4 + 1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, dst_cn);
    
    mattype = depth < CV_32S ? CV_32F : depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;
    types[INPUT][2] = CV_MAKETYPE(mattype, dst_cn);
    
    scale = 1./((cvtest::randInt(rng)%4)*50+1);
    
    if( bits & 2 )
    {
        sizes[INPUT][2] = Size(0,0);
        mat_cols += (bits & 4) != 0;
    }
    else if( bits & 4 )
        sizes[INPUT][2] = Size(1,1);
    else
    {
        if( bits & 8 )
            sizes[INPUT][2] = Size(dst_cn,1);
        else
            sizes[INPUT][2] = Size(1,dst_cn);
        types[INPUT][2] &= ~CV_MAT_CN_MASK;
    }
    diagMtx = (bits & 16) != 0;
    
    sizes[INPUT][1] = Size(mat_cols,dst_cn);
}


int Core_TransformTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        Mat& m = test_mat[INPUT][1];
        cvtest::add(m, scale, m, 0, Scalar::all(0), m, m.type() );
        if(diagMtx)
        {
            Mat mask = Mat::eye(m.rows, m.cols, CV_8U)*255;
            mask = ~mask;
            m.setTo(Scalar::all(0), mask);
        }
    }
    return code;
}


double Core_TransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[INPUT][0].depth();
    return depth <= CV_8S ? 1 : depth <= CV_32S ? 9 : Base::get_success_error_level( test_case_idx, i, j );
}

void Core_TransformTest::run_func()
{
    CvMat _m = test_mat[INPUT][1], _shift = test_mat[INPUT][2];
    cvTransform( test_array[INPUT][0], test_array[OUTPUT][0], &_m, _shift.data.ptr ? &_shift : 0);
}


void Core_TransformTest::prepare_to_validation( int )
{
    Mat transmat = test_mat[INPUT][1];
    Mat shift = test_mat[INPUT][2];
    
    cvtest::transform( test_mat[INPUT][0], test_mat[REF_OUTPUT][0], transmat, shift );
}


///////////////// PerspectiveTransform /////////////////////

class Core_PerspectiveTransformTest : public Core_MatrixTest
{
public:
    Core_PerspectiveTransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_PerspectiveTransformTest::Core_PerspectiveTransformTest() : Core_MatrixTest( 2, 1, false, false, 2 )
{
}


void Core_PerspectiveTransformTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    int depth, cn, mattype;
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    cn = CV_MAT_CN(types[INPUT][0]) + 1;
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    
    mattype = depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;
    sizes[INPUT][1] = Size(cn + 1, cn + 1);
}


double Core_PerspectiveTransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = test_mat[INPUT][0].depth();
    return depth == CV_32F ? 1e-4 : depth == CV_64F ? 1e-8 :
    Core_MatrixTest::get_success_error_level(test_case_idx, i, j);
}


void Core_PerspectiveTransformTest::run_func()
{
    CvMat _m = test_mat[INPUT][1];
    cvPerspectiveTransform( test_array[INPUT][0], test_array[OUTPUT][0], &_m );
}


static void cvTsPerspectiveTransform( const CvArr* _src, CvArr* _dst, const CvMat* transmat )
{
    int i, j, cols;
    int cn, depth, mat_depth;
    CvMat astub, bstub, *a, *b;
    double mat[16];
    
    a = cvGetMat( _src, &astub, 0, 0 );
    b = cvGetMat( _dst, &bstub, 0, 0 );
    
    cn = CV_MAT_CN(a->type);
    depth = CV_MAT_DEPTH(a->type);
    mat_depth = CV_MAT_DEPTH(transmat->type);
    cols = transmat->cols;
    
    // prepare cn x (cn + 1) transform matrix
    if( mat_depth == CV_32F )
    {
        for( i = 0; i < transmat->rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = ((float*)(transmat->data.ptr + transmat->step*i))[j];
    }
    else
    {
        assert( mat_depth == CV_64F );
        for( i = 0; i < transmat->rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = ((double*)(transmat->data.ptr + transmat->step*i))[j];
    }
    
    // transform data
    cols = a->cols * cn;
    vector<double> buf(cols);
    
    for( i = 0; i < a->rows; i++ )
    {
        uchar* src = a->data.ptr + i*a->step;
        uchar* dst = b->data.ptr + i*b->step;
        
        switch( depth )
        {
            case CV_32F:
                for( j = 0; j < cols; j++ )
                    buf[j] = ((float*)src)[j];
                break;
            case CV_64F:
                for( j = 0; j < cols; j++ )
                    buf[j] = ((double*)src)[j];
                break;
            default:
                assert(0);
        }
        
        switch( cn )
        {
            case 2:
                for( j = 0; j < cols; j += 2 )
                {
                    double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + mat[2];
                    double t1 = buf[j]*mat[3] + buf[j+1]*mat[4] + mat[5];
                    double w = buf[j]*mat[6] + buf[j+1]*mat[7] + mat[8];
                    w = w ? 1./w : 0;
                    buf[j] = t0*w;
                    buf[j+1] = t1*w;
                }
                break;
            case 3:
                for( j = 0; j < cols; j += 3 )
                {
                    double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + buf[j+2]*mat[2] + mat[3];
                    double t1 = buf[j]*mat[4] + buf[j+1]*mat[5] + buf[j+2]*mat[6] + mat[7];
                    double t2 = buf[j]*mat[8] + buf[j+1]*mat[9] + buf[j+2]*mat[10] + mat[11];
                    double w = buf[j]*mat[12] + buf[j+1]*mat[13] + buf[j+2]*mat[14] + mat[15];
                    w = w ? 1./w : 0;
                    buf[j] = t0*w;
                    buf[j+1] = t1*w;
                    buf[j+2] = t2*w;
                }
                break;
            default:
                assert(0);
        }
        
        switch( depth )
        {
            case CV_32F:
                for( j = 0; j < cols; j++ )
                    ((float*)dst)[j] = (float)buf[j];
                break;
            case CV_64F:
                for( j = 0; j < cols; j++ )
                    ((double*)dst)[j] = buf[j];
                break;
            default:
                assert(0);
        }
    }
}


void Core_PerspectiveTransformTest::prepare_to_validation( int )
{
    CvMat transmat = test_mat[INPUT][1];
    cvTsPerspectiveTransform( test_array[INPUT][0], test_array[REF_OUTPUT][0], &transmat );
}

///////////////// Mahalanobis /////////////////////

class Core_MahalanobisTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_MahalanobisTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_MahalanobisTest::Core_MahalanobisTest() : Core_MatrixTest( 3, 1, false, true, 1 )
{
    test_case_count = 100;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_MahalanobisTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    if( cvtest::randInt(rng) & 1 )
        sizes[INPUT][0].width = sizes[INPUT][1].width = 1;
    else
        sizes[INPUT][0].height = sizes[INPUT][1].height = 1;
    
    sizes[TEMP][0] = sizes[TEMP][1] = sizes[INPUT][0];
    sizes[INPUT][2].width = sizes[INPUT][2].height = sizes[INPUT][0].width + sizes[INPUT][0].height - 1;
    sizes[TEMP][2] = sizes[INPUT][2];
    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[INPUT][0];
}

int Core_MahalanobisTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        // make sure that the inverted "covariation" matrix is symmetrix and positively defined.
        cvtest::gemm( test_mat[INPUT][2], test_mat[INPUT][2], 1., Mat(), 0., test_mat[TEMP][2], GEMM_2_T );
        cvtest::copy( test_mat[TEMP][2], test_mat[INPUT][2] );
    }
    
    return code;
}


void Core_MahalanobisTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) =
    cvRealScalar(cvMahalanobis(test_array[INPUT][0], test_array[INPUT][1], test_array[INPUT][2]));
}

void Core_MahalanobisTest::prepare_to_validation( int )
{
    cvtest::add( test_mat[INPUT][0], 1., test_mat[INPUT][1], -1.,
                Scalar::all(0), test_mat[TEMP][0], test_mat[TEMP][0].type() );
    if( test_mat[INPUT][0].rows == 1 )
        cvtest::gemm( test_mat[TEMP][0], test_mat[INPUT][2], 1.,
                 Mat(), 0., test_mat[TEMP][1], 0 );
    else
        cvtest::gemm( test_mat[INPUT][2], test_mat[TEMP][0], 1.,
                 Mat(), 0., test_mat[TEMP][1], 0 );
    
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = cvRealScalar(sqrt(cvtest::crossCorr(test_mat[TEMP][0], test_mat[TEMP][1])));
}


///////////////// covarmatrix /////////////////////

class Core_CovarMatrixTest : public Core_MatrixTest
{
public:
    Core_CovarMatrixTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    vector<void*> temp_hdrs;
    vector<uchar> hdr_data;
    int flags, t_flag, len, count;
    bool are_images;
};


Core_CovarMatrixTest::Core_CovarMatrixTest() : Core_MatrixTest( 1, 1, true, false, 1 ),
flags(0), t_flag(0), are_images(false)
{
    test_case_count = 100;
    test_array[INPUT_OUTPUT].push_back(NULL);
    test_array[REF_INPUT_OUTPUT].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_CovarMatrixTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    int i, single_matrix;
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    flags = bits & (CV_COVAR_NORMAL | CV_COVAR_USE_AVG | CV_COVAR_SCALE | CV_COVAR_ROWS );
    single_matrix = flags & CV_COVAR_ROWS;
    t_flag = (bits & 256) != 0;
    
    const int min_count = 2;
    
    if( !t_flag )
    {
        len = sizes[INPUT][0].width;
        count = sizes[INPUT][0].height;
        count = MAX(count, min_count);
        sizes[INPUT][0] = Size(len, count);
    }
    else
    {
        len = sizes[INPUT][0].height;
        count = sizes[INPUT][0].width;
        count = MAX(count, min_count);
        sizes[INPUT][0] = Size(count, len);
    }
    
    if( single_matrix && t_flag )
        flags = (flags & ~CV_COVAR_ROWS) | CV_COVAR_COLS;
    
    if( CV_MAT_DEPTH(types[INPUT][0]) == CV_32S )
        types[INPUT][0] = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK) | CV_32F;
    
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = flags & CV_COVAR_NORMAL ? Size(len,len) : Size(count,count);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = !t_flag ? Size(len,1) : Size(1,len);
    sizes[TEMP][0] = sizes[INPUT][0];
    
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] =
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[TEMP][0] =
    CV_MAT_DEPTH(types[INPUT][0]) == CV_64F || (bits & 512) ? CV_64F : CV_32F;
    
    are_images = (bits & 1024) != 0;
    for( i = 0; i < (single_matrix ? 1 : count); i++ )
        temp_hdrs.push_back(NULL);
}


int Core_CovarMatrixTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        int i;
        int single_matrix = flags & (CV_COVAR_ROWS|CV_COVAR_COLS);
        int hdr_size = are_images ? sizeof(IplImage) : sizeof(CvMat);
        
        hdr_data.resize(count*hdr_size);
        uchar* _hdr_data = &hdr_data[0];
        if( single_matrix )
        {
            if( !are_images )
                *((CvMat*)_hdr_data) = test_mat[INPUT][0];
            else
                *((IplImage*)_hdr_data) = test_mat[INPUT][0];
            temp_hdrs[0] = _hdr_data;
        }
        else
            for( i = 0; i < count; i++ )
            {
                Mat part;
                void* ptr = _hdr_data + i*hdr_size;
                
                if( !t_flag )
                    part = test_mat[INPUT][0].row(i);
                else
                    part = test_mat[INPUT][0].col(i);
                
                if( !are_images )
                    *((CvMat*)ptr) = part;
                else
                    *((IplImage*)ptr) = part;
                
                temp_hdrs[i] = ptr;
            }
    }
    
    return code;
}


void Core_CovarMatrixTest::run_func()
{
    cvCalcCovarMatrix( (const void**)&temp_hdrs[0], count,
                      test_array[OUTPUT][0], test_array[INPUT_OUTPUT][0], flags );
}


void Core_CovarMatrixTest::prepare_to_validation( int )
{
    Mat& avg = test_mat[REF_INPUT_OUTPUT][0];
    double scale = 1.;
    
    if( !(flags & CV_COVAR_USE_AVG) )
    {
        Mat hdrs0 = cvarrToMat(temp_hdrs[0]);
        
        int i;
        avg = Scalar::all(0);
        
        for( i = 0; i < count; i++ )
        {
            Mat vec;
            if( flags & CV_COVAR_ROWS )
                vec = hdrs0.row(i);
            else if( flags & CV_COVAR_COLS )
                vec = hdrs0.col(i);
            else
                vec = cvarrToMat(temp_hdrs[i]);
            
            cvtest::add(avg, 1, vec, 1, Scalar::all(0), avg, avg.type());
        }
        
        cvtest::add(avg, 1./count, avg, 0., Scalar::all(0), avg, avg.type());
    }
    
    if( flags & CV_COVAR_SCALE )
    {
        scale = 1./count;
    }
    
    Mat& temp0 = test_mat[TEMP][0];
    cv::repeat( avg, temp0.rows/avg.rows, temp0.cols/avg.cols, temp0 );
    cvtest::add( test_mat[INPUT][0], 1, temp0, -1, Scalar::all(0), temp0, temp0.type());
    
    cvtest::gemm( temp0, temp0, scale, Mat(), 0., test_mat[REF_OUTPUT][0],
             t_flag ^ ((flags & CV_COVAR_NORMAL) != 0) ? CV_GEMM_A_T : CV_GEMM_B_T );
    temp_hdrs.clear();
}


static void cvTsFloodWithZeros( Mat& mat, RNG& rng )
{
    int k, total = mat.rows*mat.cols, type = mat.type();
    int zero_total = cvtest::randInt(rng) % total;
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );
    
    for( k = 0; k < zero_total; k++ )
    {
        int i = cvtest::randInt(rng) % mat.rows;
        int j = cvtest::randInt(rng) % mat.cols;
        
        if( type == CV_32FC1 )
            mat.at<float>(i,j) = 0.f;
        else
            mat.at<double>(i,j) = 0.;
    }
}


///////////////// determinant /////////////////////

class Core_DetTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_DetTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


Core_DetTest::Core_DetTest() : Core_MatrixTest( 1, 1, false, true, 1 )
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push_back(NULL);
}


void Core_DetTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    
    sizes[INPUT][0].width = sizes[INPUT][0].height = sizes[INPUT][0].height;
    sizes[TEMP][0] = sizes[INPUT][0];
    types[TEMP][0] = CV_64FC1;
}


void Core_DetTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-2.);
    high = cvScalarAll(2.);
}


double Core_DetTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0])) == CV_32F ? 1e-2 : 1e-5;
}


int Core_DetTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
        cvTsFloodWithZeros( test_mat[INPUT][0], ts->get_rng() );
    
    return code;
}


void Core_DetTest::run_func()
{
    test_mat[OUTPUT][0].at<Scalar>(0,0) = cvRealScalar(cvDet(test_array[INPUT][0]));
}


// LU method that chooses the optimal in a column pivot element
static double cvTsLU( CvMat* a, CvMat* b=NULL, CvMat* x=NULL, int* rank=0 )
{
    int i, j, k, N = a->rows, N1 = a->cols, Nm = MIN(N, N1), step = a->step/sizeof(double);
    int M = b ? b->cols : 0, b_step = b ? b->step/sizeof(double) : 0;
    int x_step = x ? x->step/sizeof(double) : 0;
    double *a0 = a->data.db, *b0 = b ? b->data.db : 0;
    double *x0 = x ? x->data.db : 0;
    double t, det = 1.;
    assert( CV_MAT_TYPE(a->type) == CV_64FC1 &&
           (!b || CV_ARE_TYPES_EQ(a,b)) && (!x || CV_ARE_TYPES_EQ(a,x)));
    
    for( i = 0; i < Nm; i++ )
    {
        double max_val = fabs(a0[i*step + i]);
        double *a1, *a2, *b1 = 0, *b2 = 0;
        k = i;
        
        for( j = i+1; j < N; j++ )
        {
            t = fabs(a0[j*step + i]);
            if( max_val < t )
            {
                max_val = t;
                k = j;
            }
        }
        
        if( k != i )
        {
            for( j = i; j < N1; j++ )
                CV_SWAP( a0[i*step + j], a0[k*step + j], t );
            
            for( j = 0; j < M; j++ )
                CV_SWAP( b0[i*b_step + j], b0[k*b_step + j], t );
            det = -det;
        }
        
        if( max_val == 0 )
        {
            if( rank )
                *rank = i;
            return 0.;
        }
        
        a1 = a0 + i*step;
        a2 = a1 + step;
        b1 = b0 + i*b_step;
        b2 = b1 + b_step;
        
        for( j = i+1; j < N; j++, a2 += step, b2 += b_step )
        {
            t = a2[i]/a1[i];
            for( k = i+1; k < N1; k++ )
                a2[k] -= t*a1[k];
            
            for( k = 0; k < M; k++ )
                b2[k] -= t*b1[k];
        }
        
        det *= a1[i];
    }
    
    if( x )
    {
        assert( b );
        
        for( i = N-1; i >= 0; i-- )
        {
            double* a1 = a0 + i*step;
            double* b1 = b0 + i*b_step;
            for( j = 0; j < M; j++ )
            {
                t = b1[j];
                for( k = i+1; k < N1; k++ )
                    t -= a1[k]*x0[k*x_step + j];
                x0[i*x_step + j] = t/a1[i];
            }
        }
    }
    
    if( rank )
        *rank = i;
    return det;
}


void Core_DetTest::prepare_to_validation( int )
{
    test_mat[INPUT][0].convertTo(test_mat[TEMP][0], test_mat[TEMP][0].type());
    CvMat temp0 = test_mat[TEMP][0];
    test_mat[REF_OUTPUT][0].at<Scalar>(0,0) = cvRealScalar(cvTsLU(&temp0, 0, 0));
}


///////////////// invert /////////////////////

class Core_InvertTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_InvertTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


Core_InvertTest::Core_InvertTest()
: Core_MatrixTest( 1, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_InvertTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );
    
    if( (bits & 3) == 0 )
    {
        method = CV_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = Size(min_size, min_size);
            if( bits & 16 )
                method = CV_CHOLESKY;
        }
    }
    else
    {
        method = CV_LU;
        sizes[INPUT][0] = Size(min_size, min_size);
    }
    
    sizes[TEMP][0].width = sizes[INPUT][0].height;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(min_size, min_size);
}


double Core_InvertTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[OUTPUT][0])) == CV_32F ? 1e-2 : 1e-6;
}

int Core_InvertTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        cvTsFloodWithZeros( test_mat[INPUT][0], ts->get_rng() );
        
        if( method == CV_CHOLESKY )
        {
            cvtest::gemm( test_mat[INPUT][0], test_mat[INPUT][0], 1.,
                     Mat(), 0., test_mat[TEMP][0], CV_GEMM_B_T );
            cvtest::copy( test_mat[TEMP][0], test_mat[INPUT][0] );
        }
    }
    
    return code;
}



void Core_InvertTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-1.);
    high = cvScalarAll(1.);
}


void Core_InvertTest::run_func()
{
    result = cvInvert(test_array[INPUT][0], test_array[TEMP][0], method);
}


static double cvTsSVDet( CvMat* mat, double* ratio )
{
    int type = CV_MAT_TYPE(mat->type);
    int i, nm = MIN( mat->rows, mat->cols );
    CvMat* w = cvCreateMat( nm, 1, type );
    double det = 1.;
    
    cvSVD( mat, w, 0, 0, 0 );
    
    if( type == CV_32FC1 )
    {
        for( i = 0; i < nm; i++ )
            det *= w->data.fl[i];
        *ratio = w->data.fl[nm-1] < FLT_EPSILON ? 0 : w->data.fl[nm-1]/w->data.fl[0];
    }
    else
    {
        for( i = 0; i < nm; i++ )
            det *= w->data.db[i];
        *ratio = w->data.db[nm-1] < FLT_EPSILON ? 0 : w->data.db[nm-1]/w->data.db[0];
    }
    
    cvReleaseMat( &w );
    return det;
}

void Core_InvertTest::prepare_to_validation( int )
{
    Mat& input = test_mat[INPUT][0];
    Mat& temp0 = test_mat[TEMP][0];
    Mat& temp1 = test_mat[TEMP][1];
    Mat& dst0 = test_mat[REF_OUTPUT][0];
    Mat& dst = test_mat[OUTPUT][0];
    CvMat _input = input;
    double ratio = 0, det = cvTsSVDet( &_input, &ratio );
    double threshold = (input.depth() == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;
    
    cvtest::convert( input, temp1, temp1.type() );
    
    if( det < threshold ||
       ((method == CV_LU || method == CV_CHOLESKY) && (result == 0 || ratio < threshold)) ||
       ((method == CV_SVD || method == CV_SVD_SYM) && result < threshold) )
    {
        dst = Scalar::all(0);
        dst0 = Scalar::all(0);
        return;
    }
    
    if( input.rows >= input.cols )
        cvtest::gemm( temp0, input, 1., Mat(), 0., dst, 0 );
    else
        cvtest::gemm( input, temp0, 1., Mat(), 0., dst, 0 );
    
    cv::setIdentity( dst0, Scalar::all(1) );
}


///////////////// solve /////////////////////

class Core_SolveTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_SolveTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


Core_SolveTest::Core_SolveTest() : Core_MatrixTest( 2, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_SolveTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize in_sz = sizes[INPUT][0];
    if( in_sz.width > in_sz.height )
        in_sz = cvSize(in_sz.height, in_sz.width);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = in_sz;
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );
    
    if( (bits & 3) == 0 )
    {
        method = CV_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = Size(min_size, min_size);
            /*if( bits & 8 )
             method = CV_SVD_SYM;*/
        }
    }
    else
    {
        method = CV_LU;
        sizes[INPUT][0] = Size(min_size, min_size);
    }
    
    sizes[INPUT][1].height = sizes[INPUT][0].height;
    sizes[TEMP][0].width = sizes[INPUT][1].width;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(sizes[INPUT][1].width, min_size);
}


int Core_SolveTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    
    /*if( method == CV_SVD_SYM )
     {
     cvTsGEMM( test_array[INPUT][0], test_array[INPUT][0], 1.,
     0, 0., test_array[TEMP][0], CV_GEMM_B_T );
     cvTsCopy( test_array[TEMP][0], test_array[INPUT][0] );
     }*/
    
    return code;
}


void Core_SolveTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-1.);
    high = cvScalarAll(1.);
}


double Core_SolveTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[OUTPUT][0])) == CV_32F ? 5e-2 : 1e-8;
}


void Core_SolveTest::run_func()
{
    result = cvSolve(test_array[INPUT][0], test_array[INPUT][1], test_array[TEMP][0], method);
}

void Core_SolveTest::prepare_to_validation( int )
{
    //int rank = test_mat[REF_OUTPUT][0].rows;
    Mat& input = test_mat[INPUT][0];
    Mat& dst = test_mat[OUTPUT][0];
    Mat& dst0 = test_mat[REF_OUTPUT][0];
    
    if( method == CV_LU )
    {
        if( result == 0 )
        {
            Mat& temp1 = test_mat[TEMP][1];
            cvtest::convert(input, temp1, temp1.type());
            dst = Scalar::all(0);
            CvMat _temp1 = temp1;
            double det = cvTsLU( &_temp1, 0, 0 );
            dst0 = Scalar::all(det != 0);
            return;
        }
        
        double threshold = (input.type() == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;
        CvMat _input = input;
        double ratio = 0, det = cvTsSVDet( &_input, &ratio );
        if( det < threshold || ratio < threshold )
        {
            dst = Scalar::all(0);
            dst0 = Scalar::all(0);
            return;
        }
    }
    
    Mat* pdst = input.rows <= input.cols ? &test_mat[OUTPUT][0] : &test_mat[INPUT][1];
    
    cvtest::gemm( input, test_mat[TEMP][0], 1., test_mat[INPUT][1], -1., *pdst, 0 );
    if( pdst != &dst )
        cvtest::gemm( input, *pdst, 1., Mat(), 0., dst, CV_GEMM_A_T );
    dst0 = Scalar::all(0);
}


///////////////// SVD /////////////////////

class Core_SVDTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_SVDTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int flags;
    bool have_u, have_v, symmetric, compact, vector_w;
};


Core_SVDTest::Core_SVDTest() :
Core_MatrixTest( 1, 4, false, false, 1 ),
flags(0), have_u(false), have_v(false), symmetric(false), compact(false), vector_w(false)
{
    test_case_count = 100;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_SVDTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    Core_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size, i, m, n;
    
    min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );
    
    flags = bits & (CV_SVD_MODIFY_A+CV_SVD_U_T+CV_SVD_V_T);
    have_u = (bits & 8) != 0;
    have_v = (bits & 16) != 0;
    symmetric = (bits & 32) != 0;
    compact = (bits & 64) != 0;
    vector_w = (bits & 128) != 0;
    
    if( symmetric )
        sizes[INPUT][0] = Size(min_size, min_size);
    
    m = sizes[INPUT][0].height;
    n = sizes[INPUT][0].width;
    
    if( compact )
        sizes[TEMP][0] = Size(min_size, min_size);
    else
        sizes[TEMP][0] = sizes[INPUT][0];
    sizes[TEMP][3] = Size(0,0);
    
    if( vector_w )
    {
        sizes[TEMP][3] = sizes[TEMP][0];
        if( bits & 256 )
            sizes[TEMP][0] = Size(1, min_size);
        else
            sizes[TEMP][0] = Size(min_size, 1);
    }
    
    if( have_u )
    {
        sizes[TEMP][1] = compact ? Size(min_size, m) : Size(m, m);
        
        if( flags & CV_SVD_U_T )
            CV_SWAP( sizes[TEMP][1].width, sizes[TEMP][1].height, i );
    }
    else
        sizes[TEMP][1] = Size(0,0);
    
    if( have_v )
    {
        sizes[TEMP][2] = compact ? Size(n, min_size) : Size(n, n);
        
        if( !(flags & CV_SVD_V_T) )
            CV_SWAP( sizes[TEMP][2].width, sizes[TEMP][2].height, i );
    }
    else
        sizes[TEMP][2] = Size(0,0);
    
    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[TEMP][3] = types[INPUT][0];
    types[OUTPUT][0] = types[OUTPUT][1] = types[OUTPUT][2] = types[INPUT][0];
    types[OUTPUT][3] = CV_8UC1;
    sizes[OUTPUT][0] = !have_u || !have_v ? Size(0,0) : sizes[INPUT][0];
    sizes[OUTPUT][1] = !have_u ? Size(0,0) : compact ? Size(min_size,min_size) : Size(m,m);
    sizes[OUTPUT][2] = !have_v ? Size(0,0) : compact ? Size(min_size,min_size) : Size(n,n);
    sizes[OUTPUT][3] = Size(min_size,1);
    
    for( i = 0; i < 4; i++ )
    {
        sizes[REF_OUTPUT][i] = sizes[OUTPUT][i];
        types[REF_OUTPUT][i] = types[OUTPUT][i];
    }
}


int Core_SVDTest::prepare_test_case( int test_case_idx )
{
    int code = Core_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        Mat& input = test_mat[INPUT][0];
        cvTsFloodWithZeros( input, ts->get_rng() );
        
        if( symmetric && (have_u || have_v) )
        {
            Mat& temp = test_mat[TEMP][have_u ? 1 : 2];
            cvtest::gemm( input, input, 1., Mat(), 0., temp, CV_GEMM_B_T );
            cvtest::copy( temp, input );
        }
        
        if( (flags & CV_SVD_MODIFY_A) && test_array[OUTPUT][0] )
            cvtest::copy( input, test_mat[OUTPUT][0] );
    }
    
    return code;
}


void Core_SVDTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-2.);
    high = cvScalarAll(2.);
}

double Core_SVDTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int input_depth = CV_MAT_DEPTH(cvGetElemType( test_array[INPUT][0] ));
    double input_precision = input_depth < CV_32F ? 0 : input_depth == CV_32F ? 1e-5 : 5e-11;
    double output_precision = Base::get_success_error_level( test_case_idx, i, j );
    return MAX(input_precision, output_precision);
}

void Core_SVDTest::run_func()
{
    CvArr* src = test_array[!(flags & CV_SVD_MODIFY_A) ? INPUT : OUTPUT][0];
    if( !src )
        src = test_array[INPUT][0];
    cvSVD( src, test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2], flags );
}


void Core_SVDTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& input = test_mat[INPUT][0];
    int depth = input.depth();
    int m = input.rows, n = input.cols, min_size = MIN(m, n);
    Mat *src, *dst, *w;
    double prev = 0, threshold = depth == CV_32F ? FLT_EPSILON : DBL_EPSILON;
    int i, step;
    
    if( have_u )
    {
        src = &test_mat[TEMP][1];
        dst = &test_mat[OUTPUT][1];
        cvtest::gemm( *src, *src, 1., Mat(), 0., *dst, src->rows == dst->rows ? CV_GEMM_B_T : CV_GEMM_A_T );
        cv::setIdentity( test_mat[REF_OUTPUT][1], Scalar::all(1.) );
    }
    
    if( have_v )
    {
        src = &test_mat[TEMP][2];
        dst = &test_mat[OUTPUT][2];
        cvtest::gemm( *src, *src, 1., Mat(), 0., *dst, src->rows == dst->rows ? CV_GEMM_B_T : CV_GEMM_A_T );
        cv::setIdentity( test_mat[REF_OUTPUT][2], Scalar::all(1.) );
    }
    
    w = &test_mat[TEMP][0];
    step = w->rows == 1 ? 1 : (int)w->step1();
    for( i = 0; i < min_size; i++ )
    {
        double normval = 0, aii;
        if( w->rows > 1 && w->cols > 1 )
        {
            normval = cvtest::norm( w->row(i), NORM_L1 );
            aii = depth == CV_32F ? w->at<float>(i,i) : w->at<double>(i,i);
        }
        else
        {
            normval = aii = depth == CV_32F ? w->at<float>(i) : w->at<double>(i);
        }
        
        normval = fabs(normval - aii);
        test_mat[OUTPUT][3].at<uchar>(i) = aii >= 0 && normval < threshold && (i == 0 || aii <= prev);
        prev = aii;
    }
    
    test_mat[REF_OUTPUT][3] = Scalar::all(1);
    
    if( have_u && have_v )
    {
        if( vector_w )
        {
            test_mat[TEMP][3] = Scalar::all(0);
            for( i = 0; i < min_size; i++ )
            {
                double val = depth == CV_32F ? w->at<float>(i) : w->at<double>(i);
                cvSetReal2D( test_array[TEMP][3], i, i, val );
            }
            w = &test_mat[TEMP][3];
        }
        
        if( m >= n )
        {
            cvtest::gemm( test_mat[TEMP][1], *w, 1., Mat(), 0., test_mat[REF_OUTPUT][0],
                     flags & CV_SVD_U_T ? CV_GEMM_A_T : 0 );
            cvtest::gemm( test_mat[REF_OUTPUT][0], test_mat[TEMP][2], 1., Mat(), 0.,
                     test_mat[OUTPUT][0], flags & CV_SVD_V_T ? 0 : CV_GEMM_B_T );
        }
        else
        {
            cvtest::gemm( *w, test_mat[TEMP][2], 1., Mat(), 0., test_mat[REF_OUTPUT][0],
                     flags & CV_SVD_V_T ? 0 : CV_GEMM_B_T );
            cvtest::gemm( test_mat[TEMP][1], test_mat[REF_OUTPUT][0], 1., Mat(), 0.,
                     test_mat[OUTPUT][0], flags & CV_SVD_U_T ? CV_GEMM_A_T : 0 );
        }
        
        cvtest::copy( test_mat[INPUT][0], test_mat[REF_OUTPUT][0] );
    }
}



///////////////// SVBkSb /////////////////////

class Core_SVBkSbTest : public Core_MatrixTest
{
public:
    typedef Core_MatrixTest Base;
    Core_SVBkSbTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int flags;
    bool have_b, symmetric, compact, vector_w;
};


Core_SVBkSbTest::Core_SVBkSbTest() : Core_MatrixTest( 2, 1, false, false, 1 ),
flags(0), have_b(false), symmetric(false), compact(false), vector_w(false)
{
    test_case_count = 100;
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
    test_array[TEMP].push_back(NULL);
}


void Core_SVBkSbTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes,
                                                      vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int bits = cvtest::randInt(rng);
    Base::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size, i, m, n;
    CvSize b_size;
    
    min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );
    
    flags = bits & (CV_SVD_MODIFY_A+CV_SVD_U_T+CV_SVD_V_T);
    have_b = (bits & 16) != 0;
    symmetric = (bits & 32) != 0;
    compact = (bits & 64) != 0;
    vector_w = (bits & 128) != 0;
    
    if( symmetric )
        sizes[INPUT][0] = Size(min_size, min_size);
    
    m = sizes[INPUT][0].height;
    n = sizes[INPUT][0].width;
    
    sizes[INPUT][1] = Size(0,0);
    b_size = Size(m,m);
    if( have_b )
    {
        sizes[INPUT][1].height = sizes[INPUT][0].height;
        sizes[INPUT][1].width = cvtest::randInt(rng) % 100 + 1;
        b_size = sizes[INPUT][1];
    }
    
    if( compact )
        sizes[TEMP][0] = Size(min_size, min_size);
    else
        sizes[TEMP][0] = sizes[INPUT][0];
    
    if( vector_w )
    {
        if( bits & 256 )
            sizes[TEMP][0] = Size(1, min_size);
        else
            sizes[TEMP][0] = Size(min_size, 1);
    }
    
    sizes[TEMP][1] = compact ? Size(min_size, m) : Size(m, m);
    
    if( flags & CV_SVD_U_T )
        CV_SWAP( sizes[TEMP][1].width, sizes[TEMP][1].height, i );
    
    sizes[TEMP][2] = compact ? Size(n, min_size) : Size(n, n);
    
    if( !(flags & CV_SVD_V_T) )
        CV_SWAP( sizes[TEMP][2].width, sizes[TEMP][2].height, i );
    
    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[INPUT][0];
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0];
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size( b_size.width, n );
}


int Core_SVBkSbTest::prepare_test_case( int test_case_idx )
{
    int code = Base::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        Mat& input = test_mat[INPUT][0];
        cvTsFloodWithZeros( input, ts->get_rng() );
        
        if( symmetric )
        {
            Mat& temp = test_mat[TEMP][1];
            cvtest::gemm( input, input, 1., Mat(), 0., temp, CV_GEMM_B_T );
            cvtest::copy( temp, input );
        }
        
        CvMat _input = input;
        cvSVD( &_input, test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2], flags );
    }
    
    return code;
}


void Core_SVBkSbTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, Scalar& low, Scalar& high )
{
    low = cvScalarAll(-2.);
    high = cvScalarAll(2.);
}


double Core_SVBkSbTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0])) == CV_32F ? 1e-3 : 1e-7;
}


void Core_SVBkSbTest::run_func()
{
    cvSVBkSb( test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2],
             test_array[INPUT][1], test_array[OUTPUT][0], flags );
}


void Core_SVBkSbTest::prepare_to_validation( int )
{
    Mat& input = test_mat[INPUT][0];
    int i, m = input.rows, n = input.cols, min_size = MIN(m, n);
    bool is_float = input.type() == CV_32F;
    Size w_size = compact ? Size(min_size,min_size) : Size(m,n);
    Mat& w = test_mat[TEMP][0];
    Mat wdb( w_size.height, w_size.width, CV_64FC1 );
    CvMat _w = w, _wdb = wdb;
    // use exactly the same threshold as in icvSVD... ,
    // so the changes in the library and here should be synchronized.
    double threshold = cv::sum(w)[0]*(DBL_EPSILON*2);//(is_float ? FLT_EPSILON*10 : DBL_EPSILON*2);
    
    wdb = Scalar::all(0);
    for( i = 0; i < min_size; i++ )
    {
        double wii = vector_w ? cvGetReal1D(&_w,i) : cvGetReal2D(&_w,i,i);
        cvSetReal2D( &_wdb, i, i, wii > threshold ? 1./wii : 0. );
    }
    
    Mat u = test_mat[TEMP][1];
    Mat v = test_mat[TEMP][2];
    Mat b = test_mat[INPUT][1];
    
    if( is_float )
    {
        test_mat[TEMP][1].convertTo(u, CV_64F);
        test_mat[TEMP][2].convertTo(v, CV_64F);
        if( !b.empty() )
            test_mat[INPUT][1].convertTo(b, CV_64F);
    }
    
    Mat t0, t1;
    
    if( !b.empty() )
        cvtest::gemm( u, b, 1., Mat(), 0., t0, !(flags & CV_SVD_U_T) ? CV_GEMM_A_T : 0 );
    else if( flags & CV_SVD_U_T )
        cvtest::copy( u, t0 );
    else
        cvtest::transpose( u, t0 );
    
    cvtest::gemm( wdb, t0, 1, Mat(), 0, t1, 0 );
    
    cvtest::gemm( v, t1, 1, Mat(), 0, t0, flags & CV_SVD_V_T ? CV_GEMM_A_T : 0 );
    Mat& dst0 = test_mat[REF_OUTPUT][0];
    t0.convertTo(dst0, dst0.type() );
}


typedef std::complex<double> complex_type;

struct pred_complex
{
    bool operator() (const complex_type& lhs, const complex_type& rhs) const
    {
        return fabs(lhs.real() - rhs.real()) > fabs(rhs.real())*FLT_EPSILON ? lhs.real() < rhs.real() : lhs.imag() < rhs.imag();
    }
};

struct pred_double
{
    bool operator() (const double& lhs, const double& rhs) const
    {
        return lhs < rhs;
    }
};

class Core_SolvePolyTest : public cvtest::BaseTest
{
public:
    Core_SolvePolyTest();
    ~Core_SolvePolyTest();
protected:
    virtual void run( int start_from );
};

Core_SolvePolyTest::Core_SolvePolyTest() {}

Core_SolvePolyTest::~Core_SolvePolyTest() {}

void Core_SolvePolyTest::run( int )
{
    RNG& rng = ts->get_rng();
    int fig = 100;
    double range = 50;
    double err_eps = 1e-4;
    
    for (int idx = 0, max_idx = 1000, progress = 0; idx < max_idx; ++idx)
    {
        progress = update_progress(progress, idx-1, max_idx, 0);
        int n = cvtest::randInt(rng) % 13 + 1;
        std::vector<complex_type> r(n), ar(n), c(n + 1, 0);
        std::vector<double> a(n + 1), u(n * 2), ar1(n), ar2(n);
        
        int rr_odds = 3; // odds that we get a real root
        for (int j = 0; j < n;)
        {
            if (cvtest::randInt(rng) % rr_odds == 0 || j == n - 1)
	            r[j++] = cvtest::randReal(rng) * range;
            else
            {
	            r[j] = complex_type(cvtest::randReal(rng) * range,
                                    cvtest::randReal(rng) * range + 1);
	            r[j + 1] = std::conj(r[j]);
	            j += 2;
            }
        }
        
        for (int j = 0, k = 1 << n, jj, kk; j < k; ++j)
        {
            int p = 0;
            complex_type v(1);
            for (jj = 0, kk = 1; jj < n && !(j & kk); ++jj, ++p, kk <<= 1)
                ;
            for (; jj < n; ++jj, kk <<= 1)
            {
	            if (j & kk)
	                v *= -r[jj];
	            else
	                ++p;
            }
            c[p] += v;
        }
        
        bool pass = false;
        double div = 0, s = 0;
        int cubic_case = idx & 1;
        for (int maxiter = 100; !pass && maxiter < 10000; maxiter *= 2, cubic_case = (cubic_case + 1) % 2)
        {
            for (int j = 0; j < n + 1; ++j)
	            a[j] = c[j].real();
            
            CvMat amat, umat;
            cvInitMatHeader(&amat, n + 1, 1, CV_64FC1, &a[0]);
            cvInitMatHeader(&umat, n, 1, CV_64FC2, &u[0]);
            cvSolvePoly(&amat, &umat, maxiter, fig);
            
            for (int j = 0; j < n; ++j)
	            ar[j] = complex_type(u[j * 2], u[j * 2 + 1]);
            
            std::sort(r.begin(), r.end(), pred_complex());
            std::sort(ar.begin(), ar.end(), pred_complex());
            
            pass = true;
            if( n == 3 )
            {
                ar2.resize(n);
                cv::Mat _umat2(3, 1, CV_64F, &ar2[0]), umat2 = _umat2;
                cvFlip(&amat, &amat, 0);
                int nr2;
                if( cubic_case == 0 )
                    nr2 = cv::solveCubic(cv::Mat(&amat),umat2);
                else
                    nr2 = cv::solveCubic(cv::Mat_<float>(cv::Mat(&amat)), umat2);
                cvFlip(&amat, &amat, 0);
                if(nr2 > 0)
                    std::sort(ar2.begin(), ar2.begin()+nr2, pred_double());
                ar2.resize(nr2);
                
                int nr1 = 0;
                for(int j = 0; j < n; j++)
                    if( fabs(r[j].imag()) < DBL_EPSILON )
                        ar1[nr1++] = r[j].real();
                
                pass = pass && nr1 == nr2;
                if( nr2 > 0 )
                {
                    div = s = 0;
                    for(int j = 0; j < nr1; j++)
                    {
                        s += fabs(ar1[j]);
                        div += fabs(ar1[j] - ar2[j]);
                    }
                    div /= s;
                    pass = pass && div < err_eps;
                }
            }
            
            div = s = 0;
            for (int j = 0; j < n; ++j)
            {
                s += fabs(r[j].real()) + fabs(r[j].imag());
                div += sqrt(pow(r[j].real() - ar[j].real(), 2) + pow(r[j].imag() - ar[j].imag(), 2));
            }
            div /= s;
            pass = pass && div < err_eps;
        }
        
        if (!pass)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            ts->printf( cvtest::TS::LOG, "too big diff = %g\n", div );
            
            for (size_t j=0;j<ar2.size();++j)
                ts->printf( cvtest::TS::LOG, "ar2[%d]=%g\n", j, ar2[j]);
            ts->printf(cvtest::TS::LOG, "\n");
            
            for (size_t j=0;j<r.size();++j)
	            ts->printf( cvtest::TS::LOG, "r[%d]=(%g, %g)\n", j, r[j].real(), r[j].imag());
            ts->printf( cvtest::TS::LOG, "\n" );
            for (size_t j=0;j<ar.size();++j)
	            ts->printf( cvtest::TS::LOG, "ar[%d]=(%g, %g)\n", j, ar[j].real(), ar[j].imag());
            break;
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Core_CovarMatrix, accuracy) { Core_CovarMatrixTest test; test.safe_run(); }
TEST(Core_CrossProduct, accuracy) { Core_CrossProductTest test; test.safe_run(); }
TEST(Core_Determinant, accuracy) { Core_DetTest test; test.safe_run(); }
TEST(Core_DotProduct, accuracy) { Core_DotProductTest test; test.safe_run(); }
TEST(Core_GEMM, accuracy) { Core_GEMMTest test; test.safe_run(); }
TEST(Core_Invert, accuracy) { Core_InvertTest test; test.safe_run(); }
TEST(Core_Mahalanobis, accuracy) { Core_MahalanobisTest test; test.safe_run(); }
TEST(Core_MulTransposed, accuracy) { Core_MulTransposedTest test; test.safe_run(); }
TEST(Core_Transform, accuracy) { Core_TransformTest test; test.safe_run(); }
TEST(Core_PerspectiveTransform, accuracy) { Core_PerspectiveTransformTest test; test.safe_run(); }
TEST(Core_Pow, accuracy) { Core_PowTest test; test.safe_run(); }
TEST(Core_SolveLinearSystem, accuracy) { Core_SolveTest test; test.safe_run(); }
TEST(Core_SVD, accuracy) { Core_SVDTest test; test.safe_run(); }
TEST(Core_SVBkSb, accuracy) { Core_SVBkSbTest test; test.safe_run(); }
TEST(Core_Trace, accuracy) { Core_TraceTest test; test.safe_run(); }
TEST(Core_SolvePoly, accuracy) { Core_SolvePolyTest test; test.safe_run(); }

// TODO: eigenvv, invsqrt, cbrt, fastarctan, (round, floor, ceil(?)),

/* End of file. */

