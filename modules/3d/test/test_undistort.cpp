/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

class CV_DefaultNewCameraMatrixTest : public cvtest::ArrayTest
{
public:
    CV_DefaultNewCameraMatrixTest();
protected:
    int prepare_test_case (int test_case_idx);
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void run_func();

private:
    cv::Size img_size;
    cv::Mat camera_mat;
    cv::Mat new_camera_mat;

    int matrix_type;

    bool center_principal_point;

    static const int MAX_X = 2048;
    static const int MAX_Y = 2048;
    //static const int MAX_VAL = 10000;
};

CV_DefaultNewCameraMatrixTest::CV_DefaultNewCameraMatrixTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);

    matrix_type = 0;
    center_principal_point = false;
}

void CV_DefaultNewCameraMatrixTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx,sizes,types);
    RNG& rng = ts->get_rng();
    matrix_type = types[INPUT][0] = types[OUTPUT][0]= types[REF_OUTPUT][0] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(3,3);
}

int CV_DefaultNewCameraMatrixTest::prepare_test_case(int test_case_idx)
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );

    if (code <= 0)
        return code;

    RNG& rng = ts->get_rng();

    img_size.width = cvtest::randInt(rng) % MAX_X + 1;
    img_size.height = cvtest::randInt(rng) % MAX_Y + 1;

    center_principal_point = ((cvtest::randInt(rng) % 2)!=0);

    // Generating camera_mat matrix
    double sz = MAX(img_size.width, img_size.height);
    double aspect_ratio = cvtest::randReal(rng)*0.6 + 0.7;
    double a[9] = {0,0,0,0,0,0,0,0,1};
    Mat _a(3,3,CV_64F,a);
    a[2] = (img_size.width - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[5] = (img_size.height - 1)*0.5 + cvtest::randReal(rng)*10 - 5;
    a[0] = sz/(0.9 - cvtest::randReal(rng)*0.6);
    a[4] = aspect_ratio*a[0];

    Mat& _a0 = test_mat[INPUT][0];
    cvtest::convert(_a, _a0, _a0.type());
    camera_mat = _a0;

    return code;

}

void CV_DefaultNewCameraMatrixTest::run_func()
{
    new_camera_mat = cv::getDefaultNewCameraMatrix(camera_mat,img_size,center_principal_point);
}

void CV_DefaultNewCameraMatrixTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];
    Mat& test_output = test_mat[OUTPUT][0];
    Mat& output = new_camera_mat;
    cvtest::convert( output, test_output, test_output.type() );
    if (!center_principal_point)
    {
        cvtest::copy(src, dst);
    }
    else
    {
        double a[9] = {0,0,0,0,0,0,0,0,1};
        Mat _a(3,3,CV_64F,a);
        if (matrix_type == CV_64F)
        {
            a[0] = src.at<double>(0,0);
            a[4] = src.at<double>(1,1);
        }
        else
        {
            a[0] = src.at<float>(0,0);
            a[4] = src.at<float>(1,1);
        }
        a[2] = (img_size.width - 1)*0.5;
        a[5] = (img_size.height - 1)*0.5;
        cvtest::convert( _a, dst, dst.type() );
    }
}

//---------

class CV_GetOptimalNewCameraMatrixNoDistortionTest : public cvtest::ArrayTest
{
public:
    CV_GetOptimalNewCameraMatrixNoDistortionTest();
protected:
    int prepare_test_case (int test_case_idx);
    void prepare_to_validation(int test_case_idx);
    void get_test_array_types_and_sizes(int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types);
    void run_func();

private:
    cv::Mat camera_mat;
    cv::Mat distortion_coeffs;
    cv::Mat new_camera_mat;

    cv::Size img_size;
    double alpha;
    bool center_principal_point;

    int matrix_type;

    static const int MAX_X = 2000;
    static const int MAX_Y = 2000;
};

CV_GetOptimalNewCameraMatrixNoDistortionTest::CV_GetOptimalNewCameraMatrixNoDistortionTest()
{
    test_array[INPUT].push_back(NULL); // camera_mat
    test_array[INPUT].push_back(NULL); // distortion_coeffs
    test_array[OUTPUT].push_back(NULL); // new_camera_mat
    test_array[REF_OUTPUT].push_back(NULL);

    alpha = 0.0;
    center_principal_point = false;
    matrix_type = 0;
}

void CV_GetOptimalNewCameraMatrixNoDistortionTest::get_test_array_types_and_sizes(int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types)
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx, sizes, types);
    RNG& rng = ts->get_rng();
    matrix_type = types[INPUT][0] = types[INPUT][1] = types[OUTPUT][0] = types[REF_OUTPUT][0] = cvtest::randInt(rng)%2 ? CV_64F : CV_32F;
    sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = Size(3,3);
    sizes[INPUT][1] = Size(1,4);
}

int CV_GetOptimalNewCameraMatrixNoDistortionTest::prepare_test_case(int test_case_idx)
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );

    if (code <= 0)
        return code;

    RNG& rng = ts->get_rng();

    alpha = cvtest::randReal(rng);
    center_principal_point = ((cvtest::randInt(rng) % 2)!=0);

    // Generate random camera matrix. Use floating point precision for source to avoid precision loss
    img_size.width = cvtest::randInt(rng) % MAX_X + 1;
    img_size.height = cvtest::randInt(rng) % MAX_Y + 1;
    const float aspect_ratio = static_cast<float>(img_size.width) / img_size.height;
    float cam_array[9] = {0,0,0,0,0,0,0,0,1};
    cam_array[2] = static_cast<float>((img_size.width - 1)*0.5);  // center
    cam_array[5] = static_cast<float>((img_size.height - 1)*0.5); // center
    cam_array[0] = static_cast<float>(MAX(img_size.width, img_size.height)/(0.9 - cvtest::randReal(rng)*0.6));
    cam_array[4] = aspect_ratio*cam_array[0];

    Mat& input_camera_mat = test_mat[INPUT][0];
    cvtest::convert(Mat(3, 3, CV_32F, cam_array), input_camera_mat, input_camera_mat.type());
    camera_mat = input_camera_mat;

    // Generate zero distortion matrix
    const Mat zero_dist_coeffs = Mat::zeros(1, 4, CV_32F);
    Mat& input_dist_coeffs = test_mat[INPUT][1];
    cvtest::convert(zero_dist_coeffs, input_dist_coeffs, input_dist_coeffs.type());
    distortion_coeffs = input_dist_coeffs;

    return code;
}

void CV_GetOptimalNewCameraMatrixNoDistortionTest::run_func()
{
    new_camera_mat = cv::getOptimalNewCameraMatrix(camera_mat, distortion_coeffs, img_size, alpha, img_size, NULL, center_principal_point);
}

void CV_GetOptimalNewCameraMatrixNoDistortionTest::prepare_to_validation(int /*test_case_idx*/)
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];
    cvtest::copy(src, dst);

    Mat& output = test_mat[OUTPUT][0];
    cvtest::convert(new_camera_mat, output, output.type());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_DefaultNewCameraMatrix, accuracy) { CV_DefaultNewCameraMatrixTest test; test.safe_run(); }
TEST(Calib3d_GetOptimalNewCameraMatrixNoDistortion, accuracy) { CV_GetOptimalNewCameraMatrixNoDistortionTest test; test.safe_run(); }

////////////////////////////// undistort /////////////////////////////////

static void test_remap( const Mat& src, Mat& dst, const Mat& mapx, const Mat& mapy,
                        Mat* mask=0, int interpolation=cv::INTER_LINEAR )
{
    int x, y, k;
    int drows = dst.rows, dcols = dst.cols;
    int srows = src.rows, scols = src.cols;
    const uchar* sptr0 = src.ptr();
    int depth = src.depth(), cn = src.channels();
    int elem_size = (int)src.elemSize();
    int step = (int)(src.step / CV_ELEM_SIZE(depth));
    int delta;

    if( interpolation != cv::INTER_CUBIC )
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
                CV_Assert(0);
            }
        }
    }
}

class CV_ImgWarpBaseTest : public cvtest::ArrayTest
{
public:
    CV_ImgWarpBaseTest( bool warp_matrix );

protected:
    int read_params( const cv::FileStorage& fs );
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


int CV_ImgWarpBaseTest::read_params( const cv::FileStorage& fs )
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
            CV_Assert(0);
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
            CV_Assert(0);
        }*/
        cv::Mat src(1, cols*cn, CV_32F, &buffer[0]);
        cv::Mat dst(1, cols*cn, depth, ptr);
        src.convertTo(dst, dst.type());
    }

    return code;
}


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
    sizes[INPUT][1] = Size(3,3);
    sizes[INPUT][2] = cvtest::randInt(rng)%2 ? Size(4,1) : Size(1,4);
    types[INPUT][3] =  types[INPUT][1];
    sizes[INPUT][3] = sizes[INPUT][1];
    interpolation = cv::INTER_LINEAR;
}


void CV_UndistortTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    if( i != INPUT )
        CV_ImgWarpBaseTest::fill_array( test_case_idx, i, j, arr );
}


void CV_UndistortTest::run_func()
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
    //useCPlus = ((cvtest::randInt(rng) % 2)!=0);
    input0 = test_mat[INPUT][0];
    input1 = test_mat[INPUT][1];
    input2 = test_mat[INPUT][2];
    input_new_cam = test_mat[INPUT][3];

    return code;
}


void CV_UndistortTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat& output = test_mat[INPUT_OUTPUT][0];
    input_output.convertTo(output, output.type());
    Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_INPUT_OUTPUT][0];
    Mat& dst0 = test_mat[INPUT_OUTPUT][0];
    Mat mapx, mapy;
    cvtest::initUndistortMap( test_mat[INPUT][1], test_mat[INPUT][2],
                              Mat(), Mat(), dst.size(), mapx, mapy, CV_32F );
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

    Size sz = sizes[OUTPUT][0];
    types[INPUT][0] = types[INPUT][1] = depth;
    dualChannel = cvtest::randInt(rng)%2 == 0;
    types[OUTPUT][0] = types[OUTPUT][1] =
        types[REF_OUTPUT][0] = types[REF_OUTPUT][1] = dualChannel ? CV_32FC2 : CV_32F;
    sizes[INPUT][0] = Size(3,3);
    sizes[INPUT][1] = cvtest::randInt(rng)%2 ? Size(4,1) : Size(1,4);

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
    cv::Mat a = test_mat[INPUT][0], k = test_mat[INPUT][1];
    cv::Mat &mapx = test_mat[OUTPUT][0], &mapy = !dualChannel ? test_mat[OUTPUT][1] : mapx;
    cv::Size mapsz = test_mat[OUTPUT][0].size();

    cv::initUndistortRectifyMap(a, k, cv::Mat(), a,
        mapsz, dualChannel ? CV_32FC2 : CV_32FC1,
        mapx, !dualChannel ? cv::_InputOutputArray(mapy) : cv::noArray());
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
    cvtest::initUndistortMap( test_mat[INPUT][0], test_mat[INPUT][1], Mat(), Mat(),
                              test_mat[REF_OUTPUT][0].size(), mapx, mapy, CV_32F );
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

TEST(Calib3d_UndistortImgproc, accuracy) { CV_UndistortTest test; test.safe_run(); }
TEST(Calib3d_InitUndistortMap, accuracy) { CV_UndistortMapTest test; test.safe_run(); }

TEST(Calib3d_UndistortPoints, inputShape)
{
    //https://github.com/opencv/opencv/issues/14423
    Matx33d cameraMatrix = Matx33d::eye();
    {
        //2xN 1-channel
        Mat imagePoints(2, 3, CV_32FC1);
        imagePoints.at<float>(0,0) = 320; imagePoints.at<float>(1,0) = 240;
        imagePoints.at<float>(0,1) = 0;   imagePoints.at<float>(1,1) = 240;
        imagePoints.at<float>(0,2) = 320; imagePoints.at<float>(1,2) = 0;

        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(normalized.size()), imagePoints.cols);
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints.at<float>(0,i), std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints.at<float>(1,i), std::numeric_limits<float>::epsilon());
        }
    }
    {
        //Nx2 1-channel
        Mat imagePoints(3, 2, CV_32FC1);
        imagePoints.at<float>(0,0) = 320; imagePoints.at<float>(0,1) = 240;
        imagePoints.at<float>(1,0) = 0;   imagePoints.at<float>(1,1) = 240;
        imagePoints.at<float>(2,0) = 320; imagePoints.at<float>(2,1) = 0;

        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(normalized.size()), imagePoints.rows);
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints.at<float>(i,0), std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints.at<float>(i,1), std::numeric_limits<float>::epsilon());
        }
    }
    {
        //1xN 2-channel
        Mat imagePoints(1, 3, CV_32FC2);
        imagePoints.at<Vec2f>(0,0) = Vec2f(320, 240);
        imagePoints.at<Vec2f>(0,1) = Vec2f(0, 240);
        imagePoints.at<Vec2f>(0,2) = Vec2f(320, 0);

        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(normalized.size()), imagePoints.cols);
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints.at<Vec2f>(0,i)(0), std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints.at<Vec2f>(0,i)(1), std::numeric_limits<float>::epsilon());
        }
    }
    {
        //Nx1 2-channel
        Mat imagePoints(3, 1, CV_32FC2);
        imagePoints.at<Vec2f>(0,0) = Vec2f(320, 240);
        imagePoints.at<Vec2f>(1,0) = Vec2f(0, 240);
        imagePoints.at<Vec2f>(2,0) = Vec2f(320, 0);

        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(normalized.size()), imagePoints.rows);
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints.at<Vec2f>(i,0)(0), std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints.at<Vec2f>(i,0)(1), std::numeric_limits<float>::epsilon());
        }
    }
    {
        //vector<Point2f>
        vector<Point2f> imagePoints;
        imagePoints.push_back(Point2f(320, 240));
        imagePoints.push_back(Point2f(0,   240));
        imagePoints.push_back(Point2f(320, 0));

        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(normalized.size(), imagePoints.size());
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints[i].x, std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints[i].y, std::numeric_limits<float>::epsilon());
        }
    }
    {
        //vector<Point2d>
        vector<Point2d> imagePoints;
        imagePoints.push_back(Point2d(320, 240));
        imagePoints.push_back(Point2d(0,   240));
        imagePoints.push_back(Point2d(320, 0));

        vector<Point2d> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(normalized.size(), imagePoints.size());
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints[i].x, std::numeric_limits<double>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints[i].y, std::numeric_limits<double>::epsilon());
        }
    }
}

TEST(Calib3d_UndistortPoints, outputShape)
{
    Matx33d cameraMatrix = Matx33d::eye();
    {
        vector<Point2f> imagePoints;
        imagePoints.push_back(Point2f(320, 240));
        imagePoints.push_back(Point2f(0,   240));
        imagePoints.push_back(Point2f(320, 0));

        //Mat --> will be Nx1 2-channel
        Mat normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(imagePoints.size()), normalized.rows);
        for (int i = 0; i < normalized.rows; i++) {
            EXPECT_NEAR(normalized.at<Vec2f>(i,0)(0), imagePoints[i].x, std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized.at<Vec2f>(i,0)(1), imagePoints[i].y, std::numeric_limits<float>::epsilon());
        }
    }
    {
        vector<Point2f> imagePoints;
        imagePoints.push_back(Point2f(320, 240));
        imagePoints.push_back(Point2f(0,   240));
        imagePoints.push_back(Point2f(320, 0));

        //Nx1 2-channel
        Mat normalized(static_cast<int>(imagePoints.size()), 1, CV_32FC2);
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(imagePoints.size()), normalized.rows);
        for (int i = 0; i < normalized.rows; i++) {
            EXPECT_NEAR(normalized.at<Vec2f>(i,0)(0), imagePoints[i].x, std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized.at<Vec2f>(i,0)(1), imagePoints[i].y, std::numeric_limits<float>::epsilon());
        }
    }
    {
        vector<Point2f> imagePoints;
        imagePoints.push_back(Point2f(320, 240));
        imagePoints.push_back(Point2f(0,   240));
        imagePoints.push_back(Point2f(320, 0));

        //1xN 2-channel
        Mat normalized(1, static_cast<int>(imagePoints.size()), CV_32FC2);
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(static_cast<int>(imagePoints.size()), normalized.cols);
        for (int i = 0; i < normalized.rows; i++) {
            EXPECT_NEAR(normalized.at<Vec2f>(0,i)(0), imagePoints[i].x, std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized.at<Vec2f>(0,i)(1), imagePoints[i].y, std::numeric_limits<float>::epsilon());
        }
    }
    {
        vector<Point2f> imagePoints;
        imagePoints.push_back(Point2f(320, 240));
        imagePoints.push_back(Point2f(0,   240));
        imagePoints.push_back(Point2f(320, 0));

        //vector<Point2f>
        vector<Point2f> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(imagePoints.size(), normalized.size());
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints[i].x, std::numeric_limits<float>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints[i].y, std::numeric_limits<float>::epsilon());
        }
    }
    {
        vector<Point2d> imagePoints;
        imagePoints.push_back(Point2d(320, 240));
        imagePoints.push_back(Point2d(0,   240));
        imagePoints.push_back(Point2d(320, 0));

        //vector<Point2d>
        vector<Point2d> normalized;
        undistortPoints(imagePoints, normalized, cameraMatrix, noArray());
        EXPECT_EQ(imagePoints.size(), normalized.size());
        for (int i = 0; i < static_cast<int>(normalized.size()); i++) {
            EXPECT_NEAR(normalized[i].x, imagePoints[i].x, std::numeric_limits<double>::epsilon());
            EXPECT_NEAR(normalized[i].y, imagePoints[i].y, std::numeric_limits<double>::epsilon());
        }
    }
}

TEST(Imgproc_undistort, regression_15286)
{
    double kmat_data[9] = { 3217, 0, 1592, 0, 3217, 1201, 0, 0, 1 };
    Mat kmat(3, 3, CV_64F, kmat_data);
    double dist_coeff_data[5] = { 0.04, -0.4, -0.01, 0.04, 0.7 };
    Mat dist_coeffs(5, 1, CV_64F, dist_coeff_data);

    Mat img = Mat::zeros(512, 512, CV_8UC1);
    img.at<uchar>(128, 128) = 255;
    img.at<uchar>(128, 384) = 255;
    img.at<uchar>(384, 384) = 255;
    img.at<uchar>(384, 128) = 255;

    Mat ref = Mat::zeros(512, 512, CV_8UC1);
    ref.at<uchar>(Point(24, 98)) = 78;
    ref.at<uchar>(Point(24, 99)) = 114;
    ref.at<uchar>(Point(25, 98)) = 36;
    ref.at<uchar>(Point(25, 99)) = 60;
    ref.at<uchar>(Point(27, 361)) = 6;
    ref.at<uchar>(Point(28, 361)) = 188;
    ref.at<uchar>(Point(28, 362)) = 49;
    ref.at<uchar>(Point(29, 361)) = 44;
    ref.at<uchar>(Point(29, 362)) = 16;
    ref.at<uchar>(Point(317, 366)) = 134;
    ref.at<uchar>(Point(317, 367)) = 78;
    ref.at<uchar>(Point(318, 366)) = 40;
    ref.at<uchar>(Point(318, 367)) = 29;
    ref.at<uchar>(Point(310, 104)) = 106;
    ref.at<uchar>(Point(310, 105)) = 30;
    ref.at<uchar>(Point(311, 104)) = 112;
    ref.at<uchar>(Point(311, 105)) = 38;

    Mat img_undist;
    undistort(img, img_undist, kmat, dist_coeffs);

    ASSERT_EQ(0.0, cvtest::norm(img_undist, ref, cv::NORM_INF));
}

TEST(Calib3d_initUndistortRectifyMap, regression_14467)
{
    Size size_w_h(512 + 3, 512);
    Matx33f k(
        6200, 0, size_w_h.width / 2.0f,
        0, 6200, size_w_h.height / 2.0f,
        0, 0, 1
    );

    Mat mesh_uv(size_w_h, CV_32FC2);
    for (int i = 0; i < size_w_h.height; i++)
    {
        for (int j = 0; j < size_w_h.width; j++)
        {
            mesh_uv.at<Vec2f>(i, j) = Vec2f((float)j, (float)i);
        }
    }

    Matx<double, 1, 14> d(
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0.09, 0.0
    );
    Mat mapxy, dst;
    initUndistortRectifyMap(k, d, noArray(), k, size_w_h, CV_32FC2, mapxy, noArray());
    undistortPoints(mapxy.reshape(2, (int)mapxy.total()), dst, k, d, noArray(), k);
    dst = dst.reshape(2, mapxy.rows);
    EXPECT_LE(cvtest::norm(dst, mesh_uv, NORM_INF), 1e-3);
}

TEST(Calib3d_initInverseRectificationMap, regression_20165)
{
    Size size_w_h(1280, 800);
    Mat dst(size_w_h, CV_32FC2); // Reference for validation
    Mat mapxy; // Output of initInverseRectificationMap()

    // Camera Matrix
    double k[9]={
        1.5393951443032472e+03, 0., 6.7491727003047140e+02,
        0., 1.5400748240626747e+03, 5.1226968329123963e+02,
        0., 0., 1.
    };
    Mat _K(3, 3, CV_64F, k);

    // Distortion
    // double d[5]={0,0,0,0,0}; // Zero Distortion
    double d[5]={ // Non-zero distortion
        -3.4134571357400023e-03, 2.9733267766101856e-03, // K1, K2
        3.6653586399031184e-03, -3.1960714017365702e-03, // P1, P2
        0. // K3
    };
    Mat _d(1, 5, CV_64F, d);

    // Rotation
    //double R[9]={1., 0., 0., 0., 1., 0., 0., 0., 1.}; // Identity transform (none)
    double R[9]={ // Random transform
        9.6625486010428052e-01, 1.6055789378989216e-02, 2.5708706103628531e-01,
        -8.0300261706161002e-03, 9.9944797497929860e-01, -3.2237617614807819e-02,
       -2.5746274294459848e-01, 2.9085338870243265e-02, 9.6585039165403186e-01
    };
    Mat _R(3, 3, CV_64F, R);

    // --- Validation --- //
    initInverseRectificationMap(_K, _d, _R, _K, size_w_h, CV_32FC2, mapxy, noArray());

    // Copy camera matrix
    double fx, fy, cx, cy, ifx, ify, cxn, cyn;
    fx = k[0]; fy = k[4]; cx = k[2]; cy = k[5];

    // Copy new camera matrix
    ifx = k[0]; ify = k[4]; cxn = k[2]; cyn = k[5];

    // Distort Points
    for( int v = 0; v < size_w_h.height; v++ )
    {
        for( int u = 0; u < size_w_h.width; u++ )
        {
            // Convert from image to pin-hole coordinates
            double x = (u - cx)/fx;
            double y = (v - cy)/fy;

            // Undistort
            double x2 = x*x, y2 = y*y;
            double r2 = x2 + y2;
            double cdist = 1./(1. + (d[0] + (d[1] + d[4]*r2)*r2)*r2); // (1. + (d[5] + (d[6] + d[7]*r2)*r2)*r2) == 1 as d[5-7]=0;
            double x_ = (x - (d[2]*2.*x*y + d[3]*(r2 + 2.*x2)))*cdist;
            double y_ = (y - (d[3]*2.*x*y + d[2]*(r2 + 2.*y2)))*cdist;

            // Rectify
            double X = R[0]*x_ + R[1]*y_ + R[2];
            double Y = R[3]*x_ + R[4]*y_ + R[5];
            double Z = R[6]*x_ + R[7]*y_ + R[8];
            double x__ = X/Z;
            double y__ = Y/Z;

            // Convert from pin-hole to image coordinates
            dst.at<Vec2f>(v, u) = Vec2f((float)(x__*ifx + cxn), (float)(y__*ify + cyn));
        }
    }

    // Check Result
    EXPECT_LE(cvtest::norm(dst, mapxy, NORM_INF), 2e-1);
}

}} // namespace
