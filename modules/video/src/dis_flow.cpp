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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_video.hpp"

using namespace std;
#define EPS 0.001F
#define INF 1E+10F

namespace cv
{

class DISOpticalFlowImpl CV_FINAL : public DISOpticalFlow
{
  public:
    DISOpticalFlowImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;

  protected: //!< algorithm parameters
    int finest_scale, coarsest_scale;
    int patch_size;
    int patch_stride;
    int grad_descent_iter;
    int variational_refinement_iter;
    float variational_refinement_alpha;
    float variational_refinement_gamma;
    float variational_refinement_delta;
    bool use_mean_normalization;
    bool use_spatial_propagation;

  protected: //!< some auxiliary variables
    int border_size;
    int w, h;   //!< flow buffer width and height on the current scale
    int ws, hs; //!< sparse flow buffer width and height on the current scale

  public:
    int getFinestScale() const CV_OVERRIDE { return finest_scale; }
    void setFinestScale(int val) CV_OVERRIDE { finest_scale = val; }
    int getPatchSize() const CV_OVERRIDE { return patch_size; }
    void setPatchSize(int val) CV_OVERRIDE { patch_size = val; }
    int getPatchStride() const CV_OVERRIDE { return patch_stride; }
    void setPatchStride(int val) CV_OVERRIDE { patch_stride = val; }
    int getGradientDescentIterations() const CV_OVERRIDE { return grad_descent_iter; }
    void setGradientDescentIterations(int val) CV_OVERRIDE { grad_descent_iter = val; }
    int getVariationalRefinementIterations() const CV_OVERRIDE { return variational_refinement_iter; }
    void setVariationalRefinementIterations(int val) CV_OVERRIDE { variational_refinement_iter = val; }
    float getVariationalRefinementAlpha() const CV_OVERRIDE { return variational_refinement_alpha; }
    void setVariationalRefinementAlpha(float val) CV_OVERRIDE { variational_refinement_alpha = val; }
    float getVariationalRefinementDelta() const CV_OVERRIDE { return variational_refinement_delta; }
    void setVariationalRefinementDelta(float val) CV_OVERRIDE { variational_refinement_delta = val; }
    float getVariationalRefinementGamma() const CV_OVERRIDE { return variational_refinement_gamma; }
    void setVariationalRefinementGamma(float val) CV_OVERRIDE { variational_refinement_gamma = val; }

    bool getUseMeanNormalization() const CV_OVERRIDE { return use_mean_normalization; }
    void setUseMeanNormalization(bool val) CV_OVERRIDE { use_mean_normalization = val; }
    bool getUseSpatialPropagation() const CV_OVERRIDE { return use_spatial_propagation; }
    void setUseSpatialPropagation(bool val) CV_OVERRIDE { use_spatial_propagation = val; }

  protected:                      //!< internal buffers
    vector<Mat_<uchar> > I0s;     //!< Gaussian pyramid for the current frame
    vector<Mat_<uchar> > I1s;     //!< Gaussian pyramid for the next frame
    vector<Mat_<uchar> > I1s_ext; //!< I1s with borders

    vector<Mat_<short> > I0xs; //!< Gaussian pyramid for the x gradient of the current frame
    vector<Mat_<short> > I0ys; //!< Gaussian pyramid for the y gradient of the current frame

    vector<Mat_<float> > Ux; //!< x component of the flow vectors
    vector<Mat_<float> > Uy; //!< y component of the flow vectors

    vector<Mat_<float> > initial_Ux; //!< x component of the initial flow field, if one was passed as an input
    vector<Mat_<float> > initial_Uy; //!< y component of the initial flow field, if one was passed as an input

    Mat_<Vec2f> U; //!< a buffer for the merged flow

    Mat_<float> Sx; //!< intermediate sparse flow representation (x component)
    Mat_<float> Sy; //!< intermediate sparse flow representation (y component)

    /* Structure tensor components: */
    Mat_<float> I0xx_buf; //!< sum of squares of x gradient values
    Mat_<float> I0yy_buf; //!< sum of squares of y gradient values
    Mat_<float> I0xy_buf; //!< sum of x and y gradient products

    /* Extra buffers that are useful if patch mean-normalization is used: */
    Mat_<float> I0x_buf; //!< sum of x gradient values
    Mat_<float> I0y_buf; //!< sum of y gradient values

    /* Auxiliary buffers used in structure tensor computation: */
    Mat_<float> I0xx_buf_aux;
    Mat_<float> I0yy_buf_aux;
    Mat_<float> I0xy_buf_aux;
    Mat_<float> I0x_buf_aux;
    Mat_<float> I0y_buf_aux;

    vector<Ptr<VariationalRefinement> > variational_refinement_processors;

  private: //!< private methods and parallel sections
    void prepareBuffers(Mat &I0, Mat &I1, Mat &flow, bool use_flow);
    void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x, Mat &dst_I0y, Mat &I0x,
                                   Mat &I0y);

    struct PatchInverseSearch_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        int nstripes, stripe_sz;
        int hs;
        Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
        int num_iter, pyr_level;

        PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                   Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y, int _num_iter,
                                   int _pyr_level);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct Densification_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        int nstripes, stripe_sz;
        int h;
        Mat *Ux, *Uy, *Sx, *Sy, *I0, *I1;

        Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h, Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx,
                              Mat &src_Sy, Mat &_I0, Mat &_I1);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

#ifdef HAVE_OPENCL
    vector<UMat> u_I0s;     //!< Gaussian pyramid for the current frame
    vector<UMat> u_I1s;     //!< Gaussian pyramid for the next frame
    vector<UMat> u_I1s_ext; //!< I1s with borders

    vector<UMat> u_I0xs; //!< Gaussian pyramid for the x gradient of the current frame
    vector<UMat> u_I0ys; //!< Gaussian pyramid for the y gradient of the current frame

    vector<UMat> u_Ux; //!< x component of the flow vectors
    vector<UMat> u_Uy; //!< y component of the flow vectors

    vector<UMat> u_initial_Ux; //!< x component of the initial flow field, if one was passed as an input
    vector<UMat> u_initial_Uy; //!< y component of the initial flow field, if one was passed as an input

    UMat u_U; //!< a buffer for the merged flow

    UMat u_Sx; //!< intermediate sparse flow representation (x component)
    UMat u_Sy; //!< intermediate sparse flow representation (y component)

    /* Structure tensor components: */
    UMat u_I0xx_buf; //!< sum of squares of x gradient values
    UMat u_I0yy_buf; //!< sum of squares of y gradient values
    UMat u_I0xy_buf; //!< sum of x and y gradient products

    /* Extra buffers that are useful if patch mean-normalization is used: */
    UMat u_I0x_buf; //!< sum of x gradient values
    UMat u_I0y_buf; //!< sum of y gradient values

    /* Auxiliary buffers used in structure tensor computation: */
    UMat u_I0xx_buf_aux;
    UMat u_I0yy_buf_aux;
    UMat u_I0xy_buf_aux;
    UMat u_I0x_buf_aux;
    UMat u_I0y_buf_aux;

    bool ocl_precomputeStructureTensor(UMat &dst_I0xx, UMat &dst_I0yy, UMat &dst_I0xy,
                                       UMat &dst_I0x, UMat &dst_I0y, UMat &I0x, UMat &I0y);
    void ocl_prepareBuffers(UMat &I0, UMat &I1, UMat &flow, bool use_flow);
    bool ocl_calc(InputArray I0, InputArray I1, InputOutputArray flow);
    bool ocl_Densification(UMat &dst_Ux, UMat &dst_Uy, UMat &src_Sx, UMat &src_Sy, UMat &_I0, UMat &_I1);
    bool ocl_PatchInverseSearch(UMat &src_Ux, UMat &src_Uy,
                                UMat &I0, UMat &I1, UMat &I0x, UMat &I0y, int num_iter, int pyr_level);
#endif
};

DISOpticalFlowImpl::DISOpticalFlowImpl()
{
    finest_scale = 2;
    patch_size = 8;
    patch_stride = 4;
    grad_descent_iter = 16;
    variational_refinement_iter = 5;
    variational_refinement_alpha = 20.f;
    variational_refinement_gamma = 10.f;
    variational_refinement_delta = 5.f;

    border_size = 16;
    use_mean_normalization = true;
    use_spatial_propagation = true;

    /* Use separate variational refinement instances for different scales to avoid repeated memory allocation: */
    int max_possible_scales = 10;
    for (int i = 0; i < max_possible_scales; i++)
        variational_refinement_processors.push_back(VariationalRefinement::create());
}

void DISOpticalFlowImpl::prepareBuffers(Mat &I0, Mat &I1, Mat &flow, bool use_flow)
{
    I0s.resize(coarsest_scale + 1);
    I1s.resize(coarsest_scale + 1);
    I1s_ext.resize(coarsest_scale + 1);
    I0xs.resize(coarsest_scale + 1);
    I0ys.resize(coarsest_scale + 1);
    Ux.resize(coarsest_scale + 1);
    Uy.resize(coarsest_scale + 1);

    Mat flow_uv[2];
    if (use_flow)
    {
        split(flow, flow_uv);
        initial_Ux.resize(coarsest_scale + 1);
        initial_Uy.resize(coarsest_scale + 1);
    }

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        /* Avoid initializing the pyramid levels above the finest scale, as they won't be used anyway */
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            /* These buffers are reused in each scale so we initialize them once on the finest scale: */
            Sx.create(cur_rows / patch_stride, cur_cols / patch_stride);
            Sy.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0x_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0y_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);

            I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0x_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0y_buf_aux.create(cur_rows, cur_cols / patch_stride);

            U.create(cur_rows, cur_cols);
        }
        else if (i > finest_scale)
        {
            cur_rows = I0s[i - 1].rows / 2;
            cur_cols = I0s[i - 1].cols / 2;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0s[i - 1], I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1s[i - 1], I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }

        if (i >= finest_scale)
        {
            I1s_ext[i].create(cur_rows + 2 * border_size, cur_cols + 2 * border_size);
            copyMakeBorder(I1s[i], I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);
            I0xs[i].create(cur_rows, cur_cols);
            I0ys[i].create(cur_rows, cur_cols);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);
            Ux[i].create(cur_rows, cur_cols);
            Uy[i].create(cur_rows, cur_cols);
            variational_refinement_processors[i]->setAlpha(variational_refinement_alpha);
            variational_refinement_processors[i]->setDelta(variational_refinement_delta);
            variational_refinement_processors[i]->setGamma(variational_refinement_gamma);
            variational_refinement_processors[i]->setSorIterations(5);
            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);

            if (use_flow)
            {
                resize(flow_uv[0], initial_Ux[i], Size(cur_cols, cur_rows));
                initial_Ux[i] /= fraction;
                resize(flow_uv[1], initial_Uy[i], Size(cur_cols, cur_rows));
                initial_Uy[i] /= fraction;
            }
        }

        fraction *= 2;
    }
}

/* This function computes the structure tensor elements (local sums of I0x^2, I0x*I0y and I0y^2).
 * A simple box filter is not used instead because we need to compute these sums on a sparse grid
 * and store them densely in the output buffers.
 */
void DISOpticalFlowImpl::precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x,
                                                   Mat &dst_I0y, Mat &I0x, Mat &I0y)
{
    float *I0xx_ptr = dst_I0xx.ptr<float>();
    float *I0yy_ptr = dst_I0yy.ptr<float>();
    float *I0xy_ptr = dst_I0xy.ptr<float>();
    float *I0x_ptr = dst_I0x.ptr<float>();
    float *I0y_ptr = dst_I0y.ptr<float>();

    float *I0xx_aux_ptr = I0xx_buf_aux.ptr<float>();
    float *I0yy_aux_ptr = I0yy_buf_aux.ptr<float>();
    float *I0xy_aux_ptr = I0xy_buf_aux.ptr<float>();
    float *I0x_aux_ptr = I0x_buf_aux.ptr<float>();
    float *I0y_aux_ptr = I0y_buf_aux.ptr<float>();

    /* Separable box filter: horizontal pass */
    for (int i = 0; i < h; i++)
    {
        float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        short *x_row = I0x.ptr<short>(i);
        short *y_row = I0y.ptr<short>(i);
        for (int j = 0; j < patch_size; j++)
        {
            sum_xx += x_row[j] * x_row[j];
            sum_yy += y_row[j] * y_row[j];
            sum_xy += x_row[j] * y_row[j];
            sum_x += x_row[j];
            sum_y += y_row[j];
        }
        I0xx_aux_ptr[i * ws] = sum_xx;
        I0yy_aux_ptr[i * ws] = sum_yy;
        I0xy_aux_ptr[i * ws] = sum_xy;
        I0x_aux_ptr[i * ws] = sum_x;
        I0y_aux_ptr[i * ws] = sum_y;
        int js = 1;
        for (int j = patch_size; j < w; j++)
        {
            sum_xx += (x_row[j] * x_row[j] - x_row[j - patch_size] * x_row[j - patch_size]);
            sum_yy += (y_row[j] * y_row[j] - y_row[j - patch_size] * y_row[j - patch_size]);
            sum_xy += (x_row[j] * y_row[j] - x_row[j - patch_size] * y_row[j - patch_size]);
            sum_x += (x_row[j] - x_row[j - patch_size]);
            sum_y += (y_row[j] - y_row[j - patch_size]);
            if ((j - patch_size + 1) % patch_stride == 0)
            {
                I0xx_aux_ptr[i * ws + js] = sum_xx;
                I0yy_aux_ptr[i * ws + js] = sum_yy;
                I0xy_aux_ptr[i * ws + js] = sum_xy;
                I0x_aux_ptr[i * ws + js] = sum_x;
                I0y_aux_ptr[i * ws + js] = sum_y;
                js++;
            }
        }
    }

    AutoBuffer<float> sum_xx(ws), sum_yy(ws), sum_xy(ws), sum_x(ws), sum_y(ws);
    for (int j = 0; j < ws; j++)
    {
        sum_xx[j] = 0.0f;
        sum_yy[j] = 0.0f;
        sum_xy[j] = 0.0f;
        sum_x[j] = 0.0f;
        sum_y[j] = 0.0f;
    }

    /* Separable box filter: vertical pass */
    for (int i = 0; i < patch_size; i++)
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += I0xx_aux_ptr[i * ws + j];
            sum_yy[j] += I0yy_aux_ptr[i * ws + j];
            sum_xy[j] += I0xy_aux_ptr[i * ws + j];
            sum_x[j] += I0x_aux_ptr[i * ws + j];
            sum_y[j] += I0y_aux_ptr[i * ws + j];
        }
    for (int j = 0; j < ws; j++)
    {
        I0xx_ptr[j] = sum_xx[j];
        I0yy_ptr[j] = sum_yy[j];
        I0xy_ptr[j] = sum_xy[j];
        I0x_ptr[j] = sum_x[j];
        I0y_ptr[j] = sum_y[j];
    }
    int is = 1;
    for (int i = patch_size; i < h; i++)
    {
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
            sum_yy[j] += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
            sum_xy[j] += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
            sum_x[j] += (I0x_aux_ptr[i * ws + j] - I0x_aux_ptr[(i - patch_size) * ws + j]);
            sum_y[j] += (I0y_aux_ptr[i * ws + j] - I0y_aux_ptr[(i - patch_size) * ws + j]);
        }
        if ((i - patch_size + 1) % patch_stride == 0)
        {
            for (int j = 0; j < ws; j++)
            {
                I0xx_ptr[is * ws + j] = sum_xx[j];
                I0yy_ptr[is * ws + j] = sum_yy[j];
                I0xy_ptr[is * ws + j] = sum_xy[j];
                I0x_ptr[is * ws + j] = sum_x[j];
                I0y_ptr[is * ws + j] = sum_y[j];
            }
            is++;
        }
    }
}

DISOpticalFlowImpl::PatchInverseSearch_ParBody::PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes,
                                                                           int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                                                           Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1,
                                                                           Mat &_I0x, Mat &_I0y, int _num_iter,
                                                                           int _pyr_level)
    : dis(&_dis), nstripes(_nstripes), hs(_hs), Sx(&dst_Sx), Sy(&dst_Sy), Ux(&src_Ux), Uy(&src_Uy), I0(&_I0), I1(&_I1),
      I0x(&_I0x), I0y(&_I0y), num_iter(_num_iter), pyr_level(_pyr_level)
{
    stripe_sz = (int)ceil(hs / (double)nstripes);
}

/////////////////////////////////////////////* Patch processing functions */////////////////////////////////////////////

/* Some auxiliary macros */
#define HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION                                                                         \
    v_float32x4 w00v = v_setall_f32(w00);                                                                              \
    v_float32x4 w01v = v_setall_f32(w01);                                                                              \
    v_float32x4 w10v = v_setall_f32(w10);                                                                              \
    v_float32x4 w11v = v_setall_f32(w11);                                                                              \
                                                                                                                       \
    v_uint8x16 I0_row_16, I1_row_16, I1_row_shifted_16, I1_row_next_16, I1_row_next_shifted_16;                        \
    v_uint16x8 I0_row_8, I1_row_8, I1_row_shifted_8, I1_row_next_8, I1_row_next_shifted_8, tmp;                        \
    v_uint32x4 I0_row_4_left, I1_row_4_left, I1_row_shifted_4_left, I1_row_next_4_left, I1_row_next_shifted_4_left;    \
    v_uint32x4 I0_row_4_right, I1_row_4_right, I1_row_shifted_4_right, I1_row_next_4_right,                            \
      I1_row_next_shifted_4_right;                                                                                     \
    v_float32x4 I_diff_left, I_diff_right;                                                                             \
                                                                                                                       \
    /* Preload and expand the first row of I1: */                                                                      \
    I1_row_16 = v_load(I1_ptr);                                                                                        \
    I1_row_shifted_16 = v_extract<1>(I1_row_16, I1_row_16);                                                            \
    v_expand(I1_row_16, I1_row_8, tmp);                                                                                \
    v_expand(I1_row_shifted_16, I1_row_shifted_8, tmp);                                                                \
    v_expand(I1_row_8, I1_row_4_left, I1_row_4_right);                                                                 \
    v_expand(I1_row_shifted_8, I1_row_shifted_4_left, I1_row_shifted_4_right);                                         \
    I1_ptr += I1_stride;

#define HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION                                                                      \
    /* Load the next row of I1: */                                                                                     \
    I1_row_next_16 = v_load(I1_ptr);                                                                                   \
    /* Circular shift left by 1 element: */                                                                            \
    I1_row_next_shifted_16 = v_extract<1>(I1_row_next_16, I1_row_next_16);                                             \
    /* Expand to 8 ushorts (we only need the first 8 values): */                                                       \
    v_expand(I1_row_next_16, I1_row_next_8, tmp);                                                                      \
    v_expand(I1_row_next_shifted_16, I1_row_next_shifted_8, tmp);                                                      \
    /* Separate the left and right halves: */                                                                          \
    v_expand(I1_row_next_8, I1_row_next_4_left, I1_row_next_4_right);                                                  \
    v_expand(I1_row_next_shifted_8, I1_row_next_shifted_4_left, I1_row_next_shifted_4_right);                          \
                                                                                                                       \
    /* Load current row of I0: */                                                                                      \
    I0_row_16 = v_load(I0_ptr);                                                                                        \
    v_expand(I0_row_16, I0_row_8, tmp);                                                                                \
    v_expand(I0_row_8, I0_row_4_left, I0_row_4_right);                                                                 \
                                                                                                                       \
    /* Compute diffs between I0 and bilinearly interpolated I1: */                                                     \
    I_diff_left = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_left)) +                                              \
                  w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_left)) +                                      \
                  w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_left)) +                                         \
                  w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_left)) -                                 \
                  v_cvt_f32(v_reinterpret_as_s32(I0_row_4_left));                                                      \
    I_diff_right = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_right)) +                                            \
                   w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_right)) +                                    \
                   w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_right)) +                                       \
                   w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_right)) -                               \
                   v_cvt_f32(v_reinterpret_as_s32(I0_row_4_right));

#define HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW                                                                     \
    I0_ptr += I0_stride;                                                                                               \
    I1_ptr += I1_stride;                                                                                               \
                                                                                                                       \
    I1_row_4_left = I1_row_next_4_left;                                                                                \
    I1_row_4_right = I1_row_next_4_right;                                                                              \
    I1_row_shifted_4_left = I1_row_next_shifted_4_left;                                                                \
    I1_row_shifted_4_right = I1_row_next_shifted_4_right;

/* This function essentially performs one iteration of gradient descent when finding the most similar patch in I1 for a
 * given one in I0. It assumes that I0_ptr and I1_ptr already point to the corresponding patches and w00, w01, w10, w11
 * are precomputed bilinear interpolation weights. It returns the SSD (sum of squared differences) between these patches
 * and computes the values (dst_dUx, dst_dUy) that are used in the flow vector update. HAL acceleration is implemented
 * only for the default patch size (8x8). Everything is processed in floats as using fixed-point approximations harms
 * the quality significantly.
 */
inline float processPatch(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr, short *I0y_ptr,
                          int I0_stride, int I1_stride, float w00, float w01, float w10, float w11, int patch_sz)
{
    float SSD = 0.0f;
#if CV_SIMD128
    if (patch_sz == 8)
    {
        /* Variables to accumulate the sums */
        v_float32x4 Ux_vec = v_setall_f32(0);
        v_float32x4 Uy_vec = v_setall_f32(0);
        v_float32x4 SSD_vec = v_setall_f32(0);

        v_int16x8 I0x_row, I0y_row;
        v_int32x4 I0x_row_4_left, I0x_row_4_right, I0y_row_4_left, I0y_row_4_right;

        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            I0x_row = v_load(I0x_ptr);
            v_expand(I0x_row, I0x_row_4_left, I0x_row_4_right);
            I0y_row = v_load(I0y_ptr);
            v_expand(I0y_row, I0y_row_4_left, I0y_row_4_right);

            /* Update the sums: */
            Ux_vec += I_diff_left * v_cvt_f32(I0x_row_4_left) + I_diff_right * v_cvt_f32(I0x_row_4_right);
            Uy_vec += I_diff_left * v_cvt_f32(I0y_row_4_left) + I_diff_right * v_cvt_f32(I0y_row_4_right);
            SSD_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;

            I0x_ptr += I0_stride;
            I0y_ptr += I0_stride;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }

        /* Final reduce operations: */
        dst_dUx = v_reduce_sum(Ux_vec);
        dst_dUy = v_reduce_sum(Uy_vec);
        SSD = v_reduce_sum(SSD_vec);
    }
    else
    {
#endif
        dst_dUx = 0.0f;
        dst_dUy = 0.0f;
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                SSD += diff * diff;
                dst_dUx += diff * I0x_ptr[i * I0_stride + j];
                dst_dUy += diff * I0y_ptr[i * I0_stride + j];
            }
#if CV_SIMD128
    }
#endif
    return SSD;
}

/* Same as processPatch, but with patch mean normalization, which improves robustness under changing
 * lighting conditions
 */
inline float processPatchMeanNorm(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr,
                                  short *I0y_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                                  float w11, int patch_sz, float x_grad_sum, float y_grad_sum)
{
    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;
    float n = (float)patch_sz * patch_sz;

#if CV_SIMD128
    if (patch_sz == 8)
    {
        /* Variables to accumulate the sums */
        v_float32x4 sum_I0x_mul_vec = v_setall_f32(0);
        v_float32x4 sum_I0y_mul_vec = v_setall_f32(0);
        v_float32x4 sum_diff_vec = v_setall_f32(0);
        v_float32x4 sum_diff_sq_vec = v_setall_f32(0);

        v_int16x8 I0x_row, I0y_row;
        v_int32x4 I0x_row_4_left, I0x_row_4_right, I0y_row_4_left, I0y_row_4_right;

        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            I0x_row = v_load(I0x_ptr);
            v_expand(I0x_row, I0x_row_4_left, I0x_row_4_right);
            I0y_row = v_load(I0y_ptr);
            v_expand(I0y_row, I0y_row_4_left, I0y_row_4_right);

            /* Update the sums: */
            sum_I0x_mul_vec += I_diff_left * v_cvt_f32(I0x_row_4_left) + I_diff_right * v_cvt_f32(I0x_row_4_right);
            sum_I0y_mul_vec += I_diff_left * v_cvt_f32(I0y_row_4_left) + I_diff_right * v_cvt_f32(I0y_row_4_right);
            sum_diff_sq_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            sum_diff_vec += I_diff_left + I_diff_right;

            I0x_ptr += I0_stride;
            I0y_ptr += I0_stride;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }

        /* Final reduce operations: */
        sum_I0x_mul = v_reduce_sum(sum_I0x_mul_vec);
        sum_I0y_mul = v_reduce_sum(sum_I0y_mul_vec);
        sum_diff = v_reduce_sum(sum_diff_vec);
        sum_diff_sq = v_reduce_sum(sum_diff_sq_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                sum_diff += diff;
                sum_diff_sq += diff * diff;

                sum_I0x_mul += diff * I0x_ptr[i * I0_stride + j];
                sum_I0y_mul += diff * I0y_ptr[i * I0_stride + j];
            }
#if CV_SIMD128
    }
#endif
    dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum / n;
    dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum / n;
    return sum_diff_sq - sum_diff * sum_diff / n;
}

/* Similar to processPatch, but compute only the sum of squared differences (SSD) between the patches */
inline float computeSSD(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                        float w11, int patch_sz)
{
    float SSD = 0.0f;
#if CV_SIMD128
    if (patch_sz == 8)
    {
        v_float32x4 SSD_vec = v_setall_f32(0);
        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            SSD_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }
        SSD = v_reduce_sum(SSD_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];
                SSD += diff * diff;
            }
#if CV_SIMD128
    }
#endif
    return SSD;
}

/* Same as computeSSD, but with patch mean normalization */
inline float computeSSDMeanNorm(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01,
                                float w10, float w11, int patch_sz)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    float n = (float)patch_sz * patch_sz;
#if CV_SIMD128
    if (patch_sz == 8)
    {
        v_float32x4 sum_diff_vec = v_setall_f32(0);
        v_float32x4 sum_diff_sq_vec = v_setall_f32(0);
        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            sum_diff_sq_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            sum_diff_vec += I_diff_left + I_diff_right;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }
        sum_diff = v_reduce_sum(sum_diff_vec);
        sum_diff_sq = v_reduce_sum(sum_diff_sq_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                sum_diff += diff;
                sum_diff_sq += diff * diff;
            }
#if CV_SIMD128
    }
#endif
    return sum_diff_sq - sum_diff * sum_diff / n;
}

#undef HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION
#undef HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION
#undef HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DISOpticalFlowImpl::PatchInverseSearch_ParBody::operator()(const Range &range) const
{
    // force separate processing of stripes if we are using spatial propagation:
    if (dis->use_spatial_propagation && range.end > range.start + 1)
    {
        for (int n = range.start; n < range.end; n++)
            (*this)(Range(n, n + 1));
        return;
    }
    int psz = dis->patch_size;
    int psz2 = psz / 2;
    int w_ext = dis->w + 2 * dis->border_size; //!< width of I1_ext
    int bsz = dis->border_size;

    /* Input dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    /* Output sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    short *I0x_ptr = I0x->ptr<short>();
    short *I0y_ptr = I0y->ptr<short>();

    /* Precomputed structure tensor */
    float *xx_ptr = dis->I0xx_buf.ptr<float>();
    float *yy_ptr = dis->I0yy_buf.ptr<float>();
    float *xy_ptr = dis->I0xy_buf.ptr<float>();
    /* And extra buffers for mean-normalization: */
    float *x_ptr = dis->I0x_buf.ptr<float>();
    float *y_ptr = dis->I0y_buf.ptr<float>();

    bool use_temporal_candidates = false;
    float *initial_Ux_ptr = NULL, *initial_Uy_ptr = NULL;
    if (!dis->initial_Ux.empty())
    {
        initial_Ux_ptr = dis->initial_Ux[pyr_level].ptr<float>();
        initial_Uy_ptr = dis->initial_Uy[pyr_level].ptr<float>();
        use_temporal_candidates = true;
    }

    int i, j, dir;
    int start_is, end_is, start_js, end_js;
    int start_i, start_j;
    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + dis->h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + dis->w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;

#define INIT_BILINEAR_WEIGHTS(Ux, Uy)                                                                                  \
    i_I1 = min(max(i + Uy + bsz, i_lower_limit), i_upper_limit);                                                       \
    j_I1 = min(max(j + Ux + bsz, j_lower_limit), j_upper_limit);                                                       \
                                                                                                                       \
    w11 = (i_I1 - floor(i_I1)) * (j_I1 - floor(j_I1));                                                                 \
    w10 = (i_I1 - floor(i_I1)) * (floor(j_I1) + 1 - j_I1);                                                             \
    w01 = (floor(i_I1) + 1 - i_I1) * (j_I1 - floor(j_I1));                                                             \
    w00 = (floor(i_I1) + 1 - i_I1) * (floor(j_I1) + 1 - j_I1);

#define COMPUTE_SSD(dst, Ux, Uy)                                                                                       \
    INIT_BILINEAR_WEIGHTS(Ux, Uy);                                                                                     \
    if (dis->use_mean_normalization)                                                                                   \
        dst = computeSSDMeanNorm(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00,  \
                                 w01, w10, w11, psz);                                                                  \
    else                                                                                                               \
        dst = computeSSD(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00, w01,     \
                         w10, w11, psz);

    int num_inner_iter = (int)floor(dis->grad_descent_iter / (float)num_iter);
    for (int iter = 0; iter < num_iter; iter++)
    {
        if (iter % 2 == 0)
        {
            dir = 1;
            start_is = min(range.start * stripe_sz, hs);
            end_is = min(range.end * stripe_sz, hs);
            start_js = 0;
            end_js = dis->ws;
            start_i = start_is * dis->patch_stride;
            start_j = 0;
        }
        else
        {
            dir = -1;
            start_is = min(range.end * stripe_sz, hs) - 1;
            end_is = min(range.start * stripe_sz, hs) - 1;
            start_js = dis->ws - 1;
            end_js = -1;
            start_i = start_is * dis->patch_stride;
            start_j = (dis->ws - 1) * dis->patch_stride;
        }

        i = start_i;
        for (int is = start_is; dir * is < dir * end_is; is += dir)
        {
            j = start_j;
            for (int js = start_js; dir * js < dir * end_js; js += dir)
            {
                if (iter == 0)
                {
                    /* Using result form the previous pyramid level as the very first approximation: */
                    Sx_ptr[is * dis->ws + js] = Ux_ptr[(i + psz2) * dis->w + j + psz2];
                    Sy_ptr[is * dis->ws + js] = Uy_ptr[(i + psz2) * dis->w + j + psz2];
                }

                float min_SSD = INF, cur_SSD;
                if (use_temporal_candidates || dis->use_spatial_propagation)
                {
                    COMPUTE_SSD(min_SSD, Sx_ptr[is * dis->ws + js], Sy_ptr[is * dis->ws + js]);
                }

                if (use_temporal_candidates)
                {
                    /* Try temporal candidates (vectors from the initial flow field that was passed to the function) */
                    COMPUTE_SSD(cur_SSD, initial_Ux_ptr[(i + psz2) * dis->w + j + psz2],
                                initial_Uy_ptr[(i + psz2) * dis->w + j + psz2]);
                    if (cur_SSD < min_SSD)
                    {
                        min_SSD = cur_SSD;
                        Sx_ptr[is * dis->ws + js] = initial_Ux_ptr[(i + psz2) * dis->w + j + psz2];
                        Sy_ptr[is * dis->ws + js] = initial_Uy_ptr[(i + psz2) * dis->w + j + psz2];
                    }
                }

                if (dis->use_spatial_propagation)
                {
                    /* Try spatial candidates: */
                    if (dir * js > dir * start_js)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[is * dis->ws + js - dir], Sy_ptr[is * dis->ws + js - dir]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[is * dis->ws + js - dir];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[is * dis->ws + js - dir];
                        }
                    }
                    /* Flow vectors won't actually propagate across different stripes, which is the reason for keeping
                     * the number of stripes constant. It works well enough in practice and doesn't introduce any
                     * visible seams.
                     */
                    if (dir * is > dir * start_is)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[(is - dir) * dis->ws + js], Sy_ptr[(is - dir) * dis->ws + js]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[(is - dir) * dis->ws + js];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[(is - dir) * dis->ws + js];
                        }
                    }
                }

                /* Use the best candidate as a starting point for the gradient descent: */
                float cur_Ux = Sx_ptr[is * dis->ws + js];
                float cur_Uy = Sy_ptr[is * dis->ws + js];

                /* Computing the inverse of the structure tensor: */
                float detH = xx_ptr[is * dis->ws + js] * yy_ptr[is * dis->ws + js] -
                             xy_ptr[is * dis->ws + js] * xy_ptr[is * dis->ws + js];
                if (abs(detH) < EPS)
                    detH = EPS;
                float invH11 = yy_ptr[is * dis->ws + js] / detH;
                float invH12 = -xy_ptr[is * dis->ws + js] / detH;
                float invH22 = xx_ptr[is * dis->ws + js] / detH;
                float prev_SSD = INF, SSD;
                float x_grad_sum = x_ptr[is * dis->ws + js];
                float y_grad_sum = y_ptr[is * dis->ws + js];

                for (int t = 0; t < num_inner_iter; t++)
                {
                    INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
                    if (dis->use_mean_normalization)
                        SSD = processPatchMeanNorm(dUx, dUy, I0_ptr + i * dis->w + j,
                                                   I1_ptr + (int)i_I1 * w_ext + (int)j_I1, I0x_ptr + i * dis->w + j,
                                                   I0y_ptr + i * dis->w + j, dis->w, w_ext, w00, w01, w10, w11, psz,
                                                   x_grad_sum, y_grad_sum);
                    else
                        SSD = processPatch(dUx, dUy, I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                           I0x_ptr + i * dis->w + j, I0y_ptr + i * dis->w + j, dis->w, w_ext, w00, w01,
                                           w10, w11, psz);

                    dx = invH11 * dUx + invH12 * dUy;
                    dy = invH12 * dUx + invH22 * dUy;
                    cur_Ux -= dx;
                    cur_Uy -= dy;

                    /* Break when patch distance stops decreasing */
                    if (SSD >= prev_SSD)
                        break;
                    prev_SSD = SSD;
                }

                /* If gradient descent converged to a flow vector that is very far from the initial approximation
                 * (more than patch size) then we don't use it. Noticeably improves the robustness.
                 */
                if (norm(Vec2f(cur_Ux - Sx_ptr[is * dis->ws + js], cur_Uy - Sy_ptr[is * dis->ws + js])) <= psz)
                {
                    Sx_ptr[is * dis->ws + js] = cur_Ux;
                    Sy_ptr[is * dis->ws + js] = cur_Uy;
                }
                j += dir * dis->patch_stride;
            }
            i += dir * dis->patch_stride;
        }
    }
#undef INIT_BILINEAR_WEIGHTS
#undef COMPUTE_SSD
}

DISOpticalFlowImpl::Densification_ParBody::Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h,
                                                                 Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx, Mat &src_Sy,
                                                                 Mat &_I0, Mat &_I1)
    : dis(&_dis), nstripes(_nstripes), h(_h), Ux(&dst_Ux), Uy(&dst_Uy), Sx(&src_Sx), Sy(&src_Sy), I0(&_I0), I1(&_I1)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function transforms a sparse optical flow field obtained by PatchInverseSearch (which computes flow values
 * on a sparse grid defined by patch_stride) into a dense optical flow field by weighted averaging of values from the
 * overlapping patches.
 */
void DISOpticalFlowImpl::Densification_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    /* Input sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    /* Output dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();

    int psz = dis->patch_size;
    int pstr = dis->patch_stride;
    int i_l, i_u;
    int j_l, j_u;
    float i_m, j_m, diff;

    /* These values define the set of sparse grid locations that contain patches overlapping with the current dense flow
     * location */
    int start_is, end_is;
    int start_js, end_js;

/* Some helper macros for updating this set of sparse grid locations */
#define UPDATE_SPARSE_I_COORDINATES                                                                                    \
    if (i % pstr == 0 && i + psz <= h)                                                                                 \
        end_is++;                                                                                                      \
    if (i - psz >= 0 && (i - psz) % pstr == 0 && start_is < end_is)                                                    \
        start_is++;

#define UPDATE_SPARSE_J_COORDINATES                                                                                    \
    if (j % pstr == 0 && j + psz <= dis->w)                                                                            \
        end_js++;                                                                                                      \
    if (j - psz >= 0 && (j - psz) % pstr == 0 && start_js < end_js)                                                    \
        start_js++;

    start_is = 0;
    end_is = -1;
    for (int i = 0; i < start_i; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
    }
    for (int i = start_i; i < end_i; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
        start_js = 0;
        end_js = -1;
        for (int j = 0; j < dis->w; j++)
        {
            UPDATE_SPARSE_J_COORDINATES;
            float coef, sum_coef = 0.0f;
            float sum_Ux = 0.0f;
            float sum_Uy = 0.0f;

            /* Iterate through all the patches that overlap the current location (i,j) */
            for (int is = start_is; is <= end_is; is++)
                for (int js = start_js; js <= end_js; js++)
                {
                    j_m = min(max(j + Sx_ptr[is * dis->ws + js], 0.0f), dis->w - 1.0f - EPS);
                    i_m = min(max(i + Sy_ptr[is * dis->ws + js], 0.0f), dis->h - 1.0f - EPS);
                    j_l = (int)j_m;
                    j_u = j_l + 1;
                    i_l = (int)i_m;
                    i_u = i_l + 1;
                    diff = (j_m - j_l) * (i_m - i_l) * I1_ptr[i_u * dis->w + j_u] +
                           (j_u - j_m) * (i_m - i_l) * I1_ptr[i_u * dis->w + j_l] +
                           (j_m - j_l) * (i_u - i_m) * I1_ptr[i_l * dis->w + j_u] +
                           (j_u - j_m) * (i_u - i_m) * I1_ptr[i_l * dis->w + j_l] - I0_ptr[i * dis->w + j];
                    coef = 1 / max(1.0f, abs(diff));
                    sum_Ux += coef * Sx_ptr[is * dis->ws + js];
                    sum_Uy += coef * Sy_ptr[is * dis->ws + js];
                    sum_coef += coef;
                }
            Ux_ptr[i * dis->w + j] = sum_Ux / sum_coef;
            Uy_ptr[i * dis->w + j] = sum_Uy / sum_coef;
        }
    }
#undef UPDATE_SPARSE_I_COORDINATES
#undef UPDATE_SPARSE_J_COORDINATES
}

#ifdef HAVE_OPENCL
bool DISOpticalFlowImpl::ocl_PatchInverseSearch(UMat &src_Ux, UMat &src_Uy,
                                                UMat &I0, UMat &I1, UMat &I0x, UMat &I0y, int num_iter, int pyr_level)
{
    size_t globalSize[] = {(size_t)ws, (size_t)hs};
    size_t localSize[]  = {16, 16};
    int idx;
    int num_inner_iter = (int)floor(grad_descent_iter / (float)num_iter);

    for (int iter = 0; iter < num_iter; iter++)
    {
        if (iter == 0)
        {
            ocl::Kernel k1("dis_patch_inverse_search_fwd_1", ocl::video::dis_flow_oclsrc);
            size_t global_sz[] = {(size_t)hs * 8};
            size_t local_sz[]  = {8};
            idx = 0;

            idx = k1.set(idx, ocl::KernelArg::PtrReadOnly(src_Ux));
            idx = k1.set(idx, ocl::KernelArg::PtrReadOnly(src_Uy));
            idx = k1.set(idx, ocl::KernelArg::PtrReadOnly(I0));
            idx = k1.set(idx, ocl::KernelArg::PtrReadOnly(I1));
            idx = k1.set(idx, (int)border_size);
            idx = k1.set(idx, (int)patch_size);
            idx = k1.set(idx, (int)patch_stride);
            idx = k1.set(idx, (int)w);
            idx = k1.set(idx, (int)h);
            idx = k1.set(idx, (int)ws);
            idx = k1.set(idx, (int)hs);
            idx = k1.set(idx, (int)pyr_level);
            idx = k1.set(idx, ocl::KernelArg::PtrWriteOnly(u_Sx));
            idx = k1.set(idx, ocl::KernelArg::PtrWriteOnly(u_Sy));
            if (!k1.run(1, global_sz, local_sz, false))
                return false;

            ocl::Kernel k2("dis_patch_inverse_search_fwd_2", ocl::video::dis_flow_oclsrc);
            idx = 0;

            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(src_Ux));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(src_Uy));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(I0));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(I1));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(I0x));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(I0y));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(u_I0xx_buf));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(u_I0yy_buf));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(u_I0xy_buf));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(u_I0x_buf));
            idx = k2.set(idx, ocl::KernelArg::PtrReadOnly(u_I0y_buf));
            idx = k2.set(idx, (int)border_size);
            idx = k2.set(idx, (int)patch_size);
            idx = k2.set(idx, (int)patch_stride);
            idx = k2.set(idx, (int)w);
            idx = k2.set(idx, (int)h);
            idx = k2.set(idx, (int)ws);
            idx = k2.set(idx, (int)hs);
            idx = k2.set(idx, (int)num_inner_iter);
            idx = k2.set(idx, (int)pyr_level);
            idx = k2.set(idx, ocl::KernelArg::PtrReadWrite(u_Sx));
            idx = k2.set(idx, ocl::KernelArg::PtrReadWrite(u_Sy));
            if (!k2.run(2, globalSize, localSize, false))
                return false;
        }
        else
        {
            ocl::Kernel k3("dis_patch_inverse_search_bwd_1", ocl::video::dis_flow_oclsrc);
            size_t global_sz[] = {(size_t)hs * 8};
            size_t local_sz[]  = {8};
            idx = 0;

            idx = k3.set(idx, ocl::KernelArg::PtrReadOnly(I0));
            idx = k3.set(idx, ocl::KernelArg::PtrReadOnly(I1));
            idx = k3.set(idx, (int)border_size);
            idx = k3.set(idx, (int)patch_size);
            idx = k3.set(idx, (int)patch_stride);
            idx = k3.set(idx, (int)w);
            idx = k3.set(idx, (int)h);
            idx = k3.set(idx, (int)ws);
            idx = k3.set(idx, (int)hs);
            idx = k3.set(idx, (int)pyr_level);
            idx = k3.set(idx, ocl::KernelArg::PtrReadWrite(u_Sx));
            idx = k3.set(idx, ocl::KernelArg::PtrReadWrite(u_Sy));
            if (!k3.run(1, global_sz, local_sz, false))
                return false;

            ocl::Kernel k4("dis_patch_inverse_search_bwd_2", ocl::video::dis_flow_oclsrc);
            idx = 0;

            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(I0));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(I1));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(I0x));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(I0y));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(u_I0xx_buf));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(u_I0yy_buf));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(u_I0xy_buf));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(u_I0x_buf));
            idx = k4.set(idx, ocl::KernelArg::PtrReadOnly(u_I0y_buf));
            idx = k4.set(idx, (int)border_size);
            idx = k4.set(idx, (int)patch_size);
            idx = k4.set(idx, (int)patch_stride);
            idx = k4.set(idx, (int)w);
            idx = k4.set(idx, (int)h);
            idx = k4.set(idx, (int)ws);
            idx = k4.set(idx, (int)hs);
            idx = k4.set(idx, (int)num_inner_iter);
            idx = k4.set(idx, ocl::KernelArg::PtrReadWrite(u_Sx));
            idx = k4.set(idx, ocl::KernelArg::PtrReadWrite(u_Sy));
            if (!k4.run(2, globalSize, localSize, false))
                return false;
        }
    }
    return true;
}

bool DISOpticalFlowImpl::ocl_Densification(UMat &dst_Ux, UMat &dst_Uy, UMat &src_Sx, UMat &src_Sy, UMat &_I0, UMat &_I1)
{
    size_t globalSize[] = {(size_t)w, (size_t)h};
    size_t localSize[]  = {16, 16};

    ocl::Kernel kernel("dis_densification", ocl::video::dis_flow_oclsrc);
    kernel.args(ocl::KernelArg::PtrReadOnly(src_Sx),
                ocl::KernelArg::PtrReadOnly(src_Sy),
                ocl::KernelArg::PtrReadOnly(_I0),
                ocl::KernelArg::PtrReadOnly(_I1),
                (int)patch_size, (int)patch_stride,
                (int)w, (int)h, (int)ws,
                ocl::KernelArg::PtrWriteOnly(dst_Ux),
                ocl::KernelArg::PtrWriteOnly(dst_Uy));
    return kernel.run(2, globalSize, localSize, false);
}

void DISOpticalFlowImpl::ocl_prepareBuffers(UMat &I0, UMat &I1, UMat &flow, bool use_flow)
{
    u_I0s.resize(coarsest_scale + 1);
    u_I1s.resize(coarsest_scale + 1);
    u_I1s_ext.resize(coarsest_scale + 1);
    u_I0xs.resize(coarsest_scale + 1);
    u_I0ys.resize(coarsest_scale + 1);
    u_Ux.resize(coarsest_scale + 1);
    u_Uy.resize(coarsest_scale + 1);

    vector<UMat> flow_uv(2);
    if (use_flow)
    {
        split(flow, flow_uv);
        u_initial_Ux.resize(coarsest_scale + 1);
        u_initial_Uy.resize(coarsest_scale + 1);
    }

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        /* Avoid initializing the pyramid levels above the finest scale, as they won't be used anyway */
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            u_I0s[i].create(cur_rows, cur_cols, CV_8UC1);
            resize(I0, u_I0s[i], u_I0s[i].size(), 0.0, 0.0, INTER_AREA);
            u_I1s[i].create(cur_rows, cur_cols, CV_8UC1);
            resize(I1, u_I1s[i], u_I1s[i].size(), 0.0, 0.0, INTER_AREA);

            /* These buffers are reused in each scale so we initialize them once on the finest scale: */
            u_Sx.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_Sy.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_I0x_buf.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);
            u_I0y_buf.create(cur_rows / patch_stride, cur_cols / patch_stride, CV_32FC1);

            u_I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride, CV_32FC1);
            u_I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride, CV_32FC1);
            u_I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride, CV_32FC1);
            u_I0x_buf_aux.create(cur_rows, cur_cols / patch_stride, CV_32FC1);
            u_I0y_buf_aux.create(cur_rows, cur_cols / patch_stride, CV_32FC1);

            u_U.create(cur_rows, cur_cols, CV_32FC2);
        }
        else if (i > finest_scale)
        {
            cur_rows = u_I0s[i - 1].rows / 2;
            cur_cols = u_I0s[i - 1].cols / 2;
            u_I0s[i].create(cur_rows, cur_cols, CV_8UC1);
            resize(u_I0s[i - 1], u_I0s[i], u_I0s[i].size(), 0.0, 0.0, INTER_AREA);
            u_I1s[i].create(cur_rows, cur_cols, CV_8UC1);
            resize(u_I1s[i - 1], u_I1s[i], u_I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }

        if (i >= finest_scale)
        {
            u_I1s_ext[i].create(cur_rows + 2 * border_size, cur_cols + 2 * border_size, CV_8UC1);
            copyMakeBorder(u_I1s[i], u_I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);
            u_I0xs[i].create(cur_rows, cur_cols, CV_16SC1);
            u_I0ys[i].create(cur_rows, cur_cols, CV_16SC1);
            spatialGradient(u_I0s[i], u_I0xs[i], u_I0ys[i]);
            u_Ux[i].create(cur_rows, cur_cols, CV_32FC1);
            u_Uy[i].create(cur_rows, cur_cols, CV_32FC1);
            variational_refinement_processors[i]->setAlpha(variational_refinement_alpha);
            variational_refinement_processors[i]->setDelta(variational_refinement_delta);
            variational_refinement_processors[i]->setGamma(variational_refinement_gamma);
            variational_refinement_processors[i]->setSorIterations(5);
            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);

            if (use_flow)
            {
                resize(flow_uv[0], u_initial_Ux[i], Size(cur_cols, cur_rows));
                divide(u_initial_Ux[i], static_cast<float>(fraction), u_initial_Ux[i]);
                resize(flow_uv[1], u_initial_Uy[i], Size(cur_cols, cur_rows));
                divide(u_initial_Uy[i], static_cast<float>(fraction), u_initial_Uy[i]);
            }
        }

        fraction *= 2;
    }
}

bool DISOpticalFlowImpl::ocl_precomputeStructureTensor(UMat &dst_I0xx, UMat &dst_I0yy, UMat &dst_I0xy,
                                                       UMat &dst_I0x, UMat &dst_I0y, UMat &I0x, UMat &I0y)
{
    size_t globalSizeX[] = {(size_t)h};
    size_t localSizeX[]  = {16};

    ocl::Kernel kernelX("dis_precomputeStructureTensor_hor", ocl::video::dis_flow_oclsrc);
    kernelX.args(ocl::KernelArg::PtrReadOnly(I0x),
                 ocl::KernelArg::PtrReadOnly(I0y),
                 (int)patch_size, (int)patch_stride,
                 (int)w, (int)h, (int)ws,
                 ocl::KernelArg::PtrWriteOnly(u_I0xx_buf_aux),
                 ocl::KernelArg::PtrWriteOnly(u_I0yy_buf_aux),
                 ocl::KernelArg::PtrWriteOnly(u_I0xy_buf_aux),
                 ocl::KernelArg::PtrWriteOnly(u_I0x_buf_aux),
                 ocl::KernelArg::PtrWriteOnly(u_I0y_buf_aux));
    if (!kernelX.run(1, globalSizeX, localSizeX, false))
        return false;

    size_t globalSizeY[] = {(size_t)ws};
    size_t localSizeY[]  = {16};

    ocl::Kernel kernelY("dis_precomputeStructureTensor_ver", ocl::video::dis_flow_oclsrc);
    kernelY.args(ocl::KernelArg::PtrReadOnly(u_I0xx_buf_aux),
                 ocl::KernelArg::PtrReadOnly(u_I0yy_buf_aux),
                 ocl::KernelArg::PtrReadOnly(u_I0xy_buf_aux),
                 ocl::KernelArg::PtrReadOnly(u_I0x_buf_aux),
                 ocl::KernelArg::PtrReadOnly(u_I0y_buf_aux),
                 (int)patch_size, (int)patch_stride,
                 (int)w, (int)h, (int)ws,
                 ocl::KernelArg::PtrWriteOnly(dst_I0xx),
                 ocl::KernelArg::PtrWriteOnly(dst_I0yy),
                 ocl::KernelArg::PtrWriteOnly(dst_I0xy),
                 ocl::KernelArg::PtrWriteOnly(dst_I0x),
                 ocl::KernelArg::PtrWriteOnly(dst_I0y));
    return kernelY.run(1, globalSizeY, localSizeY, false);
}

bool DISOpticalFlowImpl::ocl_calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    UMat I0Mat = I0.getUMat();
    UMat I1Mat = I1.getUMat();
    bool use_input_flow = false;
    if (flow.sameSize(I0) && flow.depth() == CV_32F && flow.channels() == 2)
        use_input_flow = true;
    else
        flow.create(I1Mat.size(), CV_32FC2);
    UMat &u_flowMat = flow.getUMatRef();
    coarsest_scale = min((int)(log(max(I0Mat.cols, I0Mat.rows) / (4.0 * patch_size)) / log(2.0) + 0.5), /* Original code serach for maximal movement of width/4 */
                         (int)(log(min(I0Mat.cols, I0Mat.rows) / patch_size) / log(2.0)));              /* Deepest pyramid level greater or equal than patch*/

    ocl_prepareBuffers(I0Mat, I1Mat, u_flowMat, use_input_flow);
    u_Ux[coarsest_scale].setTo(0.0f);
    u_Uy[coarsest_scale].setTo(0.0f);

    for (int i = coarsest_scale; i >= finest_scale; i--)
    {
        w = u_I0s[i].cols;
        h = u_I0s[i].rows;
        ws = 1 + (w - patch_size) / patch_stride;
        hs = 1 + (h - patch_size) / patch_stride;

        if (!ocl_precomputeStructureTensor(u_I0xx_buf, u_I0yy_buf, u_I0xy_buf,
                                           u_I0x_buf, u_I0y_buf, u_I0xs[i], u_I0ys[i]))
            return false;

        if (!ocl_PatchInverseSearch(u_Ux[i], u_Uy[i], u_I0s[i], u_I1s_ext[i], u_I0xs[i], u_I0ys[i], 2, i))
            return false;

        if (!ocl_Densification(u_Ux[i], u_Uy[i], u_Sx, u_Sy, u_I0s[i], u_I1s[i]))
            return false;

        if (variational_refinement_iter > 0)
            variational_refinement_processors[i]->calcUV(u_I0s[i], u_I1s[i],
                                                         u_Ux[i].getMat(ACCESS_WRITE), u_Uy[i].getMat(ACCESS_WRITE));

        if (i > finest_scale)
        {
            resize(u_Ux[i], u_Ux[i - 1], u_Ux[i - 1].size());
            resize(u_Uy[i], u_Uy[i - 1], u_Uy[i - 1].size());
            multiply(u_Ux[i - 1], 2, u_Ux[i - 1]);
            multiply(u_Uy[i - 1], 2, u_Uy[i - 1]);
        }
    }
    vector<UMat> uxy(2);
    uxy[0] = u_Ux[finest_scale];
    uxy[1] = u_Uy[finest_scale];
    merge(uxy, u_U);
    resize(u_U, u_flowMat, u_flowMat.size());
    multiply(u_flowMat, 1 << finest_scale, u_flowMat);

    return true;
}
#endif

void DISOpticalFlowImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert(I0.isContinuous());
    CV_Assert(I1.isContinuous());

    CV_OCL_RUN(ocl::Device::getDefault().isIntel() && flow.isUMat() &&
               (patch_size == 8) && (use_spatial_propagation == true),
               ocl_calc(I0, I1, flow));

    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    bool use_input_flow = false;
    if (flow.sameSize(I0) && flow.depth() == CV_32F && flow.channels() == 2)
        use_input_flow = true;
    else
        flow.create(I1Mat.size(), CV_32FC2);
    Mat flowMat = flow.getMat();
    coarsest_scale = min((int)(log(max(I0Mat.cols, I0Mat.rows) / (4.0 * patch_size)) / log(2.0) + 0.5), /* Original code serach for maximal movement of width/4 */
                         (int)(log(min(I0Mat.cols, I0Mat.rows) / patch_size) / log(2.0)));              /* Deepest pyramid level greater or equal than patch*/
    int num_stripes = getNumThreads();

    prepareBuffers(I0Mat, I1Mat, flowMat, use_input_flow);
    Ux[coarsest_scale].setTo(0.0f);
    Uy[coarsest_scale].setTo(0.0f);

    for (int i = coarsest_scale; i >= finest_scale; i--)
    {
        w = I0s[i].cols;
        h = I0s[i].rows;
        ws = 1 + (w - patch_size) / patch_stride;
        hs = 1 + (h - patch_size) / patch_stride;

        precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0x_buf, I0y_buf, I0xs[i], I0ys[i]);
        if (use_spatial_propagation)
        {
            /* Use a fixed number of stripes regardless the number of threads to make inverse search
             * with spatial propagation reproducible
             */
            parallel_for_(Range(0, 8), PatchInverseSearch_ParBody(*this, 8, hs, Sx, Sy, Ux[i], Uy[i], I0s[i],
                                                                  I1s_ext[i], I0xs[i], I0ys[i], 2, i));
        }
        else
        {
            parallel_for_(Range(0, num_stripes),
                          PatchInverseSearch_ParBody(*this, num_stripes, hs, Sx, Sy, Ux[i], Uy[i], I0s[i], I1s_ext[i],
                                                     I0xs[i], I0ys[i], 1, i));
        }

        parallel_for_(Range(0, num_stripes),
                      Densification_ParBody(*this, num_stripes, I0s[i].rows, Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]));
        if (variational_refinement_iter > 0)
            variational_refinement_processors[i]->calcUV(I0s[i], I1s[i], Ux[i], Uy[i]);

        if (i > finest_scale)
        {
            resize(Ux[i], Ux[i - 1], Ux[i - 1].size());
            resize(Uy[i], Uy[i - 1], Uy[i - 1].size());
            Ux[i - 1] *= 2;
            Uy[i - 1] *= 2;
        }
    }
    Mat uxy[] = {Ux[finest_scale], Uy[finest_scale]};
    merge(uxy, 2, U);
    resize(U, flowMat, flowMat.size());
    flowMat *= 1 << finest_scale;
}

void DISOpticalFlowImpl::collectGarbage()
{
    I0s.clear();
    I1s.clear();
    I1s_ext.clear();
    I0xs.clear();
    I0ys.clear();
    Ux.clear();
    Uy.clear();
    U.release();
    Sx.release();
    Sy.release();
    I0xx_buf.release();
    I0yy_buf.release();
    I0xy_buf.release();
    I0xx_buf_aux.release();
    I0yy_buf_aux.release();
    I0xy_buf_aux.release();

#ifdef HAVE_OPENCL
    u_I0s.clear();
    u_I1s.clear();
    u_I1s_ext.clear();
    u_I0xs.clear();
    u_I0ys.clear();
    u_Ux.clear();
    u_Uy.clear();
    u_U.release();
    u_Sx.release();
    u_Sy.release();
    u_I0xx_buf.release();
    u_I0yy_buf.release();
    u_I0xy_buf.release();
    u_I0xx_buf_aux.release();
    u_I0yy_buf_aux.release();
    u_I0xy_buf_aux.release();
#endif

    for (int i = finest_scale; i <= coarsest_scale; i++)
        variational_refinement_processors[i]->collectGarbage();
    variational_refinement_processors.clear();
}

Ptr<DISOpticalFlow> DISOpticalFlow::create(int preset)
{
    Ptr<DISOpticalFlow> dis = makePtr<DISOpticalFlowImpl>();
    dis->setPatchSize(8);
    if (preset == DISOpticalFlow::PRESET_ULTRAFAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(12);
        dis->setVariationalRefinementIterations(0);
    }
    else if (preset == DISOpticalFlow::PRESET_FAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(16);
        dis->setVariationalRefinementIterations(5);
    }
    else if (preset == DISOpticalFlow::PRESET_MEDIUM)
    {
        dis->setFinestScale(1);
        dis->setPatchStride(3);
        dis->setGradientDescentIterations(25);
        dis->setVariationalRefinementIterations(5);
    }

    return dis;
}
}
