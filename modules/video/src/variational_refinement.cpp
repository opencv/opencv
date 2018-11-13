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

using namespace std;

namespace cv
{

class VariationalRefinementImpl CV_FINAL : public VariationalRefinement
{
  public:
    VariationalRefinementImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;

  protected: //!< algorithm parameters
    int fixedPointIterations, sorIterations;
    float omega;
    float alpha, delta, gamma;
    float zeta, epsilon;

  public:
    int getFixedPointIterations() const CV_OVERRIDE { return fixedPointIterations; }
    void setFixedPointIterations(int val) CV_OVERRIDE { fixedPointIterations = val; }
    int getSorIterations() const CV_OVERRIDE { return sorIterations; }
    void setSorIterations(int val) CV_OVERRIDE { sorIterations = val; }
    float getOmega() const CV_OVERRIDE { return omega; }
    void setOmega(float val) CV_OVERRIDE { omega = val; }
    float getAlpha() const CV_OVERRIDE { return alpha; }
    void setAlpha(float val) CV_OVERRIDE { alpha = val; }
    float getDelta() const CV_OVERRIDE { return delta; }
    void setDelta(float val) CV_OVERRIDE { delta = val; }
    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

  protected: //!< internal buffers
    /* This struct defines a special data layout for Mat_<float>. Original buffer is split into two: one for "red"
     * elements (sum of indices is even) and one for "black" (sum of indices is odd) in a checkerboard pattern. It
     * allows for more efficient processing in SOR iterations, more natural SIMD vectorization and parallelization
     * (Red-Black SOR). Additionally, it simplifies border handling by adding repeated borders to both red and
     * black buffers.
     */
    struct RedBlackBuffer
    {
        Mat_<float> red;   //!< (i+j)%2==0
        Mat_<float> black; //!< (i+j)%2==1

        /* Width of even and odd rows may be different */
        int red_even_len, red_odd_len;
        int black_even_len, black_odd_len;

        void create(Size s);
        void release();
    };

    Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;                            //!< image derivative buffers
    RedBlackBuffer Ix_rb, Iy_rb, Iz_rb, Ixx_rb, Ixy_rb, Iyy_rb, Ixz_rb, Iyz_rb; //!< corresponding red-black buffers

    RedBlackBuffer A11, A12, A22, b1, b2; //!< main linear system coefficients
    RedBlackBuffer weights;               //!< smoothness term weights in the current fixed point iteration

    Mat_<float> mapX, mapY; //!< auxiliary buffers for remapping

    RedBlackBuffer tempW_u, tempW_v; //!< flow buffers that are modified in each fixed point iteration
    RedBlackBuffer dW_u, dW_v;       //!< optical flow increment
    RedBlackBuffer W_u_rb, W_v_rb;   //!< red-black-buffer version of the input flow

  private: //!< private methods and parallel sections
    void splitCheckerboard(RedBlackBuffer &dst, Mat &src);
    void mergeCheckerboard(Mat &dst, RedBlackBuffer &src);
    void updateRepeatedBorders(RedBlackBuffer &dst);
    void warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v);
    void prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v);

    /* Parallelizing arbitrary operations with 3 input/output arguments */
    typedef void (VariationalRefinementImpl::*Op)(void *op1, void *op2, void *op3);
    struct ParallelOp_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        vector<Op> ops;
        vector<void *> op1s;
        vector<void *> op2s;
        vector<void *> op3s;

        ParallelOp_ParBody(VariationalRefinementImpl &_var, vector<Op> _ops, vector<void *> &_op1s,
                           vector<void *> &_op2s, vector<void *> &_op3s);
        void operator()(const Range &range) const CV_OVERRIDE;
    };
    void gradHorizAndSplitOp(void *src, void *dst, void *dst_split)
    {
        Sobel(*(Mat *)src, *(Mat *)dst, -1, 1, 0, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
    }
    void gradVertAndSplitOp(void *src, void *dst, void *dst_split)
    {
        Sobel(*(Mat *)src, *(Mat *)dst, -1, 0, 1, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
    }
    void averageOp(void *src1, void *src2, void *dst)
    {
        addWeighted(*(Mat *)src1, 0.5, *(Mat *)src2, 0.5, 0.0, *(Mat *)dst, CV_32F);
    }
    void subtractOp(void *src1, void *src2, void *dst)
    {
        subtract(*(Mat *)src1, *(Mat *)src2, *(Mat *)dst, noArray(), CV_32F);
    }

    struct ComputeDataTerm_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        ComputeDataTerm_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                                RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeSmoothnessTermHorPass_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
        bool red_pass;

        ComputeSmoothnessTermHorPass_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h,
                                             RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, RedBlackBuffer &_tempW_u,
                                             RedBlackBuffer &_tempW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeSmoothnessTermVertPass_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v;
        bool red_pass;

        ComputeSmoothnessTermVertPass_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h,
                                              RedBlackBuffer &W_u, RedBlackBuffer &_W_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct RedBlackSOR_ParBody : public ParallelLoopBody
    {
        VariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        RedBlackSOR_ParBody(VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                            RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };
};

VariationalRefinementImpl::VariationalRefinementImpl()
{
    fixedPointIterations = 5;
    sorIterations = 5;
    alpha = 20.0f;
    delta = 5.0f;
    gamma = 10.0f;
    omega = 1.6f;
    zeta = 0.1f;
    epsilon = 0.001f;
}

/* This function converts an input Mat into the RedBlackBuffer format, which involves
 * splitting the input buffer into two and adding repeated borders. Assumes that enough
 * memory in dst is already allocated.
 */
void VariationalRefinementImpl::splitCheckerboard(RedBlackBuffer &dst, Mat &src)
{
    int buf_j, j;
    int buf_w = (int)ceil(src.cols / 2.0) + 2; //!< max width of red/black buffers with borders

    /* Rows of red and black buffers can have different actual width, some extra repeated values are
     * added for padding in such cases.
     */
    for (int i = 0; i < src.rows; i++)
    {
        float *src_buf = src.ptr<float>(i);
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);
        r_buf[0] = b_buf[0] = src_buf[0];
        buf_j = 1;
        if (i % 2 == 0)
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                r_buf[buf_j] = src_buf[j];
                b_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        else
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                b_buf[buf_j] = src_buf[j];
                r_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        r_buf[buf_w - 1] = b_buf[buf_w - 1] = src_buf[j];
    }

    /* Fill top and bottom borders: */
    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

/* The inverse of splitCheckerboard, i.e. converting the RedBlackBuffer back into Mat.
 * Assumes that enough memory in dst is already allocated.
 */
void VariationalRefinementImpl::mergeCheckerboard(Mat &dst, RedBlackBuffer &src)
{
    int buf_j, j;
    for (int i = 0; i < dst.rows; i++)
    {
        float *src_r_buf = src.red.ptr<float>(i + 1);
        float *src_b_buf = src.black.ptr<float>(i + 1);
        float *dst_buf = dst.ptr<float>(i);
        buf_j = 1;

        if (i % 2 == 0)
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_r_buf[buf_j];
                dst_buf[j + 1] = src_b_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_r_buf[buf_j];
        }
        else
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_b_buf[buf_j];
                dst_buf[j + 1] = src_r_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_b_buf[buf_j];
        }
    }
}

/* An auxiliary function that updates the borders. Used to enforce that border values repeat
 * the ones adjacent to the border.
 */
void VariationalRefinementImpl::updateRepeatedBorders(RedBlackBuffer &dst)
{
    int buf_w = dst.red.cols;
    for (int i = 0; i < dst.red.rows - 2; i++)
    {
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);

        if (i % 2 == 0)
        {
            b_buf[0] = r_buf[1];
            if (dst.red_even_len > dst.black_even_len)
                b_buf[dst.black_even_len + 1] = r_buf[dst.red_even_len];
            else
                r_buf[dst.red_even_len + 1] = b_buf[dst.black_even_len];
        }
        else
        {
            r_buf[0] = b_buf[1];
            if (dst.red_odd_len < dst.black_odd_len)
                r_buf[dst.red_odd_len + 1] = b_buf[dst.black_odd_len];
            else
                b_buf[dst.black_odd_len + 1] = r_buf[dst.red_odd_len];
        }
    }
    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

void VariationalRefinementImpl::RedBlackBuffer::create(Size s)
{
    /* Allocate enough memory to include borders */
    int w = (int)ceil(s.width / 2.0) + 2;
    red.create(s.height + 2, w);
    black.create(s.height + 2, w);

    if (s.width % 2 == 0)
        red_even_len = red_odd_len = black_even_len = black_odd_len = w - 2;
    else
    {
        red_even_len = black_odd_len = w - 2;
        red_odd_len = black_even_len = w - 3;
    }
}

void VariationalRefinementImpl::RedBlackBuffer::release()
{
    red.release();
    black.release();
}

VariationalRefinementImpl::ParallelOp_ParBody::ParallelOp_ParBody(VariationalRefinementImpl &_var, vector<Op> _ops,
                                                                  vector<void *> &_op1s, vector<void *> &_op2s,
                                                                  vector<void *> &_op3s)
    : var(&_var), ops(_ops), op1s(_op1s), op2s(_op2s), op3s(_op3s)
{
}

void VariationalRefinementImpl::ParallelOp_ParBody::operator()(const Range &range) const
{
    for (int i = range.start; i < range.end; i++)
        (var->*ops[i])(op1s[i], op2s[i], op3s[i]);
}

void VariationalRefinementImpl::warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v)
{
    for (int i = 0; i < flow_u.rows; i++)
    {
        float *pFlowU = flow_u.ptr<float>(i);
        float *pFlowV = flow_v.ptr<float>(i);
        float *pMapX = mapX.ptr<float>(i);
        float *pMapY = mapY.ptr<float>(i);
        for (int j = 0; j < flow_u.cols; j++)
        {
            pMapX[j] = j + pFlowU[j];
            pMapY[j] = i + pFlowV[j];
        }
    }
    remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_REPLICATE);
}

void VariationalRefinementImpl::prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v)
{
    Size s = I0.size();
    A11.create(s);
    A12.create(s);
    A22.create(s);
    b1.create(s);
    b2.create(s);
    weights.create(s);
    weights.red.setTo(0.0f);
    weights.black.setTo(0.0f);
    tempW_u.create(s);
    tempW_v.create(s);
    dW_u.create(s);
    dW_v.create(s);
    W_u_rb.create(s);
    W_v_rb.create(s);

    Ix.create(s);
    Iy.create(s);
    Iz.create(s);
    Ixx.create(s);
    Ixy.create(s);
    Iyy.create(s);
    Ixz.create(s);
    Iyz.create(s);

    Ix_rb.create(s);
    Iy_rb.create(s);
    Iz_rb.create(s);
    Ixx_rb.create(s);
    Ixy_rb.create(s);
    Iyy_rb.create(s);
    Ixz_rb.create(s);
    Iyz_rb.create(s);

    mapX.create(s);
    mapY.create(s);

    /* Floating point warps work significantly better than fixed-point */
    Mat I1flt, warpedI;
    I1.convertTo(I1flt, CV_32F);
    warpImage(warpedI, I1flt, W_u, W_v);

    /* Computing an average of the current and warped next frames (to compute the derivatives on) and
     * temporal derivative Iz
     */
    Mat averagedI;
    {
        vector<void *> op1s;
        op1s.push_back((void *)&I0);
        op1s.push_back((void *)&warpedI);
        vector<void *> op2s;
        op2s.push_back((void *)&warpedI);
        op2s.push_back((void *)&I0);
        vector<void *> op3s;
        op3s.push_back((void *)&averagedI);
        op3s.push_back((void *)&Iz);
        vector<Op> ops;
        ops.push_back(&VariationalRefinementImpl::averageOp);
        ops.push_back(&VariationalRefinementImpl::subtractOp);
        parallel_for_(Range(0, 2), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }
    splitCheckerboard(Iz_rb, Iz);

    /* Computing first-order derivatives */
    {
        vector<void *> op1s;
        op1s.push_back((void *)&averagedI);
        op1s.push_back((void *)&averagedI);
        op1s.push_back((void *)&Iz);
        op1s.push_back((void *)&Iz);
        vector<void *> op2s;
        op2s.push_back((void *)&Ix);
        op2s.push_back((void *)&Iy);
        op2s.push_back((void *)&Ixz);
        op2s.push_back((void *)&Iyz);
        vector<void *> op3s;
        op3s.push_back((void *)&Ix_rb);
        op3s.push_back((void *)&Iy_rb);
        op3s.push_back((void *)&Ixz_rb);
        op3s.push_back((void *)&Iyz_rb);
        vector<Op> ops;
        ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 4), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }

    /* Computing second-order derivatives */
    {
        vector<void *> op1s;
        op1s.push_back((void *)&Ix);
        op1s.push_back((void *)&Ix);
        op1s.push_back((void *)&Iy);
        vector<void *> op2s;
        op2s.push_back((void *)&Ixx);
        op2s.push_back((void *)&Ixy);
        op2s.push_back((void *)&Iyy);
        vector<void *> op3s;
        op3s.push_back((void *)&Ixx_rb);
        op3s.push_back((void *)&Ixy_rb);
        op3s.push_back((void *)&Iyy_rb);
        vector<Op> ops;
        ops.push_back(&VariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&VariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 3), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }
}

VariationalRefinementImpl::ComputeDataTerm_ParBody::ComputeDataTerm_ParBody(VariationalRefinementImpl &_var,
                                                                            int _nstripes, int _h,
                                                                            RedBlackBuffer &_dW_u,
                                                                            RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function computes parts of the main linear system coefficients A11,A12,A22,b1,b1
 * that correspond to the data term, which includes color and gradient constancy assumptions.
 */
void VariationalRefinementImpl::ComputeDataTerm_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float zeta_squared = var->zeta * var->zeta;
    float epsilon_squared = var->epsilon * var->epsilon;
    float gamma2 = var->gamma / 2;
    float delta2 = var->delta / 2;

    float *pIx, *pIy, *pIz;
    float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;

    float derivNorm, derivNorm2;
    float Ik1z, Ik1zx, Ik1zy;
    float weight;
    int len;
    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(color)                                                                                       \
    pIx = var->Ix_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIy = var->Iy_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIz = var->Iz_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIxx = var->Ixx_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxy = var->Ixy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyy = var->Iyy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxz = var->Ixz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyz = var->Iyz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pa11 = var->A11.color.ptr<float>(i + 1) + 1;                                                                       \
    pa12 = var->A12.color.ptr<float>(i + 1) + 1;                                                                       \
    pa22 = var->A22.color.ptr<float>(i + 1) + 1;                                                                       \
    pb1 = var->b1.color.ptr<float>(i + 1) + 1;                                                                         \
    pb2 = var->b2.color.ptr<float>(i + 1) + 1;                                                                         \
    pdU = dW_u->color.ptr<float>(i + 1) + 1;                                                                           \
    pdV = dW_v->color.ptr<float>(i + 1) + 1;                                                                           \
    if (i % 2 == 0)                                                                                                    \
        len = var->Ix_rb.color##_even_len;                                                                             \
    else                                                                                                               \
        len = var->Ix_rb.color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red);
        }
        else
        {
            INIT_ROW_POINTERS(black);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 zeta_vec = v_setall_f32(zeta_squared);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 delta_vec = v_setall_f32(delta2);
        v_float32x4 gamma_vec = v_setall_f32(gamma2);
        v_float32x4 zero_vec = v_setall_f32(0.0f);
        v_float32x4 pIx_vec, pIy_vec, pIz_vec, pdU_vec, pdV_vec;
        v_float32x4 pIxx_vec, pIxy_vec, pIyy_vec, pIxz_vec, pIyz_vec;
        v_float32x4 derivNorm_vec, derivNorm2_vec, weight_vec;
        v_float32x4 Ik1z_vec, Ik1zx_vec, Ik1zy_vec;
        v_float32x4 pa11_vec, pa12_vec, pa22_vec, pb1_vec, pb2_vec;

        for (; j < len - 3; j += 4)
        {
            pIx_vec = v_load(pIx + j);
            pIy_vec = v_load(pIy + j);
            pIz_vec = v_load(pIz + j);
            pdU_vec = v_load(pdU + j);
            pdV_vec = v_load(pdV + j);

            derivNorm_vec = pIx_vec * pIx_vec + pIy_vec * pIy_vec + zeta_vec;
            Ik1z_vec = pIz_vec + pIx_vec * pdU_vec + pIy_vec * pdV_vec;
            weight_vec = (delta_vec / v_sqrt(Ik1z_vec * Ik1z_vec / derivNorm_vec + eps_vec)) / derivNorm_vec;

            pa11_vec = weight_vec * (pIx_vec * pIx_vec) + zeta_vec;
            pa12_vec = weight_vec * (pIx_vec * pIy_vec);
            pa22_vec = weight_vec * (pIy_vec * pIy_vec) + zeta_vec;
            pb1_vec = zero_vec - weight_vec * (pIz_vec * pIx_vec);
            pb2_vec = zero_vec - weight_vec * (pIz_vec * pIy_vec);

            pIxx_vec = v_load(pIxx + j);
            pIxy_vec = v_load(pIxy + j);
            pIyy_vec = v_load(pIyy + j);
            pIxz_vec = v_load(pIxz + j);
            pIyz_vec = v_load(pIyz + j);

            derivNorm_vec = pIxx_vec * pIxx_vec + pIxy_vec * pIxy_vec + zeta_vec;
            derivNorm2_vec = pIyy_vec * pIyy_vec + pIxy_vec * pIxy_vec + zeta_vec;
            Ik1zx_vec = pIxz_vec + pIxx_vec * pdU_vec + pIxy_vec * pdV_vec;
            Ik1zy_vec = pIyz_vec + pIxy_vec * pdU_vec + pIyy_vec * pdV_vec;
            weight_vec = gamma_vec / v_sqrt(Ik1zx_vec * Ik1zx_vec / derivNorm_vec +
                                            Ik1zy_vec * Ik1zy_vec / derivNorm2_vec + eps_vec);

            pa11_vec += weight_vec * (pIxx_vec * pIxx_vec / derivNorm_vec + pIxy_vec * pIxy_vec / derivNorm2_vec);
            pa12_vec += weight_vec * (pIxx_vec * pIxy_vec / derivNorm_vec + pIxy_vec * pIyy_vec / derivNorm2_vec);
            pa22_vec += weight_vec * (pIxy_vec * pIxy_vec / derivNorm_vec + pIyy_vec * pIyy_vec / derivNorm2_vec);
            pb1_vec -= weight_vec * (pIxx_vec * pIxz_vec / derivNorm_vec + pIxy_vec * pIyz_vec / derivNorm2_vec);
            pb2_vec -= weight_vec * (pIxy_vec * pIxz_vec / derivNorm_vec + pIyy_vec * pIyz_vec / derivNorm2_vec);

            v_store(pa11 + j, pa11_vec);
            v_store(pa12 + j, pa12_vec);
            v_store(pa22 + j, pa22_vec);
            v_store(pb1 + j, pb1_vec);
            v_store(pb2 + j, pb2_vec);
        }
#endif
        for (; j < len; j++)
        {
            /* Step 1: Compute color constancy terms */
            /* Normalization factor:*/
            derivNorm = pIx[j] * pIx[j] + pIy[j] * pIy[j] + zeta_squared;
            /* Color constancy penalty (computed by Taylor expansion):*/
            Ik1z = pIz[j] + pIx[j] * pdU[j] + pIy[j] * pdV[j];
            /* Weight of the color constancy term in the current fixed-point iteration divided by derivNorm: */
            weight = (delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared)) / derivNorm;
            /* Add respective color constancy terms to the linear system coefficients: */
            pa11[j] = weight * (pIx[j] * pIx[j]) + zeta_squared;
            pa12[j] = weight * (pIx[j] * pIy[j]);
            pa22[j] = weight * (pIy[j] * pIy[j]) + zeta_squared;
            pb1[j] = -weight * (pIz[j] * pIx[j]);
            pb2[j] = -weight * (pIz[j] * pIy[j]);

            /* Step 2: Compute gradient constancy terms */
            /* Normalization factor for x gradient: */
            derivNorm = pIxx[j] * pIxx[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Normalization factor for y gradient: */
            derivNorm2 = pIyy[j] * pIyy[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Gradient constancy penalties (computed by Taylor expansion): */
            Ik1zx = pIxz[j] + pIxx[j] * pdU[j] + pIxy[j] * pdV[j];
            Ik1zy = pIyz[j] + pIxy[j] * pdU[j] + pIyy[j] * pdV[j];
            /* Weight of the gradient constancy term in the current fixed-point iteration: */
            weight = gamma2 / sqrt(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + epsilon_squared);
            /* Add respective gradient constancy components to the linear system coefficients: */
            pa11[j] += weight * (pIxx[j] * pIxx[j] / derivNorm + pIxy[j] * pIxy[j] / derivNorm2);
            pa12[j] += weight * (pIxx[j] * pIxy[j] / derivNorm + pIxy[j] * pIyy[j] / derivNorm2);
            pa22[j] += weight * (pIxy[j] * pIxy[j] / derivNorm + pIyy[j] * pIyy[j] / derivNorm2);
            pb1[j] += -weight * (pIxx[j] * pIxz[j] / derivNorm + pIxy[j] * pIyz[j] / derivNorm2);
            pb2[j] += -weight * (pIxy[j] * pIxz[j] / derivNorm + pIyy[j] * pIyz[j] / derivNorm2);
        }
    }
}

VariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::ComputeSmoothnessTermHorPass_ParBody(
  VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
  RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), W_u(&_W_u), W_v(&_W_v), curW_u(&_tempW_u), curW_v(&_tempW_v),
      red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function updates the linear system coefficients A11,A22,b1,b1 according to the
 * flow smoothness term and computes corresponding weights for the current fixed point iteration.
 * A11,A22,b1,b1 are updated only partially (horizontal pass). Doing both horizontal and vertical
 * passes in one loop complicates parallelization (different threads write to the same elements).
 */
void VariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float epsilon_squared = var->epsilon * var->epsilon;
    float alpha2 = var->alpha / 2;
    float *pWeight;
    float *pA_u, *pA_u_next, *pA_v, *pA_v_next;
    float *pB_u, *pB_u_next, *pB_v, *pB_v_next;
    float *cW_u, *cW_u_next, *cW_u_next_row;
    float *cW_v, *cW_v_next, *cW_v_next_row;
    float *pW_u, *pW_u_next;
    float *pW_v, *pW_v_next;
    float ux, uy, vx, vy;
    int len;
    bool touches_right_border = true;

#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd, bool_default)                          \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_u = curW_u->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_v = curW_v->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    cW_u_next_row = curW_u->next_color.ptr<float>(i + 2) + 1;                                                          \
    cW_v_next_row = curW_v->next_color.ptr<float>(i + 2) + 1;                                                          \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = bool_default;                                                                       \
        else                                                                                                           \
            touches_right_border = !bool_default;                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = !bool_default;                                                                      \
        else                                                                                                           \
            touches_right_border = bool_default;                                                                       \
    }

    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2, true);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1, false);
        }
#undef INIT_ROW_POINTERS

#define COMPUTE                                                                                                        \
    /* Gradients for the flow on the current fixed-point iteration: */                                                 \
    ux = cW_u_next[j] - cW_u[j];                                                                                       \
    vx = cW_v_next[j] - cW_v[j];                                                                                       \
    uy = cW_u_next_row[j] - cW_u[j];                                                                                   \
    vy = cW_v_next_row[j] - cW_v[j];                                                                                   \
    /* Weight of the smoothness term in the current fixed-point iteration: */                                          \
    pWeight[j] = alpha2 / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon_squared);                               \
    /* Gradients for initial raw flow multiplied by weight:*/                                                          \
    ux = pWeight[j] * (pW_u_next[j] - pW_u[j]);                                                                        \
    vx = pWeight[j] * (pW_v_next[j] - pW_v[j]);

#define UPDATE                                                                                                         \
    pB_u[j] += ux;                                                                                                     \
    pA_u[j] += pWeight[j];                                                                                             \
    pB_v[j] += vx;                                                                                                     \
    pA_v[j] += pWeight[j];                                                                                             \
    pB_u_next[j] -= ux;                                                                                                \
    pA_u_next[j] += pWeight[j];                                                                                        \
    pB_v_next[j] -= vx;                                                                                                \
    pA_v_next[j] += pWeight[j];

        int j = 0;
#if CV_SIMD128
        v_float32x4 alpha2_vec = v_setall_f32(alpha2);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 cW_u_vec, cW_v_vec;
        v_float32x4 pWeight_vec, ux_vec, vx_vec, uy_vec, vy_vec;

        for (; j < len - 4; j += 4)
        {
            cW_u_vec = v_load(cW_u + j);
            cW_v_vec = v_load(cW_v + j);

            ux_vec = v_load(cW_u_next + j) - cW_u_vec;
            vx_vec = v_load(cW_v_next + j) - cW_v_vec;
            uy_vec = v_load(cW_u_next_row + j) - cW_u_vec;
            vy_vec = v_load(cW_v_next_row + j) - cW_v_vec;
            pWeight_vec =
              alpha2_vec / v_sqrt(ux_vec * ux_vec + vx_vec * vx_vec + uy_vec * uy_vec + vy_vec * vy_vec + eps_vec);
            v_store(pWeight + j, pWeight_vec);

            ux_vec = pWeight_vec * (v_load(pW_u_next + j) - v_load(pW_u + j));
            vx_vec = pWeight_vec * (v_load(pW_v_next + j) - v_load(pW_v + j));

            v_store(pA_u + j, v_load(pA_u + j) + pWeight_vec);
            v_store(pA_v + j, v_load(pA_v + j) + pWeight_vec);
            v_store(pB_u + j, v_load(pB_u + j) + ux_vec);
            v_store(pB_v + j, v_load(pB_v + j) + vx_vec);

            v_store(pA_u_next + j, v_load(pA_u_next + j) + pWeight_vec);
            v_store(pA_v_next + j, v_load(pA_v_next + j) + pWeight_vec);
            v_store(pB_u_next + j, v_load(pB_u_next + j) - ux_vec);
            v_store(pB_v_next + j, v_load(pB_v_next + j) - vx_vec);
        }
#endif
        for (; j < len - 1; j++)
        {
            COMPUTE;
            UPDATE;
        }

        /* Omit the update on the rightmost elements */
        if (touches_right_border)
        {
            COMPUTE;
        }
        else
        {
            COMPUTE;
            UPDATE;
        }
    }
#undef COMPUTE
#undef UPDATE
}

VariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::ComputeSmoothnessTermVertPass_ParBody(
  VariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), W_u(&_W_u), W_v(&_W_v), red_pass(_red_pass)
{
    /* Omit the last row in the vertical pass */
    h = _h - 1;
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function adds the last remaining terms to the linear system coefficients A11,A22,b1,b1. */
void VariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float *pWeight;
    float *pA_u, *pA_u_next_row, *pA_v, *pA_v_next_row;
    float *pB_u, *pB_u_next_row, *pB_v, *pB_v_next_row;
    float *pW_u, *pW_u_next_row, *pW_v, *pW_v_next_row;
    float vy, uy;
    int len;

    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color)                                                                       \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pA_u_next_row = var->A11.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_u_next_row = var->b1.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_u_next_row = W_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pA_v_next_row = var->A22.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_v_next_row = var->b2.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_v_next_row = W_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
        len = var->A11.cur_color##_even_len;                                                                           \
    else                                                                                                               \
        len = var->A11.cur_color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black);
        }
        else
        {
            INIT_ROW_POINTERS(black, red);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 pWeight_vec, uy_vec, vy_vec;
        for (; j < len - 3; j += 4)
        {
            pWeight_vec = v_load(pWeight + j);
            uy_vec = pWeight_vec * (v_load(pW_u_next_row + j) - v_load(pW_u + j));
            vy_vec = pWeight_vec * (v_load(pW_v_next_row + j) - v_load(pW_v + j));

            v_store(pA_u + j, v_load(pA_u + j) + pWeight_vec);
            v_store(pA_v + j, v_load(pA_v + j) + pWeight_vec);
            v_store(pB_u + j, v_load(pB_u + j) + uy_vec);
            v_store(pB_v + j, v_load(pB_v + j) + vy_vec);

            v_store(pA_u_next_row + j, v_load(pA_u_next_row + j) + pWeight_vec);
            v_store(pA_v_next_row + j, v_load(pA_v_next_row + j) + pWeight_vec);
            v_store(pB_u_next_row + j, v_load(pB_u_next_row + j) - uy_vec);
            v_store(pB_v_next_row + j, v_load(pB_v_next_row + j) - vy_vec);
        }
#endif
        for (; j < len; j++)
        {
            uy = pWeight[j] * (pW_u_next_row[j] - pW_u[j]);
            vy = pWeight[j] * (pW_v_next_row[j] - pW_v[j]);
            pB_u[j] += uy;
            pA_u[j] += pWeight[j];
            pB_v[j] += vy;
            pA_v[j] += pWeight[j];
            pB_u_next_row[j] -= uy;
            pA_u_next_row[j] += pWeight[j];
            pB_v_next_row[j] -= vy;
            pA_v_next_row[j] += pWeight[j];
        }
    }
}

VariationalRefinementImpl::RedBlackSOR_ParBody::RedBlackSOR_ParBody(VariationalRefinementImpl &_var, int _nstripes,
                                                                    int _h, RedBlackBuffer &_dW_u,
                                                                    RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function implements the Red-Black SOR (successive-over relaxation) method for solving the main
 * linear system in the current fixed-point iteration.
 */
void VariationalRefinementImpl::RedBlackSOR_ParBody::operator()(const Range &range) const
{
    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);

    float *pa11, *pa12, *pa22, *pb1, *pb2, *pW, *pdu, *pdv;
    float *pW_next, *pdu_next, *pdv_next;
    float *pW_prev_row, *pdu_prev_row, *pdv_prev_row;
    float *pdu_next_row, *pdv_next_row;

    float sigmaU, sigmaV;
    int j, len;
    for (int i = start; i < end; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd)                                        \
    pW = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                                 \
    pa11 = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa12 = var->A12.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa22 = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pb1 = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pb2 = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pdu = dW_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pdv = dW_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pdu_next_row = dW_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pdv_next_row = dW_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    pW_prev_row = var->weights.next_color.ptr<float>(i) + 1;                                                           \
    pdu_prev_row = dW_u->next_color.ptr<float>(i) + 1;                                                                 \
    pdv_prev_row = dW_v->next_color.ptr<float>(i) + 1;                                                                 \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pW_next = var->weights.next_color.ptr<float>(i + 1) + next_offs_even;                                          \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pW_next = var->weights.next_color.ptr<float>(i + 1) + next_offs_odd;                                           \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
    }
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1);
        }
#undef INIT_ROW_POINTERS

        j = 0;
#if CV_SIMD128
        v_float32x4 pW_prev_vec = v_setall_f32(pW_next[-1]);
        v_float32x4 pdu_prev_vec = v_setall_f32(pdu_next[-1]);
        v_float32x4 pdv_prev_vec = v_setall_f32(pdv_next[-1]);
        v_float32x4 omega_vec = v_setall_f32(var->omega);
        v_float32x4 pW_vec, pW_next_vec, pW_prev_row_vec;
        v_float32x4 pdu_next_vec, pdu_prev_row_vec, pdu_next_row_vec;
        v_float32x4 pdv_next_vec, pdv_prev_row_vec, pdv_next_row_vec;
        v_float32x4 pW_shifted_vec, pdu_shifted_vec, pdv_shifted_vec;
        v_float32x4 pa12_vec, sigmaU_vec, sigmaV_vec, pdu_vec, pdv_vec;
        for (; j < len - 3; j += 4)
        {
            pW_vec = v_load(pW + j);
            pW_next_vec = v_load(pW_next + j);
            pW_prev_row_vec = v_load(pW_prev_row + j);
            pdu_next_vec = v_load(pdu_next + j);
            pdu_prev_row_vec = v_load(pdu_prev_row + j);
            pdu_next_row_vec = v_load(pdu_next_row + j);
            pdv_next_vec = v_load(pdv_next + j);
            pdv_prev_row_vec = v_load(pdv_prev_row + j);
            pdv_next_row_vec = v_load(pdv_next_row + j);
            pa12_vec = v_load(pa12 + j);
            pW_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pW_prev_vec), v_reinterpret_as_s32(pW_next_vec)));
            pdu_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdu_prev_vec), v_reinterpret_as_s32(pdu_next_vec)));
            pdv_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdv_prev_vec), v_reinterpret_as_s32(pdv_next_vec)));

            sigmaU_vec = pW_shifted_vec * pdu_shifted_vec + pW_vec * pdu_next_vec + pW_prev_row_vec * pdu_prev_row_vec +
                         pW_vec * pdu_next_row_vec;
            sigmaV_vec = pW_shifted_vec * pdv_shifted_vec + pW_vec * pdv_next_vec + pW_prev_row_vec * pdv_prev_row_vec +
                         pW_vec * pdv_next_row_vec;

            pdu_vec = v_load(pdu + j);
            pdv_vec = v_load(pdv + j);
            pdu_vec += omega_vec * ((sigmaU_vec + v_load(pb1 + j) - pdv_vec * pa12_vec) / v_load(pa11 + j) - pdu_vec);
            pdv_vec += omega_vec * ((sigmaV_vec + v_load(pb2 + j) - pdu_vec * pa12_vec) / v_load(pa22 + j) - pdv_vec);
            v_store(pdu + j, pdu_vec);
            v_store(pdv + j, pdv_vec);

            pW_prev_vec = pW_next_vec;
            pdu_prev_vec = pdu_next_vec;
            pdv_prev_vec = pdv_next_vec;
        }
#endif
        for (; j < len; j++)
        {
            sigmaU = pW_next[j - 1] * pdu_next[j - 1] + pW[j] * pdu_next[j] + pW_prev_row[j] * pdu_prev_row[j] +
                     pW[j] * pdu_next_row[j];
            sigmaV = pW_next[j - 1] * pdv_next[j - 1] + pW[j] * pdv_next[j] + pW_prev_row[j] * pdv_prev_row[j] +
                     pW[j] * pdv_next_row[j];
            pdu[j] += var->omega * ((sigmaU + pb1[j] - pdv[j] * pa12[j]) / pa11[j] - pdu[j]);
            pdv[j] += var->omega * ((sigmaV + pb2[j] - pdu[j] * pa12[j]) / pa22[j] - pdv[j]);
        }
    }
}

void VariationalRefinementImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) || (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!flow.empty() && flow.depth() == CV_32F && flow.channels() == 2);
    CV_Assert(I0.sameSize(flow));

    Mat uv[2];
    Mat &flowMat = flow.getMatRef();
    split(flowMat, uv);
    calcUV(I0, I1, uv[0], uv[1]);
    merge(uv, 2, flowMat);
}

void VariationalRefinementImpl::calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v)
{
    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) || (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!flow_u.empty() && flow_u.depth() == CV_32F && flow_u.channels() == 1);
    CV_Assert(!flow_v.empty() && flow_v.depth() == CV_32F && flow_v.channels() == 1);
    CV_Assert(I0.sameSize(flow_u));
    CV_Assert(flow_u.sameSize(flow_v));

    int num_stripes = getNumThreads();
    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    Mat &W_u = flow_u.getMatRef();
    Mat &W_v = flow_v.getMatRef();
    prepareBuffers(I0Mat, I1Mat, W_u, W_v);

    splitCheckerboard(W_u_rb, W_u);
    splitCheckerboard(W_v_rb, W_v);
    W_u_rb.red.copyTo(tempW_u.red);
    W_u_rb.black.copyTo(tempW_u.black);
    W_v_rb.red.copyTo(tempW_v.red);
    W_v_rb.black.copyTo(tempW_v.black);
    dW_u.red.setTo(0.0f);
    dW_u.black.setTo(0.0f);
    dW_v.red.setTo(0.0f);
    dW_v.black.setTo(0.0f);

    for (int i = 0; i < fixedPointIterations; i++)
    {
        parallel_for_(Range(0, num_stripes), ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));

        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(
                                               *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, tempW_u, tempW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(
                                               *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, tempW_u, tempW_v, false));

        parallel_for_(Range(0, num_stripes),
                      ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, true));
        parallel_for_(Range(0, num_stripes),
                      ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, false));

        for (int j = 0; j < sorIterations; j++)
        {
            parallel_for_(Range(0, num_stripes), RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
            parallel_for_(Range(0, num_stripes), RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));
        }

        tempW_u.red = W_u_rb.red + dW_u.red;
        tempW_u.black = W_u_rb.black + dW_u.black;
        updateRepeatedBorders(tempW_u);
        tempW_v.red = W_v_rb.red + dW_v.red;
        tempW_v.black = W_v_rb.black + dW_v.black;
        updateRepeatedBorders(tempW_v);
    }
    mergeCheckerboard(W_u, tempW_u);
    mergeCheckerboard(W_v, tempW_v);
}
void VariationalRefinementImpl::collectGarbage()
{
    Ix.release();
    Iy.release();
    Iz.release();
    Ixx.release();
    Ixy.release();
    Iyy.release();
    Ixz.release();
    Iyz.release();

    Ix_rb.release();
    Iy_rb.release();
    Iz_rb.release();
    Ixx_rb.release();
    Ixy_rb.release();
    Iyy_rb.release();
    Ixz_rb.release();
    Iyz_rb.release();

    A11.release();
    A12.release();
    A22.release();
    b1.release();
    b2.release();
    weights.release();

    mapX.release();
    mapY.release();

    tempW_u.release();
    tempW_v.release();
    dW_u.release();
    dW_v.release();
    W_u_rb.release();
    W_v_rb.release();
}

Ptr<VariationalRefinement> VariationalRefinement::create()
{ return makePtr<VariationalRefinementImpl>(); }

}
