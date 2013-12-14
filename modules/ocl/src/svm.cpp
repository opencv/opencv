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
// Copyright (C) 2010-2013, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Erping Pang, erping@multicorewareinc.com
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
#include "opencl_kernels.hpp"

// TODO Remove this after HAVE_CLAMDBLAS eliminating
#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

using namespace cv;
using namespace ocl;

namespace cv { namespace ocl {

#if 1
typedef float Qfloat;
#define QFLOAT_TYPE CV_32F
#else
typedef double Qfloat;
#define QFLOAT_TYPE CV_64F
#endif

class CvSVMKernel_ocl: public CvSVMKernel
{
public:
    typedef void (CvSVMKernel_ocl::*Calc_ocl)( int vec_count, const int row_idx, Qfloat* results, Mat& src);
    CvSVMKernel_ocl(const CvSVMParams* params, Calc_ocl _calc_func , Calc _calc_func1);

    Calc_ocl calc_func_ocl;
    bool create( const CvSVMParams* params, Calc_ocl _calc_func, Calc _calc_func1);

    void calc( int vcount, const int row_idx, Qfloat* results, Mat& src);
    void calc_linear( int vec_count, const int row_idx, Qfloat* results, Mat& src);

    void calc_poly( int vec_count, const int row_idx, Qfloat* results, Mat& src);
    void calc_sigmoid( int vec_count, const int row_idx, Qfloat* results, Mat& src);
    void calc_non_rbf_base( int vec_count, const int row_idx, Qfloat* results, Mat& src);
    void calc_rbf( int vec_count, const int row_idx, Qfloat* results, Mat& src);
};

class CvSVMSolver_ocl: public CvSVMSolver
{
public:
    CvSVMSolver_ocl();
    CvSVMSolver_ocl(const CvSVMParams *);
    float* get_row_base( int i, bool* _existed, Mat& src);
    bool solve_generic( CvSVMSolutionInfo& si );
    float* get_row( int i, float* dst, Mat& src);
};

typedef struct CvSparseVecElem32f
{
    int idx;
    float val;
} CvSparseVecElem32f;

static int icvCmpSparseVecElems( const void* a, const void* b )
{
    return ((CvSparseVecElem32f*)a)->idx - ((CvSparseVecElem32f*)b)->idx;
}

void cvPreparePredictData( const CvArr* sample, int dims_all, const CvMat* comp_idx,
                           int class_count, const CvMat* prob, float** row_sample,
                           int as_sparse CV_DEFAULT(0) );

void  cvPreparePredictData( const CvArr* _sample, int dims_all,
                            const CvMat* comp_idx, int class_count,
                            const CvMat* prob, float** _row_sample,
                            int as_sparse )
{
    float* row_sample = 0;
    int* inverse_comp_idx = 0;

    CV_FUNCNAME( "cvPreparePredictData" );

    __BEGIN__;

    const CvMat* sample = (const CvMat*)_sample;
    float* sample_data;
    int sample_step;
    int is_sparse = CV_IS_SPARSE_MAT(sample);
    int d, sizes[CV_MAX_DIM];
    int i, dims_selected;
    int vec_size;

    if( !is_sparse && !CV_IS_MAT(sample) )
    {
        CV_ERROR( !sample ? CV_StsNullPtr : CV_StsBadArg, "The sample is not a valid vector" );
    }

    if( cvGetElemType( sample ) != CV_32FC1 )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input sample must have 32fC1 type" );
    }

    CV_CALL( d = cvGetDims( sample, sizes ));

    if( !((is_sparse && d == 1) || (!is_sparse && d == 2 && (sample->rows == 1 || sample->cols == 1))) )
    {
        CV_ERROR( CV_StsBadSize, "Input sample must be 1-dimensional vector" );
    }

    if( d == 1 )
        sizes[1] = 1;

    if( sizes[0] + sizes[1] - 1 != dims_all )
        CV_ERROR( CV_StsUnmatchedSizes,
                  "The sample size is different from what has been used for training" );

    if( !_row_sample )
    {
        CV_ERROR( CV_StsNullPtr, "INTERNAL ERROR: The row_sample pointer is NULL" );
    }

    if( comp_idx && (!CV_IS_MAT(comp_idx) || comp_idx->rows != 1 ||
                     CV_MAT_TYPE(comp_idx->type) != CV_32SC1) )
    {
        CV_ERROR( CV_StsBadArg, "INTERNAL ERROR: invalid comp_idx" );
    }

    dims_selected = comp_idx ? comp_idx->cols : dims_all;

    if( prob )
    {
        if( !CV_IS_MAT(prob) )
        {
            CV_ERROR( CV_StsBadArg, "The output matrix of probabilities is invalid" );
        }

        if( (prob->rows != 1 && prob->cols != 1) ||
                (CV_MAT_TYPE(prob->type) != CV_32FC1 &&
                 CV_MAT_TYPE(prob->type) != CV_64FC1) )
            CV_ERROR( CV_StsBadSize,
                      "The matrix of probabilities must be 1-dimensional vector of 32fC1 type" );

        if( prob->rows + prob->cols - 1 != class_count )
            CV_ERROR( CV_StsUnmatchedSizes,
                      "The vector of probabilities must contain as many elements as "
                      "the number of classes in the training set" );
    }

    vec_size = !as_sparse ? dims_selected * sizeof(row_sample[0]) :
               (dims_selected + 1) * sizeof(CvSparseVecElem32f);

    if( CV_IS_MAT(sample) )
    {
        sample_data = sample->data.fl;
        sample_step = CV_IS_MAT_CONT(sample->type) ? 1 : sample->step / sizeof(row_sample[0]);

        if( !comp_idx && CV_IS_MAT_CONT(sample->type) && !as_sparse )
            *_row_sample = sample_data;
        else
        {
            CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

            if( !comp_idx )
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step * i];
            else
            {
                int* comp = comp_idx->data.i;
                for( i = 0; i < dims_selected; i++ )
                    row_sample[i] = sample_data[sample_step * comp[i]];
            }

            *_row_sample = row_sample;
        }

        if( as_sparse )
        {
            const float* src = (const float*)row_sample;
            CvSparseVecElem32f* dst = (CvSparseVecElem32f*)row_sample;

            dst[dims_selected].idx = -1;
            for( i = dims_selected - 1; i >= 0; i-- )
            {
                dst[i].idx = i;
                dst[i].val = src[i];
            }
        }
    }
    else
    {
        CvSparseNode* node;
        CvSparseMatIterator mat_iterator;
        const CvSparseMat* sparse = (const CvSparseMat*)sample;
        assert( is_sparse );

        node = cvInitSparseMatIterator( sparse, &mat_iterator );
        CV_CALL( row_sample = (float*)cvAlloc( vec_size ));

        if( comp_idx )
        {
            CV_CALL( inverse_comp_idx = (int*)cvAlloc( dims_all * sizeof(int) ));
            memset( inverse_comp_idx, -1, dims_all * sizeof(int) );
            for( i = 0; i < dims_selected; i++ )
                inverse_comp_idx[comp_idx->data.i[i]] = i;
        }

        if( !as_sparse )
        {
            memset( row_sample, 0, vec_size );

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                row_sample[idx] = *(float*)CV_NODE_VAL( sparse, node );
            }
        }
        else
        {
            CvSparseVecElem32f* ptr = (CvSparseVecElem32f*)row_sample;

            for( ; node != 0; node = cvGetNextSparseNode(&mat_iterator) )
            {
                int idx = *CV_NODE_IDX( sparse, node );
                if( inverse_comp_idx )
                {
                    idx = inverse_comp_idx[idx];
                    if( idx < 0 )
                        continue;
                }
                ptr->idx = idx;
                ptr->val = *(float*)CV_NODE_VAL( sparse, node );
                ptr++;
            }

            qsort( row_sample, ptr - (CvSparseVecElem32f*)row_sample,
                   sizeof(ptr[0]), icvCmpSparseVecElems );
            ptr->idx = -1;
        }

        *_row_sample = row_sample;
    }

    __END__;

    if( inverse_comp_idx )
        cvFree( &inverse_comp_idx );

    if( cvGetErrStatus() < 0 && _row_sample )
    {
        cvFree( &row_sample );
        *_row_sample = 0;
    }
}

float CvSVM_OCL::predict( const int row_index, int row_len, Mat& src, bool returnDFVal ) const
{
    assert( kernel );

    (void)row_len;

    int class_count = class_labels ? class_labels->cols :
                      params.svm_type == ONE_CLASS ? 1 : 0;

    float result = 0;
    cv::AutoBuffer<float> _buffer(sv_total + (class_count + 1) * 2);
    float* buffer = _buffer;

    if( params.svm_type == EPS_SVR ||
            params.svm_type == NU_SVR ||
            params.svm_type == ONE_CLASS )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int i, sv_count = df->sv_count;
        double sum = -df->rho;

        ((CvSVMKernel_ocl*)kernel)->calc( sv_count, row_index, buffer, src);
        for( i = 0; i < sv_count; i++ )
            sum += buffer[i] * df->alpha[i];

        result = params.svm_type == ONE_CLASS ? (float)(sum > 0) : (float)sum;
    }
    else if( params.svm_type == C_SVC ||
             params.svm_type == NU_SVC )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int* vote = (int*)(buffer + sv_total);
        int i, j, k;

        memset( vote, 0, class_count * sizeof(vote[0]));
        ((CvSVMKernel_ocl*)kernel)->calc( sv_total, row_index, buffer, src);
        double sum = 0.;

        for( i = 0; i < class_count; i++ )
            for( j = i + 1; j < class_count; j++, df++ )
            {
                sum = -df->rho;
                int sv_count = df->sv_count;
                for( k = 0; k < sv_count; k++ )
                    sum += df->alpha[k] * buffer[df->sv_index[k]];

                vote[sum > 0 ? i : j]++;
            }

        for( i = 1, k = 0; i < class_count; i++ )
            if( vote[i] > vote[k] )
                k = i;

        result = returnDFVal && class_count == 2 ? (float)sum : (float)(class_labels->data.i[k]);
    }
    else
        CV_Error( CV_StsBadArg, "INTERNAL ERROR: Unknown SVM type, "
                  "the SVM structure is probably corrupted" );

    return result;
}

float CvSVM_OCL::predict( const Mat& _sample, bool returnDFVal ) const
{
    CvMat sample = _sample;
    return CvSVM::predict(&sample, returnDFVal);
}

float CvSVM_OCL::predict( const int row_index, Mat& src, bool returnDFVal) const
{
    float result = 0;

    result = predict( row_index, get_var_count(), src, returnDFVal);

    return result;
}

#undef get_C
#define get_C(i) (C[y[i]>0])
#undef is_upper_bound
#define is_upper_bound(i) (alpha_status[i] > 0)
#undef is_lower_bound
#define is_lower_bound(i) (alpha_status[i] < 0)
#undef update_alpha_status
#define update_alpha_status(i) \
    alpha_status[i] = (schar)(alpha[i] >= get_C(i) ? 1 : alpha[i] <= 0 ? -1 : 0)

CvSVMSolver_ocl::CvSVMSolver_ocl(const CvSVMParams* _params)
{
    params = _params;
}

float* CvSVMSolver_ocl::get_row( int i, float* dst, Mat& src )
{
    bool existed = false;
    float* row = get_row_base( i, &existed, src);
    return (this->*get_row_func)( i, row, dst, existed );
}

float* CvSVMSolver_ocl::get_row_base( int i, bool* _existed, Mat& src )
{
    int i1 = i < sample_count ? i : i - sample_count;
    CvSVMKernelRow* row = rows + i1;
    bool existed = row->data != 0;
    Qfloat* data;

    if( existed || cache_size <= 0 )
    {
        CvSVMKernelRow* del_row = existed ? row : lru_list.prev;
        data = del_row->data;
        assert( data != 0 );

        // delete row from the LRU list
        del_row->data = 0;
        del_row->prev->next = del_row->next;
        del_row->next->prev = del_row->prev;
    }
    else
    {
        data = (Qfloat*)cvMemStorageAlloc( storage, cache_line_size );
        cache_size -= cache_line_size;
    }

    // insert row into the LRU list
    row->data = data;
    row->prev = &lru_list;
    row->next = lru_list.next;
    row->prev->next = row->next->prev = row;

    if( !existed )
        ((CvSVMKernel_ocl*)kernel)->calc( sample_count, i1, row->data, src);

    if( _existed )
        *_existed = existed;

    return row->data;
}

#ifndef HAVE_CLAMDBLAS

static void matmul_sigmod(oclMat & src, oclMat & src2, oclMat & dst, int src_rows, int src2_cols, int var_count, double alpha1, double beta1)
{
    Context *clCxt = Context::getContext();
    string kernelName = "svm_sigmod";
    int src_step = (int)src.step / src.elemSize();
    int src2_step = (int)src2.step / src2.elemSize();
    int dst_step = (int)dst.step / dst.elemSize();
    int x = MIN(16, src_rows);
    int y = MIN(16, src2_cols);
    size_t localThreads[] = {x, y, 1};
    size_t globalThreads[] = {src2_cols, src_rows, 1};
    int width = var_count;

    vector< pair<size_t, const void *> > args;
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src2.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&dst.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_rows));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_cols));
    args.push_back(make_pair(sizeof(cl_int), (void* )&width));

    float alpha = 0.0f, beta = 0.0f;
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        alpha = (float)alpha1;
        beta = (float)beta1;
        args.push_back(make_pair(sizeof(cl_float), (void* )&alpha));
        args.push_back(make_pair(sizeof(cl_float), (void* )&beta));
    }
    else
    {
        args.push_back(make_pair(sizeof(cl_double), (void* )&alpha1));
        args.push_back(make_pair(sizeof(cl_double), (void* )&beta1));
    }
    openCLExecuteKernel(clCxt, &svm, kernelName, globalThreads, localThreads, args, -1, -1);
}

static void matmul_poly(oclMat & src, oclMat & src2, oclMat & dst, int src_rows, int src2_cols, int var_count, double alpha1, double beta1, double degree1, bool flag)
{
    Context *clCxt = Context::getContext();
    string kernelName = "svm_poly";
    int src_step = (int)src.step / src.elemSize();
    int src2_step = (int)src2.step / src2.elemSize();
    int dst_step = (int)dst.step / dst.elemSize();
    int x = MIN(16, src_rows);
    int y = MIN(16, src2_cols);
    size_t localThreads[] = {x, y, 1};
    size_t globalThreads[] = {src2_cols, src_rows, 1};
    int width = var_count;

    char build_options[50];

    if(flag)
    {
        sprintf(build_options, "-D ADDPOW");
    }
    vector< pair<size_t, const void *> > args;
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src2.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&dst.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_rows));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_cols));
    args.push_back(make_pair(sizeof(cl_int), (void* )&width));

    float alpha = 0.0f, beta = 0.0f, degree = 0.0f;
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        alpha = (float)alpha1;
        beta = (float)beta1;
        degree = (float)degree1;
        args.push_back(make_pair(sizeof(cl_float), (void* )&alpha));
        args.push_back(make_pair(sizeof(cl_float), (void* )&beta));
        args.push_back(make_pair(sizeof(cl_float), (void* )&degree));
    }
    else
    {
        args.push_back(make_pair(sizeof(cl_double), (void* )&alpha1));
        args.push_back(make_pair(sizeof(cl_double), (void* )&beta1));
        args.push_back(make_pair(sizeof(cl_double), (void* )&degree1));
    }
    openCLExecuteKernel(clCxt, &svm, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

static void matmul_linear(oclMat & src, oclMat & src2, oclMat & dst, int src_rows, int src2_cols, int var_count, double alpha1, double beta1)
{
    Context *clCxt = Context::getContext();
    string kernelName = "svm_linear";
    int src_step = (int)src.step / src.elemSize();
    int src2_step = (int)src2.step / src2.elemSize();
    int dst_step = (int)dst.step / dst.elemSize();
    int x = MIN(16, src_rows);
    int y = MIN(16, src2_cols);
    size_t localThreads[] = {x, y, 1};
    size_t globalThreads[] = {src2_cols, src_rows, 1};
    int width = var_count;

    vector< pair<size_t, const void *> > args;
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src2.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&dst.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_rows));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_cols));
    args.push_back(make_pair(sizeof(cl_int), (void* )&width));

    float alpha = 0.0f, beta = 0.0f;
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        alpha = (float)alpha1;
        beta = (float)beta1;
        args.push_back(make_pair(sizeof(cl_float), (void* )&alpha));
        args.push_back(make_pair(sizeof(cl_float), (void* )&beta));
    }
    else
    {
        args.push_back(make_pair(sizeof(cl_double), (void* )&alpha1));
        args.push_back(make_pair(sizeof(cl_double), (void* )&beta1));
    }
    openCLExecuteKernel(clCxt, &svm, kernelName, globalThreads, localThreads, args, -1, -1);
}

#endif // #ifndef HAVE_CLAMDBLAS

static void matmul_rbf(oclMat& src, oclMat& src_e, oclMat& dst, int src_rows, int src2_cols, int var_count, double gamma1, bool flag)
{

    Context *clCxt = Context::getContext();

    string kernelName = "svm_rbf";

    int width = var_count;
    int src_step = (int)src.step / src.elemSize();
    int src_e_step = (int)src_e.step / src_e.elemSize();
    int dst_step = (int)dst.step / dst.elemSize();

    int x = MIN(16, src_rows);
    int y = MIN(16, src2_cols);
    size_t localThreads[] = {x, y, 1};
    size_t globalThreads[] = {src2_cols,  src_rows, 1};
    char build_options[50];

    if(flag)
        sprintf(build_options, "-D ADDEXP");

    vector< pair<size_t, const void *> > args;
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&src_e.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_e_step));
    args.push_back(make_pair(sizeof(cl_mem), (void* )&dst.data));
    args.push_back(make_pair(sizeof(cl_int), (void* )&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src_rows));
    args.push_back(make_pair(sizeof(cl_int), (void* )&src2_cols));
    args.push_back(make_pair(sizeof(cl_int), (void* )&width));
    float gamma = 0.0f;
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        gamma = (float)gamma1;
        args.push_back(make_pair(sizeof(cl_float), (void* )&gamma));
    }
    else
        args.push_back(make_pair(sizeof(cl_double), (void* )&gamma1));

    openCLExecuteKernel(clCxt, &svm, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

float CvSVM_OCL::predict(const CvMat* samples, CV_OUT CvMat* results) const
{
    int var_count = get_var_count();
    int sample_count = samples->rows;

    //float* row_sample = 0;
    Mat src_temp = Mat(sample_count, var_count, CV_32FC1);
    CV_FUNCNAME( "CvSVM::predict" );


    for(int i = 0; i < samples->rows; i++)
    {
        __BEGIN__;
        CvMat sample;
        float* row_sample = 0;
        cvGetRow( samples, &sample, i );
        int class_count;
        if( !kernel )
        {
            CV_ERROR( CV_StsBadArg, "The SVM should be trained first" );
        }

        class_count = class_labels ? class_labels->cols :
                      params.svm_type == ONE_CLASS ? 1 : 0;

        CV_CALL( cvPreparePredictData(&sample, var_all, var_idx,
                                      class_count, 0, &row_sample ));
        for(int j = 0; j < var_count; ++j)
            src_temp.at<float>(i, j) = row_sample[j];
        __END__;
    }

    Mat dst1;
    double alpha1 = 0.0, beta1 = 0.0, gamma1 = 0.0;
    if(params.kernel_type == CvSVM::LINEAR)
    {
        alpha1 = 1;
        beta1 = 0;
    }
    if(params.kernel_type == CvSVM::POLY)
    {
        alpha1 = params.gamma;
        beta1 = params.coef0;
    }
    if(params.kernel_type == CvSVM::SIGMOID)
    {
        alpha1 = - 2 * params.gamma;
        beta1 = - 2 * params.coef0;
    }
    if(params.kernel_type == CvSVM::RBF)
        gamma1 = - params.gamma;

    Mat sv_temp = Mat(sv_total, var_count, CV_32FC1, Scalar::all(0));


    for(int i = 0; i < sv_total; ++i)
        for(int j = 0; j < var_count; ++j)
            sv_temp.at<float>(i, j) = sv[i][j];

    oclMat src(sample_count, var_count, CV_32FC1, Scalar::all(0));
    oclMat sv_;

    src.upload(src_temp);
    oclMat dst;

#ifdef HAVE_CLAMDBLAS

    dst = oclMat(sample_count, sv_total, CV_32FC1);
    oclMat src3(sample_count, sv_total, CV_32FC1, Scalar::all(1));
    if(params.kernel_type != CvSVM::RBF)
    {
        Mat sv_temp1;
        transpose(sv_temp, sv_temp1);
        sv_.upload(sv_temp1);
        gemm(src, sv_, alpha1, src3, beta1, dst);
    }

#else
    double degree1 = 0.0;
    if (params.kernel_type == CvSVM::POLY)
        degree1 = params.degree;

    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
        dst = oclMat(sample_count, sv_total, CV_32FC1);
    else
        dst = oclMat(sample_count, sv_total, CV_64FC1);

    if(params.kernel_type == CvSVM::LINEAR)
    {
        sv_.upload(sv_temp);
        matmul_linear(src, sv_, dst, sample_count, sv_total, var_count, alpha1, beta1);
    }
    if( params.kernel_type == CvSVM::SIGMOID)
    {
        sv_.upload(sv_temp);
        matmul_sigmod(src, sv_, dst, sample_count, sv_total, var_count, alpha1, beta1);
    }

    if(params.kernel_type == CvSVM::POLY)
    {
        sv_.upload(sv_temp);
        if(sample_count > 0)
            matmul_poly(src, sv_, dst, sample_count, sv_total, var_count, alpha1, beta1, degree1, true);
        else
            matmul_poly(src, sv_, dst, sample_count, sv_total, var_count, alpha1, beta1, degree1, false);
    }
#endif

    if(params.kernel_type == CvSVM::RBF)
    {
        sv_.upload(sv_temp);
        if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
            dst = oclMat(sample_count, sv_total, CV_32FC1);
        else
            dst = oclMat(sample_count, sv_total, CV_64FC1);

        if(sample_count > 0)
            matmul_rbf(src, sv_, dst, sample_count, sv_total, var_count, gamma1, true);
        else
            matmul_rbf(src, sv_, dst, sample_count, sv_total, var_count, gamma1, false);
    }
    dst.download(dst1);

    float result = 0;
    for(int i = 0; i < samples->rows; i++ )
    {
        int r = (int)this->predict(i, dst1);
        if (results)
            results->data.fl[i] = (float)r;
        if (i == 0)
            result = (float)r;
    }
    return result;
}

void CvSVM_OCL::predict( cv::InputArray _samples, cv::OutputArray _results ) const
{
    _results.create(_samples.size().height, 1, CV_32F);
    CvMat samples = _samples.getMat(), results = _results.getMat();
    predict(&samples, &results);
}

bool CvSVMSolver_ocl::solve_generic( CvSVMSolutionInfo& si )
{
    int iter = 0;
    int i, j, k;

    // 1. initialize gradient and alpha status
    for( i = 0; i < alpha_count; i++ )
    {
        update_alpha_status(i);
        G[i] = b[i];
        if( fabs(G[i]) > 1e200 )
        {
            return false;
        }
    }
    Mat dst1;
    double alpha1 = 0.0, beta1 = 0.0, gamma1 = 0.0;
    if(params->kernel_type == CvSVM::LINEAR)
    {
        alpha1 = 1;
        beta1 = 0;
    }
    if(params->kernel_type == CvSVM::POLY)
    {
        alpha1 = params->gamma;
        beta1 = params->coef0;
    }
    if(params->kernel_type == CvSVM::SIGMOID)
    {
        alpha1 = -2 * params->gamma;
        beta1 = -2 * params->coef0;
    }
    if(params->kernel_type == CvSVM::RBF)
    {
        gamma1 = -params->gamma;
    }
    Mat src1 = Mat(sample_count, var_count, CV_32FC1);

    for(int i = 0; i < sample_count; ++i)
    {
        for(int j = 0; j < var_count; ++j)
        {
            src1.at<float>(i, j) = samples[i][j];
        }
    }
    oclMat src, src_e;
    src.upload(src1);
    oclMat dst;

#ifdef HAVE_CLAMDBLAS

    dst = oclMat(sample_count, sample_count, CV_32FC1);
    oclMat src3(sample_count, sample_count, CV_32FC1, Scalar::all(1));
    if(params->kernel_type != CvSVM::RBF)
    {
        ocl::transpose(src, src_e);
        gemm(src, src_e, alpha1, src3, beta1, dst);
    }

#else
    double degree1 = 0.0;
    if(params->kernel_type == CvSVM::POLY)
        degree1 = params->degree;

    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
        dst = oclMat(sample_count, sample_count, CV_32FC1);
    else
        dst = oclMat(sample_count, sample_count, CV_64FC1);

    if(params->kernel_type == CvSVM::LINEAR )
    {
        src_e = src;
        matmul_linear(src, src_e, dst, sample_count, sample_count, var_count, alpha1, beta1);
    }
    if( params->kernel_type == CvSVM::SIGMOID)
    {
        src_e = src;
        matmul_sigmod(src, src_e, dst, sample_count, sample_count, var_count, alpha1, beta1);
    }

    if(params->kernel_type == CvSVM::POLY)
    {
        src_e = src;
        if(sample_count > 0)
            matmul_poly(src, src_e, dst, sample_count, sample_count, var_count, alpha1, beta1, degree1, true);
        else
            matmul_poly(src, src_e, dst, sample_count, sample_count, var_count, alpha1, beta1, degree1, false);
    }

#endif

    if(params->kernel_type == CvSVM::RBF)
    {
        src_e = src;
        if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
            dst = oclMat(sample_count, sample_count, CV_32FC1);
        else
            dst = oclMat(sample_count, sample_count, CV_64FC1);

        if(sample_count > 0)
            matmul_rbf(src, src_e, dst, sample_count, sample_count, var_count, gamma1, true);
        else
            matmul_rbf(src, src_e, dst, sample_count, sample_count, var_count, gamma1, false);
    }
    dst.download(dst1);
    for( i = 0; i < alpha_count; i++ )
    {
        if( !is_lower_bound(i) )
        {
            const Qfloat *Q_i = CvSVMSolver::get_row( i, buf[0]);
            double alpha_i = alpha[i];

            for( j = 0; j < alpha_count; j++ )
                G[j] += alpha_i * Q_i[j];
        }
    }

    // 2. optimization loop
    for(;;)
    {
        const Qfloat *Q_i, *Q_j;
        double C_i, C_j;
        double old_alpha_i, old_alpha_j, alpha_i, alpha_j;
        double delta_alpha_i, delta_alpha_j;

#ifdef _DEBUG
        for( i = 0; i < alpha_count; i++ )
        {
            if( fabs(G[i]) > 1e+300 )
                return false;

            if( fabs(alpha[i]) > 1e16 )
                return false;
        }
#endif

        if( (this->*select_working_set_func)( i, j ) != 0 || iter++ >= max_iter )
        {
            break;
        }
        Q_i = get_row( i, buf[0], dst1);
        Q_j = get_row( j, buf[1], dst1);

        C_i = get_C(i);
        C_j = get_C(j);

        alpha_i = old_alpha_i = alpha[i];
        alpha_j = old_alpha_j = alpha[j];

        if( y[i] != y[j] )
        {
            double denom = Q_i[i] + Q_j[j] + 2 * Q_i[j];
            double delta = (-G[i] - G[j]) / MAX(fabs(denom), FLT_EPSILON);
            double diff = alpha_i - alpha_j;
            alpha_i += delta;
            alpha_j += delta;

            if( diff > 0 && alpha_j < 0 )
            {
                alpha_j = 0;
                alpha_i = diff;
            }
            else if( diff <= 0 && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = -diff;
            }

            if( diff > C_i - C_j && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = C_i - diff;
            }
            else if( diff <= C_i - C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = C_j + diff;
            }
        }
        else
        {
            double denom = Q_i[i] + Q_j[j] - 2 * Q_i[j];
            double delta = (G[i] - G[j]) / MAX(fabs(denom), FLT_EPSILON);
            double sum = alpha_i + alpha_j;
            alpha_i -= delta;
            alpha_j += delta;

            if( sum > C_i && alpha_i > C_i )
            {
                alpha_i = C_i;
                alpha_j = sum - C_i;
            }
            else if( sum <= C_i && alpha_j < 0)
            {
                alpha_j = 0;
                alpha_i = sum;
            }

            if( sum > C_j && alpha_j > C_j )
            {
                alpha_j = C_j;
                alpha_i = sum - C_j;
            }
            else if( sum <= C_j && alpha_i < 0 )
            {
                alpha_i = 0;
                alpha_j = sum;
            }
        }
        // update alpha
        alpha[i] = alpha_i;
        alpha[j] = alpha_j;
        update_alpha_status(i);
        update_alpha_status(j);

        // update G
        delta_alpha_i = alpha_i - old_alpha_i;
        delta_alpha_j = alpha_j - old_alpha_j;

        for( k = 0; k < alpha_count; k++ )
            G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
    }

    // calculate rho
    (this->*calc_rho_func)( si.rho, si.r );

    // calculate objective value
    for( i = 0, si.obj = 0; i < alpha_count; i++ )
        si.obj += alpha[i] * (G[i] + b[i]);

    si.obj *= 0.5;

    si.upper_bound_p = C[1];
    si.upper_bound_n = C[0];

    return true;
}

void CvSVMKernel_ocl::calc( int vcount, const int row_idx, Qfloat* results, Mat& src)
{
    //const Qfloat max_val = (Qfloat)(FLT_MAX*1e-3);
    //int j;
    (this->*calc_func_ocl)( vcount, row_idx, results, src);

#if !defined(HAVE_CLAMDBLAS)
    // nothing
#else
    const Qfloat max_val = (Qfloat)(FLT_MAX * 1e-3);
    int j;
    for( j = 0; j < vcount; j++ )
        if( results[j] > max_val )
            results[j] = max_val;
#endif
}

bool CvSVMKernel_ocl::create( const CvSVMParams* _params, Calc_ocl _calc_func, Calc _calc_func1 )
{
    clear();
    params = _params;
    calc_func_ocl = _calc_func;
    calc_func = _calc_func1;
    if( !calc_func_ocl )
        calc_func_ocl = params->kernel_type == CvSVM::RBF ? &CvSVMKernel_ocl::calc_rbf :
                        params->kernel_type == CvSVM::POLY ? &CvSVMKernel_ocl::calc_poly :
                        params->kernel_type == CvSVM::SIGMOID ? &CvSVMKernel_ocl::calc_sigmoid :
                        &CvSVMKernel_ocl::calc_linear;
    if( !calc_func)
        calc_func = params->kernel_type == CvSVM::RBF ? &CvSVMKernel::calc_rbf :
                    params->kernel_type == CvSVM::POLY ? &CvSVMKernel::calc_poly :
                    params->kernel_type == CvSVM::SIGMOID ? &CvSVMKernel::calc_sigmoid :
                    &CvSVMKernel::calc_linear;
    return true;
}
CvSVMKernel_ocl::CvSVMKernel_ocl(const CvSVMParams* params, CvSVMKernel_ocl::Calc_ocl _calc_func, CvSVMKernel::Calc _calc_func1)
{
    CvSVMKernel::clear();
    CvSVMKernel_ocl::create( params, _calc_func, _calc_func1 );
}

void CvSVMKernel_ocl::calc_non_rbf_base( int vcount, const int row_idx, Qfloat* results, Mat& src)
{
#ifdef HAVE_CLAMDBLAS

    for(int i = 0; i < vcount; i++)
    {
        results[i] = (Qfloat) * src.ptr<float>(row_idx, i);
    }
#else
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        for(int i = 0; i < vcount; i++)
        {
            results[i] = (Qfloat) * src.ptr<float>(row_idx, i);
        }
    }
    else
    {
        for(int i = 0; i < vcount; i++)
        {
            results[i] = (Qfloat) * src.ptr<double>(row_idx, i);
        }
    }
#endif
}

void CvSVMKernel_ocl::calc_rbf( int vcount, const int row_idx, Qfloat* results, Mat& src)
{
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
        for(int m = 0; m < vcount; m++)
            results[m] = (Qfloat) * src.ptr<float>(row_idx, m);
    else
        for(int m = 0; m < vcount; m++)
            results[m] = (Qfloat) * src.ptr<double>(row_idx, m);
}

void CvSVMKernel_ocl::calc_linear( int vcount, const int row_idx, Qfloat* results, Mat& src )
{
    calc_non_rbf_base( vcount, row_idx, results, src);
}

void CvSVMKernel_ocl::calc_poly( int vcount, const int row_idx, Qfloat* results, Mat& src)
{
    calc_non_rbf_base( vcount, row_idx, results, src);

#if !defined(HAVE_CLAMDBLAS)
    // nothing
#else
    CvMat R = cvMat( 1, vcount, QFLOAT_TYPE, results );
    if( vcount > 0 )
        cvPow( &R, &R, params->degree );
#endif
}


void CvSVMKernel_ocl::calc_sigmoid( int vcount, const int row_idx, Qfloat* results, Mat& src)
{
    calc_non_rbf_base( vcount, row_idx, results, src);
    // TODO: speedup this
#if !defined(HAVE_CLAMDBLAS)
    // nothing
#else
    for(int j = 0; j < vcount; j++ )
    {
        Qfloat t = results[j];
        double e = ::exp(-fabs(t));
        if( t > 0 )
            results[j] = (Qfloat)((1. - e) / (1. + e));
        else
            results[j] = (Qfloat)((e - 1.) / (e + 1.));
    }
#endif
}

CvSVM_OCL::CvSVM_OCL()
{
    CvSVM();
}

CvSVM_OCL::CvSVM_OCL( const Mat& _train_data, const Mat& _responses,
                      const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params )
{
    decision_func = 0;
    class_labels = 0;
    class_weights = 0;
    storage = 0;
    var_idx = 0;
    kernel = 0;
    solver = 0;
    default_model_name = "my_svm";

    train( _train_data, _responses, _var_idx, _sample_idx, _params );
}

void CvSVM_OCL::create_kernel()
{
    kernel = new CvSVMKernel_ocl(&params, 0, 0);
}

void CvSVM_OCL::create_solver( )
{
    solver = new CvSVMSolver_ocl(&params);
}

} }
