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

CvANN_MLP_TrainParams::CvANN_MLP_TrainParams()
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 );
    train_method = RPROP;
    bp_dw_scale = bp_moment_scale = 0.1;
    rp_dw0 = 0.1; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
    rp_dw_min = FLT_EPSILON; rp_dw_max = 50.;
}


CvANN_MLP_TrainParams::CvANN_MLP_TrainParams( CvTermCriteria _term_crit,
                                              int _train_method,
                                              double _param1, double _param2 )
{
    term_crit = _term_crit;
    train_method = _train_method;
    bp_dw_scale = bp_moment_scale = 0.1;
    rp_dw0 = 1.; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
    rp_dw_min = FLT_EPSILON; rp_dw_max = 50.;

    if( train_method == RPROP )
    {
        rp_dw0 = _param1;
        if( rp_dw0 < FLT_EPSILON )
            rp_dw0 = 1.;
        rp_dw_min = _param2;
        rp_dw_min = MAX( rp_dw_min, 0 );
    }
    else if( train_method == BACKPROP )
    {
        bp_dw_scale = _param1;
        if( bp_dw_scale <= 0 )
            bp_dw_scale = 0.1;
        bp_dw_scale = MAX( bp_dw_scale, 1e-3 );
        bp_dw_scale = MIN( bp_dw_scale, 1 );
        bp_moment_scale = _param2;
        if( bp_moment_scale < 0 )
            bp_moment_scale = 0.1;
        bp_moment_scale = MIN( bp_moment_scale, 1 );
    }
    else
        train_method = RPROP;
}


CvANN_MLP_TrainParams::~CvANN_MLP_TrainParams()
{
}


CvANN_MLP::CvANN_MLP()
{
    layer_sizes = wbuf = 0;
    min_val = max_val = min_val1 = max_val1 = 0.;
    weights = 0;
    rng = cvRNG(-1);
    default_model_name = "my_nn";
    clear();
}


CvANN_MLP::CvANN_MLP( const CvMat* _layer_sizes,
                      int _activ_func,
                      double _f_param1, double _f_param2 )
{
    layer_sizes = wbuf = 0;
    min_val = max_val = min_val1 = max_val1 = 0.;
    weights = 0;
    rng = cvRNG(-1);
    default_model_name = "my_nn";
    create( _layer_sizes, _activ_func, _f_param1, _f_param2 );
}


CvANN_MLP::~CvANN_MLP()
{
    clear();
}


void CvANN_MLP::clear()
{
    cvReleaseMat( &layer_sizes );
    cvReleaseMat( &wbuf );
    cvFree( &weights );
    activ_func = SIGMOID_SYM;
    f_param1 = f_param2 = 1;
    max_buf_sz = 1 << 12;
}


void CvANN_MLP::set_activ_func( int _activ_func, double _f_param1, double _f_param2 )
{
    CV_FUNCNAME( "CvANN_MLP::set_activ_func" );

    __BEGIN__;

    if( _activ_func < 0 || _activ_func > GAUSSIAN )
        CV_ERROR( CV_StsOutOfRange, "Unknown activation function" );

    activ_func = _activ_func;

    switch( activ_func )
    {
    case SIGMOID_SYM:
        max_val = 0.95; min_val = -max_val;
        max_val1 = 0.98; min_val1 = -max_val1;
        if( fabs(_f_param1) < FLT_EPSILON )
            _f_param1 = 2./3;
        if( fabs(_f_param2) < FLT_EPSILON )
            _f_param2 = 1.7159;
        break;
    case GAUSSIAN:
        max_val = 1.; min_val = 0.05;
        max_val1 = 1.; min_val1 = 0.02;
        if( fabs(_f_param1) < FLT_EPSILON )
            _f_param1 = 1.;
        if( fabs(_f_param2) < FLT_EPSILON )
            _f_param2 = 1.;
        break;
    default:
        min_val = max_val = min_val1 = max_val1 = 0.;
        _f_param1 = 1.;
        _f_param2 = 0.;
    }

    f_param1 = _f_param1;
    f_param2 = _f_param2;

    __END__;
}


void CvANN_MLP::init_weights()
{
    int i, j, k;

    for( i = 1; i < layer_sizes->cols; i++ )
    {
        int n1 = layer_sizes->data.i[i-1];
        int n2 = layer_sizes->data.i[i];
        double val = 0, G = n2 > 2 ? 0.7*pow((double)n1,1./(n2-1)) : 1.;
        double* w = weights[i];

        // initialize weights using Nguyen-Widrow algorithm
        for( j = 0; j < n2; j++ )
        {
            double s = 0;
            for( k = 0; k <= n1; k++ )
            {
                val = cvRandReal(&rng)*2-1.;
                w[k*n2 + j] = val;
                s += fabs(val);
            }

            if( i < layer_sizes->cols - 1 )
            {
                s = 1./(s - fabs(val));
                for( k = 0; k <= n1; k++ )
                    w[k*n2 + j] *= s;
                w[n1*n2 + j] *= G*(-1+j*2./n2);
            }
        }
    }
}


void CvANN_MLP::create( const CvMat* _layer_sizes, int _activ_func,
                        double _f_param1, double _f_param2 )
{
    CV_FUNCNAME( "CvANN_MLP::create" );

    __BEGIN__;

    int i, l_step, l_count, buf_sz = 0;
    int *l_src, *l_dst;

    clear();

    if( !CV_IS_MAT(_layer_sizes) ||
        (_layer_sizes->cols != 1 && _layer_sizes->rows != 1) ||
        CV_MAT_TYPE(_layer_sizes->type) != CV_32SC1 )
        CV_ERROR( CV_StsBadArg,
        "The array of layer neuron counters must be an integer vector" );

    CV_CALL( set_activ_func( _activ_func, _f_param1, _f_param2 ));

    l_count = _layer_sizes->rows + _layer_sizes->cols - 1;
    l_src = _layer_sizes->data.i;
    l_step = CV_IS_MAT_CONT(_layer_sizes->type) ? 1 :
                _layer_sizes->step / sizeof(l_src[0]);
    CV_CALL( layer_sizes = cvCreateMat( 1, l_count, CV_32SC1 ));
    l_dst = layer_sizes->data.i;

    max_count = 0;
    for( i = 0; i < l_count; i++ )
    {
        int n = l_src[i*l_step];
        if( n < 1 + (0 < i && i < l_count-1))
            CV_ERROR( CV_StsOutOfRange,
            "there should be at least one input and one output "
            "and every hidden layer must have more than 1 neuron" );
        l_dst[i] = n;
        max_count = MAX( max_count, n );
        if( i > 0 )
            buf_sz += (l_dst[i-1]+1)*n;
    }

    buf_sz += (l_dst[0] + l_dst[l_count-1]*2)*2;

    CV_CALL( wbuf = cvCreateMat( 1, buf_sz, CV_64F ));
    CV_CALL( weights = (double**)cvAlloc( (l_count+1)*sizeof(weights[0]) ));

    weights[0] = wbuf->data.db;
    weights[1] = weights[0] + l_dst[0]*2;
    for( i = 1; i < l_count; i++ )
        weights[i+1] = weights[i] + (l_dst[i-1] + 1)*l_dst[i];
    weights[l_count+1] = weights[l_count] + l_dst[l_count-1]*2;

    __END__;
}


float CvANN_MLP::predict( const CvMat* _inputs, CvMat* _outputs ) const
{
    CV_FUNCNAME( "CvANN_MLP::predict" );

    __BEGIN__;

    double* buf;
    int i, j, n, dn = 0, l_count, dn0, buf_sz, min_buf_sz;

    if( !layer_sizes )
        CV_ERROR( CV_StsError, "The network has not been initialized" );

    if( !CV_IS_MAT(_inputs) || !CV_IS_MAT(_outputs) ||
        !CV_ARE_TYPES_EQ(_inputs,_outputs) ||
        (CV_MAT_TYPE(_inputs->type) != CV_32FC1 &&
        CV_MAT_TYPE(_inputs->type) != CV_64FC1) ||
        _inputs->rows != _outputs->rows )
        CV_ERROR( CV_StsBadArg, "Both input and output must be floating-point matrices "
                                "of the same type and have the same number of rows" );

    if( _inputs->cols != layer_sizes->data.i[0] )
        CV_ERROR( CV_StsBadSize, "input matrix must have the same number of columns as "
                                 "the number of neurons in the input layer" );

    if( _outputs->cols != layer_sizes->data.i[layer_sizes->cols - 1] )
        CV_ERROR( CV_StsBadSize, "output matrix must have the same number of columns as "
                                 "the number of neurons in the output layer" );
    n = dn0 = _inputs->rows;
    min_buf_sz = 2*max_count;
    buf_sz = n*min_buf_sz;

    if( buf_sz > max_buf_sz )
    {
        dn0 = max_buf_sz/min_buf_sz;
        dn0 = MAX( dn0, 1 );
        buf_sz = dn0*min_buf_sz;
    }

    buf = (double*)cvStackAlloc( buf_sz*sizeof(buf[0]) );
    l_count = layer_sizes->cols;

    for( i = 0; i < n; i += dn )
    {
        CvMat hdr[2], _w, *layer_in = &hdr[0], *layer_out = &hdr[1], *temp;
        dn = MIN( dn0, n - i );

        cvGetRows( _inputs, layer_in, i, i + dn );
        cvInitMatHeader( layer_out, dn, layer_in->cols, CV_64F, buf );

        scale_input( layer_in, layer_out );
        CV_SWAP( layer_in, layer_out, temp );

        for( j = 1; j < l_count; j++ )
        {
            double* data = buf + (j&1 ? max_count*dn0 : 0);
            int cols = layer_sizes->data.i[j];

            cvInitMatHeader( layer_out, dn, cols, CV_64F, data );
            cvInitMatHeader( &_w, layer_in->cols, layer_out->cols, CV_64F, weights[j] );
            cvGEMM( layer_in, &_w, 1, 0, 0, layer_out );
            calc_activ_func( layer_out, _w.data.db + _w.rows*_w.cols );

            CV_SWAP( layer_in, layer_out, temp );
        }

        cvGetRows( _outputs, layer_out, i, i + dn );
        scale_output( layer_in, layer_out );
    }

    __END__;

    return 0.f;
}


void CvANN_MLP::scale_input( const CvMat* _src, CvMat* _dst ) const
{
    int i, j, cols = _src->cols;
    double* dst = _dst->data.db;
    const double* w = weights[0];
    int step = _src->step;

    if( CV_MAT_TYPE( _src->type ) == CV_32F )
    {
        const float* src = _src->data.fl;
        step /= sizeof(src[0]);

        for( i = 0; i < _src->rows; i++, src += step, dst += cols )
            for( j = 0; j < cols; j++ )
                dst[j] = src[j]*w[j*2] + w[j*2+1];
    }
    else
    {
        const double* src = _src->data.db;
        step /= sizeof(src[0]);

        for( i = 0; i < _src->rows; i++, src += step, dst += cols )
            for( j = 0; j < cols; j++ )
                dst[j] = src[j]*w[j*2] + w[j*2+1];
    }
}


void CvANN_MLP::scale_output( const CvMat* _src, CvMat* _dst ) const
{
    int i, j, cols = _src->cols;
    const double* src = _src->data.db;
    const double* w = weights[layer_sizes->cols];
    int step = _dst->step;

    if( CV_MAT_TYPE( _dst->type ) == CV_32F )
    {
        float* dst = _dst->data.fl;
        step /= sizeof(dst[0]);

        for( i = 0; i < _src->rows; i++, src += cols, dst += step )
            for( j = 0; j < cols; j++ )
                dst[j] = (float)(src[j]*w[j*2] + w[j*2+1]);
    }
    else
    {
        double* dst = _dst->data.db;
        step /= sizeof(dst[0]);

        for( i = 0; i < _src->rows; i++, src += cols, dst += step )
            for( j = 0; j < cols; j++ )
                dst[j] = src[j]*w[j*2] + w[j*2+1];
    }
}


void CvANN_MLP::calc_activ_func( CvMat* sums, const double* bias ) const
{
    int i, j, n = sums->rows, cols = sums->cols;
    double* data = sums->data.db;
    double scale = 0, scale2 = f_param2;

    switch( activ_func )
    {
    case IDENTITY:
        scale = 1.;
        break;
    case SIGMOID_SYM:
        scale = -f_param1;
        break;
    case GAUSSIAN:
        scale = -f_param1*f_param1;
        break;
    default:
        ;
    }

    assert( CV_IS_MAT_CONT(sums->type) );

    if( activ_func != GAUSSIAN )
    {
        for( i = 0; i < n; i++, data += cols )
            for( j = 0; j < cols; j++ )
                data[j] = (data[j] + bias[j])*scale;

        if( activ_func == IDENTITY )
            return;
    }
    else
    {
        for( i = 0; i < n; i++, data += cols )
            for( j = 0; j < cols; j++ )
            {
                double t = data[j] + bias[j];
                data[j] = t*t*scale;
            }
    }

    cvExp( sums, sums );

    n *= cols;
    data -= n;

    switch( activ_func )
    {
    case SIGMOID_SYM:
        for( i = 0; i <= n - 4; i += 4 )
        {
            double x0 = 1.+data[i], x1 = 1.+data[i+1], x2 = 1.+data[i+2], x3 = 1.+data[i+3];
            double a = x0*x1, b = x2*x3, d = scale2/(a*b), t0, t1;
            a *= d; b *= d;
            t0 = (2 - x0)*b*x1; t1 = (2 - x1)*b*x0;
            data[i] = t0; data[i+1] = t1;
            t0 = (2 - x2)*a*x3; t1 = (2 - x3)*a*x2;
            data[i+2] = t0; data[i+3] = t1;
        }

        for( ; i < n; i++ )
        {
            double t = scale2*(1. - data[i])/(1. + data[i]);
            data[i] = t;
        }
        break;

    case GAUSSIAN:
        for( i = 0; i < n; i++ )
            data[i] = scale2*data[i];
        break;

    default:
        ;
    }
}


void CvANN_MLP::calc_activ_func_deriv( CvMat* _xf, CvMat* _df,
                                       const double* bias ) const
{
    int i, j, n = _xf->rows, cols = _xf->cols;
    double* xf = _xf->data.db;
    double* df = _df->data.db;
    double scale, scale2 = f_param2;
    assert( CV_IS_MAT_CONT( _xf->type & _df->type ) );

    if( activ_func == IDENTITY )
    {
        for( i = 0; i < n; i++, xf += cols, df += cols )
            for( j = 0; j < cols; j++ )
            {
                xf[j] += bias[j];
                df[j] = 1;
            }
        return;
    }
    else if( activ_func == GAUSSIAN )
    {
        scale = -f_param1*f_param1;
        scale2 *= scale;
        for( i = 0; i < n; i++, xf += cols, df += cols )
            for( j = 0; j < cols; j++ )
            {
                double t = xf[j] + bias[j];
                df[j] = t*2*scale2;
                xf[j] = t*t*scale;
            }
        cvExp( _xf, _xf );

        n *= cols;
        xf -= n; df -= n;
        
        for( i = 0; i < n; i++ )
            df[i] *= xf[i];
    }
    else
    {
        scale = f_param1;
        for( i = 0; i < n; i++, xf += cols, df += cols )
            for( j = 0; j < cols; j++ )
            {
                xf[j] = (xf[j] + bias[j])*scale;
                df[j] = -fabs(xf[j]);
            }
        
        cvExp( _df, _df );

        n *= cols;
        xf -= n; df -= n;

        // ((1+exp(-ax))^-1)'=a*((1+exp(-ax))^-2)*exp(-ax);
        // ((1-exp(-ax))/(1+exp(-ax)))'=(a*exp(-ax)*(1+exp(-ax)) + a*exp(-ax)*(1-exp(-ax)))/(1+exp(-ax))^2=
        // 2*a*exp(-ax)/(1+exp(-ax))^2
        scale *= 2*f_param2;
        for( i = 0; i < n; i++ )
        {
            int s0 = xf[i] > 0 ? 1 : -1;
            double t0 = 1./(1. + df[i]);
            double t1 = scale*df[i]*t0*t0;
            t0 *= scale2*(1. - df[i])*s0;
            df[i] = t1;
            xf[i] = t0;
        }
    }
}


void CvANN_MLP::calc_input_scale( const CvVectors* vecs, int flags )
{
    bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
    bool no_scale = (flags & NO_INPUT_SCALE) != 0;
    double* scale = weights[0];
    int count = vecs->count;

    if( reset_weights )
    {
        int i, j, vcount = layer_sizes->data.i[0];
        int type = vecs->type;
        double a = no_scale ? 1. : 0.;

        for( j = 0; j < vcount; j++ )
            scale[2*j] = a, scale[j*2+1] = 0.;

        if( no_scale )
            return;

        for( i = 0; i < count; i++ )
        {
            const float* f = vecs->data.fl[i];
            const double* d = vecs->data.db[i];
            for( j = 0; j < vcount; j++ )
            {
                double t = type == CV_32F ? (double)f[j] : d[j];
                scale[j*2] += t;
                scale[j*2+1] += t*t;
            }
        }

        for( j = 0; j < vcount; j++ )
        {
            double s = scale[j*2], s2 = scale[j*2+1];
            double m = s/count, sigma2 = s2/count - m*m;
            scale[j*2] = sigma2 < DBL_EPSILON ? 1 : 1./sqrt(sigma2);
            scale[j*2+1] = -m*scale[j*2];
        }
    }
}


void CvANN_MLP::calc_output_scale( const CvVectors* vecs, int flags )
{
    int i, j, vcount = layer_sizes->data.i[layer_sizes->cols-1];
    int type = vecs->type;
    double m = min_val, M = max_val, m1 = min_val1, M1 = max_val1;
    bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
    bool no_scale = (flags & NO_OUTPUT_SCALE) != 0;
    int l_count = layer_sizes->cols;
    double* scale = weights[l_count];
    double* inv_scale = weights[l_count+1];
    int count = vecs->count;

    CV_FUNCNAME( "CvANN_MLP::calc_output_scale" );

    __BEGIN__;

    if( reset_weights )
    {
        double a0 = no_scale ? 1 : DBL_MAX, b0 = no_scale ? 0 : -DBL_MAX;

        for( j = 0; j < vcount; j++ )
        {
            scale[2*j] = inv_scale[2*j] = a0;
            scale[j*2+1] = inv_scale[2*j+1] = b0;
        }

        if( no_scale )
            EXIT;
    }

    for( i = 0; i < count; i++ )
    {
        const float* f = vecs->data.fl[i];
        const double* d = vecs->data.db[i];

        for( j = 0; j < vcount; j++ )
        {
            double t = type == CV_32F ? (double)f[j] : d[j];

            if( reset_weights )
            {
                double mj = scale[j*2], Mj = scale[j*2+1];
                if( mj > t ) mj = t;
                if( Mj < t ) Mj = t;

                scale[j*2] = mj;
                scale[j*2+1] = Mj;
            }
            else
            {
                t = t*scale[j*2] + scale[2*j+1];
                if( t < m1 || t > M1 )
                    CV_ERROR( CV_StsOutOfRange,
                    "Some of new output training vector components run exceed the original range too much" );
            }
        }
    }

    if( reset_weights )
        for( j = 0; j < vcount; j++ )
        {
            // map mj..Mj to m..M
            double mj = scale[j*2], Mj = scale[j*2+1];
            double a, b;
            double delta = Mj - mj;
            if( delta < DBL_EPSILON )
                a = 1, b = (M + m - Mj - mj)*0.5;
            else
                a = (M - m)/delta, b = m - mj*a;
            inv_scale[j*2] = a; inv_scale[j*2+1] = b;
            a = 1./a; b = -b*a;
            scale[j*2] = a; scale[j*2+1] = b;
        }

    __END__;
}


bool CvANN_MLP::prepare_to_train( const CvMat* _inputs, const CvMat* _outputs,
            const CvMat* _sample_weights, const CvMat* _sample_idx,
            CvVectors* _ivecs, CvVectors* _ovecs, double** _sw, int _flags )
{
    bool ok = false;
    CvMat* sample_idx = 0;
    CvVectors ivecs, ovecs;
    double* sw = 0;
    int count = 0;

    CV_FUNCNAME( "CvANN_MLP::prepare_to_train" );

    ivecs.data.ptr = ovecs.data.ptr = 0;
    assert( _ivecs && _ovecs );

    __BEGIN__;

    const int* sidx = 0;
    int i, sw_type = 0, sw_count = 0;
    int sw_step = 0;
    double sw_sum = 0;

    if( !layer_sizes )
        CV_ERROR( CV_StsError,
        "The network has not been created. Use method create or the appropriate constructor" );

    if( !CV_IS_MAT(_inputs) || (CV_MAT_TYPE(_inputs->type) != CV_32FC1 &&
        CV_MAT_TYPE(_inputs->type) != CV_64FC1) || _inputs->cols != layer_sizes->data.i[0] )
        CV_ERROR( CV_StsBadArg,
        "input training data should be a floating-point matrix with"
        "the number of rows equal to the number of training samples and "
        "the number of columns equal to the size of 0-th (input) layer" );

    if( !CV_IS_MAT(_outputs) || (CV_MAT_TYPE(_outputs->type) != CV_32FC1 &&
        CV_MAT_TYPE(_outputs->type) != CV_64FC1) ||
        _outputs->cols != layer_sizes->data.i[layer_sizes->cols - 1] )
        CV_ERROR( CV_StsBadArg,
        "output training data should be a floating-point matrix with"
        "the number of rows equal to the number of training samples and "
        "the number of columns equal to the size of last (output) layer" );

    if( _inputs->rows != _outputs->rows )
        CV_ERROR( CV_StsUnmatchedSizes, "The numbers of input and output samples do not match" );

    if( _sample_idx )
    {
        CV_CALL( sample_idx = cvPreprocessIndexArray( _sample_idx, _inputs->rows ));
        sidx = sample_idx->data.i;
        count = sample_idx->cols + sample_idx->rows - 1;
    }
    else
        count = _inputs->rows;

    if( _sample_weights )
    {
        if( !CV_IS_MAT(_sample_weights) )
            CV_ERROR( CV_StsBadArg, "sample_weights (if passed) must be a valid matrix" );

        sw_type = CV_MAT_TYPE(_sample_weights->type);
        sw_count = _sample_weights->cols + _sample_weights->rows - 1;

        if( (sw_type != CV_32FC1 && sw_type != CV_64FC1) ||
            (_sample_weights->cols != 1 && _sample_weights->rows != 1) ||
            (sw_count != count && sw_count != _inputs->rows) )
            CV_ERROR( CV_StsBadArg,
            "sample_weights must be 1d floating-point vector containing weights "
            "of all or selected training samples" );

        sw_step = CV_IS_MAT_CONT(_sample_weights->type) ? 1 :
            _sample_weights->step/CV_ELEM_SIZE(sw_type);

        CV_CALL( sw = (double*)cvAlloc( count*sizeof(sw[0]) ));
    }

    CV_CALL( ivecs.data.ptr = (uchar**)cvAlloc( count*sizeof(ivecs.data.ptr[0]) ));
    CV_CALL( ovecs.data.ptr = (uchar**)cvAlloc( count*sizeof(ovecs.data.ptr[0]) ));

    ivecs.type = CV_MAT_TYPE(_inputs->type);
    ovecs.type = CV_MAT_TYPE(_outputs->type);
    ivecs.count = ovecs.count = count;

    for( i = 0; i < count; i++ )
    {
        int idx = sidx ? sidx[i] : i;
        ivecs.data.ptr[i] = _inputs->data.ptr + idx*_inputs->step;
        ovecs.data.ptr[i] = _outputs->data.ptr + idx*_outputs->step;
        if( sw )
        {
            int si = sw_count == count ? i : idx;
            double w = sw_type == CV_32FC1 ?
                (double)_sample_weights->data.fl[si*sw_step] :
                _sample_weights->data.db[si*sw_step];
            sw[i] = w;
            if( w < 0 )
                CV_ERROR( CV_StsOutOfRange, "some of sample weights are negative" );
            sw_sum += w;
        }
    }

    // normalize weights
    if( sw )
    {
        sw_sum = sw_sum > DBL_EPSILON ? 1./sw_sum : 0;
        for( i = 0; i < count; i++ )
            sw[i] *= sw_sum;
    }

    calc_input_scale( &ivecs, _flags );
    CV_CALL( calc_output_scale( &ovecs, _flags ));

    ok = true;

    __END__;

    if( !ok )
    {
        cvFree( &ivecs.data.ptr );
        cvFree( &ovecs.data.ptr );
        cvFree( &sw );
    }

    cvReleaseMat( &sample_idx );
    *_ivecs = ivecs;
    *_ovecs = ovecs;
    *_sw = sw;

    return ok;
}


int CvANN_MLP::train( const CvMat* _inputs, const CvMat* _outputs,
                      const CvMat* _sample_weights, const CvMat* _sample_idx,
                      CvANN_MLP_TrainParams _params, int flags )
{
    const int MAX_ITER = 1000;
    const double DEFAULT_EPSILON = FLT_EPSILON;

    double* sw = 0;
    CvVectors x0, u;
    int iter = -1;

    x0.data.ptr = u.data.ptr = 0;

    CV_FUNCNAME( "CvANN_MLP::train" );

    __BEGIN__;

    int max_iter;
    double epsilon;

    params = _params;

    // initialize training data
    CV_CALL( prepare_to_train( _inputs, _outputs, _sample_weights,
                               _sample_idx, &x0, &u, &sw, flags ));

    // ... and link weights
    if( !(flags & UPDATE_WEIGHTS) )
        init_weights();

    max_iter = params.term_crit.type & CV_TERMCRIT_ITER ? params.term_crit.max_iter : MAX_ITER;
    max_iter = MIN( max_iter, MAX_ITER );
    max_iter = MAX( max_iter, 1 );

    epsilon = params.term_crit.type & CV_TERMCRIT_EPS ? params.term_crit.epsilon : DEFAULT_EPSILON;
    epsilon = MAX(epsilon, DBL_EPSILON);

    params.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    params.term_crit.max_iter = max_iter;
    params.term_crit.epsilon = epsilon;

    if( params.train_method == CvANN_MLP_TrainParams::BACKPROP )
    {
        CV_CALL( iter = train_backprop( x0, u, sw ));
    }
    else
    {
        CV_CALL( iter = train_rprop( x0, u, sw ));
    }

    __END__;

    cvFree( &x0.data.ptr );
    cvFree( &u.data.ptr );
    cvFree( &sw );

    return iter;
}


int CvANN_MLP::train_backprop( CvVectors x0, CvVectors u, const double* sw )
{
    CvMat* dw = 0;
    CvMat* buf = 0;
    double **x = 0, **df = 0;
    CvMat* _idx = 0;
    int iter = -1, count = x0.count;

    CV_FUNCNAME( "CvANN_MLP::train_backprop" );

    __BEGIN__;

    int i, j, k, ivcount, ovcount, l_count, total = 0, max_iter;
    double *buf_ptr;
    double prev_E = DBL_MAX*0.5, E = 0, epsilon;

    max_iter = params.term_crit.max_iter*count;
    epsilon = params.term_crit.epsilon*count;

    l_count = layer_sizes->cols;
    ivcount = layer_sizes->data.i[0];
    ovcount = layer_sizes->data.i[l_count-1];

    // allocate buffers
    for( i = 0; i < l_count; i++ )
        total += layer_sizes->data.i[i] + 1;

    CV_CALL( dw = cvCreateMat( wbuf->rows, wbuf->cols, wbuf->type ));
    cvZero( dw );
    CV_CALL( buf = cvCreateMat( 1, (total + max_count)*2, CV_64F ));
    CV_CALL( _idx = cvCreateMat( 1, count, CV_32SC1 ));
    for( i = 0; i < count; i++ )
        _idx->data.i[i] = i;

    CV_CALL( x = (double**)cvAlloc( total*2*sizeof(x[0]) ));
    df = x + total;
    buf_ptr = buf->data.db;

    for( j = 0; j < l_count; j++ )
    {
        x[j] = buf_ptr;
        df[j] = x[j] + layer_sizes->data.i[j];
        buf_ptr += (df[j] - x[j])*2;
    }

    // run back-propagation loop
    /*
        y_i = w_i*x_{i-1}
        x_i = f(y_i)
        E = 1/2*||u - x_N||^2
        grad_N = (x_N - u)*f'(y_i)
        dw_i(t) = momentum*dw_i(t-1) + dw_scale*x_{i-1}*grad_i
        w_i(t+1) = w_i(t) + dw_i(t)
        grad_{i-1} = w_i^t*grad_i
    */
    for( iter = 0; iter < max_iter; iter++ )
    {
        int idx = iter % count;
        double* w = weights[0];
        double sweight = sw ? count*sw[idx] : 1.;
        CvMat _w, _dw, hdr1, hdr2, ghdr1, ghdr2, _df;
        CvMat *x1 = &hdr1, *x2 = &hdr2, *grad1 = &ghdr1, *grad2 = &ghdr2, *temp;

        if( idx == 0 )
        {
            //printf("%d. E = %g\n", iter/count, E);
            if( fabs(prev_E - E) < epsilon )
                break;
            prev_E = E;
            E = 0;

            // shuffle indices
            for( i = 0; i < count; i++ )
            {
                int tt;
                j = (unsigned)cvRandInt(&rng) % count;
                k = (unsigned)cvRandInt(&rng) % count;
                CV_SWAP( _idx->data.i[j], _idx->data.i[k], tt );
            }
        }

        idx = _idx->data.i[idx];

        if( x0.type == CV_32F )
        {
            const float* x0data = x0.data.fl[idx];
            for( j = 0; j < ivcount; j++ )
                x[0][j] = x0data[j]*w[j*2] + w[j*2 + 1];
        }
        else
        {
            const double* x0data = x0.data.db[idx];
            for( j = 0; j < ivcount; j++ )
                x[0][j] = x0data[j]*w[j*2] + w[j*2 + 1];
        }

        cvInitMatHeader( x1, 1, ivcount, CV_64F, x[0] );

        // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
        for( i = 1; i < l_count; i++ )
        {
            cvInitMatHeader( x2, 1, layer_sizes->data.i[i], CV_64F, x[i] );
            cvInitMatHeader( &_w, x1->cols, x2->cols, CV_64F, weights[i] );
            cvGEMM( x1, &_w, 1, 0, 0, x2 );
            _df = *x2;
            _df.data.db = df[i];
            calc_activ_func_deriv( x2, &_df, _w.data.db + _w.rows*_w.cols );
            CV_SWAP( x1, x2, temp );
        }

        cvInitMatHeader( grad1, 1, ovcount, CV_64F, buf_ptr );
        *grad2 = *grad1;
        grad2->data.db = buf_ptr + max_count;

        w = weights[l_count+1];

        // calculate error
        if( u.type == CV_32F )
        {
            const float* udata = u.data.fl[idx];
            for( k = 0; k < ovcount; k++ )
            {
                double t = udata[k]*w[k*2] + w[k*2+1] - x[l_count-1][k];
                grad1->data.db[k] = t*sweight;
                E += t*t;
            }
        }
        else
        {
            const double* udata = u.data.db[idx];
            for( k = 0; k < ovcount; k++ )
            {
                double t = udata[k]*w[k*2] + w[k*2+1] - x[l_count-1][k];
                grad1->data.db[k] = t*sweight;
                E += t*t;
            }
        }
        E *= sweight;

        // backward pass, update weights
        for( i = l_count-1; i > 0; i-- )
        {
            int n1 = layer_sizes->data.i[i-1], n2 = layer_sizes->data.i[i];
            cvInitMatHeader( &_df, 1, n2, CV_64F, df[i] );
            cvMul( grad1, &_df, grad1 );
            cvInitMatHeader( &_w, n1+1, n2, CV_64F, weights[i] );
            cvInitMatHeader( &_dw, n1+1, n2, CV_64F, dw->data.db + (weights[i] - weights[0]) );
            cvInitMatHeader( x1, n1+1, 1, CV_64F, x[i-1] );
            x[i-1][n1] = 1.;
            cvGEMM( x1, grad1, params.bp_dw_scale, &_dw, params.bp_moment_scale, &_dw );
            cvAdd( &_w, &_dw, &_w );
            if( i > 1 )
            {
                grad2->cols = n1;
                _w.rows = n1;
                cvGEMM( grad1, &_w, 1, 0, 0, grad2, CV_GEMM_B_T );
            }
            CV_SWAP( grad1, grad2, temp );
        }
    }

    iter /= count;

    __END__;

    cvReleaseMat( &dw );
    cvReleaseMat( &buf );
    cvReleaseMat( &_idx );
    cvFree( &x );

    return iter;
}


int CvANN_MLP::train_rprop( CvVectors x0, CvVectors u, const double* sw )
{
    const int max_buf_sz = 1 << 16;
    CvMat* dw = 0;
    CvMat* dEdw = 0;
    CvMat* prev_dEdw_sign = 0;
    CvMat* buf = 0;
    double **x = 0, **df = 0;
    int iter = -1, count = x0.count;

    CV_FUNCNAME( "CvANN_MLP::train" );

    __BEGIN__;

    int i, ivcount, ovcount, l_count, total = 0, max_iter, buf_sz, dcount0, dcount=0;
    double *buf_ptr;
    double prev_E = DBL_MAX*0.5, epsilon;
    double dw_plus, dw_minus, dw_min, dw_max;
    double inv_count;

    max_iter = params.term_crit.max_iter;
    epsilon = params.term_crit.epsilon;
    dw_plus = params.rp_dw_plus;
    dw_minus = params.rp_dw_minus;
    dw_min = params.rp_dw_min;
    dw_max = params.rp_dw_max;

    l_count = layer_sizes->cols;
    ivcount = layer_sizes->data.i[0];
    ovcount = layer_sizes->data.i[l_count-1];

    // allocate buffers
    for( i = 0; i < l_count; i++ )
        total += layer_sizes->data.i[i];

    CV_CALL( dw = cvCreateMat( wbuf->rows, wbuf->cols, wbuf->type ));
    cvSet( dw, cvScalarAll(params.rp_dw0) );
    CV_CALL( dEdw = cvCreateMat( wbuf->rows, wbuf->cols, wbuf->type ));
    cvZero( dEdw );
    CV_CALL( prev_dEdw_sign = cvCreateMat( wbuf->rows, wbuf->cols, CV_8SC1 ));
    cvZero( prev_dEdw_sign );

    inv_count = 1./count;
    dcount0 = max_buf_sz/(2*total);
    dcount0 = MAX( dcount0, 1 );
    dcount0 = MIN( dcount0, count );
    buf_sz = dcount0*(total + max_count)*2;

    CV_CALL( buf = cvCreateMat( 1, buf_sz, CV_64F ));

    CV_CALL( x = (double**)cvAlloc( total*2*sizeof(x[0]) ));
    df = x + total;
    buf_ptr = buf->data.db;

    for( i = 0; i < l_count; i++ )
    {
        x[i] = buf_ptr;
        df[i] = x[i] + layer_sizes->data.i[i]*dcount0;
        buf_ptr += (df[i] - x[i])*2;
    }

    // run rprop loop
    /*
        y_i(t) = w_i(t)*x_{i-1}(t)
        x_i(t) = f(y_i(t))
        E = sum_over_all_samples(1/2*||u - x_N||^2)
        grad_N = (x_N - u)*f'(y_i)

                      MIN(dw_i{jk}(t)*dw_plus, dw_max), if dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) > 0
        dw_i{jk}(t) = MAX(dw_i{jk}(t)*dw_minus, dw_min), if dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) < 0
                      dw_i{jk}(t-1) else

        if (dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) < 0)
           dE/dw_i{jk}(t)<-0
        else
           w_i{jk}(t+1) = w_i{jk}(t) + dw_i{jk}(t)
        grad_{i-1}(t) = w_i^t(t)*grad_i(t)
    */
    for( iter = 0; iter < max_iter; iter++ )
    {
        int n1, n2, si, j, k;
        double* w;
        CvMat _w, _dEdw, hdr1, hdr2, ghdr1, ghdr2, _df;
        CvMat *x1, *x2, *grad1, *grad2, *temp;
        double E = 0;

        // first, iterate through all the samples and compute dEdw
        for( si = 0; si < count; si += dcount )
        {
            dcount = MIN( count - si, dcount0 );
            w = weights[0];
            grad1 = &ghdr1; grad2 = &ghdr2;
            x1 = &hdr1; x2 = &hdr2;

            // grab and preprocess input data
            if( x0.type == CV_32F )
                for( i = 0; i < dcount; i++ )
                {
                    const float* x0data = x0.data.fl[si+i];
                    double* xdata = x[0]+i*ivcount;
                    for( j = 0; j < ivcount; j++ )
                        xdata[j] = x0data[j]*w[j*2] + w[j*2+1];
                }
            else
                for( i = 0; i < dcount; i++ )
                {
                    const double* x0data = x0.data.db[si+i];
                    double* xdata = x[0]+i*ivcount;
                    for( j = 0; j < ivcount; j++ )
                        xdata[j] = x0data[j]*w[j*2] + w[j*2+1];
                }

            cvInitMatHeader( x1, dcount, ivcount, CV_64F, x[0] );

            // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
            for( i = 1; i < l_count; i++ )
            {
                cvInitMatHeader( x2, dcount, layer_sizes->data.i[i], CV_64F, x[i] );
                cvInitMatHeader( &_w, x1->cols, x2->cols, CV_64F, weights[i] );
                cvGEMM( x1, &_w, 1, 0, 0, x2 );
                _df = *x2;
                _df.data.db = df[i];
                calc_activ_func_deriv( x2, &_df, _w.data.db + _w.rows*_w.cols );
                CV_SWAP( x1, x2, temp );
            }

            cvInitMatHeader( grad1, dcount, ovcount, CV_64F, buf_ptr );
            w = weights[l_count+1];
            grad2->data.db = buf_ptr + max_count*dcount;

            // calculate error
            if( u.type == CV_32F )
                for( i = 0; i < dcount; i++ )
                {
                    const float* udata = u.data.fl[si+i];
                    const double* xdata = x[l_count-1] + i*ovcount;
                    double* gdata = grad1->data.db + i*ovcount;
                    double sweight = sw ? sw[si+i] : inv_count, E1 = 0;

                    for( j = 0; j < ovcount; j++ )
                    {
                        double t = udata[j]*w[j*2] + w[j*2+1] - xdata[j];
                        gdata[j] = t*sweight;
                        E1 += t*t;
                    }
                    E += sweight*E1;
                }
            else
                for( i = 0; i < dcount; i++ )
                {
                    const double* udata = u.data.db[si+i];
                    const double* xdata = x[l_count-1] + i*ovcount;
                    double* gdata = grad1->data.db + i*ovcount;
                    double sweight = sw ? sw[si+i] : inv_count, E1 = 0;

                    for( j = 0; j < ovcount; j++ )
                    {
                        double t = udata[j]*w[j*2] + w[j*2+1] - xdata[j];
                        gdata[j] = t*sweight;
                        E1 += t*t;
                    }
                    E += sweight*E1;
                }

            // backward pass, update dEdw
            for( i = l_count-1; i > 0; i-- )
            {
                n1 = layer_sizes->data.i[i-1]; n2 = layer_sizes->data.i[i];
                cvInitMatHeader( &_df, dcount, n2, CV_64F, df[i] );
                cvMul( grad1, &_df, grad1 );
                cvInitMatHeader( &_dEdw, n1, n2, CV_64F, dEdw->data.db+(weights[i]-weights[0]) );
                cvInitMatHeader( x1, dcount, n1, CV_64F, x[i-1] );
                cvGEMM( x1, grad1, 1, &_dEdw, 1, &_dEdw, CV_GEMM_A_T );
                // update bias part of dEdw
                for( k = 0; k < dcount; k++ )
                {
                    double* dst = _dEdw.data.db + n1*n2;
                    const double* src = grad1->data.db + k*n2;
                    for( j = 0; j < n2; j++ )
                        dst[j] += src[j];
                }
                cvInitMatHeader( &_w, n1, n2, CV_64F, weights[i] );
                cvInitMatHeader( grad2, dcount, n1, CV_64F, grad2->data.db );

                if( i > 1 )
                    cvGEMM( grad1, &_w, 1, 0, 0, grad2, CV_GEMM_B_T );
                CV_SWAP( grad1, grad2, temp );
            }
        }

        // now update weights
        for( i = 1; i < l_count; i++ )
        {
            n1 = layer_sizes->data.i[i-1]; n2 = layer_sizes->data.i[i];
            for( k = 0; k <= n1; k++ )
            {
                double* wk = weights[i]+k*n2;
                size_t delta = wk - weights[0];
                double* dwk = dw->data.db + delta;
                double* dEdwk = dEdw->data.db + delta;
                char* prevEk = (char*)(prev_dEdw_sign->data.ptr + delta);

                for( j = 0; j < n2; j++ )
                {
                    double Eval = dEdwk[j];
                    double dval = dwk[j];
                    double wval = wk[j];
                    int s = CV_SIGN(Eval);
                    int ss = prevEk[j]*s;
                    if( ss > 0 )
                    {
                        dval *= dw_plus;
                        dval = MIN( dval, dw_max );
                        dwk[j] = dval;
                        wk[j] = wval + dval*s;
                    }
                    else if( ss < 0 )
                    {
                        dval *= dw_minus;
                        dval = MAX( dval, dw_min );
                        prevEk[j] = 0;
                        dwk[j] = dval;
                        wk[j] = wval + dval*s;
                    }
                    else
                    {
                        prevEk[j] = (char)s;
                        wk[j] = wval + dval*s;
                    }
                    dEdwk[j] = 0.;
                }
            }
        }

        //printf("%d. E = %g\n", iter, E);
        if( fabs(prev_E - E) < epsilon )
            break;
        prev_E = E;
        E = 0;
    }

    __END__;

    cvReleaseMat( &dw );
    cvReleaseMat( &dEdw );
    cvReleaseMat( &prev_dEdw_sign );
    cvReleaseMat( &buf );
    cvFree( &x );

    return iter;
}


void CvANN_MLP::write_params( CvFileStorage* fs ) const
{
    //CV_FUNCNAME( "CvANN_MLP::write_params" );

    __BEGIN__;

    const char* activ_func_name = activ_func == IDENTITY ? "IDENTITY" :
                            activ_func == SIGMOID_SYM ? "SIGMOID_SYM" :
                            activ_func == GAUSSIAN ? "GAUSSIAN" : 0;

    if( activ_func_name )
        cvWriteString( fs, "activation_function", activ_func_name );
    else
        cvWriteInt( fs, "activation_function", activ_func );

    if( activ_func != IDENTITY )
    {
        cvWriteReal( fs, "f_param1", f_param1 );
        cvWriteReal( fs, "f_param2", f_param2 );
    }

    cvWriteReal( fs, "min_val", min_val );
    cvWriteReal( fs, "max_val", max_val );
    cvWriteReal( fs, "min_val1", min_val1 );
    cvWriteReal( fs, "max_val1", max_val1 );

    cvStartWriteStruct( fs, "training_params", CV_NODE_MAP );
    if( params.train_method == CvANN_MLP_TrainParams::BACKPROP )
    {
        cvWriteString( fs, "train_method", "BACKPROP" );
        cvWriteReal( fs, "dw_scale", params.bp_dw_scale );
        cvWriteReal( fs, "moment_scale", params.bp_moment_scale );
    }
    else if( params.train_method == CvANN_MLP_TrainParams::RPROP )
    {
        cvWriteString( fs, "train_method", "RPROP" );
        cvWriteReal( fs, "dw0", params.rp_dw0 );
        cvWriteReal( fs, "dw_plus", params.rp_dw_plus );
        cvWriteReal( fs, "dw_minus", params.rp_dw_minus );
        cvWriteReal( fs, "dw_min", params.rp_dw_min );
        cvWriteReal( fs, "dw_max", params.rp_dw_max );
    }

    cvStartWriteStruct( fs, "term_criteria", CV_NODE_MAP + CV_NODE_FLOW );
    if( params.term_crit.type & CV_TERMCRIT_EPS )
        cvWriteReal( fs, "epsilon", params.term_crit.epsilon );
    if( params.term_crit.type & CV_TERMCRIT_ITER )
        cvWriteInt( fs, "iterations", params.term_crit.max_iter );
    cvEndWriteStruct( fs );

    cvEndWriteStruct( fs );

    __END__;
}


void CvANN_MLP::write( CvFileStorage* fs, const char* name ) const
{
    CV_FUNCNAME( "CvANN_MLP::write" );

    __BEGIN__;

    int i, l_count = layer_sizes->cols;

    if( !layer_sizes )
        CV_ERROR( CV_StsError, "The network has not been initialized" );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_ANN_MLP );

    cvWrite( fs, "layer_sizes", layer_sizes );

    write_params( fs );

    cvStartWriteStruct( fs, "input_scale", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, weights[0], layer_sizes->data.i[0]*2, "d" );
    cvEndWriteStruct( fs );

    cvStartWriteStruct( fs, "output_scale", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, weights[l_count], layer_sizes->data.i[l_count-1]*2, "d" );
    cvEndWriteStruct( fs );

    cvStartWriteStruct( fs, "inv_output_scale", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, weights[l_count+1], layer_sizes->data.i[l_count-1]*2, "d" );
    cvEndWriteStruct( fs );

    cvStartWriteStruct( fs, "weights", CV_NODE_SEQ );
    for( i = 1; i < l_count; i++ )
    {
        cvStartWriteStruct( fs, 0, CV_NODE_SEQ + CV_NODE_FLOW );
        cvWriteRawData( fs, weights[i], (layer_sizes->data.i[i-1]+1)*layer_sizes->data.i[i], "d" );
        cvEndWriteStruct( fs );
    }

    cvEndWriteStruct( fs );

    __END__;
}


void CvANN_MLP::read_params( CvFileStorage* fs, CvFileNode* node )
{
    //CV_FUNCNAME( "CvANN_MLP::read_params" );

    __BEGIN__;

    const char* activ_func_name = cvReadStringByName( fs, node, "activation_function", 0 );
    CvFileNode* tparams_node;

    if( activ_func_name )
        activ_func = strcmp( activ_func_name, "SIGMOID_SYM" ) == 0 ? SIGMOID_SYM :
                     strcmp( activ_func_name, "IDENTITY" ) == 0 ? IDENTITY :
                     strcmp( activ_func_name, "GAUSSIAN" ) == 0 ? GAUSSIAN : 0;
    else
        activ_func = cvReadIntByName( fs, node, "activation_function" );

    f_param1 = cvReadRealByName( fs, node, "f_param1", 0 );
    f_param2 = cvReadRealByName( fs, node, "f_param2", 0 );

    set_activ_func( activ_func, f_param1, f_param2 );

    min_val = cvReadRealByName( fs, node, "min_val", 0. );
    max_val = cvReadRealByName( fs, node, "max_val", 1. );
    min_val1 = cvReadRealByName( fs, node, "min_val1", 0. );
    max_val1 = cvReadRealByName( fs, node, "max_val1", 1. );

    tparams_node = cvGetFileNodeByName( fs, node, "training_params" );
    params = CvANN_MLP_TrainParams();

    if( tparams_node )
    {
        const char* tmethod_name = cvReadStringByName( fs, tparams_node, "train_method", "" );
        CvFileNode* tcrit_node;

        if( strcmp( tmethod_name, "BACKPROP" ) == 0 )
        {
            params.train_method = CvANN_MLP_TrainParams::BACKPROP;
            params.bp_dw_scale = cvReadRealByName( fs, tparams_node, "dw_scale", 0 );
            params.bp_moment_scale = cvReadRealByName( fs, tparams_node, "moment_scale", 0 );
        }
        else if( strcmp( tmethod_name, "RPROP" ) == 0 )
        {
            params.train_method = CvANN_MLP_TrainParams::RPROP;
            params.rp_dw0 = cvReadRealByName( fs, tparams_node, "dw0", 0 );
            params.rp_dw_plus = cvReadRealByName( fs, tparams_node, "dw_plus", 0 );
            params.rp_dw_minus = cvReadRealByName( fs, tparams_node, "dw_minus", 0 );
            params.rp_dw_min = cvReadRealByName( fs, tparams_node, "dw_min", 0 );
            params.rp_dw_max = cvReadRealByName( fs, tparams_node, "dw_max", 0 );
        }

        tcrit_node = cvGetFileNodeByName( fs, tparams_node, "term_criteria" );
        if( tcrit_node )
        {
            params.term_crit.epsilon = cvReadRealByName( fs, tcrit_node, "epsilon", -1 );
            params.term_crit.max_iter = cvReadIntByName( fs, tcrit_node, "iterations", -1 );
            params.term_crit.type = (params.term_crit.epsilon >= 0 ? CV_TERMCRIT_EPS : 0) +
                                   (params.term_crit.max_iter >= 0 ? CV_TERMCRIT_ITER : 0);
        }
    }

    __END__;
}


void CvANN_MLP::read( CvFileStorage* fs, CvFileNode* node )
{
    CvMat* _layer_sizes = 0;

    CV_FUNCNAME( "CvANN_MLP::read" );

    __BEGIN__;

    CvFileNode* w;
    CvSeqReader reader;
    int i, l_count;

    _layer_sizes = (CvMat*)cvReadByName( fs, node, "layer_sizes" );
    CV_CALL( create( _layer_sizes, SIGMOID_SYM, 0, 0 ));
    l_count = layer_sizes->cols;

    CV_CALL( read_params( fs, node ));

    w = cvGetFileNodeByName( fs, node, "input_scale" );
    if( !w || CV_NODE_TYPE(w->tag) != CV_NODE_SEQ ||
        w->data.seq->total != layer_sizes->data.i[0]*2 )
        CV_ERROR( CV_StsParseError, "input_scale tag is not found or is invalid" );

    CV_CALL( cvReadRawData( fs, w, weights[0], "d" ));

    w = cvGetFileNodeByName( fs, node, "output_scale" );
    if( !w || CV_NODE_TYPE(w->tag) != CV_NODE_SEQ ||
        w->data.seq->total != layer_sizes->data.i[l_count-1]*2 )
        CV_ERROR( CV_StsParseError, "output_scale tag is not found or is invalid" );

    CV_CALL( cvReadRawData( fs, w, weights[l_count], "d" ));

    w = cvGetFileNodeByName( fs, node, "inv_output_scale" );
    if( !w || CV_NODE_TYPE(w->tag) != CV_NODE_SEQ ||
        w->data.seq->total != layer_sizes->data.i[l_count-1]*2 )
        CV_ERROR( CV_StsParseError, "inv_output_scale tag is not found or is invalid" );

    CV_CALL( cvReadRawData( fs, w, weights[l_count+1], "d" ));

    w = cvGetFileNodeByName( fs, node, "weights" );
    if( !w || CV_NODE_TYPE(w->tag) != CV_NODE_SEQ ||
        w->data.seq->total != l_count - 1 )
        CV_ERROR( CV_StsParseError, "weights tag is not found or is invalid" );

    cvStartReadSeq( w->data.seq, &reader );

    for( i = 1; i < l_count; i++ )
    {
        w = (CvFileNode*)reader.ptr;
        CV_CALL( cvReadRawData( fs, w, weights[i], "d" ));
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
    }

    __END__;
}

using namespace cv;

CvANN_MLP::CvANN_MLP( const Mat& _layer_sizes, int _activ_func,
                      double _f_param1, double _f_param2 )
{
    layer_sizes = wbuf = 0;
    min_val = max_val = min_val1 = max_val1 = 0.;
    weights = 0;
    rng = cvRNG(-1);
    default_model_name = "my_nn";
    create( _layer_sizes, _activ_func, _f_param1, _f_param2 );
}

void CvANN_MLP::create( const Mat& _layer_sizes, int _activ_func,
                       double _f_param1, double _f_param2 )
{
    CvMat layer_sizes = _layer_sizes;
    create( &layer_sizes, _activ_func, _f_param1, _f_param2 );
}

int CvANN_MLP::train( const Mat& _inputs, const Mat& _outputs,
                     const Mat& _sample_weights, const Mat& _sample_idx,
                     CvANN_MLP_TrainParams _params, int flags )
{
    CvMat inputs = _inputs, outputs = _outputs, sweights = _sample_weights, sidx = _sample_idx;
    return train(&inputs, &outputs, sweights.data.ptr ? &sweights : 0,
                 sidx.data.ptr ? &sidx : 0, _params, flags); 
}

float CvANN_MLP::predict( const Mat& _inputs, Mat& _outputs ) const
{
    CV_Assert(layer_sizes != 0);
    _outputs.create(_inputs.rows, layer_sizes->data.i[layer_sizes->cols-1], _inputs.type());
    CvMat inputs = _inputs, outputs = _outputs;
    
    return predict(&inputs, &outputs); 
}

/* End of file. */
