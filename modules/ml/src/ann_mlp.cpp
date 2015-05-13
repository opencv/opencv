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

namespace cv { namespace ml {

struct AnnParams
{
    AnnParams()
    {
        termCrit = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 1000, 0.01 );
        trainMethod = ANN_MLP::RPROP;
        bpDWScale = bpMomentScale = 0.1;
        rpDW0 = 0.1; rpDWPlus = 1.2; rpDWMinus = 0.5;
        rpDWMin = FLT_EPSILON; rpDWMax = 50.;
    }

    TermCriteria termCrit;
    int trainMethod;

    double bpDWScale;
    double bpMomentScale;

    double rpDW0;
    double rpDWPlus;
    double rpDWMinus;
    double rpDWMin;
    double rpDWMax;
};

template <typename T>
inline T inBounds(T val, T min_val, T max_val)
{
    return std::min(std::max(val, min_val), max_val);
}

class ANN_MLPImpl : public ANN_MLP
{
public:
    ANN_MLPImpl()
    {
        clear();
        setActivationFunction( SIGMOID_SYM, 0, 0 );
        setLayerSizes(Mat());
        setTrainMethod(ANN_MLP::RPROP, 0.1, FLT_EPSILON);
    }

    virtual ~ANN_MLPImpl() {}

    CV_IMPL_PROPERTY(TermCriteria, TermCriteria, params.termCrit)
    CV_IMPL_PROPERTY(double, BackpropWeightScale, params.bpDWScale)
    CV_IMPL_PROPERTY(double, BackpropMomentumScale, params.bpMomentScale)
    CV_IMPL_PROPERTY(double, RpropDW0, params.rpDW0)
    CV_IMPL_PROPERTY(double, RpropDWPlus, params.rpDWPlus)
    CV_IMPL_PROPERTY(double, RpropDWMinus, params.rpDWMinus)
    CV_IMPL_PROPERTY(double, RpropDWMin, params.rpDWMin)
    CV_IMPL_PROPERTY(double, RpropDWMax, params.rpDWMax)

    void clear()
    {
        min_val = max_val = min_val1 = max_val1 = 0.;
        rng = RNG((uint64)-1);
        weights.clear();
        trained = false;
        max_buf_sz = 1 << 12;
    }

    int layer_count() const { return (int)layer_sizes.size(); }

    void setTrainMethod(int method, double param1, double param2)
    {
        if (method != ANN_MLP::RPROP && method != ANN_MLP::BACKPROP)
            method = ANN_MLP::RPROP;
        params.trainMethod = method;
        if(method == ANN_MLP::RPROP )
        {
            if( param1 < FLT_EPSILON )
                param1 = 1.;
            params.rpDW0 = param1;
            params.rpDWMin = std::max( param2, 0. );
        }
        else if(method == ANN_MLP::BACKPROP )
        {
            if( param1 <= 0 )
                param1 = 0.1;
            params.bpDWScale = inBounds<double>(param1, 1e-3, 1.);
            if( param2 < 0 )
                param2 = 0.1;
            params.bpMomentScale = std::min( param2, 1. );
        }
    }

    int getTrainMethod() const
    {
        return params.trainMethod;
    }

    void setActivationFunction(int _activ_func, double _f_param1, double _f_param2 )
    {
        if( _activ_func < 0 || _activ_func > GAUSSIAN )
            CV_Error( CV_StsOutOfRange, "Unknown activation function" );

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
    }


    void init_weights()
    {
        int i, j, k, l_count = layer_count();

        for( i = 1; i < l_count; i++ )
        {
            int n1 = layer_sizes[i-1];
            int n2 = layer_sizes[i];
            double val = 0, G = n2 > 2 ? 0.7*pow((double)n1,1./(n2-1)) : 1.;
            double* w = weights[i].ptr<double>();

            // initialize weights using Nguyen-Widrow algorithm
            for( j = 0; j < n2; j++ )
            {
                double s = 0;
                for( k = 0; k <= n1; k++ )
                {
                    val = rng.uniform(0., 1.)*2-1.;
                    w[k*n2 + j] = val;
                    s += fabs(val);
                }

                if( i < l_count - 1 )
                {
                    s = 1./(s - fabs(val));
                    for( k = 0; k <= n1; k++ )
                        w[k*n2 + j] *= s;
                    w[n1*n2 + j] *= G*(-1+j*2./n2);
                }
            }
        }
    }

    Mat getLayerSizes() const
    {
        return Mat_<int>(layer_sizes, true);
    }

    void setLayerSizes( InputArray _layer_sizes )
    {
        clear();

        _layer_sizes.copyTo(layer_sizes);
        int l_count = layer_count();

        weights.resize(l_count + 2);
        max_lsize = 0;

        if( l_count > 0 )
        {
            for( int i = 0; i < l_count; i++ )
            {
                int n = layer_sizes[i];
                if( n < 1 + (0 < i && i < l_count-1))
                    CV_Error( CV_StsOutOfRange,
                             "there should be at least one input and one output "
                             "and every hidden layer must have more than 1 neuron" );
                max_lsize = std::max( max_lsize, n );
                if( i > 0 )
                    weights[i].create(layer_sizes[i-1]+1, n, CV_64F);
            }

            int ninputs = layer_sizes.front();
            int noutputs = layer_sizes.back();
            weights[0].create(1, ninputs*2, CV_64F);
            weights[l_count].create(1, noutputs*2, CV_64F);
            weights[l_count+1].create(1, noutputs*2, CV_64F);
        }
    }

    float predict( InputArray _inputs, OutputArray _outputs, int ) const
    {
        if( !trained )
            CV_Error( CV_StsError, "The network has not been trained or loaded" );

        Mat inputs = _inputs.getMat();
        int type = inputs.type(), l_count = layer_count();
        int n = inputs.rows, dn0 = n;

        CV_Assert( (type == CV_32F || type == CV_64F) && inputs.cols == layer_sizes[0] );
        int noutputs = layer_sizes[l_count-1];
        Mat outputs;

        int min_buf_sz = 2*max_lsize;
        int buf_sz = n*min_buf_sz;

        if( buf_sz > max_buf_sz )
        {
            dn0 = max_buf_sz/min_buf_sz;
            dn0 = std::max( dn0, 1 );
            buf_sz = dn0*min_buf_sz;
        }

        cv::AutoBuffer<double> _buf(buf_sz+noutputs);
        double* buf = _buf;

        if( !_outputs.needed() )
        {
            CV_Assert( n == 1 );
            outputs = Mat(n, noutputs, type, buf + buf_sz);
        }
        else
        {
            _outputs.create(n, noutputs, type);
            outputs = _outputs.getMat();
        }

        int dn = 0;
        for( int i = 0; i < n; i += dn )
        {
            dn = std::min( dn0, n - i );

            Mat layer_in = inputs.rowRange(i, i + dn);
            Mat layer_out( dn, layer_in.cols, CV_64F, buf);

            scale_input( layer_in, layer_out );
            layer_in = layer_out;

            for( int j = 1; j < l_count; j++ )
            {
                double* data = buf + ((j&1) ? max_lsize*dn0 : 0);
                int cols = layer_sizes[j];

                layer_out = Mat(dn, cols, CV_64F, data);
                Mat w = weights[j].rowRange(0, layer_in.cols);
                gemm(layer_in, w, 1, noArray(), 0, layer_out);
                calc_activ_func( layer_out, weights[j] );

                layer_in = layer_out;
            }

            layer_out = outputs.rowRange(i, i + dn);
            scale_output( layer_in, layer_out );
        }

        if( n == 1 )
        {
            int maxIdx[] = {0, 0};
            minMaxIdx(outputs, 0, 0, 0, maxIdx);
            return (float)(maxIdx[0] + maxIdx[1]);
        }

        return 0.f;
    }

    void scale_input( const Mat& _src, Mat& _dst ) const
    {
        int cols = _src.cols;
        const double* w = weights[0].ptr<double>();

        if( _src.type() == CV_32F )
        {
            for( int i = 0; i < _src.rows; i++ )
            {
                const float* src = _src.ptr<float>(i);
                double* dst = _dst.ptr<double>(i);
                for( int j = 0; j < cols; j++ )
                    dst[j] = src[j]*w[j*2] + w[j*2+1];
            }
        }
        else
        {
            for( int i = 0; i < _src.rows; i++ )
            {
                const float* src = _src.ptr<float>(i);
                double* dst = _dst.ptr<double>(i);
                for( int j = 0; j < cols; j++ )
                    dst[j] = src[j]*w[j*2] + w[j*2+1];
            }
        }
    }

    void scale_output( const Mat& _src, Mat& _dst ) const
    {
        int cols = _src.cols;
        const double* w = weights[layer_count()].ptr<double>();

        if( _dst.type() == CV_32F )
        {
            for( int i = 0; i < _src.rows; i++ )
            {
                const double* src = _src.ptr<double>(i);
                float* dst = _dst.ptr<float>(i);
                for( int j = 0; j < cols; j++ )
                    dst[j] = (float)(src[j]*w[j*2] + w[j*2+1]);
            }
        }
        else
        {
            for( int i = 0; i < _src.rows; i++ )
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( int j = 0; j < cols; j++ )
                    dst[j] = src[j]*w[j*2] + w[j*2+1];
            }
        }
    }

    void calc_activ_func( Mat& sums, const Mat& w ) const
    {
        const double* bias = w.ptr<double>(w.rows-1);
        int i, j, n = sums.rows, cols = sums.cols;
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

        CV_Assert( sums.isContinuous() );

        if( activ_func != GAUSSIAN )
        {
            for( i = 0; i < n; i++ )
            {
                double* data = sums.ptr<double>(i);
                for( j = 0; j < cols; j++ )
                    data[j] = (data[j] + bias[j])*scale;
            }

            if( activ_func == IDENTITY )
                return;
        }
        else
        {
            for( i = 0; i < n; i++ )
            {
                double* data = sums.ptr<double>(i);
                for( j = 0; j < cols; j++ )
                {
                    double t = data[j] + bias[j];
                    data[j] = t*t*scale;
                }
            }
        }

        exp( sums, sums );

        if( sums.isContinuous() )
        {
            cols *= n;
            n = 1;
        }

        switch( activ_func )
        {
            case SIGMOID_SYM:
                for( i = 0; i < n; i++ )
                {
                    double* data = sums.ptr<double>(i);
                    for( j = 0; j < cols; j++ )
                    {
                        double t = scale2*(1. - data[j])/(1. + data[j]);
                        data[j] = t;
                    }
                }
                break;

            case GAUSSIAN:
                for( i = 0; i < n; i++ )
                {
                    double* data = sums.ptr<double>(i);
                    for( j = 0; j < cols; j++ )
                        data[j] = scale2*data[j];
                }
                break;

            default:
                ;
        }
    }

    void calc_activ_func_deriv( Mat& _xf, Mat& _df, const Mat& w ) const
    {
        const double* bias = w.ptr<double>(w.rows-1);
        int i, j, n = _xf.rows, cols = _xf.cols;

        if( activ_func == IDENTITY )
        {
            for( i = 0; i < n; i++ )
            {
                double* xf = _xf.ptr<double>(i);
                double* df = _df.ptr<double>(i);

                for( j = 0; j < cols; j++ )
                {
                    xf[j] += bias[j];
                    df[j] = 1;
                }
            }
        }
        else if( activ_func == GAUSSIAN )
        {
            double scale = -f_param1*f_param1;
            double scale2 = scale*f_param2;
            for( i = 0; i < n; i++ )
            {
                double* xf = _xf.ptr<double>(i);
                double* df = _df.ptr<double>(i);

                for( j = 0; j < cols; j++ )
                {
                    double t = xf[j] + bias[j];
                    df[j] = t*2*scale2;
                    xf[j] = t*t*scale;
                }
            }
            exp( _xf, _xf );

            for( i = 0; i < n; i++ )
            {
                double* xf = _xf.ptr<double>(i);
                double* df = _df.ptr<double>(i);

                for( j = 0; j < cols; j++ )
                    df[j] *= xf[j];
            }
        }
        else
        {
            double scale = f_param1;
            double scale2 = f_param2;

            for( i = 0; i < n; i++ )
            {
                double* xf = _xf.ptr<double>(i);
                double* df = _df.ptr<double>(i);

                for( j = 0; j < cols; j++ )
                {
                    xf[j] = (xf[j] + bias[j])*scale;
                    df[j] = -fabs(xf[j]);
                }
            }

            exp( _df, _df );

            // ((1+exp(-ax))^-1)'=a*((1+exp(-ax))^-2)*exp(-ax);
            // ((1-exp(-ax))/(1+exp(-ax)))'=(a*exp(-ax)*(1+exp(-ax)) + a*exp(-ax)*(1-exp(-ax)))/(1+exp(-ax))^2=
            // 2*a*exp(-ax)/(1+exp(-ax))^2
            scale *= 2*f_param2;
            for( i = 0; i < n; i++ )
            {
                double* xf = _xf.ptr<double>(i);
                double* df = _df.ptr<double>(i);

                for( j = 0; j < cols; j++ )
                {
                    int s0 = xf[j] > 0 ? 1 : -1;
                    double t0 = 1./(1. + df[j]);
                    double t1 = scale*df[j]*t0*t0;
                    t0 *= scale2*(1. - df[j])*s0;
                    df[j] = t1;
                    xf[j] = t0;
                }
            }
        }
    }

    void calc_input_scale( const Mat& inputs, int flags )
    {
        bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
        bool no_scale = (flags & NO_INPUT_SCALE) != 0;
        double* scale = weights[0].ptr<double>();
        int count = inputs.rows;

        if( reset_weights )
        {
            int i, j, vcount = layer_sizes[0];
            int type = inputs.type();
            double a = no_scale ? 1. : 0.;

            for( j = 0; j < vcount; j++ )
                scale[2*j] = a, scale[j*2+1] = 0.;

            if( no_scale )
                return;

            for( i = 0; i < count; i++ )
            {
                const uchar* p = inputs.ptr(i);
                const float* f = (const float*)p;
                const double* d = (const double*)p;
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

    void calc_output_scale( const Mat& outputs, int flags )
    {
        int i, j, vcount = layer_sizes.back();
        int type = outputs.type();
        double m = min_val, M = max_val, m1 = min_val1, M1 = max_val1;
        bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
        bool no_scale = (flags & NO_OUTPUT_SCALE) != 0;
        int l_count = layer_count();
        double* scale = weights[l_count].ptr<double>();
        double* inv_scale = weights[l_count+1].ptr<double>();
        int count = outputs.rows;

        if( reset_weights )
        {
            double a0 = no_scale ? 1 : DBL_MAX, b0 = no_scale ? 0 : -DBL_MAX;

            for( j = 0; j < vcount; j++ )
            {
                scale[2*j] = inv_scale[2*j] = a0;
                scale[j*2+1] = inv_scale[2*j+1] = b0;
            }

            if( no_scale )
                return;
        }

        for( i = 0; i < count; i++ )
        {
            const uchar* p = outputs.ptr(i);
            const float* f = (const float*)p;
            const double* d = (const double*)p;

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
                else if( !no_scale )
                {
                    t = t*inv_scale[j*2] + inv_scale[2*j+1];
                    if( t < m1 || t > M1 )
                        CV_Error( CV_StsOutOfRange,
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
    }

    void prepare_to_train( const Mat& inputs, const Mat& outputs,
                           Mat& sample_weights, int flags )
    {
        if( layer_sizes.empty() )
            CV_Error( CV_StsError,
                     "The network has not been created. Use method create or the appropriate constructor" );

        if( (inputs.type() != CV_32F && inputs.type() != CV_64F) ||
            inputs.cols != layer_sizes[0] )
            CV_Error( CV_StsBadArg,
                     "input training data should be a floating-point matrix with "
                     "the number of rows equal to the number of training samples and "
                     "the number of columns equal to the size of 0-th (input) layer" );

        if( (outputs.type() != CV_32F && outputs.type() != CV_64F) ||
            outputs.cols != layer_sizes.back() )
            CV_Error( CV_StsBadArg,
                     "output training data should be a floating-point matrix with "
                     "the number of rows equal to the number of training samples and "
                     "the number of columns equal to the size of last (output) layer" );

        if( inputs.rows != outputs.rows )
            CV_Error( CV_StsUnmatchedSizes, "The numbers of input and output samples do not match" );

        Mat temp;
        double s = sum(sample_weights)[0];
        sample_weights.convertTo(temp, CV_64F, 1./s);
        sample_weights = temp;

        calc_input_scale( inputs, flags );
        calc_output_scale( outputs, flags );
    }

    bool train( const Ptr<TrainData>& trainData, int flags )
    {
        const int MAX_ITER = 1000;
        const double DEFAULT_EPSILON = FLT_EPSILON;

        // initialize training data
        Mat inputs = trainData->getTrainSamples();
        Mat outputs = trainData->getTrainResponses();
        Mat sw = trainData->getTrainSampleWeights();
        prepare_to_train( inputs, outputs, sw, flags );

        // ... and link weights
        if( !(flags & UPDATE_WEIGHTS) )
            init_weights();

        TermCriteria termcrit;
        termcrit.type = TermCriteria::COUNT + TermCriteria::EPS;
        termcrit.maxCount = std::max((params.termCrit.type & CV_TERMCRIT_ITER ? params.termCrit.maxCount : MAX_ITER), 1);
        termcrit.epsilon = std::max((params.termCrit.type & CV_TERMCRIT_EPS ? params.termCrit.epsilon : DEFAULT_EPSILON), DBL_EPSILON);

        int iter = params.trainMethod == ANN_MLP::BACKPROP ?
            train_backprop( inputs, outputs, sw, termcrit ) :
            train_rprop( inputs, outputs, sw, termcrit );

        trained = iter > 0;
        return trained;
    }

    int train_backprop( const Mat& inputs, const Mat& outputs, const Mat& _sw, TermCriteria termCrit )
    {
        int i, j, k;
        double prev_E = DBL_MAX*0.5, E = 0;
        int itype = inputs.type(), otype = outputs.type();

        int count = inputs.rows;

        int iter = -1, max_iter = termCrit.maxCount*count;
        double epsilon = termCrit.epsilon*count;

        int l_count = layer_count();
        int ivcount = layer_sizes[0];
        int ovcount = layer_sizes.back();

        // allocate buffers
        vector<vector<double> > x(l_count);
        vector<vector<double> > df(l_count);
        vector<Mat> dw(l_count);

        for( i = 0; i < l_count; i++ )
        {
            int n = layer_sizes[i];
            x[i].resize(n+1);
            df[i].resize(n);
            dw[i] = Mat::zeros(weights[i].size(), CV_64F);
        }

        Mat _idx_m(1, count, CV_32S);
        int* _idx = _idx_m.ptr<int>();
        for( i = 0; i < count; i++ )
            _idx[i] = i;

        AutoBuffer<double> _buf(max_lsize*2);
        double* buf[] = { _buf, (double*)_buf + max_lsize };

        const double* sw = _sw.empty() ? 0 : _sw.ptr<double>();

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
            double sweight = sw ? count*sw[idx] : 1.;

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
                    j = rng.uniform(0, count);
                    k = rng.uniform(0, count);
                    std::swap(_idx[j], _idx[k]);
                }
            }

            idx = _idx[idx];

            const uchar* x0data_p = inputs.ptr(idx);
            const float* x0data_f = (const float*)x0data_p;
            const double* x0data_d = (const double*)x0data_p;

            double* w = weights[0].ptr<double>();
            for( j = 0; j < ivcount; j++ )
                x[0][j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j])*w[j*2] + w[j*2 + 1];

            Mat x1( 1, ivcount, CV_64F, &x[0][0] );

            // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
            for( i = 1; i < l_count; i++ )
            {
                int n = layer_sizes[i];
                Mat x2(1, n, CV_64F, &x[i][0] );
                Mat _w = weights[i].rowRange(0, x1.cols);
                gemm(x1, _w, 1, noArray(), 0, x2);
                Mat _df(1, n, CV_64F, &df[i][0] );
                calc_activ_func_deriv( x2, _df, weights[i] );
                x1 = x2;
            }

            Mat grad1( 1, ovcount, CV_64F, buf[l_count&1] );
            w = weights[l_count+1].ptr<double>();

            // calculate error
            const uchar* udata_p = outputs.ptr(idx);
            const float* udata_f = (const float*)udata_p;
            const double* udata_d = (const double*)udata_p;

            double* gdata = grad1.ptr<double>();
            for( k = 0; k < ovcount; k++ )
            {
                double t = (otype == CV_32F ? (double)udata_f[k] : udata_d[k])*w[k*2] + w[k*2+1] - x[l_count-1][k];
                gdata[k] = t*sweight;
                E += t*t;
            }
            E *= sweight;

            // backward pass, update weights
            for( i = l_count-1; i > 0; i-- )
            {
                int n1 = layer_sizes[i-1], n2 = layer_sizes[i];
                Mat _df(1, n2, CV_64F, &df[i][0]);
                multiply( grad1, _df, grad1 );
                Mat _x(n1+1, 1, CV_64F, &x[i-1][0]);
                x[i-1][n1] = 1.;
                gemm( _x, grad1, params.bpDWScale, dw[i], params.bpMomentScale, dw[i] );
                add( weights[i], dw[i], weights[i] );
                if( i > 1 )
                {
                    Mat grad2(1, n1, CV_64F, buf[i&1]);
                    Mat _w = weights[i].rowRange(0, n1);
                    gemm( grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T );
                    grad1 = grad2;
                }
            }
        }

        iter /= count;
        return iter;
    }

    struct RPropLoop : public ParallelLoopBody
    {
        RPropLoop(ANN_MLPImpl* _ann,
                  const Mat& _inputs, const Mat& _outputs, const Mat& _sw,
                  int _dcount0, vector<Mat>& _dEdw, double* _E)
        {
            ann = _ann;
            inputs = _inputs;
            outputs = _outputs;
            sw = _sw.ptr<double>();
            dcount0 = _dcount0;
            dEdw = &_dEdw;
            pE = _E;
        }

        ANN_MLPImpl* ann;
        vector<Mat>* dEdw;
        Mat inputs, outputs;
        const double* sw;
        int dcount0;
        double* pE;

        void operator()( const Range& range ) const
        {
            double inv_count = 1./inputs.rows;
            int ivcount = ann->layer_sizes.front();
            int ovcount = ann->layer_sizes.back();
            int itype = inputs.type(), otype = outputs.type();
            int count = inputs.rows;
            int i, j, k, l_count = ann->layer_count();
            vector<vector<double> > x(l_count);
            vector<vector<double> > df(l_count);
            vector<double> _buf(ann->max_lsize*dcount0*2);
            double* buf[] = { &_buf[0], &_buf[ann->max_lsize*dcount0] };
            double E = 0;

            for( i = 0; i < l_count; i++ )
            {
                x[i].resize(ann->layer_sizes[i]*dcount0);
                df[i].resize(ann->layer_sizes[i]*dcount0);
            }

            for( int si = range.start; si < range.end; si++ )
            {
                int i0 = si*dcount0, i1 = std::min((si + 1)*dcount0, count);
                int dcount = i1 - i0;
                const double* w = ann->weights[0].ptr<double>();

                // grab and preprocess input data
                for( i = 0; i < dcount; i++ )
                {
                    const uchar* x0data_p = inputs.ptr(i0 + i);
                    const float* x0data_f = (const float*)x0data_p;
                    const double* x0data_d = (const double*)x0data_p;

                    double* xdata = &x[0][i*ivcount];
                    for( j = 0; j < ivcount; j++ )
                        xdata[j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j])*w[j*2] + w[j*2+1];
                }
                Mat x1(dcount, ivcount, CV_64F, &x[0][0]);

                // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
                for( i = 1; i < l_count; i++ )
                {
                    Mat x2( dcount, ann->layer_sizes[i], CV_64F, &x[i][0] );
                    Mat _w = ann->weights[i].rowRange(0, x1.cols);
                    gemm( x1, _w, 1, noArray(), 0, x2 );
                    Mat _df( x2.size(), CV_64F, &df[i][0] );
                    ann->calc_activ_func_deriv( x2, _df, ann->weights[i] );
                    x1 = x2;
                }

                Mat grad1(dcount, ovcount, CV_64F, buf[l_count & 1]);

                w = ann->weights[l_count+1].ptr<double>();

                // calculate error
                for( i = 0; i < dcount; i++ )
                {
                    const uchar* udata_p = outputs.ptr(i0+i);
                    const float* udata_f = (const float*)udata_p;
                    const double* udata_d = (const double*)udata_p;

                    const double* xdata = &x[l_count-1][i*ovcount];
                    double* gdata = grad1.ptr<double>(i);
                    double sweight = sw ? sw[si+i] : inv_count, E1 = 0;

                    for( j = 0; j < ovcount; j++ )
                    {
                        double t = (otype == CV_32F ? (double)udata_f[j] : udata_d[j])*w[j*2] + w[j*2+1] - xdata[j];
                        gdata[j] = t*sweight;
                        E1 += t*t;
                    }
                    E += sweight*E1;
                }

                for( i = l_count-1; i > 0; i-- )
                {
                    int n1 = ann->layer_sizes[i-1], n2 = ann->layer_sizes[i];
                    Mat _df(dcount, n2, CV_64F, &df[i][0]);
                    multiply(grad1, _df, grad1);

                    {
                        AutoLock lock(ann->mtx);
                        Mat _dEdw = dEdw->at(i).rowRange(0, n1);
                        x1 = Mat(dcount, n1, CV_64F, &x[i-1][0]);
                        gemm(x1, grad1, 1, _dEdw, 1, _dEdw, GEMM_1_T);

                        // update bias part of dEdw
                        double* dst = dEdw->at(i).ptr<double>(n1);
                        for( k = 0; k < dcount; k++ )
                        {
                            const double* src = grad1.ptr<double>(k);
                            for( j = 0; j < n2; j++ )
                                dst[j] += src[j];
                        }
                    }

                    Mat grad2( dcount, n1, CV_64F, buf[i&1] );
                    if( i > 1 )
                    {
                        Mat _w = ann->weights[i].rowRange(0, n1);
                        gemm(grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T);
                    }
                    grad1 = grad2;
                }
            }
            {
                AutoLock lock(ann->mtx);
                *pE += E;
            }
        }
    };

    int train_rprop( const Mat& inputs, const Mat& outputs, const Mat& _sw, TermCriteria termCrit )
    {
        const int max_buf_size = 1 << 16;
        int i, iter = -1, count = inputs.rows;

        double prev_E = DBL_MAX*0.5;

        int max_iter = termCrit.maxCount;
        double epsilon = termCrit.epsilon;
        double dw_plus = params.rpDWPlus;
        double dw_minus = params.rpDWMinus;
        double dw_min = params.rpDWMin;
        double dw_max = params.rpDWMax;

        int l_count = layer_count();

        // allocate buffers
        vector<Mat> dw(l_count), dEdw(l_count), prev_dEdw_sign(l_count);

        int total = 0;
        for( i = 0; i < l_count; i++ )
        {
            total += layer_sizes[i];
            dw[i].create(weights[i].size(), CV_64F);
            dw[i].setTo(Scalar::all(params.rpDW0));
            prev_dEdw_sign[i] = Mat::zeros(weights[i].size(), CV_8S);
            dEdw[i] = Mat::zeros(weights[i].size(), CV_64F);
        }

        int dcount0 = max_buf_size/(2*total);
        dcount0 = std::max( dcount0, 1 );
        dcount0 = std::min( dcount0, count );
        int chunk_count = (count + dcount0 - 1)/dcount0;

        // run rprop loop
        /*
         y_i(t) = w_i(t)*x_{i-1}(t)
         x_i(t) = f(y_i(t))
         E = sum_over_all_samples(1/2*||u - x_N||^2)
         grad_N = (x_N - u)*f'(y_i)

         std::min(dw_i{jk}(t)*dw_plus, dw_max), if dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) > 0
         dw_i{jk}(t) = std::max(dw_i{jk}(t)*dw_minus, dw_min), if dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) < 0
         dw_i{jk}(t-1) else

         if (dE/dw_i{jk}(t)*dE/dw_i{jk}(t-1) < 0)
         dE/dw_i{jk}(t)<-0
         else
         w_i{jk}(t+1) = w_i{jk}(t) + dw_i{jk}(t)
         grad_{i-1}(t) = w_i^t(t)*grad_i(t)
         */
        for( iter = 0; iter < max_iter; iter++ )
        {
            double E = 0;

            for( i = 0; i < l_count; i++ )
                dEdw[i].setTo(Scalar::all(0));

            // first, iterate through all the samples and compute dEdw
            RPropLoop invoker(this, inputs, outputs, _sw, dcount0, dEdw, &E);
            parallel_for_(Range(0, chunk_count), invoker);
            //invoker(Range(0, chunk_count));

            // now update weights
            for( i = 1; i < l_count; i++ )
            {
                int n1 = layer_sizes[i-1], n2 = layer_sizes[i];
                for( int k = 0; k <= n1; k++ )
                {
                    CV_Assert(weights[i].size() == Size(n2, n1+1));
                    double* wk = weights[i].ptr<double>(k);
                    double* dwk = dw[i].ptr<double>(k);
                    double* dEdwk = dEdw[i].ptr<double>(k);
                    schar* prevEk = prev_dEdw_sign[i].ptr<schar>(k);

                    for( int j = 0; j < n2; j++ )
                    {
                        double Eval = dEdwk[j];
                        double dval = dwk[j];
                        double wval = wk[j];
                        int s = CV_SIGN(Eval);
                        int ss = prevEk[j]*s;
                        if( ss > 0 )
                        {
                            dval *= dw_plus;
                            dval = std::min( dval, dw_max );
                            dwk[j] = dval;
                            wk[j] = wval + dval*s;
                        }
                        else if( ss < 0 )
                        {
                            dval *= dw_minus;
                            dval = std::max( dval, dw_min );
                            prevEk[j] = 0;
                            dwk[j] = dval;
                            wk[j] = wval + dval*s;
                        }
                        else
                        {
                            prevEk[j] = (schar)s;
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
        }

        return iter;
    }

    void write_params( FileStorage& fs ) const
    {
        const char* activ_func_name = activ_func == IDENTITY ? "IDENTITY" :
                                      activ_func == SIGMOID_SYM ? "SIGMOID_SYM" :
                                      activ_func == GAUSSIAN ? "GAUSSIAN" : 0;

        if( activ_func_name )
            fs << "activation_function" << activ_func_name;
        else
            fs << "activation_function_id" << activ_func;

        if( activ_func != IDENTITY )
        {
            fs << "f_param1" << f_param1;
            fs << "f_param2" << f_param2;
        }

        fs << "min_val" << min_val << "max_val" << max_val << "min_val1" << min_val1 << "max_val1" << max_val1;

        fs << "training_params" << "{";
        if( params.trainMethod == ANN_MLP::BACKPROP )
        {
            fs << "train_method" << "BACKPROP";
            fs << "dw_scale" << params.bpDWScale;
            fs << "moment_scale" << params.bpMomentScale;
        }
        else if( params.trainMethod == ANN_MLP::RPROP )
        {
            fs << "train_method" << "RPROP";
            fs << "dw0" << params.rpDW0;
            fs << "dw_plus" << params.rpDWPlus;
            fs << "dw_minus" << params.rpDWMinus;
            fs << "dw_min" << params.rpDWMin;
            fs << "dw_max" << params.rpDWMax;
        }
        else
            CV_Error(CV_StsError, "Unknown training method");

        fs << "term_criteria" << "{";
        if( params.termCrit.type & TermCriteria::EPS )
            fs << "epsilon" << params.termCrit.epsilon;
        if( params.termCrit.type & TermCriteria::COUNT )
            fs << "iterations" << params.termCrit.maxCount;
        fs << "}" << "}";
    }

    void write( FileStorage& fs ) const
    {
        if( layer_sizes.empty() )
            return;
        int i, l_count = layer_count();

        fs << "layer_sizes" << layer_sizes;

        write_params( fs );

        size_t esz = weights[0].elemSize();

        fs << "input_scale" << "[";
        fs.writeRaw("d", weights[0].ptr(), weights[0].total()*esz);

        fs << "]" << "output_scale" << "[";
        fs.writeRaw("d", weights[l_count].ptr(), weights[l_count].total()*esz);

        fs << "]" << "inv_output_scale" << "[";
        fs.writeRaw("d", weights[l_count+1].ptr(), weights[l_count+1].total()*esz);

        fs << "]" << "weights" << "[";
        for( i = 1; i < l_count; i++ )
        {
            fs << "[";
            fs.writeRaw("d", weights[i].ptr(), weights[i].total()*esz);
            fs << "]";
        }
        fs << "]";
    }

    void read_params( const FileNode& fn )
    {
        String activ_func_name = (String)fn["activation_function"];
        if( !activ_func_name.empty() )
        {
            activ_func = activ_func_name == "SIGMOID_SYM" ? SIGMOID_SYM :
                         activ_func_name == "IDENTITY" ? IDENTITY :
                         activ_func_name == "GAUSSIAN" ? GAUSSIAN : -1;
            CV_Assert( activ_func >= 0 );
        }
        else
            activ_func = (int)fn["activation_function_id"];

        f_param1 = (double)fn["f_param1"];
        f_param2 = (double)fn["f_param2"];

        setActivationFunction( activ_func, f_param1, f_param2 );

        min_val = (double)fn["min_val"];
        max_val = (double)fn["max_val"];
        min_val1 = (double)fn["min_val1"];
        max_val1 = (double)fn["max_val1"];

        FileNode tpn = fn["training_params"];
        params = AnnParams();

        if( !tpn.empty() )
        {
            String tmethod_name = (String)tpn["train_method"];

            if( tmethod_name == "BACKPROP" )
            {
                params.trainMethod = ANN_MLP::BACKPROP;
                params.bpDWScale = (double)tpn["dw_scale"];
                params.bpMomentScale = (double)tpn["moment_scale"];
            }
            else if( tmethod_name == "RPROP" )
            {
                params.trainMethod = ANN_MLP::RPROP;
                params.rpDW0 = (double)tpn["dw0"];
                params.rpDWPlus = (double)tpn["dw_plus"];
                params.rpDWMinus = (double)tpn["dw_minus"];
                params.rpDWMin = (double)tpn["dw_min"];
                params.rpDWMax = (double)tpn["dw_max"];
            }
            else
                CV_Error(CV_StsParseError, "Unknown training method (should be BACKPROP or RPROP)");

            FileNode tcn = tpn["term_criteria"];
            if( !tcn.empty() )
            {
                FileNode tcn_e = tcn["epsilon"];
                FileNode tcn_i = tcn["iterations"];
                params.termCrit.type = 0;
                if( !tcn_e.empty() )
                {
                    params.termCrit.type |= TermCriteria::EPS;
                    params.termCrit.epsilon = (double)tcn_e;
                }
                if( !tcn_i.empty() )
                {
                    params.termCrit.type |= TermCriteria::COUNT;
                    params.termCrit.maxCount = (int)tcn_i;
                }
            }
        }
    }

    void read( const FileNode& fn )
    {
        clear();

        vector<int> _layer_sizes;
        readVectorOrMat(fn["layer_sizes"], _layer_sizes);
        setLayerSizes( _layer_sizes );

        int i, l_count = layer_count();
        read_params(fn);

        size_t esz = weights[0].elemSize();

        FileNode w = fn["input_scale"];
        w.readRaw("d", weights[0].ptr(), weights[0].total()*esz);

        w = fn["output_scale"];
        w.readRaw("d", weights[l_count].ptr(), weights[l_count].total()*esz);

        w = fn["inv_output_scale"];
        w.readRaw("d", weights[l_count+1].ptr(), weights[l_count+1].total()*esz);

        FileNodeIterator w_it = fn["weights"].begin();

        for( i = 1; i < l_count; i++, ++w_it )
            (*w_it).readRaw("d", weights[i].ptr(), weights[i].total()*esz);
        trained = true;
    }

    Mat getWeights(int layerIdx) const
    {
        CV_Assert( 0 <= layerIdx && layerIdx < (int)weights.size() );
        return weights[layerIdx];
    }

    bool isTrained() const
    {
        return trained;
    }

    bool isClassifier() const
    {
        return false;
    }

    int getVarCount() const
    {
        return layer_sizes.empty() ? 0 : layer_sizes[0];
    }

    String getDefaultName() const
    {
        return "opencv_ml_ann_mlp";
    }

    vector<int> layer_sizes;
    vector<Mat> weights;
    double f_param1, f_param2;
    double min_val, max_val, min_val1, max_val1;
    int activ_func;
    int max_lsize, max_buf_sz;
    AnnParams params;
    RNG rng;
    Mutex mtx;
    bool trained;
};


Ptr<ANN_MLP> ANN_MLP::create()
{
    return makePtr<ANN_MLPImpl>();
}

}}

/* End of file. */
