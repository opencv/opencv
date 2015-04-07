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

ParamGrid::ParamGrid() { minVal = maxVal = 0.; logStep = 1; }
ParamGrid::ParamGrid(double _minVal, double _maxVal, double _logStep)
{
    minVal = std::min(_minVal, _maxVal);
    maxVal = std::max(_minVal, _maxVal);
    logStep = std::max(_logStep, 1.);
}

bool StatModel::empty() const { return !isTrained(); }

int StatModel::getVarCount() const { return 0; }

bool StatModel::train( const Ptr<TrainData>&, int )
{
    CV_Error(CV_StsNotImplemented, "");
    return false;
}

bool StatModel::train( InputArray samples, int layout, InputArray responses )
{
    return train(TrainData::create(samples, layout, responses));
}

float StatModel::calcError( const Ptr<TrainData>& data, bool testerr, OutputArray _resp ) const
{
    Mat samples = data->getSamples();
    int layout = data->getLayout();
    Mat sidx = testerr ? data->getTestSampleIdx() : data->getTrainSampleIdx();
    const int* sidx_ptr = sidx.ptr<int>();
    int i, n = (int)sidx.total();
    bool isclassifier = isClassifier();
    Mat responses = data->getResponses();

    if( n == 0 )
        n = data->getNSamples();

    if( n == 0 )
        return -FLT_MAX;

    Mat resp;
    if( _resp.needed() )
        resp.create(n, 1, CV_32F);

    double err = 0;
    for( i = 0; i < n; i++ )
    {
        int si = sidx_ptr ? sidx_ptr[i] : i;
        Mat sample = layout == ROW_SAMPLE ? samples.row(si) : samples.col(si);
        float val = predict(sample);
        float val0 = responses.at<float>(si);

        if( isclassifier )
            err += fabs(val - val0) > FLT_EPSILON;
        else
            err += (val - val0)*(val - val0);
        if( !resp.empty() )
            resp.at<float>(i) = val;
        /*if( i < 100 )
        {
            printf("%d. ref %.1f vs pred %.1f\n", i, val0, val);
        }*/
    }

    if( _resp.needed() )
        resp.copyTo(_resp);

    return (float)(err / n * (isclassifier ? 100 : 1));
}

/* Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S */
static void Cholesky( const Mat& A, Mat& S )
{
    CV_Assert(A.type() == CV_32F);

    int dim = A.rows;
    S.create(dim, dim, CV_32F);

    int i, j, k;

    for( i = 0; i < dim; i++ )
    {
        for( j = 0; j < i; j++ )
            S.at<float>(i,j) = 0.f;

        float sum = 0.f;
        for( k = 0; k < i; k++ )
        {
            float val = S.at<float>(k,i);
            sum += val*val;
        }

        S.at<float>(i,i) = std::sqrt(std::max(A.at<float>(i,i) - sum, 0.f));
        float ival = 1.f/S.at<float>(i, i);

        for( j = i + 1; j < dim; j++ )
        {
            sum = 0;
            for( k = 0; k < i; k++ )
                sum += S.at<float>(k, i) * S.at<float>(k, j);

            S.at<float>(i, j) = (A.at<float>(i, j) - sum)*ival;
        }
    }
}

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
void randMVNormal( InputArray _mean, InputArray _cov, int nsamples, OutputArray _samples )
{
    Mat mean = _mean.getMat(), cov = _cov.getMat();
    int dim = (int)mean.total();

    _samples.create(nsamples, dim, CV_32F);
    Mat samples = _samples.getMat();
    randu(samples, 0., 1.);

    Mat utmat;
    Cholesky(cov, utmat);
    int flags = mean.cols == 1 ? 0 : GEMM_3_T;

    for( int i = 0; i < nsamples; i++ )
    {
        Mat sample = samples.row(i);
        gemm(sample, utmat, 1, mean, 1, sample, flags);
    }
}

}}

/* End of file */
