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
    CV_TRACE_FUNCTION();
    minVal = std::min(_minVal, _maxVal);
    maxVal = std::max(_minVal, _maxVal);
    logStep = std::max(_logStep, 1.);
}

Ptr<ParamGrid> ParamGrid::create(double minval, double maxval, double logstep) {
  return makePtr<ParamGrid>(minval, maxval, logstep);
}

bool StatModel::empty() const { return !isTrained(); }

int StatModel::getVarCount() const { return 0; }

bool StatModel::train( const Ptr<TrainData>&, int )
{
    CV_TRACE_FUNCTION();
    CV_Error(CV_StsNotImplemented, "");
    return false;
}

bool StatModel::train( InputArray samples, int layout, InputArray responses )
{
    CV_TRACE_FUNCTION();
    return train(TrainData::create(samples, layout, responses));
}

class ParallelCalcError : public ParallelLoopBody
{
private:
    const Ptr<TrainData>& data;
    bool &testerr;
    Mat &resp;
    const StatModel &s;
    vector<double> &errStrip;
public:
    ParallelCalcError(const Ptr<TrainData>& d, bool &t, Mat &_r,const StatModel &w, vector<double> &e) :
        data(d),
        testerr(t),
        resp(_r),
        s(w),
        errStrip(e)
    {
    }
    virtual void operator()(const Range& range) const
    {
        int idxErr = range.start;
        CV_TRACE_FUNCTION_SKIP_NESTED();
        Mat samples = data->getSamples();
        Mat weights=testerr? data->getTestSampleWeights() : data->getTrainSampleWeights();
        int layout = data->getLayout();
        Mat sidx = testerr ? data->getTestSampleIdx() : data->getTrainSampleIdx();
        const int* sidx_ptr = sidx.ptr<int>();
        bool isclassifier = s.isClassifier();
        Mat responses = data->getResponses();
        int responses_type = responses.type();
        double err = 0;


        const float* sw = weights.empty() ? 0 : weights.ptr<float>();
        for (int i = range.start; i < range.end; i++)
        {
            int si = sidx_ptr ? sidx_ptr[i] : i;
            double sweight = sw ? static_cast<double>(sw[i]) : 1.;
            Mat sample = layout == ROW_SAMPLE ? samples.row(si) : samples.col(si);
            float val = s.predict(sample);
            float val0 = (responses_type == CV_32S) ? (float)responses.at<int>(si) : responses.at<float>(si);

            if (isclassifier)
                err += sweight * fabs(val - val0) > FLT_EPSILON;
            else
                err += sweight * (val - val0)*(val - val0);
            if (!resp.empty())
                resp.at<float>(i) = val;
        }


        errStrip[idxErr]=err ;

    };
    ParallelCalcError& operator=(const ParallelCalcError &) {
        return *this;
    };
};


float StatModel::calcError(const Ptr<TrainData>& data, bool testerr, OutputArray _resp) const
{
    CV_TRACE_FUNCTION_SKIP_NESTED();
    Mat samples = data->getSamples();
    Mat sidx = testerr ? data->getTestSampleIdx() : data->getTrainSampleIdx();
    Mat weights = testerr ? data->getTestSampleWeights() : data->getTrainSampleWeights();
    int n = (int)sidx.total();
    bool isclassifier = isClassifier();
    Mat responses = data->getResponses();

    if (n == 0)
    {
        n = data->getNSamples();
        weights = data->getTrainSampleWeights();
        testerr =false;
    }

    if (n == 0)
        return -FLT_MAX;

    Mat resp;
    if (_resp.needed())
        resp.create(n, 1, CV_32F);

    double err = 0;
    vector<double> errStrip(n,0.0);
    ParallelCalcError x(data, testerr, resp, *this,errStrip);

    parallel_for_(Range(0,n),x);

    for (size_t i = 0; i < errStrip.size(); i++)
        err += errStrip[i];
    float weightSum= weights.empty() ? n: static_cast<float>(sum(weights)(0));
    if (_resp.needed())
        resp.copyTo(_resp);

    return (float)(err/ weightSum * (isclassifier ? 100 : 1));
}

/* Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S */
static void Cholesky( const Mat& A, Mat& S )
{
    CV_TRACE_FUNCTION();
    CV_Assert(A.type() == CV_32F);

    S = A.clone();
    cv::Cholesky ((float*)S.ptr(),S.step, S.rows,NULL, 0, 0);
    S = S.t();
    for (int i=1;i<S.rows;i++)
        for (int j=0;j<i;j++)
            S.at<float>(i,j)=0;
}

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
void randMVNormal( InputArray _mean, InputArray _cov, int nsamples, OutputArray _samples )
{
    CV_TRACE_FUNCTION();
    // check mean vector and covariance matrix
    Mat mean = _mean.getMat(), cov = _cov.getMat();
    int dim = (int)mean.total();  // dimensionality
    CV_Assert(mean.rows == 1 || mean.cols == 1);
    CV_Assert(cov.rows == dim && cov.cols == dim);
    mean = mean.reshape(1,1);     // ensure a row vector

    // generate n-samples of the same dimension, from ~N(0,1)
    _samples.create(nsamples, dim, CV_32F);
    Mat samples = _samples.getMat();
    randn(samples, Scalar::all(0), Scalar::all(1));

    // decompose covariance using Cholesky: cov = U'*U
    // (cov must be square, symmetric, and positive semi-definite matrix)
    Mat utmat;
    Cholesky(cov, utmat);

    // transform random numbers using specified mean and covariance
    for( int i = 0; i < nsamples; i++ )
    {
        Mat sample = samples.row(i);
        sample = sample * utmat + mean;
    }
}

}}

/* End of file */
