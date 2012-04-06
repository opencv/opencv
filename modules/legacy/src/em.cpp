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
// Copyright( C) 2000, Intel Corporation, all rights reserved.
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
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even ifadvised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

CvEMParams::CvEMParams() : nclusters(10), cov_mat_type(CvEM::COV_MAT_DIAGONAL),
    start_step(CvEM::START_AUTO_STEP), probs(0), weights(0), means(0), covs(0)
{
    term_crit=cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON );
}

CvEMParams::CvEMParams( int _nclusters, int _cov_mat_type, int _start_step,
                        CvTermCriteria _term_crit, const CvMat* _probs,
                        const CvMat* _weights, const CvMat* _means, const CvMat** _covs ) :
                        nclusters(_nclusters), cov_mat_type(_cov_mat_type), start_step(_start_step),
                        probs(_probs), weights(_weights), means(_means), covs(_covs), term_crit(_term_crit)
{}

CvEM::CvEM() : likelihood(DBL_MAX)
{
}

CvEM::CvEM( const CvMat* samples, const CvMat* sample_idx,
            CvEMParams params, CvMat* labels ) : likelihood(DBL_MAX)
{
    train(samples, sample_idx, params, labels);
}

CvEM::~CvEM()
{
    clear();
}

void CvEM::clear()
{
    emObj.clear();
}

void CvEM::read( CvFileStorage* fs, CvFileNode* node )
{
    cv::FileNode fn(fs, node);
    emObj.read(fn);
    set_mat_hdrs();
}

void CvEM::write( CvFileStorage* _fs, const char* name ) const
{
    cv::FileStorage fs = _fs;
    if(name)
        fs << name << "{";
    emObj.write(fs);
    if(name)
        fs << "}";
}

double CvEM::calcLikelihood( const cv::Mat &input_sample ) const
{
    double likelihood;
    emObj.predict(input_sample, 0, &likelihood);
    return likelihood;
}

float
CvEM::predict( const CvMat* _sample, CvMat* _probs, bool isNormalize ) const
{
    cv::Mat prbs;
    int cls = emObj.predict(_sample, _probs ? &prbs : 0);
    if(_probs)
    {
        if(isNormalize)
            cv::normalize(prbs, prbs, 1, 0, cv::NORM_L1);
        *_probs = prbs;
    }
    return (float)cls;
}

void CvEM::set_mat_hdrs()
{
    if(emObj.isTrained())
    {
        meansHdr = emObj.getMeans();
        covsHdrs.resize(emObj.getNClusters());
        covsPtrs.resize(emObj.getNClusters());
        const std::vector<cv::Mat>& covs = emObj.getCovs();
        for(size_t i = 0; i < covsHdrs.size(); i++)
        {
            covsHdrs[i] = covs[i];
            covsPtrs[i] = &covsHdrs[i];
        }
        weightsHdr = emObj.getWeights();
        probsHdr = probs;
    }
}

static
void init_params(const CvEMParams& src, cv::EM::Params& dst,
                       cv::Mat& prbs, cv::Mat& weights,
                       cv::Mat& means, cv::vector<cv::Mat>& covsHdrs)
{
    dst.nclusters = src.nclusters;
    dst.covMatType = src.cov_mat_type;
    dst.startStep = src.start_step;
    dst.termCrit = src.term_crit;

    prbs = src.probs;
    dst.probs = &prbs;

    weights = src.weights;
    dst.weights = &weights;

    means = src.means;
    dst.means = &means;

    if(src.covs)
    {
        covsHdrs.resize(src.nclusters);
        for(size_t i = 0; i < covsHdrs.size(); i++)
            covsHdrs[i] = src.covs[i];
        dst.covs = &covsHdrs;
    }
}

bool CvEM::train( const CvMat* _samples, const CvMat* _sample_idx,
                  CvEMParams _params, CvMat* _labels )
{
    cv::EM::Params params;
    cv::Mat prbs, weights, means;
    std::vector<cv::Mat> covsHdrs;
    init_params(_params, params, prbs, weights, means, covsHdrs);

    cv::Mat lbls;
    cv::Mat likelihoods;
    bool isOk = emObj.train(_samples, _sample_idx, params, _labels ? &lbls : 0, &probs, &likelihoods );
    if(isOk)
    {
        if(_labels)
            *_labels = lbls;
        likelihood = cv::sum(likelihoods)[0];
        set_mat_hdrs();
    }

    return isOk;
}

int CvEM::get_nclusters() const
{
    return emObj.getNClusters();
}

const CvMat* CvEM::get_means() const
{
    return emObj.isTrained() ? &meansHdr : 0;
}

const CvMat** CvEM::get_covs() const
{
    return emObj.isTrained() ? (const CvMat**)&covsPtrs[0] : 0;
}

const CvMat* CvEM::get_weights() const
{
    return emObj.isTrained() ? &weightsHdr : 0;
}

const CvMat* CvEM::get_probs() const
{
    return emObj.isTrained() ? &probsHdr : 0;
}

using namespace cv;

CvEM::CvEM( const Mat& samples, const Mat& sample_idx, CvEMParams params )
{
    train(samples, sample_idx, params, 0);
}

bool CvEM::train( const Mat& _samples, const Mat& _sample_idx,
                 CvEMParams _params, Mat* _labels )
{
    cv::EM::Params params;
    cv::Mat prbs, weights, means;
    std::vector<cv::Mat> covsHdrs;
    init_params(_params, params, prbs, weights, means, covsHdrs);

    cv::Mat likelihoods;
    bool isOk = emObj.train(_samples, _sample_idx, params, _labels, &probs, &likelihoods);
    if(isOk)
    {
        likelihoods = cv::sum(likelihoods).val[0];
        set_mat_hdrs();
    }

    return isOk;
}

float
CvEM::predict( const Mat& _sample, Mat* _probs, bool isNormalize ) const
{
    int cls = emObj.predict(_sample, _probs);
    if(_probs && isNormalize)
        cv::normalize(*_probs, *_probs, 1, 0, cv::NORM_L1);

    return (float)cls;
}

int CvEM::getNClusters() const
{
    return emObj.getNClusters();
}

const Mat& CvEM::getMeans() const
{
    return emObj.getMeans();
}

void CvEM::getCovs(vector<Mat>& _covs) const
{
    _covs = emObj.getCovs();
}

const Mat& CvEM::getWeights() const
{
    return emObj.getWeights();
}

const Mat& CvEM::getProbs() const
{
    return probs;
}


/* End of file. */
