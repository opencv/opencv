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

namespace cv
{

const double minEigenValue = DBL_MIN;

///////////////////////////////////////////////////////////////////////////////////////////////////////

EM::EM(int _nclusters, int _covMatType, const TermCriteria& _criteria)
{
    nclusters = _nclusters;
    covMatType = _covMatType;
    maxIters = (_criteria.type & TermCriteria::MAX_ITER) ? _criteria.maxCount : DEFAULT_MAX_ITERS;
    epsilon = (_criteria.type & TermCriteria::EPS) ? _criteria.epsilon : 0;
}

EM::~EM()
{
    clear();
}

void EM::clear()
{
    trainSamples.release();
    trainProbs.release();
    trainLogLikelihoods.release();
    trainLabels.release();
    trainCounts.release();

    weights.release();
    means.release();
    covs.clear();

    covsEigenValues.clear();
    invCovsEigenValues.clear();
    covsRotateMats.clear();

    logWeightDivDet.release();
}

    
bool EM::train(InputArray samples,
               OutputArray labels,
               OutputArray probs,
               OutputArray logLikelihoods)
{
    Mat samplesMat = samples.getMat();
    setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0);
    return doTrain(START_AUTO_STEP, labels, probs, logLikelihoods);
}

bool EM::trainE(InputArray samples,
                InputArray _means0,
                InputArray _covs0,
                InputArray _weights0,
                OutputArray labels,
                OutputArray probs,
                OutputArray logLikelihoods)
{
    Mat samplesMat = samples.getMat();
    vector<Mat> covs0;
    _covs0.getMatVector(covs0);
    
    Mat means0 = _means0.getMat(), weights0 = _weights0.getMat();

    setTrainData(START_E_STEP, samplesMat, 0, !_means0.empty() ? &means0 : 0,
                 !_covs0.empty() ? &covs0 : 0, _weights0.empty() ? &weights0 : 0);
    return doTrain(START_E_STEP, labels, probs, logLikelihoods);
}

bool EM::trainM(InputArray samples,
                InputArray _probs0,
                OutputArray labels,
                OutputArray probs,
                OutputArray logLikelihoods)
{
    Mat samplesMat = samples.getMat();
    Mat probs0 = _probs0.getMat();
    
    setTrainData(START_M_STEP, samplesMat, !_probs0.empty() ? &probs0 : 0, 0, 0, 0);
    return doTrain(START_M_STEP, labels, probs, logLikelihoods);
}

    
int EM::predict(InputArray _sample, OutputArray _probs, double* logLikelihood) const
{
    Mat sample = _sample.getMat();
    CV_Assert(isTrained());

    CV_Assert(!sample.empty());
    if(sample.type() != CV_64FC1)
    {
        Mat tmp;
        sample.convertTo(tmp, CV_64FC1);
        sample = tmp;
    }

    int label;
    Mat probs;
    if( _probs.needed() )
    {
        _probs.create(1, nclusters, CV_64FC1);
        probs = _probs.getMat();
    }
    computeProbabilities(sample, label, !probs.empty() ? &probs : 0, logLikelihood);

    return label;
}

bool EM::isTrained() const
{
    return !means.empty();
}


static
void checkTrainData(int startStep, const Mat& samples,
                    int nclusters, int covMatType, const Mat* probs, const Mat* means,
                    const vector<Mat>* covs, const Mat* weights)
{
    // Check samples.
    CV_Assert(!samples.empty());
    CV_Assert(samples.channels() == 1);

    int nsamples = samples.rows;
    int dim = samples.cols;

    // Check training params.
    CV_Assert(nclusters > 0);
    CV_Assert(nclusters <= nsamples);
    CV_Assert(startStep == EM::START_AUTO_STEP ||
              startStep == EM::START_E_STEP ||
              startStep == EM::START_M_STEP);
    CV_Assert(covMatType == EM::COV_MAT_GENERIC ||
              covMatType == EM::COV_MAT_DIAGONAL ||
              covMatType == EM::COV_MAT_SPHERICAL);

    CV_Assert(!probs ||
        (!probs->empty() &&
         probs->rows == nsamples && probs->cols == nclusters &&
         (probs->type() == CV_32FC1 || probs->type() == CV_64FC1)));

    CV_Assert(!weights ||
        (!weights->empty() &&
         (weights->cols == 1 || weights->rows == 1) && static_cast<int>(weights->total()) == nclusters &&
         (weights->type() == CV_32FC1 || weights->type() == CV_64FC1)));

    CV_Assert(!means ||
        (!means->empty() &&
         means->rows == nclusters && means->cols == dim &&
         means->channels() == 1));

    CV_Assert(!covs ||
        (!covs->empty() &&
         static_cast<int>(covs->size()) == nclusters));
    if(covs)
    {
        const Size covSize(dim, dim);
        for(size_t i = 0; i < covs->size(); i++)
        {
            const Mat& m = (*covs)[i];
            CV_Assert(!m.empty() && m.size() == covSize && (m.channels() == 1));
        }
    }

    if(startStep == EM::START_E_STEP)
    {
        CV_Assert(means);
    }
    else if(startStep == EM::START_M_STEP)
    {
        CV_Assert(probs);
    }
}

static
void preprocessSampleData(const Mat& src, Mat& dst, int dstType, bool isAlwaysClone)
{
    if(src.type() == dstType && !isAlwaysClone)
        dst = src;
    else
        src.convertTo(dst, dstType);
}

static
void preprocessProbability(Mat& probs)
{
    max(probs, 0., probs);

    const double uniformProbability = (double)(1./probs.cols);
    for(int y = 0; y < probs.rows; y++)
    {
        Mat sampleProbs = probs.row(y);

        double maxVal = 0;
        minMaxLoc(sampleProbs, 0, &maxVal);
        if(maxVal < FLT_EPSILON)
            sampleProbs.setTo(uniformProbability);
        else
            normalize(sampleProbs, sampleProbs, 1, 0, NORM_L1);
    }
}

void EM::setTrainData(int startStep, const Mat& samples,
                      const Mat* probs0,
                      const Mat* means0,
                      const vector<Mat>* covs0,
                      const Mat* weights0)
{
    clear();

    checkTrainData(startStep, samples, nclusters, covMatType, probs0, means0, covs0, weights0);

    bool isKMeansInit = (startStep == EM::START_AUTO_STEP) || (startStep == EM::START_E_STEP && (covs0 == 0 || weights0 == 0));
    // Set checked data
    preprocessSampleData(samples, trainSamples, isKMeansInit ? CV_32FC1 : CV_64FC1, false);

    // set probs
    if(probs0 && startStep == EM::START_M_STEP)
    {
        preprocessSampleData(*probs0, trainProbs, CV_64FC1, true);
        preprocessProbability(trainProbs);
    }

    // set weights
    if(weights0 && (startStep == EM::START_E_STEP && covs0))
    {
        weights0->convertTo(weights, CV_64FC1);
        weights.reshape(1,1);
        preprocessProbability(weights);
    }

    // set means
    if(means0 && (startStep == EM::START_E_STEP/* || startStep == EM::START_AUTO_STEP*/))
        means0->convertTo(means, isKMeansInit ? CV_32FC1 : CV_64FC1);

    // set covs
    if(covs0 && (startStep == EM::START_E_STEP && weights0))
    {
        covs.resize(nclusters);
        for(size_t i = 0; i < covs0->size(); i++)
            (*covs0)[i].convertTo(covs[i], CV_64FC1);
    }
}

void EM::decomposeCovs()
{
    CV_Assert(!covs.empty());
    covsEigenValues.resize(nclusters);
    if(covMatType == EM::COV_MAT_GENERIC)
        covsRotateMats.resize(nclusters);
    invCovsEigenValues.resize(nclusters);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        CV_Assert(!covs[clusterIndex].empty());

        SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);

        if(covMatType == EM::COV_MAT_SPHERICAL)
        {
            double maxSingularVal = svd.w.at<double>(0);
            covsEigenValues[clusterIndex] = Mat(1, 1, CV_64FC1, Scalar(maxSingularVal));
        }
        else if(covMatType == EM::COV_MAT_DIAGONAL)
        {
            covsEigenValues[clusterIndex] = svd.w;
        }
        else //EM::COV_MAT_GENERIC
        {
            covsEigenValues[clusterIndex] = svd.w;
            covsRotateMats[clusterIndex] = svd.u;
        }
        max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
        invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
    }
}

void EM::clusterTrainSamples()
{
    int nsamples = trainSamples.rows;

    // Cluster samples, compute/update means
    Mat trainSamplesFlt, meansFlt;
    if(trainSamples.type() != CV_32FC1)
        trainSamples.convertTo(trainSamplesFlt, CV_32FC1);
    else
        trainSamplesFlt = trainSamples;
    if(!means.empty())
    {
        if(means.type() != CV_32FC1)
            means.convertTo(meansFlt, CV_32FC1);
        else
            meansFlt = means;
    }

    Mat labels;
    kmeans(trainSamplesFlt, nclusters, labels,  TermCriteria(TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5), 10, KMEANS_PP_CENTERS, meansFlt);

    CV_Assert(meansFlt.type() == CV_32FC1);
    if(trainSamples.type() != CV_64FC1)
    {
        Mat trainSamplesBuffer;
        trainSamplesFlt.convertTo(trainSamplesBuffer, CV_64FC1);
        trainSamples = trainSamplesBuffer;
    }
    meansFlt.convertTo(means, CV_64FC1);

    // Compute weights and covs
    weights = Mat(1, nclusters, CV_64FC1, Scalar(0));
    covs.resize(nclusters);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        Mat clusterSamples;
        for(int sampleIndex = 0; sampleIndex < nsamples; sampleIndex++)
        {
            if(labels.at<int>(sampleIndex) == clusterIndex)
            {
                const Mat sample = trainSamples.row(sampleIndex);
                clusterSamples.push_back(sample);
            }
        }
        CV_Assert(!clusterSamples.empty());

        calcCovarMatrix(clusterSamples, covs[clusterIndex], means.row(clusterIndex),
            CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_64FC1);
        weights.at<double>(clusterIndex) = static_cast<double>(clusterSamples.rows)/static_cast<double>(nsamples);
    }

    decomposeCovs();
}

void EM::computeLogWeightDivDet()
{
    CV_Assert(!covsEigenValues.empty());

    Mat logWeights;
    cv::max(weights, DBL_MIN, weights);
    log(weights, logWeights);

    logWeightDivDet.create(1, nclusters, CV_64FC1);
    // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        double logDetCov = 0.;
        for(int di = 0; di < covsEigenValues[clusterIndex].cols; di++)
            logDetCov += std::log(covsEigenValues[clusterIndex].at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0));

        logWeightDivDet.at<double>(clusterIndex) = logWeights.at<double>(clusterIndex) - 0.5 * logDetCov;
    }
}

bool EM::doTrain(int startStep, OutputArray labels, OutputArray probs, OutputArray logLikelihoods)
{
    int dim = trainSamples.cols;
    // Precompute the empty initial train data in the cases of EM::START_E_STEP and START_AUTO_STEP
    if(startStep != EM::START_M_STEP)
    {
        if(covs.empty())
        {
            CV_Assert(weights.empty());
            clusterTrainSamples();
        }
    }

    if(!covs.empty() && covsEigenValues.empty() )
    {
        CV_Assert(invCovsEigenValues.empty());
        decomposeCovs();
    }

    if(startStep == EM::START_M_STEP)
        mStep();

    double trainLogLikelihood, prevTrainLogLikelihood = 0.;
    for(int iter = 0; ; iter++)
    {
        eStep();
        trainLogLikelihood = sum(trainLogLikelihoods)[0];

        if(iter >= maxIters - 1)
            break;

        double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;
        if( iter != 0 &&
            (trainLogLikelihoodDelta < -DBL_EPSILON ||
             trainLogLikelihoodDelta < epsilon * std::fabs(trainLogLikelihood)))
            break;

        mStep();

        prevTrainLogLikelihood = trainLogLikelihood;
    }

    if( trainLogLikelihood <= -DBL_MAX/10000. )
    {
        clear();
        return false;
    }

    // postprocess covs
    covs.resize(nclusters);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        if(covMatType == EM::COV_MAT_SPHERICAL)
        {
            covs[clusterIndex].create(dim, dim, CV_64FC1);
            setIdentity(covs[clusterIndex], Scalar(covsEigenValues[clusterIndex].at<double>(0)));
        }
        else if(covMatType == EM::COV_MAT_DIAGONAL)
            covs[clusterIndex] = Mat::diag(covsEigenValues[clusterIndex].t());
    }
    
    if(labels.needed())
        trainLabels.copyTo(labels);
    if(probs.needed())
        trainProbs.copyTo(probs);
    if(logLikelihoods.needed())
        trainLogLikelihoods.copyTo(logLikelihoods);
    
    trainSamples.release();
    trainProbs.release();
    trainLabels.release();
    trainLogLikelihoods.release();
    trainCounts.release();

    return true;
}

void EM::computeProbabilities(const Mat& sample, int& label, Mat* probs, double* logLikelihood) const
{
    // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
    // q = arg(max_k(L_ik))
    // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))

    CV_Assert(!means.empty());
    CV_Assert(sample.type() == CV_64FC1);
    CV_Assert(sample.rows == 1);
    CV_Assert(sample.cols == means.cols);

    int dim = sample.cols;

    Mat L(1, nclusters, CV_64FC1);
    label = 0;
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        const Mat centeredSample = sample - means.row(clusterIndex);

        Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
                centeredSample : centeredSample * covsRotateMats[clusterIndex];

        double Lval = 0;
        for(int di = 0; di < dim; di++)
        {
            double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
            double val = rotatedCenteredSample.at<double>(di);
            Lval += w * val * val;
        }
        CV_DbgAssert(!logWeightDivDet.empty());
        Lval = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;
        L.at<double>(clusterIndex) = Lval;

        if(Lval > L.at<double>(label))
            label = clusterIndex;
    }

    if(!probs && !logLikelihood)
        return;

    Mat buf, *sampleProbs = probs ? probs : &buf;
    Mat expL_Lmax(L.size(), CV_64FC1);
    double maxLVal = L.at<double>(label);
    for(int i = 0; i < L.cols; i++)
        expL_Lmax.at<double>(i) = std::exp(L.at<double>(i) - maxLVal);

    double partSum = 0, // sum_j!=q (exp(L_ij - L_iq))
           factor;      // 1/(1 + partExpSum)
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        if(clusterIndex != label)
            partSum += expL_Lmax.at<double>(clusterIndex);
    factor = 1./(1 + partSum);

    sampleProbs->create(1, nclusters, CV_64FC1);
    expL_Lmax *= factor;
    expL_Lmax.copyTo(*sampleProbs);

    if(logLikelihood)
    {
        double logWeightProbs = std::log(std::max(DBL_MIN, sum(*sampleProbs)[0]));
        *logLikelihood = logWeightProbs;
    }
}

void EM::eStep()
{
    // Compute probs_ik from means_k, covs_k and weights_k.
    trainProbs.create(trainSamples.rows, nclusters, CV_64FC1);
    trainLabels.create(trainSamples.rows, 1, CV_32SC1);
    trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);

    computeLogWeightDivDet();

    CV_DbgAssert(trainSamples.type() == CV_64FC1);
    CV_DbgAssert(means.type() == CV_64FC1);

    for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
    {
        Mat sampleProbs = trainProbs.row(sampleIndex);
        computeProbabilities(trainSamples.row(sampleIndex), trainLabels.at<int>(sampleIndex),
                             &sampleProbs, &trainLogLikelihoods.at<double>(sampleIndex));
    }
}

void EM::mStep()
{
    trainCounts.create(1, nclusters, CV_32SC1);
    trainCounts = Scalar(0);

    for(int sampleIndex = 0; sampleIndex < trainLabels.rows; sampleIndex++)
        trainCounts.at<int>(trainLabels.at<int>(sampleIndex))++;

    if(countNonZero(trainCounts) != (int)trainCounts.total())
    {
        clusterTrainSamples();
    }
    else
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols;

        // Update weights
        // not normalized first
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        // Update means
        means.create(nclusters, dim, CV_64FC1);
        means = Scalar(0);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            Mat clusterMean = means.row(clusterIndex);
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                clusterMean += trainProbs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            clusterMean /= weights.at<double>(clusterIndex);
        }

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters);
        covsEigenValues.resize(nclusters);
        if(covMatType == EM::COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if(covMatType != EM::COV_MAT_SPHERICAL)
                covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);
            else
                covsEigenValues[clusterIndex].create(1, 1, CV_64FC1);

            if(covMatType == EM::COV_MAT_GENERIC)
                covs[clusterIndex].create(dim, dim, CV_64FC1);

            Mat clusterCov = covMatType != EM::COV_MAT_GENERIC ?
                covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = Scalar(0);

            Mat centeredSample;
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

                if(covMatType == EM::COV_MAT_GENERIC)
                    clusterCov += trainProbs.at<double>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
                else
                {
                    double p = trainProbs.at<double>(sampleIndex, clusterIndex);
                    for(int di = 0; di < dim; di++ )
                    {
                        double val = centeredSample.at<double>(di);
                        clusterCov.at<double>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0) += p*val*val;
                    }
                }
            }

            if(covMatType == EM::COV_MAT_SPHERICAL)
                clusterCov /= dim;

            clusterCov /= weights.at<double>(clusterIndex);

            // Update covsRotateMats for EM::COV_MAT_GENERIC only
            if(covMatType == EM::COV_MAT_GENERIC)
            {
                SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);

            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
        }

        // Normalize weights
        weights /= trainSamples.rows;
    }
}

void EM::read(const FileNode& fn)
{
    Algorithm::read(fn);

    decomposeCovs();
    computeLogWeightDivDet();
}

static Algorithm* createEM()
{
    return new EM;
}
static AlgorithmInfo em_info("StatModel.EM", createEM);

AlgorithmInfo* EM::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        EM obj;
        em_info.addParam(obj, "nclusters", obj.nclusters);
        em_info.addParam(obj, "covMatType", obj.covMatType);

        em_info.addParam(obj, "weights", obj.weights);
        em_info.addParam(obj, "means", obj.means);
        em_info.addParam(obj, "covs", obj.covs);

        initialized = true;
    }
    return &em_info;
}
} // namespace cv

/* End of file. */
