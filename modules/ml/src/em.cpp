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

const float minEigenValue = 1.e-3;

EM::Params::Params( int nclusters, int covMatType, int startStep, const cv::TermCriteria& termCrit,
                   const cv::Mat* probs, const cv::Mat* weights,
                   const cv::Mat* means, const std::vector<cv::Mat>* covs )
        : nclusters(nclusters), covMatType(covMatType), startStep(startStep),
        probs(probs), weights(weights), means(means), covs(covs), termCrit(termCrit)
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////

EM::EM()
{}

EM::EM(const cv::Mat& samples, const cv::Mat samplesMask,
       const EM::Params& params, cv::Mat* labels, cv::Mat* probs, cv::Mat* likelihoods)
{
    train(samples, samplesMask, params, labels, probs, likelihoods);
}

EM::~EM()
{
    clear();
}

void EM::clear()
{
    trainSamples.release();
    trainProbs.release();
    trainLikelihoods.release();
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

bool EM::train(const cv::Mat& samples, const cv::Mat& samplesMask,
               const EM::Params& params, cv::Mat* labels, cv::Mat* probs, cv::Mat* likelihoods)
{
    setTrainData(samples, samplesMask, params);

    bool isOk = doTrain(params.termCrit);

    if(isOk)
    {
        if(labels)
            cv::swap(*labels, trainLabels);
        if(probs)
            cv::swap(*probs, trainProbs);
        if(likelihoods)
            cv::swap(*likelihoods, trainLikelihoods);

        trainSamples.release();
        trainProbs.release();
        trainLabels.release();
        trainLikelihoods.release();
        trainCounts.release();
    }
    else
        clear();

    return isOk;
}

int EM::predict(const cv::Mat& sample, cv::Mat* _probs, double* _likelihood) const
{
    CV_Assert(isTrained());

    CV_Assert(!sample.empty());
    CV_Assert(sample.type() == CV_32FC1);

    int label;
    float likelihood;
    computeProbabilities(sample, label, _probs, _likelihood ? &likelihood : 0);
    if(_likelihood)
        *_likelihood = static_cast<double>(likelihood);

    return label;
}

bool EM::isTrained() const
{
    return !means.empty();
}

int EM::getNClusters() const
{
    return isTrained() ? nclusters : -1;
}

int EM::getCovMatType() const
{
    return isTrained() ? covMatType : -1;
}

const cv::Mat& EM::getWeights() const
{
    CV_Assert((isTrained() && !weights.empty()) || (!isTrained() && weights.empty()));
    return weights;
}

const cv::Mat& EM::getMeans() const
{
    CV_Assert((isTrained() && !means.empty()) || (!isTrained() && means.empty()));
    return means;
}

const std::vector<cv::Mat>& EM::getCovs() const
{
    CV_Assert((isTrained() && !covs.empty()) || (!isTrained() && covs.empty()));
    return covs;
}

static
void checkTrainData(const cv::Mat& samples, const cv::Mat& samplesMask, const EM::Params& params)
{
    // Check samples.
    CV_Assert(!samples.empty());
    CV_Assert(samples.type() == CV_32FC1);

    int nsamples = samples.rows;
    int dim = samples.cols;

    // Check samples indices.
    CV_Assert(samplesMask.empty() ||
        ((samplesMask.rows == 1 || samplesMask.cols == 1) &&
          static_cast<int>(samplesMask.total()) == nsamples && samplesMask.type() == CV_8UC1));

    // Check training params.
    CV_Assert(params.nclusters > 0);
    CV_Assert(params.nclusters <= nsamples);
    CV_Assert(params.startStep == EM::START_AUTO_STEP || params.startStep == EM::START_E_STEP || params.startStep == EM::START_M_STEP);

    CV_Assert(!params.probs ||
        (!params.probs->empty() &&
         params.probs->rows == nsamples && params.probs->cols == params.nclusters &&
         params.probs->type() == CV_32FC1));

    CV_Assert(!params.weights ||
        (!params.weights->empty() &&
         (params.weights->cols == 1 || params.weights->rows == 1) && static_cast<int>(params.weights->total()) == params.nclusters &&
         params.weights->type() == CV_32FC1));

    CV_Assert(!params.means ||
        (!params.means->empty() &&
         params.means->rows == params.nclusters && params.means->cols == dim &&
         params.means->type() == CV_32FC1));

    CV_Assert(!params.covs ||
        (!params.covs->empty() &&
         static_cast<int>(params.covs->size()) == params.nclusters));
    if(params.covs)
    {
        const cv::Size covSize(dim, dim);
        for(size_t i = 0; i < params.covs->size(); i++)
        {
            const cv::Mat& m = (*params.covs)[i];
            CV_Assert(!m.empty() && m.size() == covSize && (m.type() == CV_32FC1));
        }
    }

    if(params.startStep == EM::START_E_STEP)
    {
        CV_Assert(params.means);
    }
    else if(params.startStep == EM::START_M_STEP)
    {
        CV_Assert(params.probs);
    }
}

static
void preprocessSampleData(const cv::Mat& src, cv::Mat& dst, int dstType, const cv::Mat& samplesMask, bool isAlwaysClone)
{
    if(samplesMask.empty() || cv::countNonZero(samplesMask) == src.rows)
    {
        if(src.type() == dstType && !isAlwaysClone)
            dst = src;
        else
            src.convertTo(dst, dstType);
    }
    else
    {
        dst.release();
        for(int sampleIndex = 0; sampleIndex < src.rows; sampleIndex++)
        {
            if(samplesMask.at<uchar>(sampleIndex))
            {
                cv::Mat sample = src.row(sampleIndex);
                cv::Mat sample_dbl;
                sample.convertTo(sample_dbl, dstType);
                dst.push_back(sample_dbl);
            }
        }
    }
}

static
void preprocessProbability(cv::Mat& probs)
{
    cv::max(probs, 0., probs);

    const float uniformProbability = 1./probs.cols;
    for(int y = 0; y < probs.rows; y++)
    {
        cv::Mat sampleProbs = probs.row(y);

        double maxVal = 0;
        cv::minMaxLoc(sampleProbs, 0, &maxVal);
        if(maxVal < FLT_EPSILON)
            sampleProbs.setTo(uniformProbability);
        else
            cv::normalize(sampleProbs, sampleProbs, 1, 0, cv::NORM_L1);
    }
}

void EM::setTrainData(const cv::Mat& samples, const cv::Mat& samplesMask, const EM::Params& params)
{
    clear();

    checkTrainData(samples, samplesMask, params);

    // Set checked data

    nclusters = params.nclusters;
    covMatType = params.covMatType;
    startStep = params.startStep;

    preprocessSampleData(samples, trainSamples, CV_32FC1, samplesMask, false);

    // set probs
    if(params.probs && startStep == EM::START_M_STEP)
    {
        preprocessSampleData(*params.probs, trainProbs, CV_32FC1, samplesMask, true);
        preprocessProbability(trainProbs);
    }

    // set weights
    if(params.weights && (startStep == EM::START_E_STEP && params.covs))
    {
        params.weights->convertTo(weights, CV_32FC1);
        weights.reshape(1,1);
        preprocessProbability(weights);
    }

    // set means
    if(params.means && (startStep == EM::START_E_STEP || startStep == EM::START_AUTO_STEP))
        params.means->convertTo(means, CV_32FC1);

    // set covs
    if(params.covs && (startStep == EM::START_E_STEP && params.weights))
    {
        covs.resize(nclusters);
        for(size_t i = 0; i < params.covs->size(); i++)
            (*params.covs)[i].convertTo(covs[i], CV_32FC1);
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

        cv::SVD svd(covs[clusterIndex], cv::SVD::MODIFY_A + cv::SVD::FULL_UV);
        CV_DbgAssert(svd.w.rows == 1 || svd.w.cols == 1);
        CV_DbgAssert(svd.w.type() == CV_32FC1 && svd.u.type() == CV_32FC1);

        if(covMatType == EM::COV_MAT_SPHERICAL)
        {
            float maxSingularVal = svd.w.at<float>(0);
            covsEigenValues[clusterIndex] = cv::Mat(1, 1, CV_32FC1, cv::Scalar(maxSingularVal));
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
        cv::max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
        invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
    }
}

void EM::clusterTrainSamples()
{
    int nsamples = trainSamples.rows;

    // Cluster samples, compute/update means
    cv::Mat labels;
    cv::kmeans(trainSamples, nclusters, labels,
        cv::TermCriteria(cv::TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5),
        10, cv::KMEANS_PP_CENTERS, means);
    CV_Assert(means.type() == CV_32FC1);

    // Compute weights and covs
    weights = cv::Mat(1, nclusters, CV_32FC1, cv::Scalar(0));
    covs.resize(nclusters);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        cv::Mat clusterSamples;
        for(int sampleIndex = 0; sampleIndex < nsamples; sampleIndex++)
        {
            if(labels.at<int>(sampleIndex) == clusterIndex)
            {
                const cv::Mat sample = trainSamples.row(sampleIndex);
                clusterSamples.push_back(sample);
            }
        }
        CV_Assert(!clusterSamples.empty());

        cv::calcCovarMatrix(clusterSamples, covs[clusterIndex], means.row(clusterIndex),
            CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_32FC1);
        weights.at<float>(clusterIndex) = static_cast<float>(clusterSamples.rows)/static_cast<float>(nsamples);
    }

    decomposeCovs();
}

void EM::computeLogWeightDivDet()
{
    CV_Assert(!covsEigenValues.empty());

    cv::Mat logWeights;
    cv::log(weights, logWeights);

    logWeightDivDet.create(1, nclusters, CV_32FC1);
    // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        float logDetCov = 0.;
        for(int di = 0; di < covsEigenValues[clusterIndex].cols; di++)
            logDetCov += std::log(covsEigenValues[clusterIndex].at<float>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0));

        logWeightDivDet.at<float>(clusterIndex) = logWeights.at<float>(clusterIndex) - 0.5 * logDetCov;
    }
}

bool EM::doTrain(const cv::TermCriteria& termCrit)
{
    int dim = trainSamples.cols;
    // Precompute the empty initial train data in the cases of EM::START_E_STEP and START_AUTO_STEP
    if(startStep != EM::START_M_STEP)
    {
        if(weights.empty())
        {
            CV_Assert(covs.empty());
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

    double trainLikelihood, prevTrainLikelihood;
    for(int iter = 0; ; iter++)
    {
        eStep();
        trainLikelihood = cv::sum(trainLikelihoods)[0];

        if(iter >= termCrit.maxCount - 1)
            break;

        double trainLikelihoodDelta = trainLikelihood - (iter > 0 ? prevTrainLikelihood : 0);
        if( iter != 0 &&
            (trainLikelihoodDelta < -DBL_EPSILON ||
             trainLikelihoodDelta < termCrit.epsilon * std::fabs(trainLikelihood)))
            break;

        mStep();

        prevTrainLikelihood = trainLikelihood;
    }

    if( trainLikelihood <= -DBL_MAX/10000. )
        return false;

    // postprocess covs
    covs.resize(nclusters);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        if(covMatType == EM::COV_MAT_SPHERICAL)
        {
            covs[clusterIndex].create(dim, dim, CV_32FC1);
            cv::setIdentity(covs[clusterIndex], cv::Scalar(covsEigenValues[clusterIndex].at<float>(0)));
        }
        else if(covMatType == EM::COV_MAT_DIAGONAL)
            covs[clusterIndex] = cv::Mat::diag(covsEigenValues[clusterIndex].t());
    }

    return true;
}

void EM::computeProbabilities(const cv::Mat& sample, int& label, cv::Mat* probs, float* likelihood) const
{
    // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
    // q = arg(max_k(L_ik))
    // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_jk))

    CV_DbgAssert(sample.rows == 1);

    int dim = sample.cols;

    cv::Mat L(1, nclusters, CV_32FC1);
    cv::Mat expL(1, nclusters, CV_32FC1);

    label = 0;
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        const cv::Mat centeredSample = sample - means.row(clusterIndex);

        cv::Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
                centeredSample : centeredSample * covsRotateMats[clusterIndex];

        float Lval = 0;
        for(int di = 0; di < dim; di++)
        {
            float w = invCovsEigenValues[clusterIndex].at<float>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
            float val = rotatedCenteredSample.at<float>(di);
            Lval += w * val * val;
        }
        CV_DbgAssert(!logWeightDivDet.empty());
        Lval = logWeightDivDet.at<float>(clusterIndex) - 0.5 * Lval;
        L.at<float>(clusterIndex) = Lval;

        if(Lval > L.at<float>(label))
            label = clusterIndex;
    }

    if(!probs && !likelihood)
        return;

    // TODO maybe without finding max L value
    cv::exp(L, expL);
    float partExpSum = 0, // sum_j!=q (exp(L_jk)
           factor;         // 1/(1 + sum_j!=q (exp(L_jk))
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        if(clusterIndex != label)
            partExpSum += expL.at<float>(clusterIndex);
    }
    factor = 1./(1 + partExpSum);

    cv::exp(L - L.at<float>(label), expL);

    if(probs)
    {
        probs->create(1, nclusters, CV_32FC1);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
            probs->at<float>(clusterIndex) = expL.at<float>(clusterIndex) * factor;
    }

    if(likelihood)
    {
        // note likelihood = log (sum_j exp(L_ij)) - 0.5 * dims * ln2Pi
        *likelihood = std::log(partExpSum + expL.at<float>(label)) - 0.5 * dim * CV_LOG2PI;
    }
}

void EM::eStep()
{
    // Compute probs_ik from means_k, covs_k and weights_k.
    trainProbs.create(trainSamples.rows, nclusters, CV_32FC1);
    trainLabels.create(trainSamples.rows, 1, CV_32SC1);
    trainLikelihoods.create(trainSamples.rows, 1, CV_32FC1);

    computeLogWeightDivDet();

    for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
    {
        cv::Mat sampleProbs = trainProbs.row(sampleIndex);
        computeProbabilities(trainSamples.row(sampleIndex), trainLabels.at<int>(sampleIndex),
                             &sampleProbs, &trainLikelihoods.at<float>(sampleIndex));
    }
}

void EM::mStep()
{
    trainCounts.create(1, nclusters, CV_32SC1);
    trainCounts = cv::Scalar(0);

    for(int sampleIndex = 0; sampleIndex < trainLabels.rows; sampleIndex++)
        trainCounts.at<int>(trainLabels.at<int>(sampleIndex))++;

    if(cv::countNonZero(trainCounts) != (int)trainCounts.total())
    {
        clusterTrainSamples();
    }
    else
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols;

        // Update weights
        // not normalized first
        cv::reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        // Update means
        means.create(nclusters, dim, CV_32FC1);
        means = cv::Scalar(0);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            cv::Mat clusterMean = means.row(clusterIndex);
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                clusterMean += trainProbs.at<float>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            clusterMean /= weights.at<float>(clusterIndex);
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
                covsEigenValues[clusterIndex].create(1, dim, CV_32FC1);
            else
                covsEigenValues[clusterIndex].create(1, 1, CV_32FC1);

            if(covMatType == EM::COV_MAT_GENERIC)
                covs[clusterIndex].create(dim, dim, CV_32FC1);

            cv::Mat clusterCov = covMatType != EM::COV_MAT_GENERIC ?
                covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = cv::Scalar(0);

            cv::Mat centeredSample;
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

                if(covMatType == EM::COV_MAT_GENERIC)
                    clusterCov += trainProbs.at<float>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
                else
                {
                    float p = trainProbs.at<float>(sampleIndex, clusterIndex);
                    for(int di = 0; di < dim; di++ )
                    {
                        float val = centeredSample.at<float>(di);
                        clusterCov.at<float>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0) += p*val*val;
                    }
                }
            }

            if(covMatType == EM::COV_MAT_SPHERICAL)
                clusterCov /= dim;

            clusterCov /= weights.at<float>(clusterIndex);

            // Update covsRotateMats for EM::COV_MAT_GENERIC only
            if(covMatType == EM::COV_MAT_GENERIC)
            {
                cv::SVD svd(covs[clusterIndex], cv::SVD::MODIFY_A + cv::SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            cv::max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);

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
