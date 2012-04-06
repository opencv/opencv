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

const float minEigenValue = 1.e-3f;

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

    
bool EM::train(InputArray samples,
               OutputArray labels,
               OutputArray probs,
               OutputArray likelihoods)
{
    setTrainData(START_AUTO_STEP, samples.getMat(), 0, 0, 0, 0);
    return doTrain(START_AUTO_STEP, labels, probs, likelihoods);
}

bool EM::trainE(InputArray samples,
                InputArray _means0,
                InputArray _covs0,
                InputArray _weights0,
                OutputArray labels,
                OutputArray probs,
                OutputArray likelihoods)
{
    vector<Mat> covs0;
    _covs0.getMatVector(covs0);
    
    Mat means0 = _means0.getMat(), weights0 = _weights0.getMat();

    setTrainData(START_E_STEP, samples.getMat(), 0, !_means0.empty() ? &means0 : 0,
                 !_covs0.empty() ? &covs0 : 0, _weights0.empty() ? &weights0 : 0);
    return doTrain(START_E_STEP, labels, probs, likelihoods);
}

bool EM::trainM(InputArray samples,
                InputArray _probs0,
                OutputArray labels,
                OutputArray probs,
                OutputArray likelihoods)
{
    Mat probs0 = _probs0.getMat();
    
    setTrainData(START_M_STEP, samples.getMat(), !_probs0.empty() ? &probs0 : 0, 0, 0, 0);
    return doTrain(START_M_STEP, labels, probs, likelihoods);
}

    
int EM::predict(InputArray _sample, OutputArray _probs, double* _likelihood) const
{
    Mat sample = _sample.getMat();
    CV_Assert(isTrained());

    CV_Assert(!sample.empty());
    CV_Assert(sample.type() == CV_32FC1);

    int label;
    float likelihood = 0.f;
    Mat probs;
    if( _probs.needed() )
    {
        _probs.create(1, nclusters, CV_32FC1);
        probs = _probs.getMat();
    }
    computeProbabilities(sample, label, !probs.empty() ? &probs : 0, _likelihood ? &likelihood : 0);
    if(_likelihood)
        *_likelihood = static_cast<double>(likelihood);

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
    CV_Assert(samples.type() == CV_32FC1);

    int nsamples = samples.rows;
    int dim = samples.cols;

    // Check training params.
    CV_Assert(nclusters > 0);
    CV_Assert(nclusters <= nsamples);
    CV_Assert(startStep == EM::START_AUTO_STEP ||
              startStep == EM::START_E_STEP ||
              startStep == EM::START_M_STEP);

    CV_Assert(!probs ||
        (!probs->empty() &&
         probs->rows == nsamples && probs->cols == nclusters &&
         probs->type() == CV_32FC1));

    CV_Assert(!weights ||
        (!weights->empty() &&
         (weights->cols == 1 || weights->rows == 1) && static_cast<int>(weights->total()) == nclusters &&
         weights->type() == CV_32FC1));

    CV_Assert(!means ||
        (!means->empty() &&
         means->rows == nclusters && means->cols == dim &&
         means->type() == CV_32FC1));

    CV_Assert(!covs ||
        (!covs->empty() &&
         static_cast<int>(covs->size()) == nclusters));
    if(covs)
    {
        const Size covSize(dim, dim);
        for(size_t i = 0; i < covs->size(); i++)
        {
            const Mat& m = (*covs)[i];
            CV_Assert(!m.empty() && m.size() == covSize && (m.type() == CV_32FC1));
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

    const float uniformProbability = (float)(1./probs.cols);
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

    // Set checked data
    preprocessSampleData(samples, trainSamples, CV_32FC1, false);

    // set probs
    if(probs0 && startStep == EM::START_M_STEP)
    {
        preprocessSampleData(*probs0, trainProbs, CV_32FC1, true);
        preprocessProbability(trainProbs);
    }

    // set weights
    if(weights0 && (startStep == EM::START_E_STEP && covs0))
    {
        weights0->convertTo(weights, CV_32FC1);
        weights.reshape(1,1);
        preprocessProbability(weights);
    }

    // set means
    if(means0 && (startStep == EM::START_E_STEP || startStep == EM::START_AUTO_STEP))
        means0->convertTo(means, CV_32FC1);

    // set covs
    if(covs0 && (startStep == EM::START_E_STEP && weights0))
    {
        covs.resize(nclusters);
        for(size_t i = 0; i < covs0->size(); i++)
            (*covs0)[i].convertTo(covs[i], CV_32FC1);
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
        CV_DbgAssert(svd.w.rows == 1 || svd.w.cols == 1);
        CV_DbgAssert(svd.w.type() == CV_32FC1 && svd.u.type() == CV_32FC1);

        if(covMatType == EM::COV_MAT_SPHERICAL)
        {
            float maxSingularVal = svd.w.at<float>(0);
            covsEigenValues[clusterIndex] = Mat(1, 1, CV_32FC1, Scalar(maxSingularVal));
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
    Mat labels;
    kmeans(trainSamples, nclusters, labels,
        TermCriteria(TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5),
        10, KMEANS_PP_CENTERS, means);
    CV_Assert(means.type() == CV_32FC1);

    // Compute weights and covs
    weights = Mat(1, nclusters, CV_32FC1, Scalar(0));
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
            CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_32FC1);
        weights.at<float>(clusterIndex) = static_cast<float>(clusterSamples.rows)/static_cast<float>(nsamples);
    }

    decomposeCovs();
}

void EM::computeLogWeightDivDet()
{
    CV_Assert(!covsEigenValues.empty());

    Mat logWeights;
    log(weights, logWeights);

    logWeightDivDet.create(1, nclusters, CV_32FC1);
    // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        float logDetCov = 0.;
        for(int di = 0; di < covsEigenValues[clusterIndex].cols; di++)
            logDetCov += std::log(covsEigenValues[clusterIndex].at<float>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0));

        logWeightDivDet.at<float>(clusterIndex) = logWeights.at<float>(clusterIndex) - 0.5f * logDetCov;
    }
}

bool EM::doTrain(int startStep, OutputArray labels, OutputArray probs, OutputArray likelihoods)
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

    double trainLikelihood, prevTrainLikelihood = 0.;
    for(int iter = 0; ; iter++)
    {
        eStep();
        trainLikelihood = sum(trainLikelihoods)[0];

        if(iter >= maxIters - 1)
            break;

        double trainLikelihoodDelta = trainLikelihood - (iter > 0 ? prevTrainLikelihood : 0);
        if( iter != 0 &&
            (trainLikelihoodDelta < -DBL_EPSILON ||
             trainLikelihoodDelta < epsilon * std::fabs(trainLikelihood)))
            break;

        mStep();

        prevTrainLikelihood = trainLikelihood;
    }

    if( trainLikelihood <= -DBL_MAX/10000. )
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
            covs[clusterIndex].create(dim, dim, CV_32FC1);
            setIdentity(covs[clusterIndex], Scalar(covsEigenValues[clusterIndex].at<float>(0)));
        }
        else if(covMatType == EM::COV_MAT_DIAGONAL)
            covs[clusterIndex] = Mat::diag(covsEigenValues[clusterIndex].t());
    }
    
    if(labels.needed())
        trainLabels.copyTo(labels);
    if(probs.needed())
        trainProbs.copyTo(probs);
    if(likelihoods.needed())
        trainLikelihoods.copyTo(likelihoods);
    
    trainSamples.release();
    trainProbs.release();
    trainLabels.release();
    trainLikelihoods.release();
    trainCounts.release();

    return true;
}

void EM::computeProbabilities(const Mat& sample, int& label, Mat* probs, float* likelihood) const
{
    // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
    // q = arg(max_k(L_ik))
    // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_jk))

    CV_Assert(sample.rows == 1);

    int dim = sample.cols;

    Mat L(1, nclusters, CV_32FC1);
    Mat expL(1, nclusters, CV_32FC1);

    label = 0;
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        const Mat centeredSample = sample - means.row(clusterIndex);

        Mat rotatedCenteredSample = covMatType != EM::COV_MAT_GENERIC ?
                centeredSample : centeredSample * covsRotateMats[clusterIndex];

        float Lval = 0;
        for(int di = 0; di < dim; di++)
        {
            float w = invCovsEigenValues[clusterIndex].at<float>(covMatType != EM::COV_MAT_SPHERICAL ? di : 0);
            float val = rotatedCenteredSample.at<float>(di);
            Lval += w * val * val;
        }
        CV_DbgAssert(!logWeightDivDet.empty());
        Lval = logWeightDivDet.at<float>(clusterIndex) - 0.5f * Lval;
        L.at<float>(clusterIndex) = Lval;

        if(Lval > L.at<float>(label))
            label = clusterIndex;
    }

    if(!probs && !likelihood)
        return;

    // TODO maybe without finding max L value
    exp(L, expL);
    float partExpSum = 0, // sum_j!=q (exp(L_jk)
           factor;         // 1/(1 + sum_j!=q (exp(L_jk))
    float prevL = expL.at<float>(label);
    for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
    {
        if(clusterIndex != label)
            partExpSum += expL.at<float>(clusterIndex);
    }
    factor = 1.f/(1 + partExpSum);

    exp(L - L.at<float>(label), expL);

    if(probs)
    {
        probs->create(1, nclusters, CV_32FC1);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
            probs->at<float>(clusterIndex) = expL.at<float>(clusterIndex) * factor;
    }

    if(likelihood)
    {
        // note likelihood = log (sum_j exp(L_ij)) - 0.5 * dims * ln2Pi
        *likelihood = std::log(prevL + partExpSum) - (float)(0.5 * dim * CV_LOG2PI);
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
        Mat sampleProbs = trainProbs.row(sampleIndex);
        computeProbabilities(trainSamples.row(sampleIndex), trainLabels.at<int>(sampleIndex),
                             &sampleProbs, &trainLikelihoods.at<float>(sampleIndex));
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
        means.create(nclusters, dim, CV_32FC1);
        means = Scalar(0);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            Mat clusterMean = means.row(clusterIndex);
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

            Mat clusterCov = covMatType != EM::COV_MAT_GENERIC ?
                covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = Scalar(0);

            Mat centeredSample;
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
