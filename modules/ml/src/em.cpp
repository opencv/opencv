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
namespace ml
{

const double minEigenValue = DBL_EPSILON;

EM::Params::Params(int _nclusters, int _covMatType, const TermCriteria& _termCrit)
{
    nclusters = _nclusters;
    covMatType = _covMatType;
    termCrit = _termCrit;
}

class CV_EXPORTS EMImpl : public EM
{
public:
    EMImpl(const Params& _params)
    {
        setParams(_params);
    }

    virtual ~EMImpl() {}

    void setParams(const Params& _params)
    {
        params = _params;
        CV_Assert(params.nclusters > 1);
        CV_Assert(params.covMatType == COV_MAT_SPHERICAL ||
                  params.covMatType == COV_MAT_DIAGONAL ||
                  params.covMatType == COV_MAT_GENERIC);
    }

    Params getParams() const
    {
        return params;
    }

    void clear()
    {
        trainSamples.release();
        trainProbs.release();
        trainLogLikelihoods.release();
        trainLabels.release();

        weights.release();
        means.release();
        covs.clear();

        covsEigenValues.clear();
        invCovsEigenValues.clear();
        covsRotateMats.clear();

        logWeightDivDet.release();
    }

    bool train(const Ptr<TrainData>& data, int)
    {
        Mat samples = data->getTrainSamples(), labels;
        return train_(samples, labels, noArray(), noArray());
    }

    bool train_(InputArray samples,
               OutputArray logLikelihoods,
               OutputArray labels,
               OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0);
        return doTrain(START_AUTO_STEP, logLikelihoods, labels, probs);
    }

    bool trainE(InputArray samples,
                InputArray _means0,
                InputArray _covs0,
                InputArray _weights0,
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        std::vector<Mat> covs0;
        _covs0.getMatVector(covs0);

        Mat means0 = _means0.getMat(), weights0 = _weights0.getMat();

        setTrainData(START_E_STEP, samplesMat, 0, !_means0.empty() ? &means0 : 0,
                     !_covs0.empty() ? &covs0 : 0, !_weights0.empty() ? &weights0 : 0);
        return doTrain(START_E_STEP, logLikelihoods, labels, probs);
    }

    bool trainM(InputArray samples,
                InputArray _probs0,
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        Mat probs0 = _probs0.getMat();

        setTrainData(START_M_STEP, samplesMat, !_probs0.empty() ? &probs0 : 0, 0, 0, 0);
        return doTrain(START_M_STEP, logLikelihoods, labels, probs);
    }

    float predict(InputArray _inputs, OutputArray _outputs, int) const
    {
        bool needprobs = _outputs.needed();
        Mat samples = _inputs.getMat(), probs, probsrow;
        int ptype = CV_32F;
        float firstres = 0.f;
        int i, nsamples = samples.rows;

        if( needprobs )
        {
            if( _outputs.fixedType() )
                ptype = _outputs.type();
            _outputs.create(samples.rows, params.nclusters, ptype);
        }
        else
            nsamples = std::min(nsamples, 1);

        for( i = 0; i < nsamples; i++ )
        {
            if( needprobs )
                probsrow = probs.row(i);
            Vec2d res = computeProbabilities(samples.row(i), needprobs ? &probsrow : 0, ptype);
            if( i == 0 )
                firstres = (float)res[1];
        }
        return firstres;
    }

    Vec2d predict2(InputArray _sample, OutputArray _probs) const
    {
        int ptype = CV_32F;
        Mat sample = _sample.getMat();
        CV_Assert(isTrained());

        CV_Assert(!sample.empty());
        if(sample.type() != CV_64FC1)
        {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }
        sample.reshape(1, 1);

        Mat probs;
        if( _probs.needed() )
        {
            if( _probs.fixedType() )
                ptype = _probs.type();
            _probs.create(1, params.nclusters, ptype);
            probs = _probs.getMat();
        }

        return computeProbabilities(sample, !probs.empty() ? &probs : 0, ptype);
    }

    bool isTrained() const
    {
        return !means.empty();
    }

    bool isClassifier() const
    {
        return true;
    }

    int getVarCount() const
    {
        return means.cols;
    }

    String getDefaultModelName() const
    {
        return "opencv_ml_em";
    }

    static void checkTrainData(int startStep, const Mat& samples,
                               int nclusters, int covMatType, const Mat* probs, const Mat* means,
                               const std::vector<Mat>* covs, const Mat* weights)
    {
        // Check samples.
        CV_Assert(!samples.empty());
        CV_Assert(samples.channels() == 1);

        int nsamples = samples.rows;
        int dim = samples.cols;

        // Check training params.
        CV_Assert(nclusters > 0);
        CV_Assert(nclusters <= nsamples);
        CV_Assert(startStep == START_AUTO_STEP ||
                  startStep == START_E_STEP ||
                  startStep == START_M_STEP);
        CV_Assert(covMatType == COV_MAT_GENERIC ||
                  covMatType == COV_MAT_DIAGONAL ||
                  covMatType == COV_MAT_SPHERICAL);

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

        if(startStep == START_E_STEP)
        {
            CV_Assert(means);
        }
        else if(startStep == START_M_STEP)
        {
            CV_Assert(probs);
        }
    }

    static void preprocessSampleData(const Mat& src, Mat& dst, int dstType, bool isAlwaysClone)
    {
        if(src.type() == dstType && !isAlwaysClone)
            dst = src;
        else
            src.convertTo(dst, dstType);
    }

    static void preprocessProbability(Mat& probs)
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

    void setTrainData(int startStep, const Mat& samples,
                      const Mat* probs0,
                      const Mat* means0,
                      const std::vector<Mat>* covs0,
                      const Mat* weights0)
    {
        int nclusters = params.nclusters, covMatType = params.covMatType;
        clear();

        checkTrainData(startStep, samples, nclusters, covMatType, probs0, means0, covs0, weights0);

        bool isKMeansInit = (startStep == START_AUTO_STEP) || (startStep == START_E_STEP && (covs0 == 0 || weights0 == 0));
        // Set checked data
        preprocessSampleData(samples, trainSamples, isKMeansInit ? CV_32FC1 : CV_64FC1, false);

        // set probs
        if(probs0 && startStep == START_M_STEP)
        {
            preprocessSampleData(*probs0, trainProbs, CV_64FC1, true);
            preprocessProbability(trainProbs);
        }

        // set weights
        if(weights0 && (startStep == START_E_STEP && covs0))
        {
            weights0->convertTo(weights, CV_64FC1);
            weights.reshape(1,1);
            preprocessProbability(weights);
        }

        // set means
        if(means0 && (startStep == START_E_STEP/* || startStep == START_AUTO_STEP*/))
            means0->convertTo(means, isKMeansInit ? CV_32FC1 : CV_64FC1);

        // set covs
        if(covs0 && (startStep == START_E_STEP && weights0))
        {
            covs.resize(nclusters);
            for(size_t i = 0; i < covs0->size(); i++)
                (*covs0)[i].convertTo(covs[i], CV_64FC1);
        }
    }

    void decomposeCovs()
    {
        int nclusters = params.nclusters, covMatType = params.covMatType;
        CV_Assert(!covs.empty());
        covsEigenValues.resize(nclusters);
        if(covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            CV_Assert(!covs[clusterIndex].empty());

            SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);

            if(covMatType == COV_MAT_SPHERICAL)
            {
                double maxSingularVal = svd.w.at<double>(0);
                covsEigenValues[clusterIndex] = Mat(1, 1, CV_64FC1, Scalar(maxSingularVal));
            }
            else if(covMatType == COV_MAT_DIAGONAL)
            {
                covsEigenValues[clusterIndex] = svd.w;
            }
            else //COV_MAT_GENERIC
            {
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }
            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
            invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
        }
    }

    void clusterTrainSamples()
    {
        int nclusters = params.nclusters;
        int nsamples = trainSamples.rows;

        // Cluster samples, compute/update means

        // Convert samples and means to 32F, because kmeans requires this type.
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
        kmeans(trainSamplesFlt, nclusters, labels,
               TermCriteria(TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5),
               10, KMEANS_PP_CENTERS, meansFlt);

        // Convert samples and means back to 64F.
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

    void computeLogWeightDivDet()
    {
        int nclusters = params.nclusters;
        CV_Assert(!covsEigenValues.empty());

        Mat logWeights;
        cv::max(weights, DBL_MIN, weights);
        log(weights, logWeights);

        logWeightDivDet.create(1, nclusters, CV_64FC1);
        // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            double logDetCov = 0.;
            const int evalCount = static_cast<int>(covsEigenValues[clusterIndex].total());
            for(int di = 0; di < evalCount; di++)
                logDetCov += std::log(covsEigenValues[clusterIndex].at<double>(params.covMatType != COV_MAT_SPHERICAL ? di : 0));

            logWeightDivDet.at<double>(clusterIndex) = logWeights.at<double>(clusterIndex) - 0.5 * logDetCov;
        }
    }

    bool doTrain(int startStep, OutputArray logLikelihoods, OutputArray labels, OutputArray probs)
    {
        int nclusters = params.nclusters;
        int dim = trainSamples.cols;
        // Precompute the empty initial train data in the cases of START_E_STEP and START_AUTO_STEP
        if(startStep != START_M_STEP)
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

        if(startStep == START_M_STEP)
            mStep();

        double trainLogLikelihood, prevTrainLogLikelihood = 0.;
        int maxIters = (params.termCrit.type & TermCriteria::MAX_ITER) ?
            params.termCrit.maxCount : DEFAULT_MAX_ITERS;
        double epsilon = (params.termCrit.type & TermCriteria::EPS) ? params.termCrit.epsilon : 0.;

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
            if(params.covMatType == COV_MAT_SPHERICAL)
            {
                covs[clusterIndex].create(dim, dim, CV_64FC1);
                setIdentity(covs[clusterIndex], Scalar(covsEigenValues[clusterIndex].at<double>(0)));
            }
            else if(params.covMatType == COV_MAT_DIAGONAL)
            {
                covs[clusterIndex] = Mat::diag(covsEigenValues[clusterIndex]);
            }
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

        return true;
    }

    Vec2d computeProbabilities(const Mat& sample, Mat* probs, int ptype) const
    {
        // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
        // q = arg(max_k(L_ik))
        // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
        // see Alex Smola's blog http://blog.smola.org/page/2 for
        // details on the log-sum-exp trick

        int nclusters = params.nclusters, covMatType = params.covMatType;
        int stype = sample.type();
        CV_Assert(!means.empty());
        CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert(sample.size() == Size(means.cols, 1));

        int dim = sample.cols;

        Mat L(1, nclusters, CV_64FC1), centeredSample(1, dim, CV_64F);
        int i, label = 0;
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            const double* mptr = means.ptr<double>(clusterIndex);
            double* dptr = centeredSample.ptr<double>();
            if( stype == CV_32F )
            {
                const float* sptr = sample.ptr<float>();
                for( i = 0; i < dim; i++ )
                    dptr[i] = sptr[i] - mptr[i];
            }
            else
            {
                const double* sptr = sample.ptr<double>();
                for( i = 0; i < dim; i++ )
                    dptr[i] = sptr[i] - mptr[i];
            }

            Mat rotatedCenteredSample = covMatType != COV_MAT_GENERIC ?
                    centeredSample : centeredSample * covsRotateMats[clusterIndex];

            double Lval = 0;
            for(int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;

            if(L.at<double>(clusterIndex) > L.at<double>(label))
                label = clusterIndex;
        }

        double maxLVal = L.at<double>(label);
        double expDiffSum = 0;
        for( i = 0; i < L.cols; i++ )
        {
            double v = std::exp(L.at<double>(i) - maxLVal);
            L.at<double>(i) = v;
            expDiffSum += v; // sum_j(exp(L_ij - L_iq))
        }

        if(probs)
            L.convertTo(*probs, ptype, 1./expDiffSum);

        Vec2d res;
        res[0] = std::log(expDiffSum)  + maxLVal - 0.5 * dim * CV_LOG2PI;
        res[1] = label;

        return res;
    }

    void eStep()
    {
        // Compute probs_ik from means_k, covs_k and weights_k.
        trainProbs.create(trainSamples.rows, params.nclusters, CV_64FC1);
        trainLabels.create(trainSamples.rows, 1, CV_32SC1);
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);

        computeLogWeightDivDet();

        CV_DbgAssert(trainSamples.type() == CV_64FC1);
        CV_DbgAssert(means.type() == CV_64FC1);

        for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex);
            Vec2d res = computeProbabilities(trainSamples.row(sampleIndex), &sampleProbs, CV_64F);
            trainLogLikelihoods.at<double>(sampleIndex) = res[0];
            trainLabels.at<int>(sampleIndex) = static_cast<int>(res[1]);
        }
    }

    void mStep()
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int nclusters = params.nclusters;
        int covMatType = params.covMatType;
        int dim = trainSamples.cols;

        // Update weights
        // not normalized first
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        // Update means
        means.create(nclusters, dim, CV_64FC1);
        means = Scalar(0);

        const double minPosWeight = trainSamples.rows * DBL_EPSILON;
        double minWeight = DBL_MAX;
        int minWeightClusterIndex = -1;
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if(weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if(weights.at<double>(clusterIndex) < minWeight)
            {
                minWeight = weights.at<double>(clusterIndex);
                minWeightClusterIndex = clusterIndex;
            }

            Mat clusterMean = means.row(clusterIndex);
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                clusterMean += trainProbs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            clusterMean /= weights.at<double>(clusterIndex);
        }

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters);
        covsEigenValues.resize(nclusters);
        if(covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if(weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if(covMatType != COV_MAT_SPHERICAL)
                covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);
            else
                covsEigenValues[clusterIndex].create(1, 1, CV_64FC1);

            if(covMatType == COV_MAT_GENERIC)
                covs[clusterIndex].create(dim, dim, CV_64FC1);

            Mat clusterCov = covMatType != COV_MAT_GENERIC ?
                covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = Scalar(0);

            Mat centeredSample;
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

                if(covMatType == COV_MAT_GENERIC)
                    clusterCov += trainProbs.at<double>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
                else
                {
                    double p = trainProbs.at<double>(sampleIndex, clusterIndex);
                    for(int di = 0; di < dim; di++ )
                    {
                        double val = centeredSample.at<double>(di);
                        clusterCov.at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0) += p*val*val;
                    }
                }
            }

            if(covMatType == COV_MAT_SPHERICAL)
                clusterCov /= dim;

            clusterCov /= weights.at<double>(clusterIndex);

            // Update covsRotateMats for COV_MAT_GENERIC only
            if(covMatType == COV_MAT_GENERIC)
            {
                SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);

            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
        }

        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if(weights.at<double>(clusterIndex) <= minPosWeight)
            {
                Mat clusterMean = means.row(clusterIndex);
                means.row(minWeightClusterIndex).copyTo(clusterMean);
                covs[minWeightClusterIndex].copyTo(covs[clusterIndex]);
                covsEigenValues[minWeightClusterIndex].copyTo(covsEigenValues[clusterIndex]);
                if(covMatType == COV_MAT_GENERIC)
                    covsRotateMats[minWeightClusterIndex].copyTo(covsRotateMats[clusterIndex]);
                invCovsEigenValues[minWeightClusterIndex].copyTo(invCovsEigenValues[clusterIndex]);
            }
        }

        // Normalize weights
        weights /= trainSamples.rows;
    }

    void write_params(FileStorage& fs) const
    {
        fs << "nclusters" << params.nclusters;
        fs << "cov_mat_type" << (params.covMatType == COV_MAT_SPHERICAL ? String("spherical") :
                                 params.covMatType == COV_MAT_DIAGONAL ? String("diagonal") :
                                 params.covMatType == COV_MAT_GENERIC ? String("generic") :
                                 format("unknown_%d", params.covMatType));
        writeTermCrit(fs, params.termCrit);
    }

    void write(FileStorage& fs) const
    {
        fs << "training_params" << "{";
        write_params(fs);
        fs << "}";
        fs << "weights" << weights;
        fs << "means" << means;

        size_t i, n = covs.size();

        fs << "covs" << "[";
        for( i = 0; i < n; i++ )
            fs << covs[i];
        fs << "]";
    }

    void read_params(const FileNode& fn)
    {
        Params _params;
        _params.nclusters = (int)fn["nclusters"];
        String s = (String)fn["cov_mat_type"];
        _params.covMatType = s == "spherical" ? COV_MAT_SPHERICAL :
                             s == "diagonal" ? COV_MAT_DIAGONAL :
                             s == "generic" ? COV_MAT_GENERIC : -1;
        CV_Assert(_params.covMatType >= 0);
        _params.termCrit = readTermCrit(fn);
        setParams(_params);
    }

    void read(const FileNode& fn)
    {
        clear();
        read_params(fn["training_params"]);

        fn["weights"] >> weights;
        fn["means"] >> means;

        FileNode cfn = fn["covs"];
        FileNodeIterator cfn_it = cfn.begin();
        int i, n = (int)cfn.size();
        covs.resize(n);

        for( i = 0; i < n; i++, ++cfn_it )
            (*cfn_it) >> covs[i];

        decomposeCovs();
        computeLogWeightDivDet();
    }

    Mat getWeights() const { return weights; }
    Mat getMeans() const { return means; }
    void getCovs(std::vector<Mat>& _covs) const
    {
        _covs.resize(covs.size());
        std::copy(covs.begin(), covs.end(), _covs.begin());
    }

    Params params;

    // all inner matrices have type CV_64FC1
    Mat trainSamples;
    Mat trainProbs;
    Mat trainLogLikelihoods;
    Mat trainLabels;

    Mat weights;
    Mat means;
    std::vector<Mat> covs;

    std::vector<Mat> covsEigenValues;
    std::vector<Mat> covsRotateMats;
    std::vector<Mat> invCovsEigenValues;
    Mat logWeightDivDet;
};


Ptr<EM> EM::train(InputArray samples, OutputArray logLikelihoods,
                  OutputArray labels, OutputArray probs,
                  const EM::Params& params)
{
    Ptr<EMImpl> em = makePtr<EMImpl>(params);
    if(!em->train_(samples, logLikelihoods, labels, probs))
        em.release();
    return em;
}

Ptr<EM> EM::train_startWithE(InputArray samples, InputArray means0,
                             InputArray covs0, InputArray weights0,
                             OutputArray logLikelihoods, OutputArray labels,
                             OutputArray probs, const EM::Params& params)
{
    Ptr<EMImpl> em = makePtr<EMImpl>(params);
    if(!em->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs))
        em.release();
    return em;
}

Ptr<EM> EM::train_startWithM(InputArray samples, InputArray probs0,
                             OutputArray logLikelihoods, OutputArray labels,
                             OutputArray probs, const EM::Params& params)
{
    Ptr<EMImpl> em = makePtr<EMImpl>(params);
    if(!em->trainM(samples, probs0, logLikelihoods, labels, probs))
        em.release();
    return em;
}

Ptr<EM> EM::create(const Params& params)
{
    return makePtr<EMImpl>(params);
}

}
} // namespace cv

/* End of file. */
