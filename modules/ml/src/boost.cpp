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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

namespace cv { namespace ml {

static inline double
log_ratio( double val )
{
    const double eps = 1e-5;
    val = std::max( val, eps );
    val = std::min( val, 1. - eps );
    return log( val/(1. - val) );
}


BoostTreeParams::BoostTreeParams()
{
    boostType = Boost::REAL;
    weakCount = 100;
    weightTrimRate = 0.95;
}

BoostTreeParams::BoostTreeParams( int _boostType, int _weak_count,
                                  double _weightTrimRate)
{
    boostType = _boostType;
    weakCount = _weak_count;
    weightTrimRate = _weightTrimRate;
}

class DTreesImplForBoost CV_FINAL : public DTreesImpl
{
public:
    DTreesImplForBoost()
    {
        params.setCVFolds(0);
        params.setMaxDepth(1);
    }
    virtual ~DTreesImplForBoost() {}

    bool isClassifier() const CV_OVERRIDE { return true; }

    void clear() CV_OVERRIDE
    {
        DTreesImpl::clear();
    }

    void startTraining( const Ptr<TrainData>& trainData, int flags ) CV_OVERRIDE
    {
        CV_Assert(!trainData.empty());
        DTreesImpl::startTraining(trainData, flags);
        sumResult.assign(w->sidx.size(), 0.);

        if( bparams.boostType != Boost::DISCRETE )
        {
            _isClassifier = false;
            int i, n = (int)w->cat_responses.size();
            w->ord_responses.resize(n);

            double a = -1, b = 1;
            if( bparams.boostType == Boost::LOGIT )
            {
                a = -2, b = 2;
            }
            for( i = 0; i < n; i++ )
                w->ord_responses[i] = w->cat_responses[i] > 0 ? b : a;
        }

        normalizeWeights();
    }

    void normalizeWeights()
    {
        int i, n = (int)w->sidx.size();
        double sumw = 0, a, b;
        for( i = 0; i < n; i++ )
            sumw += w->sample_weights[w->sidx[i]];
        if( sumw > DBL_EPSILON )
        {
            a = 1./sumw;
            b = 0;
        }
        else
        {
            a = 0;
            b = 1;
        }
        for( i = 0; i < n; i++ )
        {
            double& wval = w->sample_weights[w->sidx[i]];
            wval = wval*a + b;
        }
    }

    void endTraining() CV_OVERRIDE
    {
        DTreesImpl::endTraining();
        vector<double> e;
        std::swap(sumResult, e);
    }

    void scaleTree( int root, double scale )
    {
        int nidx = root, pidx = 0;
        Node *node = 0;

        // traverse the tree and save all the nodes in depth-first order
        for(;;)
        {
            for(;;)
            {
                node = &nodes[nidx];
                node->value *= scale;
                if( node->left < 0 )
                    break;
                nidx = node->left;
            }

            for( pidx = node->parent; pidx >= 0 && nodes[pidx].right == nidx;
                 nidx = pidx, pidx = nodes[pidx].parent )
                ;

            if( pidx < 0 )
                break;

            nidx = nodes[pidx].right;
        }
    }

    void calcValue( int nidx, const vector<int>& _sidx ) CV_OVERRIDE
    {
        DTreesImpl::calcValue(nidx, _sidx);
        WNode* node = &w->wnodes[nidx];
        if( bparams.boostType == Boost::DISCRETE )
        {
            node->value = node->class_idx == 0 ? -1 : 1;
        }
        else if( bparams.boostType == Boost::REAL )
        {
            double p = (node->value+1)*0.5;
            node->value = 0.5*log_ratio(p);
        }
    }

    bool train( const Ptr<TrainData>& trainData, int flags ) CV_OVERRIDE
    {
        CV_Assert(!trainData.empty());
        startTraining(trainData, flags);
        int treeidx, ntrees = bparams.weakCount >= 0 ? bparams.weakCount : 10000;
        vector<int> sidx = w->sidx;

        for( treeidx = 0; treeidx < ntrees; treeidx++ )
        {
            int root = addTree( sidx );
            if( root < 0 )
                return false;
            updateWeightsAndTrim( treeidx, sidx );
        }
        endTraining();
        return true;
    }

    void updateWeightsAndTrim( int treeidx, vector<int>& sidx )
    {
        int i, n = (int)w->sidx.size();
        int nvars = (int)varIdx.size();
        double sumw = 0., C = 1.;
        cv::AutoBuffer<double> buf(n + nvars);
        double* result = buf.data();
        float* sbuf = (float*)(result + n);
        Mat sample(1, nvars, CV_32F, sbuf);
        int predictFlags = bparams.boostType == Boost::DISCRETE ? (PREDICT_MAX_VOTE | RAW_OUTPUT) : PREDICT_SUM;
        predictFlags |= COMPRESSED_INPUT;

        for( i = 0; i < n; i++ )
        {
            w->data->getSample(varIdx, w->sidx[i], sbuf );
            result[i] = predictTrees(Range(treeidx, treeidx+1), sample, predictFlags);
        }

        // now update weights and other parameters for each type of boosting
        if( bparams.boostType == Boost::DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))
            double err = 0.;

            for( i = 0; i < n; i++ )
            {
                int si = w->sidx[i];
                double wval = w->sample_weights[si];
                sumw += wval;
                err += wval*(result[i] != w->cat_responses[si]);
            }

            if( sumw != 0 )
                err /= sumw;
            C = -log_ratio( err );
            double scale = std::exp(C);

            sumw = 0;
            for( i = 0; i < n; i++ )
            {
                int si = w->sidx[i];
                double wval = w->sample_weights[si];
                if( result[i] != w->cat_responses[si] )
                    wval *= scale;
                sumw += wval;
                w->sample_weights[si] = wval;
            }

            scaleTree(roots[treeidx], C);
        }
        else if( bparams.boostType == Boost::REAL || bparams.boostType == Boost::GENTLE )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            for( i = 0; i < n; i++ )
            {
                int si = w->sidx[i];
                CV_Assert( std::abs(w->ord_responses[si]) == 1 );
                double wval = w->sample_weights[si]*std::exp(-result[i]*w->ord_responses[si]);
                sumw += wval;
                w->sample_weights[si] = wval;
            }
        }
        else if( bparams.boostType == Boost::LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses
            const double lb_weight_thresh = FLT_EPSILON;
            const double lb_z_max = 10.;

            for( i = 0; i < n; i++ )
            {
                int si = w->sidx[i];
                sumResult[i] += 0.5*result[i];
                double p = 1./(1 + std::exp(-2*sumResult[i]));
                double wval = std::max( p*(1 - p), lb_weight_thresh ), z;
                w->sample_weights[si] = wval;
                sumw += wval;
                if( w->ord_responses[si] > 0 )
                {
                    z = 1./p;
                    w->ord_responses[si] = std::min(z, lb_z_max);
                }
                else
                {
                    z = 1./(1-p);
                    w->ord_responses[si] = -std::min(z, lb_z_max);
                }
            }
        }
        else
            CV_Error(CV_StsNotImplemented, "Unknown boosting type");

        /*if( bparams.boostType != Boost::LOGIT )
        {
            double err = 0;
            for( i = 0; i < n; i++ )
            {
                sumResult[i] += result[i]*C;
                if( bparams.boostType != Boost::DISCRETE )
                    err += sumResult[i]*w->ord_responses[w->sidx[i]] < 0;
                else
                    err += sumResult[i]*w->cat_responses[w->sidx[i]] < 0;
            }
            printf("%d trees. C=%.2f, training error=%.1f%%, working set size=%d (out of %d)\n", (int)roots.size(), C, err*100./n, (int)sidx.size(), n);
        }*/

        // renormalize weights
        if( sumw > FLT_EPSILON )
            normalizeWeights();

        if( bparams.weightTrimRate <= 0. || bparams.weightTrimRate >= 1. )
            return;

        for( i = 0; i < n; i++ )
            result[i] = w->sample_weights[w->sidx[i]];
        std::sort(result, result + n);

        // as weight trimming occurs immediately after updating the weights,
        // where they are renormalized, we assume that the weight sum = 1.
        sumw = 1. - bparams.weightTrimRate;

        for( i = 0; i < n; i++ )
        {
            double wval = result[i];
            if( sumw <= 0 )
                break;
            sumw -= wval;
        }

        double threshold = i < n ? result[i] : DBL_MAX;
        sidx.clear();

        for( i = 0; i < n; i++ )
        {
            int si = w->sidx[i];
            if( w->sample_weights[si] >= threshold )
                sidx.push_back(si);
        }
    }

    float predictTrees( const Range& range, const Mat& sample, int flags0 ) const CV_OVERRIDE
    {
        int flags = (flags0 & ~PREDICT_MASK) | PREDICT_SUM;
        float val = DTreesImpl::predictTrees(range, sample, flags);
        if( flags != flags0 )
        {
            int ival = (int)(val > 0);
            if( !(flags0 & RAW_OUTPUT) )
                ival = classLabels[ival];
            val = (float)ival;
        }
        return val;
    }

    void writeTrainingParams( FileStorage& fs ) const CV_OVERRIDE
    {
        fs << "boosting_type" <<
        (bparams.boostType == Boost::DISCRETE ? "DiscreteAdaboost" :
        bparams.boostType == Boost::REAL ? "RealAdaboost" :
        bparams.boostType == Boost::LOGIT ? "LogitBoost" :
        bparams.boostType == Boost::GENTLE ? "GentleAdaboost" : "Unknown");

        DTreesImpl::writeTrainingParams(fs);
        fs << "weight_trimming_rate" << bparams.weightTrimRate;
    }

    void write( FileStorage& fs ) const CV_OVERRIDE
    {
        if( roots.empty() )
            CV_Error( CV_StsBadArg, "RTrees have not been trained" );

        writeFormat(fs);
        writeParams(fs);

        int k, ntrees = (int)roots.size();

        fs << "ntrees" << ntrees
        << "trees" << "[";

        for( k = 0; k < ntrees; k++ )
        {
            fs << "{";
            writeTree(fs, roots[k]);
            fs << "}";
        }

        fs << "]";
    }

    void readParams( const FileNode& fn ) CV_OVERRIDE
    {
        DTreesImpl::readParams(fn);

        FileNode tparams_node = fn["training_params"];
        // check for old layout
        String bts = (String)(fn["boosting_type"].empty() ?
                         tparams_node["boosting_type"] : fn["boosting_type"]);
        bparams.boostType = (bts == "DiscreteAdaboost" ? Boost::DISCRETE :
                             bts == "RealAdaboost" ? Boost::REAL :
                             bts == "LogitBoost" ? Boost::LOGIT :
                             bts == "GentleAdaboost" ? Boost::GENTLE : -1);
        _isClassifier = bparams.boostType == Boost::DISCRETE;
        // check for old layout
        bparams.weightTrimRate = (double)(fn["weight_trimming_rate"].empty() ?
                                    tparams_node["weight_trimming_rate"] : fn["weight_trimming_rate"]);
    }

    void read( const FileNode& fn ) CV_OVERRIDE
    {
        clear();

        int ntrees = (int)fn["ntrees"];
        readParams(fn);

        FileNode trees_node = fn["trees"];
        FileNodeIterator it = trees_node.begin();
        CV_Assert( ntrees == (int)trees_node.size() );

        for( int treeidx = 0; treeidx < ntrees; treeidx++, ++it )
        {
            FileNode nfn = (*it)["nodes"];
            readTree(nfn);
        }
    }

    BoostTreeParams bparams;
    vector<double> sumResult;
};


class BoostImpl : public Boost
{
public:
    BoostImpl() {}
    virtual ~BoostImpl() {}

    inline int getBoostType() const CV_OVERRIDE { return impl.bparams.boostType; }
    inline void setBoostType(int val) CV_OVERRIDE { impl.bparams.boostType = val; }
    inline int getWeakCount() const CV_OVERRIDE { return impl.bparams.weakCount; }
    inline void setWeakCount(int val) CV_OVERRIDE { impl.bparams.weakCount = val; }
    inline double getWeightTrimRate() const CV_OVERRIDE { return impl.bparams.weightTrimRate; }
    inline void setWeightTrimRate(double val) CV_OVERRIDE { impl.bparams.weightTrimRate = val; }

    inline int getMaxCategories() const CV_OVERRIDE { return impl.params.getMaxCategories(); }
    inline void setMaxCategories(int val) CV_OVERRIDE { impl.params.setMaxCategories(val); }
    inline int getMaxDepth() const CV_OVERRIDE { return impl.params.getMaxDepth(); }
    inline void setMaxDepth(int val) CV_OVERRIDE { impl.params.setMaxDepth(val); }
    inline int getMinSampleCount() const CV_OVERRIDE { return impl.params.getMinSampleCount(); }
    inline void setMinSampleCount(int val) CV_OVERRIDE { impl.params.setMinSampleCount(val); }
    inline int getCVFolds() const CV_OVERRIDE { return impl.params.getCVFolds(); }
    inline void setCVFolds(int val) CV_OVERRIDE { impl.params.setCVFolds(val); }
    inline bool getUseSurrogates() const CV_OVERRIDE { return impl.params.getUseSurrogates(); }
    inline void setUseSurrogates(bool val) CV_OVERRIDE { impl.params.setUseSurrogates(val); }
    inline bool getUse1SERule() const CV_OVERRIDE { return impl.params.getUse1SERule(); }
    inline void setUse1SERule(bool val) CV_OVERRIDE { impl.params.setUse1SERule(val); }
    inline bool getTruncatePrunedTree() const CV_OVERRIDE { return impl.params.getTruncatePrunedTree(); }
    inline void setTruncatePrunedTree(bool val) CV_OVERRIDE { impl.params.setTruncatePrunedTree(val); }
    inline float getRegressionAccuracy() const CV_OVERRIDE { return impl.params.getRegressionAccuracy(); }
    inline void setRegressionAccuracy(float val) CV_OVERRIDE { impl.params.setRegressionAccuracy(val); }
    inline cv::Mat getPriors() const CV_OVERRIDE { return impl.params.getPriors(); }
    inline void setPriors(const cv::Mat& val) CV_OVERRIDE { impl.params.setPriors(val); }

    String getDefaultName() const CV_OVERRIDE { return "opencv_ml_boost"; }

    bool train( const Ptr<TrainData>& trainData, int flags ) CV_OVERRIDE
    {
        CV_Assert(!trainData.empty());
        return impl.train(trainData, flags);
    }

    float predict( InputArray samples, OutputArray results, int flags ) const CV_OVERRIDE
    {
        CV_CheckEQ(samples.cols(), getVarCount(), "");
        return impl.predict(samples, results, flags);
    }

    void write( FileStorage& fs ) const CV_OVERRIDE
    {
        impl.write(fs);
    }

    void read( const FileNode& fn ) CV_OVERRIDE
    {
        impl.read(fn);
    }

    int getVarCount() const CV_OVERRIDE { return impl.getVarCount(); }

    bool isTrained() const CV_OVERRIDE { return impl.isTrained(); }
    bool isClassifier() const CV_OVERRIDE { return impl.isClassifier(); }

    const vector<int>& getRoots() const CV_OVERRIDE { return impl.getRoots(); }
    const vector<Node>& getNodes() const CV_OVERRIDE { return impl.getNodes(); }
    const vector<Split>& getSplits() const CV_OVERRIDE { return impl.getSplits(); }
    const vector<int>& getSubsets() const CV_OVERRIDE { return impl.getSubsets(); }

    DTreesImplForBoost impl;
};


Ptr<Boost> Boost::create()
{
    return makePtr<BoostImpl>();
}

Ptr<Boost> Boost::load(const String& filepath, const String& nodeName)
{
    return Algorithm::load<Boost>(filepath, nodeName);
}

}}

/* End of file. */
