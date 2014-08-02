/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//            Intel License Agreement
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

namespace cv {
namespace ml {

NormalBayesClassifier::Params::Params() {}

class NormalBayesClassifierImpl : public NormalBayesClassifier
{
public:
    NormalBayesClassifierImpl()
    {
        nallvars = 0;
    }

    void setParams(const Params&) {}
    Params getParams() const { return Params(); }

    bool train( const Ptr<TrainData>& trainData, int flags )
    {
        const float min_variation = FLT_EPSILON;
        Mat responses = trainData->getNormCatResponses();
        Mat __cls_labels = trainData->getClassLabels();
        Mat __var_idx = trainData->getVarIdx();
        Mat samples = trainData->getTrainSamples();
        int nclasses = (int)__cls_labels.total();

        int nvars = trainData->getNVars();
        int s, c1, c2, cls;

        int __nallvars = trainData->getNAllVars();
        bool update = (flags & UPDATE_MODEL) != 0;

        if( !update )
        {
            nallvars = __nallvars;
            count.resize(nclasses);
            sum.resize(nclasses);
            productsum.resize(nclasses);
            avg.resize(nclasses);
            inv_eigen_values.resize(nclasses);
            cov_rotate_mats.resize(nclasses);

            for( cls = 0; cls < nclasses; cls++ )
            {
                count[cls]            = Mat::zeros( 1, nvars, CV_32SC1 );
                sum[cls]              = Mat::zeros( 1, nvars, CV_64FC1 );
                productsum[cls]       = Mat::zeros( nvars, nvars, CV_64FC1 );
                avg[cls]              = Mat::zeros( 1, nvars, CV_64FC1 );
                inv_eigen_values[cls] = Mat::zeros( 1, nvars, CV_64FC1 );
                cov_rotate_mats[cls]  = Mat::zeros( nvars, nvars, CV_64FC1 );
            }

            var_idx = __var_idx;
            cls_labels = __cls_labels;

            c.create(1, nclasses, CV_64FC1);
        }
        else
        {
            // check that the new training data has the same dimensionality etc.
            if( nallvars != __nallvars ||
                var_idx.size() != __var_idx.size() ||
                norm(var_idx, __var_idx, NORM_INF) != 0 ||
                cls_labels.size() != __cls_labels.size() ||
                norm(cls_labels, __cls_labels, NORM_INF) != 0 )
                CV_Error( CV_StsBadArg,
                "The new training data is inconsistent with the original training data; varIdx and the class labels should be the same" );
        }

        Mat cov( nvars, nvars, CV_64FC1 );
        int nsamples = samples.rows;

        // process train data (count, sum , productsum)
        for( s = 0; s < nsamples; s++ )
        {
            cls = responses.at<int>(s);
            int* count_data = count[cls].ptr<int>();
            double* sum_data = sum[cls].ptr<double>();
            double* prod_data = productsum[cls].ptr<double>();
            const float* train_vec = samples.ptr<float>(s);

            for( c1 = 0; c1 < nvars; c1++, prod_data += nvars )
            {
                double val1 = train_vec[c1];
                sum_data[c1] += val1;
                count_data[c1]++;
                for( c2 = c1; c2 < nvars; c2++ )
                    prod_data[c2] += train_vec[c2]*val1;
            }
        }

        Mat vt;

        // calculate avg, covariance matrix, c
        for( cls = 0; cls < nclasses; cls++ )
        {
            double det = 1;
            int i, j;
            Mat& w = inv_eigen_values[cls];
            int* count_data = count[cls].ptr<int>();
            double* avg_data = avg[cls].ptr<double>();
            double* sum1 = sum[cls].ptr<double>();

            completeSymm(productsum[cls], 0);

            for( j = 0; j < nvars; j++ )
            {
                int n = count_data[j];
                avg_data[j] = n ? sum1[j] / n : 0.;
            }

            count_data = count[cls].ptr<int>();
            avg_data = avg[cls].ptr<double>();
            sum1 = sum[cls].ptr<double>();

            for( i = 0; i < nvars; i++ )
            {
                double* avg2_data = avg[cls].ptr<double>();
                double* sum2 = sum[cls].ptr<double>();
                double* prod_data = productsum[cls].ptr<double>(i);
                double* cov_data = cov.ptr<double>(i);
                double s1val = sum1[i];
                double avg1 = avg_data[i];
                int _count = count_data[i];

                for( j = 0; j <= i; j++ )
                {
                    double avg2 = avg2_data[j];
                    double cov_val = prod_data[j] - avg1 * sum2[j] - avg2 * s1val + avg1 * avg2 * _count;
                    cov_val = (_count > 1) ? cov_val / (_count - 1) : cov_val;
                    cov_data[j] = cov_val;
                }
            }

            completeSymm( cov, 1 );

            SVD::compute(cov, w, cov_rotate_mats[cls], noArray());
            transpose(cov_rotate_mats[cls], cov_rotate_mats[cls]);
            cv::max(w, min_variation, w);
            for( j = 0; j < nvars; j++ )
                det *= w.at<double>(j);

            divide(1., w, w);
            c.at<double>(cls) = det > 0 ? log(det) : -700;
        }

        return true;
    }

    class NBPredictBody : public ParallelLoopBody
    {
    public:
        NBPredictBody( const Mat& _c, const vector<Mat>& _cov_rotate_mats,
                       const vector<Mat>& _inv_eigen_values,
                       const vector<Mat>& _avg,
                       const Mat& _samples, const Mat& _vidx, const Mat& _cls_labels,
                       Mat& _results, Mat& _results_prob, bool _rawOutput )
        {
            c = &_c;
            cov_rotate_mats = &_cov_rotate_mats;
            inv_eigen_values = &_inv_eigen_values;
            avg = &_avg;
            samples = &_samples;
            vidx = &_vidx;
            cls_labels = &_cls_labels;
            results = &_results;
            results_prob = _results_prob.data ? &_results_prob : 0;
            rawOutput = _rawOutput;
        }

        const Mat* c;
        const vector<Mat>* cov_rotate_mats;
        const vector<Mat>* inv_eigen_values;
        const vector<Mat>* avg;
        const Mat* samples;
        const Mat* vidx;
        const Mat* cls_labels;

        Mat* results_prob;
        Mat* results;
        float* value;
        bool rawOutput;

        void operator()( const Range& range ) const
        {
            int cls = -1;
            int rtype = 0, rptype = 0;
            size_t rstep = 0, rpstep = 0;
            int nclasses = (int)cls_labels->total();
            int nvars = avg->at(0).cols;
            double probability = 0;
            const int* vptr = vidx && !vidx->empty() ? vidx->ptr<int>() : 0;

            if (results)
            {
                rtype = results->type();
                rstep = results->isContinuous() ? 1 : results->step/results->elemSize();
            }
            if (results_prob)
            {
                rptype = results_prob->type();
                rpstep = results_prob->isContinuous() ? 1 : results_prob->step/results_prob->elemSize();
            }
            // allocate memory and initializing headers for calculating
            cv::AutoBuffer<double> _buffer(nvars*2);
            double* _diffin = _buffer;
            double* _diffout = _buffer + nvars;
            Mat diffin( 1, nvars, CV_64FC1, _diffin );
            Mat diffout( 1, nvars, CV_64FC1, _diffout );

            for(int k = range.start; k < range.end; k++ )
            {
                double opt = FLT_MAX;

                for(int i = 0; i < nclasses; i++ )
                {
                    double cur = c->at<double>(i);
                    const Mat& u = cov_rotate_mats->at(i);
                    const Mat& w = inv_eigen_values->at(i);

                    const double* avg_data = avg->at(i).ptr<double>();
                    const float* x = samples->ptr<float>(k);

                    // cov = u w u'  -->  cov^(-1) = u w^(-1) u'
                    for(int j = 0; j < nvars; j++ )
                        _diffin[j] = avg_data[j] - x[vptr ? vptr[j] : j];

                    gemm( diffin, u, 1, noArray(), 0, diffout, GEMM_2_T );
                    for(int j = 0; j < nvars; j++ )
                    {
                        double d = _diffout[j];
                        cur += d*d*w.ptr<double>()[j];
                    }

                    if( cur < opt )
                    {
                        cls = i;
                        opt = cur;
                    }
                    probability = exp( -0.5 * cur );

                    if( results_prob )
                    {
                        if ( rptype == CV_32FC1 )
                            results_prob->ptr<float>()[k*rpstep + i] = (float)probability;
                        else
                            results_prob->ptr<double>()[k*rpstep + i] = probability;
                    }
                }

                int ival = rawOutput ? cls : cls_labels->at<int>(cls);
                if( results )
                {
                    if( rtype == CV_32SC1 )
                        results->ptr<int>()[k*rstep] = ival;
                    else
                        results->ptr<float>()[k*rstep] = (float)ival;
                }
            }
        }
    };

    float predict( InputArray _samples, OutputArray _results, int flags ) const
    {
        return predictProb(_samples, _results, noArray(), flags);
    }

    float predictProb( InputArray _samples, OutputArray _results, OutputArray _resultsProb, int flags ) const
    {
        int value=0;
        Mat samples = _samples.getMat(), results, resultsProb;
        int nsamples = samples.rows, nclasses = (int)cls_labels.total();
        bool rawOutput = (flags & RAW_OUTPUT) != 0;

        if( samples.type() != CV_32F || samples.cols != nallvars )
            CV_Error( CV_StsBadArg,
                     "The input samples must be 32f matrix with the number of columns = nallvars" );

        if( samples.rows > 1 && _results.needed() )
            CV_Error( CV_StsNullPtr,
                     "When the number of input samples is >1, the output vector of results must be passed" );

        if( _results.needed() )
        {
            _results.create(nsamples, 1, CV_32S);
            results = _results.getMat();
        }
        else
            results = Mat(1, 1, CV_32S, &value);

        if( _resultsProb.needed() )
        {
            _resultsProb.create(nsamples, nclasses, CV_32F);
            resultsProb = _resultsProb.getMat();
        }

        cv::parallel_for_(cv::Range(0, nsamples),
                          NBPredictBody(c, cov_rotate_mats, inv_eigen_values, avg, samples,
                                       var_idx, cls_labels, results, resultsProb, rawOutput));

        return (float)value;
    }

    void write( FileStorage& fs ) const
    {
        int nclasses = (int)cls_labels.total(), i;

        fs << "var_count" << (var_idx.empty() ? nallvars : (int)var_idx.total());
        fs << "var_all" << nallvars;

        if( !var_idx.empty() )
            fs << "var_idx" << var_idx;
        fs << "cls_labels" << cls_labels;

        fs << "count" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << count[i];

        fs << "]" << "sum" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << sum[i];

        fs << "]" << "productsum" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << productsum[i];

        fs << "]" << "avg" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << avg[i];

        fs << "]" << "inv_eigen_values" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << inv_eigen_values[i];

        fs << "]" << "cov_rotate_mats" << "[";
        for( i = 0; i < nclasses; i++ )
            fs << cov_rotate_mats[i];

        fs << "]";

        fs << "c" << c;
    }

    void read( const FileNode& fn )
    {
        clear();

        fn["var_all"] >> nallvars;

        if( nallvars <= 0 )
            CV_Error( CV_StsParseError,
                     "The field \"var_count\" of NBayes classifier is missing or non-positive" );

        fn["var_idx"] >> var_idx;
        fn["cls_labels"] >> cls_labels;

        int nclasses = (int)cls_labels.total(), i;

        if( cls_labels.empty() || nclasses < 1 )
            CV_Error( CV_StsParseError, "No or invalid \"cls_labels\" in NBayes classifier" );

        FileNodeIterator
            count_it = fn["count"].begin(),
            sum_it = fn["sum"].begin(),
            productsum_it = fn["productsum"].begin(),
            avg_it = fn["avg"].begin(),
            inv_eigen_values_it = fn["inv_eigen_values"].begin(),
            cov_rotate_mats_it = fn["cov_rotate_mats"].begin();

        count.resize(nclasses);
        sum.resize(nclasses);
        productsum.resize(nclasses);
        avg.resize(nclasses);
        inv_eigen_values.resize(nclasses);
        cov_rotate_mats.resize(nclasses);

        for( i = 0; i < nclasses; i++, ++count_it, ++sum_it, ++productsum_it, ++avg_it,
                                    ++inv_eigen_values_it, ++cov_rotate_mats_it )
        {
            *count_it >> count[i];
            *sum_it >> sum[i];
            *productsum_it >> productsum[i];
            *avg_it >> avg[i];
            *inv_eigen_values_it >> inv_eigen_values[i];
            *cov_rotate_mats_it >> cov_rotate_mats[i];
        }

        fn["c"] >> c;
    }

    void clear()
    {
        count.clear();
        sum.clear();
        productsum.clear();
        avg.clear();
        inv_eigen_values.clear();
        cov_rotate_mats.clear();

        var_idx.release();
        cls_labels.release();
        c.release();
        nallvars = 0;
    }

    bool isTrained() const { return !avg.empty(); }
    bool isClassifier() const { return true; }
    int getVarCount() const { return nallvars; }
    String getDefaultModelName() const { return "opencv_ml_nbayes"; }

    int nallvars;
    Mat var_idx, cls_labels, c;
    vector<Mat> count, sum, productsum, avg, inv_eigen_values, cov_rotate_mats;
};


Ptr<NormalBayesClassifier> NormalBayesClassifier::create(const Params&)
{
    Ptr<NormalBayesClassifierImpl> p = makePtr<NormalBayesClassifierImpl>();
    return p;
}

}
}

/* End of file. */
