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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_ML_HPP__
#define __OPENCV_ML_HPP__

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

#ifdef __cplusplus

#include <float.h>
#include <map>
#include <iostream>

namespace cv
{

namespace ml
{

/* Variable type */
enum
{
    VAR_NUMERICAL    =0,
    VAR_ORDERED      =0,
    VAR_CATEGORICAL  =1
};

enum
{
    TEST_ERROR = 0,
    TRAIN_ERROR = 1
};

enum
{
    ROW_SAMPLE = 0,
    COL_SAMPLE = 1
};

class CV_EXPORTS_W_MAP ParamGrid
{
public:
    ParamGrid();
    ParamGrid(double _minVal, double _maxVal, double _logStep);

    CV_PROP_RW double minVal;
    CV_PROP_RW double maxVal;
    CV_PROP_RW double logStep;
};


class CV_EXPORTS TrainData
{
public:
    static inline float missingValue() { return FLT_MAX; }
    virtual ~TrainData();

    virtual int getLayout() const = 0;
    virtual int getNTrainSamples() const = 0;
    virtual int getNTestSamples() const = 0;
    virtual int getNSamples() const = 0;
    virtual int getNVars() const = 0;
    virtual int getNAllVars() const = 0;

    virtual void getSample(InputArray varIdx, int sidx, float* buf) const = 0;
    virtual Mat getSamples() const = 0;
    virtual Mat getMissing() const = 0;
    virtual Mat getTrainSamples(int layout=ROW_SAMPLE,
                                bool compressSamples=true,
                                bool compressVars=true) const = 0;
    virtual Mat getTrainResponses() const = 0;
    virtual Mat getTrainNormCatResponses() const = 0;
    virtual Mat getTestResponses() const = 0;
    virtual Mat getTestNormCatResponses() const = 0;
    virtual Mat getResponses() const = 0;
    virtual Mat getNormCatResponses() const = 0;
    virtual Mat getSampleWeights() const = 0;
    virtual Mat getTrainSampleWeights() const = 0;
    virtual Mat getTestSampleWeights() const = 0;
    virtual Mat getVarIdx() const = 0;
    virtual Mat getVarType() const = 0;
    virtual int getResponseType() const = 0;
    virtual Mat getTrainSampleIdx() const = 0;
    virtual Mat getTestSampleIdx() const = 0;
    virtual void getValues(int vi, InputArray sidx, float* values) const = 0;
    virtual void getNormCatValues(int vi, InputArray sidx, int* values) const = 0;
    virtual Mat getDefaultSubstValues() const = 0;

    virtual int getCatCount(int vi) const = 0;
    virtual Mat getClassLabels() const = 0;

    virtual Mat getCatOfs() const = 0;
    virtual Mat getCatMap() const = 0;

    virtual void setTrainTestSplit(int count, bool shuffle=true) = 0;
    virtual void setTrainTestSplitRatio(double ratio, bool shuffle=true) = 0;
    virtual void shuffleTrainTest() = 0;

    static Mat getSubVector(const Mat& vec, const Mat& idx);
    static Ptr<TrainData> loadFromCSV(const String& filename,
                                      int headerLineCount,
                                      int responseStartIdx=-1,
                                      int responseEndIdx=-1,
                                      const String& varTypeSpec=String(),
                                      char delimiter=',',
                                      char missch='?');
    static Ptr<TrainData> create(InputArray samples, int layout, InputArray responses,
                                 InputArray varIdx=noArray(), InputArray sampleIdx=noArray(),
                                 InputArray sampleWeights=noArray(), InputArray varType=noArray());
};


class CV_EXPORTS_W StatModel : public Algorithm
{
public:
    enum { UPDATE_MODEL = 1, RAW_OUTPUT=1, COMPRESSED_INPUT=2, PREPROCESSED_INPUT=4 };
    virtual void clear();

    virtual int getVarCount() const = 0;

    virtual bool isTrained() const = 0;
    virtual bool isClassifier() const = 0;

    virtual bool train( const Ptr<TrainData>& trainData, int flags=0 );
    virtual bool train( InputArray samples, int layout, InputArray responses );
    virtual float calcError( const Ptr<TrainData>& data, bool test, OutputArray resp ) const;
    virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

    template<typename _Tp> static Ptr<_Tp> load(const String& filename)
    {
        FileStorage fs(filename, FileStorage::READ);
        Ptr<_Tp> model = _Tp::create();
        model->read(fs.getFirstTopLevelNode());
        return model->isTrained() ? model : Ptr<_Tp>();
    }

    template<typename _Tp> static Ptr<_Tp> train(const Ptr<TrainData>& data, const typename _Tp::Params& p, int flags=0)
    {
        Ptr<_Tp> model = _Tp::create(p);
        return !model.empty() && model->train(data, flags) ? model : Ptr<_Tp>();
    }

    template<typename _Tp> static Ptr<_Tp> train(InputArray samples, int layout, InputArray responses,
                                                 const typename _Tp::Params& p, int flags=0)
    {
        Ptr<_Tp> model = _Tp::create(p);
        return !model.empty() && model->train(TrainData::create(samples, layout, responses), flags) ? model : Ptr<_Tp>();
    }

    virtual void save(const String& filename) const;
    virtual String getDefaultModelName() const = 0;
};

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

/* The structure, representing the grid range of statmodel parameters.
   It is used for optimizing statmodel accuracy by varying model parameters,
   the accuracy estimate being computed by cross-validation.
   The grid is logarithmic, so <step> must be greater then 1. */

class CV_EXPORTS_W NormalBayesClassifier : public StatModel
{
public:
    class CV_EXPORTS_W Params
    {
    public:
        Params();
    };
    virtual float predictProb( InputArray inputs, OutputArray outputs,
                               OutputArray outputProbs, int flags=0 ) const = 0;
    virtual void setParams(const Params& params) = 0;
    virtual Params getParams() const = 0;

    static Ptr<NormalBayesClassifier> create(const Params& params=Params());
};

/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

// k Nearest Neighbors
class CV_EXPORTS_W KNearest : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params(int defaultK=10, bool isclassifier=true);

        CV_PROP_RW int defaultK;
        CV_PROP_RW bool isclassifier;
    };
    virtual void setParams(const Params& p) = 0;
    virtual Params getParams() const = 0;
    virtual float findNearest( InputArray samples, int k,
                               OutputArray results,
                               OutputArray neighborResponses=noArray(),
                               OutputArray dist=noArray() ) const = 0;
    static Ptr<KNearest> create(const Params& params=Params());
};

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

// SVM model
class CV_EXPORTS_W SVM : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        Params( int svm_type, int kernel_type,
                double degree, double gamma, double coef0,
                double Cvalue, double nu, double p,
                const Mat& classWeights, TermCriteria termCrit );

        CV_PROP_RW int         svmType;
        CV_PROP_RW int         kernelType;
        CV_PROP_RW double      gamma, coef0, degree;

        CV_PROP_RW double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        CV_PROP_RW double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        CV_PROP_RW double      p; // for CV_SVM_EPS_SVR
        CV_PROP_RW Mat         classWeights; // for CV_SVM_C_SVC
        CV_PROP_RW TermCriteria termCrit; // termination criteria
    };

    class CV_EXPORTS Kernel : public Algorithm
    {
    public:
        virtual int getType() const = 0;
        virtual void calc( int vcount, int n, const float* vecs, const float* another, float* results ) = 0;
    };

    // SVM type
    enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

    // SVM kernel type
    enum { CUSTOM=-1, LINEAR=0, POLY=1, RBF=2, SIGMOID=3, CHI2=4, INTER=5 };

    // SVM params type
    enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

    virtual bool trainAuto( const Ptr<TrainData>& data, int kFold = 10,
                    ParamGrid Cgrid = SVM::getDefaultGrid(SVM::C),
                    ParamGrid gammaGrid  = SVM::getDefaultGrid(SVM::GAMMA),
                    ParamGrid pGrid      = SVM::getDefaultGrid(SVM::P),
                    ParamGrid nuGrid     = SVM::getDefaultGrid(SVM::NU),
                    ParamGrid coeffGrid  = SVM::getDefaultGrid(SVM::COEF),
                    ParamGrid degreeGrid = SVM::getDefaultGrid(SVM::DEGREE),
                    bool balanced=false) = 0;

    CV_WRAP virtual Mat getSupportVectors() const = 0;

    virtual void setParams(const Params& p, const Ptr<Kernel>& customKernel=Ptr<Kernel>()) = 0;
    virtual Params getParams() const = 0;
    virtual Ptr<Kernel> getKernel() const = 0;
    virtual double getDecisionFunction(int i, OutputArray alpha, OutputArray svidx) const = 0;

    static ParamGrid getDefaultGrid( int param_id );
    static Ptr<SVM> create(const Params& p=Params(), const Ptr<Kernel>& customKernel=Ptr<Kernel>());
};

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/
class CV_EXPORTS_W EM : public StatModel
{
public:
    // Type of covariation matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    class CV_EXPORTS_W_MAP Params
    {
    public:
        explicit Params(int nclusters=DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL,
                        const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                                                  EM::DEFAULT_MAX_ITERS, 1e-6));
        CV_PROP_RW int nclusters;
        CV_PROP_RW int covMatType;
        CV_PROP_RW TermCriteria termCrit;
    };

    virtual void setParams(const Params& p) = 0;
    virtual Params getParams() const = 0;
    virtual Mat getWeights() const = 0;
    virtual Mat getMeans() const = 0;
    virtual void getCovs(std::vector<Mat>& covs) const = 0;

    CV_WRAP virtual Vec2d predict2(InputArray sample, OutputArray probs) const = 0;

    virtual bool train( const Ptr<TrainData>& trainData, int flags=0 ) = 0;

    static Ptr<EM> train(InputArray samples,
                          OutputArray logLikelihoods=noArray(),
                          OutputArray labels=noArray(),
                          OutputArray probs=noArray(),
                          const Params& params=Params());

    static Ptr<EM> train_startWithE(InputArray samples, InputArray means0,
                                     InputArray covs0=noArray(),
                                     InputArray weights0=noArray(),
                                     OutputArray logLikelihoods=noArray(),
                                     OutputArray labels=noArray(),
                                     OutputArray probs=noArray(),
                                     const Params& params=Params());

    static Ptr<EM> train_startWithM(InputArray samples, InputArray probs0,
                                     OutputArray logLikelihoods=noArray(),
                                     OutputArray labels=noArray(),
                                     OutputArray probs=noArray(),
                                     const Params& params=Params());
    static Ptr<EM> create(const Params& params=Params());
};


/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/

class CV_EXPORTS_W DTrees : public StatModel
{
public:
    enum { PREDICT_AUTO=0, PREDICT_SUM=(1<<8), PREDICT_MAX_VOTE=(2<<8), PREDICT_MASK=(3<<8) };

    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        Params( int maxDepth, int minSampleCount,
               double regressionAccuracy, bool useSurrogates,
               int maxCategories, int CVFolds,
               bool use1SERule, bool truncatePrunedTree,
               const Mat& priors );

        CV_PROP_RW int   maxCategories;
        CV_PROP_RW int   maxDepth;
        CV_PROP_RW int   minSampleCount;
        CV_PROP_RW int   CVFolds;
        CV_PROP_RW bool  useSurrogates;
        CV_PROP_RW bool  use1SERule;
        CV_PROP_RW bool  truncatePrunedTree;
        CV_PROP_RW float regressionAccuracy;
        CV_PROP_RW Mat priors;
    };

    class CV_EXPORTS Node
    {
    public:
        Node();
        double value;
        int classIdx;

        int parent;
        int left;
        int right;
        int defaultDir;

        int split;
    };

    class CV_EXPORTS Split
    {
    public:
        Split();
        int varIdx;
        bool inversed;
        float quality;
        int next;
        float c;
        int subsetOfs;
    };

    virtual void setDParams(const Params& p);
    virtual Params getDParams() const;

    virtual const std::vector<int>& getRoots() const = 0;
    virtual const std::vector<Node>& getNodes() const = 0;
    virtual const std::vector<Split>& getSplits() const = 0;
    virtual const std::vector<int>& getSubsets() const = 0;

    static Ptr<DTrees> create(const Params& params=Params());
};

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

class CV_EXPORTS_W RTrees : public DTrees
{
public:
    class CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
    public:
        Params();
        Params( int maxDepth, int minSampleCount,
                double regressionAccuracy, bool useSurrogates,
                int maxCategories, const Mat& priors,
                bool calcVarImportance, int nactiveVars,
                TermCriteria termCrit );

        CV_PROP_RW bool calcVarImportance; // true <=> RF processes variable importance
        CV_PROP_RW int nactiveVars;
        CV_PROP_RW TermCriteria termCrit;
    };

    virtual void setRParams(const Params& p) = 0;
    virtual Params getRParams() const = 0;

    virtual Mat getVarImportance() const = 0;

    static Ptr<RTrees> create(const Params& params=Params());
};

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

class CV_EXPORTS_W Boost : public DTrees
{
public:
    class CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
    public:
        CV_PROP_RW int boostType;
        CV_PROP_RW int weakCount;
        CV_PROP_RW double weightTrimRate;

        Params();
        Params( int boostType, int weakCount, double weightTrimRate,
                int maxDepth, bool useSurrogates, const Mat& priors );
    };

    // Boosting type
    enum { DISCRETE=0, REAL=1, LOGIT=2, GENTLE=3 };

    virtual Params getBParams() const = 0;
    virtual void setBParams(const Params& p) = 0;

    static Ptr<Boost> create(const Params& params=Params());
};

/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

/*class CV_EXPORTS_W GBTrees : public DTrees
{
public:
    struct CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
        CV_PROP_RW int weakCount;
        CV_PROP_RW int lossFunctionType;
        CV_PROP_RW float subsamplePortion;
        CV_PROP_RW float shrinkage;

        Params();
        Params( int lossFunctionType, int weakCount, float shrinkage,
                float subsamplePortion, int maxDepth, bool useSurrogates );
    };

    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};

    virtual void setK(int k) = 0;

    virtual float predictSerial( InputArray samples,
                                 OutputArray weakResponses, int flags) const = 0;

    static Ptr<GBTrees> create(const Params& p);
};*/

/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

class CV_EXPORTS_W ANN_MLP : public StatModel
{
public:
    struct CV_EXPORTS_W_MAP Params
    {
        Params();
        Params( const Mat& layerSizes, int activateFunc, double fparam1, double fparam2,
                TermCriteria termCrit, int trainMethod, double param1, double param2=0 );

        enum { BACKPROP=0, RPROP=1 };

        CV_PROP_RW Mat layerSizes;
        CV_PROP_RW int activateFunc;
        CV_PROP_RW double fparam1;
        CV_PROP_RW double fparam2;

        CV_PROP_RW TermCriteria termCrit;
        CV_PROP_RW int trainMethod;

        // backpropagation parameters
        CV_PROP_RW double bpDWScale, bpMomentScale;

        // rprop parameters
        CV_PROP_RW double rpDW0, rpDWPlus, rpDWMinus, rpDWMin, rpDWMax;
    };

    // possible activation functions
    enum { IDENTITY = 0, SIGMOID_SYM = 1, GAUSSIAN = 2 };

    // available training flags
    enum { UPDATE_WEIGHTS = 1, NO_INPUT_SCALE = 2, NO_OUTPUT_SCALE = 4 };

    virtual Mat getWeights(int layerIdx) const = 0;
    virtual void setParams(const Params& p) = 0;
    virtual Params getParams() const = 0;

    static Ptr<ANN_MLP> create(const Params& params=Params());
};

/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/* Generates <sample> from multivariate normal distribution, where <mean> - is an
   average row vector, <cov> - symmetric covariation matrix */
CV_EXPORTS void randMVNormal( InputArray mean, InputArray cov, int nsamples, OutputArray samples);

/* Generates sample from gaussian mixture distribution */
CV_EXPORTS void randGaussMixture( InputArray means, InputArray covs, InputArray weights,
                                  int nsamples, OutputArray samples, OutputArray sampClasses );

/* creates test set */
CV_EXPORTS void createConcentricSpheresTestSet( int nsamples, int nfeatures, int nclasses,
                                                OutputArray samples, OutputArray responses);

}
}

#endif // __cplusplus
#endif // __OPENCV_ML_HPP__

/* End of file. */
