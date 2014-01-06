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

/*
 * File cvclassifier.h
 *
 * Classifier types
 */

#ifndef _CVCLASSIFIER_H_
#define _CVCLASSIFIER_H_

#include <cmath>
#include "cxcore.h"

#define CV_BOOST_API

/* Convert matrix to vector */
#define CV_MAT2VEC( mat, vdata, vstep, num )       \
    assert( (mat).rows == 1 || (mat).cols == 1 );  \
    (vdata) = ((mat).data.ptr);                    \
    if( (mat).rows == 1 )                          \
    {                                              \
        (vstep) = CV_ELEM_SIZE( (mat).type );      \
        (num) = (mat).cols;                        \
    }                                              \
    else                                           \
    {                                              \
        (vstep) = (mat).step;                      \
        (num) = (mat).rows;                        \
    }

/* Set up <sample> matrix header to be <num> sample of <trainData> samples matrix */
#define CV_GET_SAMPLE( trainData, tdflags, num, sample )                                 \
if( CV_IS_ROW_SAMPLE( tdflags ) )                                                        \
{                                                                                        \
    cvInitMatHeader( &(sample), 1, (trainData).cols,                                     \
                     CV_MAT_TYPE( (trainData).type ),                                    \
                     ((trainData).data.ptr + (num) * (trainData).step),                  \
                     (trainData).step );                                                 \
}                                                                                        \
else                                                                                     \
{                                                                                        \
    cvInitMatHeader( &(sample), (trainData).rows, 1,                                     \
                     CV_MAT_TYPE( (trainData).type ),                                    \
                     ((trainData).data.ptr + (num) * CV_ELEM_SIZE( (trainData).type )),  \
                     (trainData).step );                                                 \
}

#define CV_GET_SAMPLE_STEP( trainData, tdflags, sstep )                                  \
(sstep) = ( ( CV_IS_ROW_SAMPLE( tdflags ) )                                              \
           ? (trainData).step : CV_ELEM_SIZE( (trainData).type ) );


#define CV_LOGRATIO_THRESHOLD 0.00001F

/* log( val / (1 - val ) ) */
CV_INLINE float cvLogRatio( float val );

CV_INLINE float cvLogRatio( float val )
{
    float tval;

    tval = MAX(CV_LOGRATIO_THRESHOLD, MIN( 1.0F - CV_LOGRATIO_THRESHOLD, (val) ));
    return logf( tval / (1.0F - tval) );
}


/* flags values for classifier consturctor flags parameter */

/* each trainData matrix column is a sample */
#define CV_COL_SAMPLE 0

/* each trainData matrix row is a sample */
#define CV_ROW_SAMPLE 1

#define CV_IS_ROW_SAMPLE( flags ) ( ( flags ) & CV_ROW_SAMPLE )

/* Classifier supports tune function */
#define CV_TUNABLE    (1 << 1)

#define CV_IS_TUNABLE( flags ) ( (flags) & CV_TUNABLE )


/* classifier fields common to all classifiers */
#define CV_CLASSIFIER_FIELDS()                                                           \
    int flags;                                                                           \
    float(*eval)( struct CvClassifier*, CvMat* );                                        \
    void (*tune)( struct CvClassifier*, CvMat*, int flags, CvMat*, CvMat*, CvMat*,       \
                  CvMat*, CvMat* );                                                      \
    int  (*save)( struct CvClassifier*, const char* file_name );                         \
    void (*release)( struct CvClassifier** );

typedef struct CvClassifier
{
    CV_CLASSIFIER_FIELDS()
} CvClassifier;

#define CV_CLASSIFIER_TRAIN_PARAM_FIELDS()
typedef struct CvClassifierTrainParams
{
    CV_CLASSIFIER_TRAIN_PARAM_FIELDS()
} CvClassifierTrainParams;


/*
 Common classifier constructor:
 CvClassifier* cvCreateMyClassifier( CvMat* trainData,
                     int flags,
                     CvMat* trainClasses,
                     CvMat* typeMask,
                      CvMat* missedMeasurementsMask CV_DEFAULT(0),
                      CvCompIdx* compIdx CV_DEFAULT(0),
                      CvMat* sampleIdx CV_DEFAULT(0),
                      CvMat* weights CV_DEFAULT(0),
                      CvClassifierTrainParams* trainParams CV_DEFAULT(0)
                    )

 */

typedef CvClassifier* (*CvClassifierConstructor)( CvMat*, int, CvMat*, CvMat*, CvMat*,
                                                  CvMat*, CvMat*, CvMat*,
                                                  CvClassifierTrainParams* );

typedef enum CvStumpType
{
    CV_CLASSIFICATION       = 0,
    CV_CLASSIFICATION_CLASS = 1,
    CV_REGRESSION           = 2
} CvStumpType;

typedef enum CvStumpError
{
    CV_MISCLASSIFICATION = 0,
    CV_GINI              = 1,
    CV_ENTROPY           = 2,
    CV_SQUARE            = 3
} CvStumpError;


typedef struct CvStumpTrainParams
{
    CV_CLASSIFIER_TRAIN_PARAM_FIELDS()
    CvStumpType  type;
    CvStumpError error;
} CvStumpTrainParams;

typedef struct CvMTStumpTrainParams
{
    CV_CLASSIFIER_TRAIN_PARAM_FIELDS()
    CvStumpType  type;
    CvStumpError error;
    int portion; /* number of components calculated in each thread */
    int numcomp; /* total number of components */

    /* callback which fills <mat> with components [first, first+num[ */
    void (*getTrainData)( CvMat* mat, CvMat* sampleIdx, CvMat* compIdx,
                          int first, int num, void* userdata );
    CvMat* sortedIdx; /* presorted samples indices */
    void* userdata; /* passed to callback */
} CvMTStumpTrainParams;

typedef struct CvStumpClassifier
{
    CV_CLASSIFIER_FIELDS()
    int compidx;

    float lerror; /* impurity of the right node */
    float rerror; /* impurity of the left  node */

    float threshold;
    float left;
    float right;
} CvStumpClassifier;

typedef struct CvCARTTrainParams
{
    CV_CLASSIFIER_TRAIN_PARAM_FIELDS()
    /* desired number of internal nodes */
    int count;
    CvClassifierTrainParams* stumpTrainParams;
    CvClassifierConstructor  stumpConstructor;

    /*
     * Split sample indices <idx>
     * on the "left" indices <left> and "right" indices <right>
     * according to samples components <compidx> values and <threshold>.
     *
     * NOTE: Matrices <left> and <right> must be allocated using cvCreateMat function
     *   since they are freed using cvReleaseMat function
     *
     * If it is NULL then the default implementation which evaluates training
     * samples from <trainData> passed to classifier constructor is used
     */
    void (*splitIdx)( int compidx, float threshold,
                      CvMat* idx, CvMat** left, CvMat** right,
                      void* userdata );
    void* userdata;
} CvCARTTrainParams;

typedef struct CvCARTClassifier
{
    CV_CLASSIFIER_FIELDS()
    /* number of internal nodes */
    int count;

    /* internal nodes (each array of <count> elements) */
    int* compidx;
    float* threshold;
    int* left;
    int* right;

    /* leaves (array of <count>+1 elements) */
    float* val;
} CvCARTClassifier;

CV_BOOST_API
void cvGetSortedIndices( CvMat* val, CvMat* idx, int sortcols CV_DEFAULT( 0 ) );

CV_BOOST_API
void cvReleaseStumpClassifier( CvClassifier** classifier );

CV_BOOST_API
float cvEvalStumpClassifier( CvClassifier* classifier, CvMat* sample );

CV_BOOST_API
CvClassifier* cvCreateStumpClassifier( CvMat* trainData,
                                       int flags,
                                       CvMat* trainClasses,
                                       CvMat* typeMask,
                                       CvMat* missedMeasurementsMask CV_DEFAULT(0),
                                       CvMat* compIdx CV_DEFAULT(0),
                                       CvMat* sampleIdx CV_DEFAULT(0),
                                       CvMat* weights CV_DEFAULT(0),
                                       CvClassifierTrainParams* trainParams CV_DEFAULT(0) );

/*
 * cvCreateMTStumpClassifier
 *
 * Multithreaded stump classifier constructor
 * Includes huge train data support through callback function
 */
CV_BOOST_API
CvClassifier* cvCreateMTStumpClassifier( CvMat* trainData,
                                         int flags,
                                         CvMat* trainClasses,
                                         CvMat* typeMask,
                                         CvMat* missedMeasurementsMask,
                                         CvMat* compIdx,
                                         CvMat* sampleIdx,
                                         CvMat* weights,
                                         CvClassifierTrainParams* trainParams );

/*
 * cvCreateCARTClassifier
 *
 * CART classifier constructor
 */
CV_BOOST_API
CvClassifier* cvCreateCARTClassifier( CvMat* trainData,
                                      int flags,
                                      CvMat* trainClasses,
                                      CvMat* typeMask,
                                      CvMat* missedMeasurementsMask,
                                      CvMat* compIdx,
                                      CvMat* sampleIdx,
                                      CvMat* weights,
                                      CvClassifierTrainParams* trainParams );

CV_BOOST_API
void cvReleaseCARTClassifier( CvClassifier** classifier );

CV_BOOST_API
float cvEvalCARTClassifier( CvClassifier* classifier, CvMat* sample );

/****************************************************************************************\
*                                        Boosting                                        *
\****************************************************************************************/

/*
 * CvBoostType
 *
 * The CvBoostType enumeration specifies the boosting type.
 *
 * Remarks
 *   Four different boosting variants for 2 class classification problems are supported:
 *   Discrete AdaBoost, Real AdaBoost, LogitBoost and Gentle AdaBoost.
 *   The L2 (2 class classification problems) and LK (K class classification problems)
 *   algorithms are close to LogitBoost but more numerically stable than last one.
 *   For regression three different loss functions are supported:
 *   Least square, least absolute deviation and huber loss.
 */
typedef enum CvBoostType
{
    CV_DABCLASS = 0, /* 2 class Discrete AdaBoost           */
    CV_RABCLASS = 1, /* 2 class Real AdaBoost               */
    CV_LBCLASS  = 2, /* 2 class LogitBoost                  */
    CV_GABCLASS = 3, /* 2 class Gentle AdaBoost             */
    CV_L2CLASS  = 4, /* classification (2 class problem)    */
    CV_LKCLASS  = 5, /* classification (K class problem)    */
    CV_LSREG    = 6, /* least squares regression            */
    CV_LADREG   = 7, /* least absolute deviation regression */
    CV_MREG     = 8  /* M-regression (Huber loss)           */
} CvBoostType;

/****************************************************************************************\
*                             Iterative training functions                               *
\****************************************************************************************/

/*
 * CvBoostTrainer
 *
 * The CvBoostTrainer structure represents internal boosting trainer.
 */
typedef struct CvBoostTrainer CvBoostTrainer;

/*
 * cvBoostStartTraining
 *
 * The cvBoostStartTraining function starts training process and calculates
 * response values and weights for the first weak classifier training.
 *
 * Parameters
 *   trainClasses
 *     Vector of classes of training samples classes. Each element must be 0 or 1 and
 *     of type CV_32FC1.
 *   weakTrainVals
 *     Vector of response values for the first trained weak classifier.
 *     Must be of type CV_32FC1.
 *   weights
 *     Weight vector of training samples for the first trained weak classifier.
 *     Must be of type CV_32FC1.
 *   type
 *     Boosting type. CV_DABCLASS, CV_RABCLASS, CV_LBCLASS, CV_GABCLASS
 *     types are supported.
 *
 * Return Values
 *   The return value is a pointer to internal trainer structure which is used
 *   to perform next training iterations.
 *
 * Remarks
 *   weakTrainVals and weights must be allocated before calling the function
 *   and of the same size as trainingClasses. Usually weights should be initialized
 *   with 1.0 value.
 *   The function calculates response values and weights for the first weak
 *   classifier training and stores them into weakTrainVals and weights
 *   respectively.
 *   Note, the training of the weak classifier using weakTrainVals, weight,
 *   trainingData is outside of this function.
 */
CV_BOOST_API
CvBoostTrainer* cvBoostStartTraining( CvMat* trainClasses,
                                      CvMat* weakTrainVals,
                                      CvMat* weights,
                                      CvMat* sampleIdx,
                                      CvBoostType type );
/*
 * cvBoostNextWeakClassifier
 *
 * The cvBoostNextWeakClassifier function performs next training
 * iteration and caluclates response values and weights for the next weak
 * classifier training.
 *
 * Parameters
 *   weakEvalVals
 *     Vector of values obtained by evaluation of each sample with
 *     the last trained weak classifier (iteration i). Must be of CV_32FC1 type.
 *   trainClasses
 *     Vector of classes of training samples. Each element must be 0 or 1,
 *     and of type CV_32FC1.
 *   weakTrainVals
 *     Vector of response values for the next weak classifier training
 *     (iteration i+1). Must be of type CV_32FC1.
 *   weights
 *     Weight vector of training samples for the next weak classifier training
 *     (iteration i+1). Must be of type CV_32FC1.
 *   trainer
 *     A pointer to internal trainer returned by the cvBoostStartTraining
 *     function call.
 *
 * Return Values
 *   The return value is the coefficient for the last trained weak classifier.
 *
 * Remarks
 *   weakTrainVals and weights must be exactly the same vectors as used in
 *   the cvBoostStartTraining function call and should not be modified.
 *   The function calculates response values and weights for the next weak
 *   classifier training and stores them into weakTrainVals and weights
 *   respectively.
 *   Note, the training of the weak classifier of iteration i+1 using
 *   weakTrainVals, weight, trainingData is outside of this function.
 */
CV_BOOST_API
float cvBoostNextWeakClassifier( CvMat* weakEvalVals,
                                 CvMat* trainClasses,
                                 CvMat* weakTrainVals,
                                 CvMat* weights,
                                 CvBoostTrainer* trainer );

/*
 * cvBoostEndTraining
 *
 * The cvBoostEndTraining function finishes training process and releases
 * internally allocated memory.
 *
 * Parameters
 *   trainer
 *     A pointer to a pointer to internal trainer returned by the cvBoostStartTraining
 *     function call.
 */
CV_BOOST_API
void cvBoostEndTraining( CvBoostTrainer** trainer );

/****************************************************************************************\
*                                    Boosted tree models                                 *
\****************************************************************************************/

/*
 * CvBtClassifier
 *
 * The CvBtClassifier structure represents boosted tree model.
 *
 * Members
 *   flags
 *     Flags. If CV_IS_TUNABLE( flags ) != 0 then the model supports tuning.
 *   eval
 *     Evaluation function. Returns sample predicted class (0, 1, etc.)
 *     for classification or predicted value for regression.
 *   tune
 *     Tune function. If the model supports tuning then tune call performs
 *     one more boosting iteration if passed to the function flags parameter
 *     is CV_TUNABLE otherwise releases internally allocated for tuning memory
 *     and makes the model untunable.
 *     NOTE: Since tuning uses the pointers to parameters,
 *     passed to the cvCreateBtClassifier function, they should not be modified
 *     or released between tune calls.
 *   save
 *     This function stores the model into given file.
 *   release
 *     This function releases the model.
 *   type
 *     Boosted tree model type.
 *   numclasses
 *     Number of classes for CV_LKCLASS type or 1 for all other types.
 *   numiter
 *     Number of iterations. Number of weak classifiers is equal to number
 *     of iterations for all types except CV_LKCLASS. For CV_LKCLASS type
 *     number of weak classifiers is (numiter * numclasses).
 *   numfeatures
 *     Number of features in sample.
 *   trees
 *     Stores weak classifiers when the model does not support tuning.
 *   seq
 *     Stores weak classifiers when the model supports tuning.
 *   trainer
 *     Pointer to internal tuning parameters if the model supports tuning.
 */
typedef struct CvBtClassifier
{
    CV_CLASSIFIER_FIELDS()

    CvBoostType type;
    int numclasses;
    int numiter;
    int numfeatures;
    union
    {
        CvCARTClassifier** trees;
        CvSeq* seq;
    };
    void* trainer;
} CvBtClassifier;

/*
 * CvBtClassifierTrainParams
 *
 * The CvBtClassifierTrainParams structure stores training parameters for
 * boosted tree model.
 *
 * Members
 *   type
 *     Boosted tree model type.
 *   numiter
 *     Desired number of iterations.
 *   param
 *     Parameter   Model Type    Parameter Meaning
 *     param[0]    Any           Shrinkage factor
 *     param[1]    CV_MREG       alpha. (1-alpha) determines "break-down" point of
 *                               the training procedure, i.e. the fraction of samples
 *                               that can be arbitrary modified without serious
 *                               degrading the quality of the result.
 *                 CV_DABCLASS,  Weight trimming factor.
 *                 CV_RABCLASS,
 *                 CV_LBCLASS,
 *                 CV_GABCLASS,
 *                 CV_L2CLASS,
 *                 CV_LKCLASS
 *   numsplits
 *     Desired number of splits in each tree.
 */
typedef struct CvBtClassifierTrainParams
{
    CV_CLASSIFIER_TRAIN_PARAM_FIELDS()

    CvBoostType type;
    int numiter;
    float param[2];
    int numsplits;
} CvBtClassifierTrainParams;

/*
 * cvCreateBtClassifier
 *
 * The cvCreateBtClassifier function creates boosted tree model.
 *
 * Parameters
 *   trainData
 *     Matrix of feature values. Must have CV_32FC1 type.
 *   flags
 *     Determines how samples are stored in trainData.
 *     One of CV_ROW_SAMPLE or CV_COL_SAMPLE.
 *     Optionally may be combined with CV_TUNABLE to make tunable model.
 *   trainClasses
 *     Vector of responses for regression or classes (0, 1, 2, etc.) for classification.
 *   typeMask,
 *   missedMeasurementsMask,
 *   compIdx
 *     Not supported. Must be NULL.
 *   sampleIdx
 *     Indices of samples used in training. If NULL then all samples are used.
 *     For CV_DABCLASS, CV_RABCLASS, CV_LBCLASS and CV_GABCLASS must be NULL.
 *   weights
 *     Not supported. Must be NULL.
 *   trainParams
 *     A pointer to CvBtClassifierTrainParams structure. Training parameters.
 *     See CvBtClassifierTrainParams description for details.
 *
 * Return Values
 *   The return value is a pointer to created boosted tree model of type CvBtClassifier.
 *
 * Remarks
 *     The function performs trainParams->numiter training iterations.
 *     If CV_TUNABLE flag is specified then created model supports tuning.
 *     In this case additional training iterations may be performed by
 *     tune function call.
 */
CV_BOOST_API
CvClassifier* cvCreateBtClassifier( CvMat* trainData,
                                    int flags,
                                    CvMat* trainClasses,
                                    CvMat* typeMask,
                                    CvMat* missedMeasurementsMask,
                                    CvMat* compIdx,
                                    CvMat* sampleIdx,
                                    CvMat* weights,
                                    CvClassifierTrainParams* trainParams );

/*
 * cvCreateBtClassifierFromFile
 *
 * The cvCreateBtClassifierFromFile function restores previously saved
 * boosted tree model from file.
 *
 * Parameters
 *   filename
 *     The name of the file with boosted tree model.
 *
 * Remarks
 *   The restored model does not support tuning.
 */
CV_BOOST_API
CvClassifier* cvCreateBtClassifierFromFile( const char* filename );

/****************************************************************************************\
*                                    Utility functions                                   *
\****************************************************************************************/

/*
 * cvTrimWeights
 *
 * The cvTrimWeights function performs weight trimming.
 *
 * Parameters
 *   weights
 *     Weights vector.
 *   idx
 *     Indices vector of weights that should be considered.
 *     If it is NULL then all weights are used.
 *   factor
 *     Weight trimming factor. Must be in [0, 1] range.
 *
 * Return Values
 *   The return value is a vector of indices. If all samples should be used then
 *   it is equal to idx. In other case the cvReleaseMat function should be called
 *   to release it.
 *
 * Remarks
 */
CV_BOOST_API
CvMat* cvTrimWeights( CvMat* weights, CvMat* idx, float factor );

/*
 * cvReadTrainData
 *
 * The cvReadTrainData function reads feature values and responses from file.
 *
 * Parameters
 *   filename
 *     The name of the file to be read.
 *   flags
 *     One of CV_ROW_SAMPLE or CV_COL_SAMPLE. Determines how feature values
 *     will be stored.
 *   trainData
 *     A pointer to a pointer to created matrix with feature values.
 *     cvReleaseMat function should be used to destroy created matrix.
 *   trainClasses
 *     A pointer to a pointer to created matrix with response values.
 *     cvReleaseMat function should be used to destroy created matrix.
 *
 * Remarks
 *   File format:
 *   ============================================
 *   m n
 *   value_1_1 value_1_2 ... value_1_n response_1
 *   value_2_1 value_2_2 ... value_2_n response_2
 *   ...
 *   value_m_1 value_m_2 ... value_m_n response_m
 *   ============================================
 *   m
 *     Number of samples
 *   n
 *     Number of features in each sample
 *   value_i_j
 *     Value of j-th feature of i-th sample
 *   response_i
 *     Response value of i-th sample
 *     For classification problems responses represent classes (0, 1, etc.)
 *   All values and classes are integer or real numbers.
 */
CV_BOOST_API
void cvReadTrainData( const char* filename,
                      int flags,
                      CvMat** trainData,
                      CvMat** trainClasses );


/*
 * cvWriteTrainData
 *
 * The cvWriteTrainData function stores feature values and responses into file.
 *
 * Parameters
 *   filename
 *     The name of the file.
 *   flags
 *     One of CV_ROW_SAMPLE or CV_COL_SAMPLE. Determines how feature values
 *     are stored.
 *   trainData
 *     Feature values matrix.
 *   trainClasses
 *     Response values vector.
 *   sampleIdx
 *     Vector of idicies of the samples that should be stored. If it is NULL
 *     then all samples will be stored.
 *
 * Remarks
 *   See the cvReadTrainData function for file format description.
 */
CV_BOOST_API
void cvWriteTrainData( const char* filename,
                       int flags,
                       CvMat* trainData,
                       CvMat* trainClasses,
                       CvMat* sampleIdx );

/*
 * cvRandShuffle
 *
 * The cvRandShuffle function perfroms random shuffling of given vector.
 *
 * Parameters
 *   vector
 *     Vector that should be shuffled.
 *     Must have CV_8UC1, CV_16SC1, CV_32SC1 or CV_32FC1 type.
 */
CV_BOOST_API
void cvRandShuffleVec( CvMat* vector );

#endif /* _CVCLASSIFIER_H_ */
