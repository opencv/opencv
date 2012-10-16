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

#if 0

ML_IMPL int
icvCmpIntegers (const void* a, const void* b) {return *(const int*)a - *(const int*)b;}

/****************************************************************************************\
*                    Cross-validation algorithms realizations                            *
\****************************************************************************************/

// Return pointer to trainIdx. Function DOES NOT FILL this matrix!
ML_IMPL
const CvMat* cvCrossValGetTrainIdxMatrix (const CvStatModel* estimateModel)
{
    CvMat* result = NULL;

        CV_FUNCNAME ("cvCrossValGetTrainIdxMatrix");
        __BEGIN__

    if (!CV_IS_CROSSVAL(estimateModel))
    {
        CV_ERROR (CV_StsBadArg, "Pointer point to not CvCrossValidationModel");
    }

    result = ((CvCrossValidationModel*)estimateModel)->sampleIdxTrain;

        __END__

    return result;
} // End of cvCrossValGetTrainIdxMatrix

/****************************************************************************************/
// Return pointer to checkIdx. Function DOES NOT FILL this matrix!
ML_IMPL
const CvMat* cvCrossValGetCheckIdxMatrix (const CvStatModel* estimateModel)
{
    CvMat* result = NULL;

        CV_FUNCNAME ("cvCrossValGetCheckIdxMatrix");
        __BEGIN__

    if (!CV_IS_CROSSVAL (estimateModel))
    {
        CV_ERROR (CV_StsBadArg, "Pointer point to not CvCrossValidationModel");
    }

    result = ((CvCrossValidationModel*)estimateModel)->sampleIdxEval;

        __END__

    return result;
} // End of cvCrossValGetCheckIdxMatrix

/****************************************************************************************/
// Create new Idx-matrix for next classifiers training and return code of result.
//   Result is 0 if function can't make next step (error input or folds are finished),
//   it is 1 if all was correct, and it is 2 if current fold wasn't' checked.
ML_IMPL
int cvCrossValNextStep (CvStatModel* estimateModel)
{
    int result = 0;

        CV_FUNCNAME ("cvCrossValGetNextTrainIdx");
        __BEGIN__

    CvCrossValidationModel* crVal = (CvCrossValidationModel*) estimateModel;
    int k, fold;

    if (!CV_IS_CROSSVAL (estimateModel))
    {
        CV_ERROR (CV_StsBadArg, "Pointer point to not CvCrossValidationModel");
    }

    fold = ++crVal->current_fold;

    if (fold >= crVal->folds_all)
    {
        if (fold == crVal->folds_all)
            EXIT;
        else
        {
            CV_ERROR (CV_StsInternal, "All iterations has end long ago");
        }
    }

    k = crVal->folds[fold + 1] - crVal->folds[fold];
    crVal->sampleIdxTrain->data.i = crVal->sampleIdxAll + crVal->folds[fold + 1];
    crVal->sampleIdxTrain->cols = crVal->samples_all - k;
    crVal->sampleIdxEval->data.i = crVal->sampleIdxAll + crVal->folds[fold];
    crVal->sampleIdxEval->cols = k;

    if (crVal->is_checked)
    {
        crVal->is_checked = 0;
        result = 1;
    }
    else
    {
        result = 2;
    }

        __END__

    return result;
}

/****************************************************************************************/
// Do checking part of loop  of cross-validations metod.
ML_IMPL
void cvCrossValCheckClassifier (CvStatModel*  estimateModel,
                          const CvStatModel*  model,
                          const CvMat*        trainData,
                                int           sample_t_flag,
                          const CvMat*        trainClasses)
{
        CV_FUNCNAME ("cvCrossValCheckClassifier ");
        __BEGIN__

    CvCrossValidationModel* crVal = (CvCrossValidationModel*) estimateModel;
    int  i, j, k;
    int* data;
    float* responses_fl;
    int    step;
    float* responses_result;
    int* responses_i;
    double te, te1;
    double sum_c, sum_p, sum_pp, sum_cp, sum_cc, sq_err;

// Check input data to correct values.
    if (!CV_IS_CROSSVAL (estimateModel))
    {
        CV_ERROR (CV_StsBadArg,"First parameter point to not CvCrossValidationModel");
    }
    if (!CV_IS_STAT_MODEL (model))
    {
        CV_ERROR (CV_StsBadArg, "Second parameter point to not CvStatModel");
    }
    if (!CV_IS_MAT (trainData))
    {
        CV_ERROR (CV_StsBadArg, "Third parameter point to not CvMat");
    }
    if (!CV_IS_MAT (trainClasses))
    {
        CV_ERROR (CV_StsBadArg, "Fifth parameter point to not CvMat");
    }
    if (crVal->is_checked)
    {
        CV_ERROR (CV_StsInternal, "This iterations already was checked");
    }

// Initialize.
    k = crVal->sampleIdxEval->cols;
    data = crVal->sampleIdxEval->data.i;

// Eval tested feature vectors.
    CV_CALL (cvStatModelMultiPredict (model, trainData, sample_t_flag,
                                         crVal->predict_results, NULL, crVal->sampleIdxEval));
// Count number if correct results.
    responses_result = crVal->predict_results->data.fl;
    if (crVal->is_regression)
    {
        sum_c = sum_p = sum_pp = sum_cp = sum_cc = sq_err = 0;
        if (CV_MAT_TYPE (trainClasses->type) == CV_32FC1)
        {
            responses_fl = trainClasses->data.fl;
            step = trainClasses->rows == 1 ? 1 : trainClasses->step / sizeof(float);
            for (i = 0; i < k; i++)
            {
                te = responses_result[*data];
                te1 = responses_fl[*data * step];
                sum_c += te1;
                sum_p += te;
                sum_cc += te1 * te1;
                sum_pp += te * te;
                sum_cp += te1 * te;
                te -= te1;
                sq_err += te  * te;

                data++;
            }
        }
        else
        {
            responses_i = trainClasses->data.i;
            step = trainClasses->rows == 1 ? 1 : trainClasses->step / sizeof(int);
            for (i = 0; i < k; i++)
            {
                te = responses_result[*data];
                te1 = responses_i[*data * step];
                sum_c += te1;
                sum_p += te;
                sum_cc += te1 * te1;
                sum_pp += te * te;
                sum_cp += te1 * te;
                te -= te1;
                sq_err += te  * te;

                data++;
            }
        }
    // Fixing new internal values of accuracy.
        crVal->sum_correct += sum_c;
        crVal->sum_predict += sum_p;
        crVal->sum_cc += sum_cc;
        crVal->sum_pp += sum_pp;
        crVal->sum_cp += sum_cp;
        crVal->sq_error += sq_err;
    }
    else
    {
        if (CV_MAT_TYPE (trainClasses->type) == CV_32FC1)
        {
            responses_fl = trainClasses->data.fl;
            step = trainClasses->rows == 1 ? 1 : trainClasses->step / sizeof(float);
            for (i = 0, j = 0; i < k; i++)
            {
                if (cvRound (responses_result[*data]) == cvRound (responses_fl[*data * step]))
                    j++;
                data++;
            }
        }
        else
        {
            responses_i = trainClasses->data.i;
            step = trainClasses->rows == 1 ? 1 : trainClasses->step / sizeof(int);
            for (i = 0, j = 0; i < k; i++)
            {
                if (cvRound (responses_result[*data]) == responses_i[*data * step])
                    j++;
                data++;
            }
        }
    // Fixing new internal values of accuracy.
        crVal->correct_results += j;
    }
// Fixing that this fold already checked.
    crVal->all_results += k;
    crVal->is_checked = 1;

        __END__
} // End of cvCrossValCheckClassifier

/****************************************************************************************/
// Return current accuracy.
ML_IMPL
float cvCrossValGetResult (const CvStatModel* estimateModel,
                                 float*       correlation)
{
    float result = 0;

        CV_FUNCNAME ("cvCrossValGetResult");
        __BEGIN__

    double te, te1;
    CvCrossValidationModel* crVal = (CvCrossValidationModel*)estimateModel;

    if (!CV_IS_CROSSVAL (estimateModel))
    {
        CV_ERROR (CV_StsBadArg, "Pointer point to not CvCrossValidationModel");
    }

    if (crVal->all_results)
    {
        if (crVal->is_regression)
        {
            result = ((float)crVal->sq_error) / crVal->all_results;
            if (correlation)
            {
                te = crVal->all_results * crVal->sum_cp -
                                             crVal->sum_correct * crVal->sum_predict;
                te *= te;
                te1 = (crVal->all_results * crVal->sum_cc -
                                    crVal->sum_correct * crVal->sum_correct) *
                           (crVal->all_results * crVal->sum_pp -
                                    crVal->sum_predict * crVal->sum_predict);
                *correlation = (float)(te / te1);

            }
        }
        else
        {
            result = ((float)crVal->correct_results) / crVal->all_results;
        }
    }

        __END__

    return result;
}

/****************************************************************************************/
// Reset cross-validation EstimateModel to state the same as it was immidiatly after
//   its creating.
ML_IMPL
void cvCrossValReset (CvStatModel* estimateModel)
{
        CV_FUNCNAME ("cvCrossValReset");
        __BEGIN__

    CvCrossValidationModel* crVal = (CvCrossValidationModel*)estimateModel;

    if (!CV_IS_CROSSVAL (estimateModel))
    {
        CV_ERROR (CV_StsBadArg, "Pointer point to not CvCrossValidationModel");
    }

    crVal->current_fold = -1;
    crVal->is_checked = 1;
    crVal->all_results = 0;
    crVal->correct_results = 0;
    crVal->sq_error = 0;
    crVal->sum_correct = 0;
    crVal->sum_predict = 0;
    crVal->sum_cc = 0;
    crVal->sum_pp = 0;
    crVal->sum_cp = 0;

        __END__
}

/****************************************************************************************/
// This function is standart CvStatModel field to release cross-validation EstimateModel.
ML_IMPL
void cvReleaseCrossValidationModel (CvStatModel** model)
{
    CvCrossValidationModel* pModel;

        CV_FUNCNAME ("cvReleaseCrossValidationModel");
        __BEGIN__

    if (!model)
    {
        CV_ERROR (CV_StsNullPtr, "");
    }

    pModel = (CvCrossValidationModel*)*model;
    if (!pModel)
    {
        return;
    }
    if (!CV_IS_CROSSVAL (pModel))
    {
        CV_ERROR (CV_StsBadArg, "");
    }

    cvFree (&pModel->sampleIdxAll);
    cvFree (&pModel->folds);
    cvReleaseMat (&pModel->sampleIdxEval);
    cvReleaseMat (&pModel->sampleIdxTrain);
    cvReleaseMat (&pModel->predict_results);

    cvFree (model);

        __END__
} // End of cvReleaseCrossValidationModel.

/****************************************************************************************/
// This function create cross-validation EstimateModel.
ML_IMPL CvStatModel*
cvCreateCrossValidationEstimateModel(
             int                samples_all,
       const CvStatModelParams* estimateParams,
       const CvMat*             sampleIdx)
{
    CvStatModel*            model   = NULL;
    CvCrossValidationModel* crVal   = NULL;

        CV_FUNCNAME ("cvCreateCrossValidationEstimateModel");
        __BEGIN__

    int  k_fold = 10;

    int  i, j, k, s_len;
    int  samples_selected;
    CvRNG rng;
    CvRNG* prng;
    int* res_s_data;
    int* te_s_data;
    int* folds;

    rng = cvRNG(cvGetTickCount());
    cvRandInt (&rng); cvRandInt (&rng); cvRandInt (&rng); cvRandInt (&rng);
// Check input parameters.
    if (estimateParams)
        k_fold = ((CvCrossValidationParams*)estimateParams)->k_fold;
    if (!k_fold)
    {
        CV_ERROR (CV_StsBadArg, "Error in parameters of cross-validation (k_fold == 0)!");
    }
    if (samples_all <= 0)
    {
        CV_ERROR (CV_StsBadArg, "<samples_all> should be positive!");
    }

// Alloc memory and fill standart StatModel's fields.
    CV_CALL (crVal = (CvCrossValidationModel*)cvCreateStatModel (
                            CV_STAT_MODEL_MAGIC_VAL | CV_CROSSVAL_MAGIC_VAL,
                            sizeof(CvCrossValidationModel),
                            cvReleaseCrossValidationModel,
                            NULL, NULL));
    crVal->current_fold    = -1;
    crVal->folds_all       = k_fold;
    if (estimateParams && ((CvCrossValidationParams*)estimateParams)->is_regression)
        crVal->is_regression = 1;
    else
        crVal->is_regression = 0;
    if (estimateParams && ((CvCrossValidationParams*)estimateParams)->rng)
        prng = ((CvCrossValidationParams*)estimateParams)->rng;
    else
        prng = &rng;

    // Check and preprocess sample indices.
    if (sampleIdx)
    {
        int s_step;
        int s_type = 0;

        if (!CV_IS_MAT (sampleIdx))
            CV_ERROR (CV_StsBadArg, "Invalid sampleIdx array");

        if (sampleIdx->rows != 1 && sampleIdx->cols != 1)
            CV_ERROR (CV_StsBadSize, "sampleIdx array must be 1-dimensional");

        s_len = sampleIdx->rows + sampleIdx->cols - 1;
        s_step = sampleIdx->rows == 1 ?
                                     1 : sampleIdx->step / CV_ELEM_SIZE(sampleIdx->type);

        s_type = CV_MAT_TYPE (sampleIdx->type);

        switch (s_type)
        {
        case CV_8UC1:
        case CV_8SC1:
            {
            uchar* s_data = sampleIdx->data.ptr;

            // sampleIdx is array of 1's and 0's -
            // i.e. it is a mask of the selected samples
            if( s_len != samples_all )
                CV_ERROR (CV_StsUnmatchedSizes,
       "Sample mask should contain as many elements as the total number of samples");

            samples_selected = 0;
            for (i = 0; i < s_len; i++)
                samples_selected += s_data[i * s_step] != 0;

            if (samples_selected == 0)
                CV_ERROR (CV_StsOutOfRange, "No samples is selected!");
            }
            s_len = samples_selected;
            break;
        case CV_32SC1:
            if (s_len > samples_all)
                CV_ERROR (CV_StsOutOfRange,
        "sampleIdx array may not contain more elements than the total number of samples");
            samples_selected = s_len;
            break;
        default:
            CV_ERROR (CV_StsUnsupportedFormat, "Unsupported sampleIdx array data type "
                                               "(it should be 8uC1, 8sC1 or 32sC1)");
        }

        // Alloc additional memory for internal Idx and fill it.
/*!!*/  CV_CALL (res_s_data = crVal->sampleIdxAll =
                                                 (int*)cvAlloc (2 * s_len * sizeof(int)));

        if (s_type < CV_32SC1)
        {
            uchar* s_data = sampleIdx->data.ptr;
            for (i = 0; i < s_len; i++)
                if (s_data[i * s_step])
                {
                    *res_s_data++ = i;
                }
            res_s_data = crVal->sampleIdxAll;
        }
        else
        {
            int* s_data = sampleIdx->data.i;
            int out_of_order = 0;

            for (i = 0; i < s_len; i++)
            {
                res_s_data[i] = s_data[i * s_step];
                if (i > 0 && res_s_data[i] < res_s_data[i - 1])
                    out_of_order = 1;
            }

            if (out_of_order)
                qsort (res_s_data, s_len, sizeof(res_s_data[0]), icvCmpIntegers);

            if (res_s_data[0] < 0 ||
                res_s_data[s_len - 1] >= samples_all)
                    CV_ERROR (CV_StsBadArg, "There are out-of-range sample indices");
            for (i = 1; i < s_len; i++)
                if (res_s_data[i] <= res_s_data[i - 1])
                    CV_ERROR (CV_StsBadArg, "There are duplicated");
        }
    }
    else // if (sampleIdx)
    {
        // Alloc additional memory for internal Idx and fill it.
        s_len = samples_all;
        CV_CALL (res_s_data = crVal->sampleIdxAll = (int*)cvAlloc (2 * s_len * sizeof(int)));
        for (i = 0; i < s_len; i++)
        {
            *res_s_data++ = i;
        }
        res_s_data = crVal->sampleIdxAll;
    } // if (sampleIdx) ... else

// Resort internal Idx.
    te_s_data = res_s_data + s_len;
    for (i = s_len; i > 1; i--)
    {
        j = cvRandInt (prng) % i;
        k = *(--te_s_data);
        *te_s_data = res_s_data[j];
        res_s_data[j] = k;
    }

// Duplicate resorted internal Idx.
// It will be used to simplify operation of getting trainIdx.
    te_s_data = res_s_data + s_len;
    for (i = 0; i < s_len; i++)
    {
        *te_s_data++ = *res_s_data++;
    }

// Cut sampleIdxAll to parts.
    if (k_fold > 0)
    {
        if (k_fold > s_len)
        {
            CV_ERROR (CV_StsBadArg,
                        "Error in parameters of cross-validation ('k_fold' > #samples)!");
        }
        folds = crVal->folds = (int*) cvAlloc ((k_fold + 1) * sizeof (int));
        *folds++ = 0;
        for (i = 1; i < k_fold; i++)
        {
            *folds++ = cvRound (i * s_len * 1. / k_fold);
        }
        *folds = s_len;
        folds = crVal->folds;

        crVal->max_fold_size = (s_len - 1) / k_fold + 1;
    }
    else
    {
        k = -k_fold;
        crVal->max_fold_size = k;
        if (k >= s_len)
        {
            CV_ERROR (CV_StsBadArg,
                      "Error in parameters of cross-validation (-'k_fold' > #samples)!");
        }
        crVal->folds_all = k = (s_len - 1) / k + 1;

        folds = crVal->folds = (int*) cvAlloc ((k + 1) * sizeof (int));
        for (i = 0; i < k; i++)
        {
            *folds++ = -i * k_fold;
        }
        *folds = s_len;
        folds = crVal->folds;
    }

// Prepare other internal fields to working.
    CV_CALL (crVal->predict_results = cvCreateMat (1, samples_all, CV_32FC1));
    CV_CALL (crVal->sampleIdxEval = cvCreateMatHeader (1, 1, CV_32SC1));
    CV_CALL (crVal->sampleIdxTrain = cvCreateMatHeader (1, 1, CV_32SC1));
    crVal->sampleIdxEval->cols = 0;
    crVal->sampleIdxTrain->cols = 0;
    crVal->samples_all = s_len;
    crVal->is_checked = 1;

    crVal->getTrainIdxMat = cvCrossValGetTrainIdxMatrix;
    crVal->getCheckIdxMat = cvCrossValGetCheckIdxMatrix;
    crVal->nextStep = cvCrossValNextStep;
    crVal->check = cvCrossValCheckClassifier;
    crVal->getResult = cvCrossValGetResult;
    crVal->reset = cvCrossValReset;

    model = (CvStatModel*)crVal;

        __END__

    if (!model)
    {
        cvReleaseCrossValidationModel ((CvStatModel**)&crVal);
    }

    return model;
} // End of cvCreateCrossValidationEstimateModel


/****************************************************************************************\
*                Extended interface with backcalls for models                            *
\****************************************************************************************/
ML_IMPL float
cvCrossValidation (const CvMat*            trueData,
                         int               tflag,
                   const CvMat*            trueClasses,
                         CvStatModel*     (*createClassifier) (const CvMat*,
                                                                     int,
                                                               const CvMat*,
                                                               const CvClassifierTrainParams*,
                                                               const CvMat*,
                                                               const CvMat*,
                                                               const CvMat*,
                                                               const CvMat*),
                   const CvClassifierTrainParams*    estimateParams,
                   const CvClassifierTrainParams*    trainParams,
                   const CvMat*            compIdx,
                   const CvMat*            sampleIdx,
                         CvStatModel**     pCrValModel,
                   const CvMat*            typeMask,
                   const CvMat*            missedMeasurementMask)
{
    CvCrossValidationModel* crVal = NULL;
    float  result = 0;
    CvStatModel* pClassifier = NULL;

        CV_FUNCNAME ("cvCrossValidation");
        __BEGIN__

    const CvMat* trainDataIdx;
    int    samples_all;

// checking input data
    if ((createClassifier) == NULL)
    {
        CV_ERROR (CV_StsNullPtr, "Null pointer to functiion which create classifier");
    }
    if (pCrValModel && *pCrValModel && !CV_IS_CROSSVAL(*pCrValModel))
    {
        CV_ERROR (CV_StsBadArg,
           "<pCrValModel> point to not cross-validation model");
    }

// initialization
    if (pCrValModel && *pCrValModel)
    {
        crVal = (CvCrossValidationModel*)*pCrValModel;
        crVal->reset ((CvStatModel*)crVal);
    }
    else
    {
        samples_all = ((tflag) ? trueData->rows : trueData->cols);
        CV_CALL (crVal = (CvCrossValidationModel*)
           cvCreateCrossValidationEstimateModel (samples_all, estimateParams, sampleIdx));
    }

    CV_CALL (trainDataIdx = crVal->getTrainIdxMat ((CvStatModel*)crVal));

// operation loop
    for (; crVal->nextStep((CvStatModel*)crVal) != 0; )
    {
        CV_CALL (pClassifier = createClassifier (trueData, tflag, trueClasses,
                    trainParams, compIdx, trainDataIdx, typeMask, missedMeasurementMask));
        CV_CALL (crVal->check ((CvStatModel*)crVal, pClassifier,
                                                           trueData, tflag, trueClasses));

        pClassifier->release (&pClassifier);
    }

// Get result and fill output field.
    CV_CALL (result = crVal->getResult ((CvStatModel*)crVal, 0));

    if (pCrValModel && !*pCrValModel)
        *pCrValModel = (CvStatModel*)crVal;

        __END__

// Free all memory that should be freed.
    if (pClassifier)
        pClassifier->release (&pClassifier);
    if (crVal && (!pCrValModel || !*pCrValModel))
        crVal->release ((CvStatModel**)&crVal);

    return result;
} // End of cvCrossValidation

#endif

/* End of file */
