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

/****************************************************************************************\

      Calculation of a texture descriptors from GLCM (Grey Level Co-occurrence Matrix'es)
      The code was submitted by Daniel Eaton [danieljameseaton@yahoo.com]

\****************************************************************************************/

#include "precomp.hpp"

#include <math.h>
#include <assert.h>

#define CV_MAX_NUM_GREY_LEVELS_8U  256

struct CvGLCM
{
    int matrixSideLength;
    int numMatrices;
    double*** matrices;

    int  numLookupTableElements;
    int  forwardLookupTable[CV_MAX_NUM_GREY_LEVELS_8U];
    int  reverseLookupTable[CV_MAX_NUM_GREY_LEVELS_8U];

    double** descriptors;
    int numDescriptors;
    int descriptorOptimizationType;
    int optimizationType;
};


static void icvCreateGLCM_LookupTable_8u_C1R( const uchar* srcImageData, int srcImageStep,
                                             CvSize srcImageSize, CvGLCM* destGLCM,
                                             int* steps, int numSteps, int* memorySteps );

static void
icvCreateGLCMDescriptors_AllowDoubleNest( CvGLCM* destGLCM, int matrixIndex );


CV_IMPL CvGLCM*
cvCreateGLCM( const IplImage* srcImage,
              int stepMagnitude,
              const int* srcStepDirections,/* should be static array..
                                          or if not the user should handle de-allocation */
              int numStepDirections,
              int optimizationType )
{
    static const int defaultStepDirections[] = { 0,1, -1,1, -1,0, -1,-1 };

    int* memorySteps = 0;
    CvGLCM* newGLCM = 0;
    int* stepDirections = 0;

    CV_FUNCNAME( "cvCreateGLCM" );

    __BEGIN__;

    uchar* srcImageData = 0;
    CvSize srcImageSize;
    int srcImageStep;
    int stepLoop;
    const int maxNumGreyLevels8u = CV_MAX_NUM_GREY_LEVELS_8U;

    if( !srcImage )
        CV_ERROR( CV_StsNullPtr, "" );

    if( srcImage->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, "Number of channels must be 1");

    if( srcImage->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, "Depth must be equal IPL_DEPTH_8U");

    // no Directions provided, use the default ones - 0 deg, 45, 90, 135
    if( !srcStepDirections )
    {
        srcStepDirections = defaultStepDirections;
    }

    CV_CALL( stepDirections = (int*)cvAlloc( numStepDirections*2*sizeof(stepDirections[0])));
    memcpy( stepDirections, srcStepDirections, numStepDirections*2*sizeof(stepDirections[0]));

    cvGetImageRawData( srcImage, &srcImageData, &srcImageStep, &srcImageSize );

    // roll together Directions and magnitudes together with knowledge of image (step)
    CV_CALL( memorySteps = (int*)cvAlloc( numStepDirections*sizeof(memorySteps[0])));

    for( stepLoop = 0; stepLoop < numStepDirections; stepLoop++ )
    {
        stepDirections[stepLoop*2 + 0] *= stepMagnitude;
        stepDirections[stepLoop*2 + 1] *= stepMagnitude;

        memorySteps[stepLoop] = stepDirections[stepLoop*2 + 0]*srcImageStep +
                                stepDirections[stepLoop*2 + 1];
    }

    CV_CALL( newGLCM = (CvGLCM*)cvAlloc(sizeof(newGLCM)));
    memset( newGLCM, 0, sizeof(*newGLCM) );

    newGLCM->matrices = 0;
    newGLCM->numMatrices = numStepDirections;
    newGLCM->optimizationType = optimizationType;

    if( optimizationType <= CV_GLCM_OPTIMIZATION_LUT )
    {
        int lookupTableLoop, imageColLoop, imageRowLoop, lineOffset = 0;

        // if optimization type is set to lut, then make one for the image
        if( optimizationType == CV_GLCM_OPTIMIZATION_LUT )
        {
            for( imageRowLoop = 0; imageRowLoop < srcImageSize.height;
                                   imageRowLoop++, lineOffset += srcImageStep )
            {
                for( imageColLoop = 0; imageColLoop < srcImageSize.width; imageColLoop++ )
                {
                    newGLCM->forwardLookupTable[srcImageData[lineOffset+imageColLoop]]=1;
                }
            }

            newGLCM->numLookupTableElements = 0;

            for( lookupTableLoop = 0; lookupTableLoop < maxNumGreyLevels8u; lookupTableLoop++ )
            {
                if( newGLCM->forwardLookupTable[ lookupTableLoop ] != 0 )
                {
                    newGLCM->forwardLookupTable[ lookupTableLoop ] =
                        newGLCM->numLookupTableElements;
                    newGLCM->reverseLookupTable[ newGLCM->numLookupTableElements ] =
                        lookupTableLoop;

                    newGLCM->numLookupTableElements++;
                }
            }
        }
        // otherwise make a "LUT" which contains all the gray-levels (for code-reuse)
        else if( optimizationType == CV_GLCM_OPTIMIZATION_NONE )
        {
            for( lookupTableLoop = 0; lookupTableLoop <maxNumGreyLevels8u; lookupTableLoop++ )
            {
                newGLCM->forwardLookupTable[ lookupTableLoop ] = lookupTableLoop;
                newGLCM->reverseLookupTable[ lookupTableLoop ] = lookupTableLoop;
            }
            newGLCM->numLookupTableElements = maxNumGreyLevels8u;
        }

        newGLCM->matrixSideLength = newGLCM->numLookupTableElements;
        icvCreateGLCM_LookupTable_8u_C1R( srcImageData, srcImageStep, srcImageSize,
                                          newGLCM, stepDirections,
                                          numStepDirections, memorySteps );
    }
    else if( optimizationType == CV_GLCM_OPTIMIZATION_HISTOGRAM )
    {
        CV_ERROR( CV_StsBadFlag, "Histogram-based method is not implemented" );

    /*  newGLCM->numMatrices *= 2;
        newGLCM->matrixSideLength = maxNumGreyLevels8u*2;

        icvCreateGLCM_Histogram_8uC1R( srcImageStep, srcImageSize, srcImageData,
                                       newGLCM, numStepDirections,
                                       stepDirections, memorySteps );
    */
    }

    __END__;

    cvFree( &memorySteps );
    cvFree( &stepDirections );

    if( cvGetErrStatus() < 0 )
    {
        cvFree( &newGLCM );
    }

    return newGLCM;
}


CV_IMPL void
cvReleaseGLCM( CvGLCM** GLCM, int flag )
{
    CV_FUNCNAME( "cvReleaseGLCM" );

    __BEGIN__;

    int matrixLoop;

    if( !GLCM )
        CV_ERROR( CV_StsNullPtr, "" );

    if( *GLCM )
        EXIT; // repeated deallocation: just skip it.

    if( (flag == CV_GLCM_GLCM || flag == CV_GLCM_ALL) && (*GLCM)->matrices )
    {
        for( matrixLoop = 0; matrixLoop < (*GLCM)->numMatrices; matrixLoop++ )
        {
            if( (*GLCM)->matrices[ matrixLoop ] )
            {
                cvFree( (*GLCM)->matrices[matrixLoop] );
                cvFree( (*GLCM)->matrices + matrixLoop );
            }
        }

        cvFree( &((*GLCM)->matrices) );
    }

    if( (flag == CV_GLCM_DESC || flag == CV_GLCM_ALL) && (*GLCM)->descriptors )
    {
        for( matrixLoop = 0; matrixLoop < (*GLCM)->numMatrices; matrixLoop++ )
        {
            cvFree( (*GLCM)->descriptors + matrixLoop );
        }
        cvFree( &((*GLCM)->descriptors) );
    }

    if( flag == CV_GLCM_ALL )
    {
        cvFree( GLCM );
    }

    __END__;
}


static void
icvCreateGLCM_LookupTable_8u_C1R( const uchar* srcImageData,
                                  int srcImageStep,
                                  CvSize srcImageSize,
                                  CvGLCM* destGLCM,
                                  int* steps,
                                  int numSteps,
                                  int* memorySteps )
{
    int* stepIncrementsCounter = 0;

    CV_FUNCNAME( "icvCreateGLCM_LookupTable_8u_C1R" );

    __BEGIN__;

    int matrixSideLength = destGLCM->matrixSideLength;
    int stepLoop, sideLoop1, sideLoop2;
    int colLoop, rowLoop, lineOffset = 0;
    double*** matrices = 0;

    // allocate memory to the matrices
    CV_CALL( destGLCM->matrices = (double***)cvAlloc( sizeof(matrices[0])*numSteps ));
    matrices = destGLCM->matrices;

    for( stepLoop=0; stepLoop<numSteps; stepLoop++ )
    {
        CV_CALL( matrices[stepLoop] = (double**)cvAlloc( sizeof(matrices[0])*matrixSideLength ));
        CV_CALL( matrices[stepLoop][0] = (double*)cvAlloc( sizeof(matrices[0][0])*
                                                  matrixSideLength*matrixSideLength ));

        memset( matrices[stepLoop][0], 0, matrixSideLength*matrixSideLength*
                                          sizeof(matrices[0][0]) );

        for( sideLoop1 = 1; sideLoop1 < matrixSideLength; sideLoop1++ )
        {
            matrices[stepLoop][sideLoop1] = matrices[stepLoop][sideLoop1-1] + matrixSideLength;
        }
    }

    CV_CALL( stepIncrementsCounter = (int*)cvAlloc( numSteps*sizeof(stepIncrementsCounter[0])));
    memset( stepIncrementsCounter, 0, numSteps*sizeof(stepIncrementsCounter[0]) );

    // generate GLCM for each step
    for( rowLoop=0; rowLoop<srcImageSize.height; rowLoop++, lineOffset+=srcImageStep )
    {
        for( colLoop=0; colLoop<srcImageSize.width; colLoop++ )
        {
            int pixelValue1 = destGLCM->forwardLookupTable[srcImageData[lineOffset + colLoop]];

            for( stepLoop=0; stepLoop<numSteps; stepLoop++ )
            {
                int col2, row2;
                row2 = rowLoop + steps[stepLoop*2 + 0];
                col2 = colLoop + steps[stepLoop*2 + 1];

                if( col2>=0 && row2>=0 && col2<srcImageSize.width && row2<srcImageSize.height )
                {
                    int memoryStep = memorySteps[ stepLoop ];
                    int pixelValue2 = destGLCM->forwardLookupTable[ srcImageData[ lineOffset + colLoop + memoryStep ] ];

                    // maintain symmetry
                    matrices[stepLoop][pixelValue1][pixelValue2] ++;
                    matrices[stepLoop][pixelValue2][pixelValue1] ++;

                    // incremenet counter of total number of increments
                    stepIncrementsCounter[stepLoop] += 2;
                }
            }
        }
    }

    // normalize matrices. each element is a probability of gray value i,j adjacency in direction/magnitude k
    for( sideLoop1=0; sideLoop1<matrixSideLength; sideLoop1++ )
    {
        for( sideLoop2=0; sideLoop2<matrixSideLength; sideLoop2++ )
        {
            for( stepLoop=0; stepLoop<numSteps; stepLoop++ )
            {
                matrices[stepLoop][sideLoop1][sideLoop2] /= double(stepIncrementsCounter[stepLoop]);
            }
        }
    }

    destGLCM->matrices = matrices;

    __END__;

    cvFree( &stepIncrementsCounter );

    if( cvGetErrStatus() < 0 )
        cvReleaseGLCM( &destGLCM, CV_GLCM_GLCM );
}


CV_IMPL void
cvCreateGLCMDescriptors( CvGLCM* destGLCM, int descriptorOptimizationType )
{
    CV_FUNCNAME( "cvCreateGLCMDescriptors" );

    __BEGIN__;

    int matrixLoop;

    if( !destGLCM )
        CV_ERROR( CV_StsNullPtr, "" );

    if( !(destGLCM->matrices) )
        CV_ERROR( CV_StsNullPtr, "Matrices are not allocated" );

    CV_CALL( cvReleaseGLCM( &destGLCM, CV_GLCM_DESC ));

    if( destGLCM->optimizationType != CV_GLCM_OPTIMIZATION_HISTOGRAM )
    {
        destGLCM->descriptorOptimizationType = destGLCM->numDescriptors = descriptorOptimizationType;
    }
    else
    {
        CV_ERROR( CV_StsBadFlag, "Histogram-based method is not implemented" );
//      destGLCM->descriptorOptimizationType = destGLCM->numDescriptors = CV_GLCMDESC_OPTIMIZATION_HISTOGRAM;
    }

    CV_CALL( destGLCM->descriptors = (double**)
            cvAlloc( destGLCM->numMatrices*sizeof(destGLCM->descriptors[0])));

    for( matrixLoop = 0; matrixLoop < destGLCM->numMatrices; matrixLoop ++ )
    {
        CV_CALL( destGLCM->descriptors[ matrixLoop ] =
                (double*)cvAlloc( destGLCM->numDescriptors*sizeof(destGLCM->descriptors[0][0])));
        memset( destGLCM->descriptors[matrixLoop], 0, destGLCM->numDescriptors*sizeof(double) );

        switch( destGLCM->descriptorOptimizationType )
        {
            case CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST:
                icvCreateGLCMDescriptors_AllowDoubleNest( destGLCM, matrixLoop );
                break;
            default:
                CV_ERROR( CV_StsBadFlag,
                "descriptorOptimizationType different from CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST\n"
                "is not supported" );
            /*
            case CV_GLCMDESC_OPTIMIZATION_ALLOWTRIPLENEST:
                icvCreateGLCMDescriptors_AllowTripleNest( destGLCM, matrixLoop );
                break;
            case CV_GLCMDESC_OPTIMIZATION_HISTOGRAM:
                if(matrixLoop < destGLCM->numMatrices>>1)
                    icvCreateGLCMDescriptors_Histogram( destGLCM, matrixLoop);
                    break;
            */
        }
    }

    __END__;

    if( cvGetErrStatus() < 0 )
        cvReleaseGLCM( &destGLCM, CV_GLCM_DESC );
}


static void
icvCreateGLCMDescriptors_AllowDoubleNest( CvGLCM* destGLCM, int matrixIndex )
{
    int sideLoop1, sideLoop2;
    int matrixSideLength = destGLCM->matrixSideLength;

    double** matrix = destGLCM->matrices[ matrixIndex ];
    double* descriptors = destGLCM->descriptors[ matrixIndex ];

    double* marginalProbability =
        (double*)cvAlloc( matrixSideLength * sizeof(marginalProbability[0]));
    memset( marginalProbability, 0, matrixSideLength * sizeof(double) );

    double maximumProbability = 0;
    double marginalProbabilityEntropy = 0;
    double correlationMean = 0, correlationStdDeviation = 0, correlationProductTerm = 0;

    for( sideLoop1=0; sideLoop1<matrixSideLength; sideLoop1++ )
    {
        int actualSideLoop1 = destGLCM->reverseLookupTable[ sideLoop1 ];

        for( sideLoop2=0; sideLoop2<matrixSideLength; sideLoop2++ )
        {
            double entryValue = matrix[ sideLoop1 ][ sideLoop2 ];

            int actualSideLoop2 = destGLCM->reverseLookupTable[ sideLoop2 ];
            int sideLoopDifference = actualSideLoop1 - actualSideLoop2;
            int sideLoopDifferenceSquared = sideLoopDifference*sideLoopDifference;

            marginalProbability[ sideLoop1 ] += entryValue;
            correlationMean += actualSideLoop1*entryValue;

            maximumProbability = MAX( maximumProbability, entryValue );

            if( actualSideLoop2 > actualSideLoop1 )
            {
                descriptors[ CV_GLCMDESC_CONTRAST ] += sideLoopDifferenceSquared * entryValue;
            }

            descriptors[ CV_GLCMDESC_HOMOGENITY ] += entryValue / ( 1.0 + sideLoopDifferenceSquared );

            if( entryValue > 0 )
            {
                descriptors[ CV_GLCMDESC_ENTROPY ] += entryValue * log( entryValue );
            }

            descriptors[ CV_GLCMDESC_ENERGY ] += entryValue*entryValue;
        }

        if( marginalProbability>0 )
            marginalProbabilityEntropy += marginalProbability[ actualSideLoop1 ]*log(marginalProbability[ actualSideLoop1 ]);
    }

    marginalProbabilityEntropy = -marginalProbabilityEntropy;

    descriptors[ CV_GLCMDESC_CONTRAST ] += descriptors[ CV_GLCMDESC_CONTRAST ];
    descriptors[ CV_GLCMDESC_ENTROPY ] = -descriptors[ CV_GLCMDESC_ENTROPY ];
    descriptors[ CV_GLCMDESC_MAXIMUMPROBABILITY ] = maximumProbability;

    double HXY = 0, HXY1 = 0, HXY2 = 0;

    HXY = descriptors[ CV_GLCMDESC_ENTROPY ];

    for( sideLoop1=0; sideLoop1<matrixSideLength; sideLoop1++ )
    {
        double sideEntryValueSum = 0;
        int actualSideLoop1 = destGLCM->reverseLookupTable[ sideLoop1 ];

        for( sideLoop2=0; sideLoop2<matrixSideLength; sideLoop2++ )
        {
            double entryValue = matrix[ sideLoop1 ][ sideLoop2 ];

            sideEntryValueSum += entryValue;

            int actualSideLoop2 = destGLCM->reverseLookupTable[ sideLoop2 ];

            correlationProductTerm += (actualSideLoop1 - correlationMean) * (actualSideLoop2 - correlationMean) * entryValue;

            double clusterTerm = actualSideLoop1 + actualSideLoop2 - correlationMean - correlationMean;

            descriptors[ CV_GLCMDESC_CLUSTERTENDENCY ] += clusterTerm * clusterTerm * entryValue;
            descriptors[ CV_GLCMDESC_CLUSTERSHADE ] += clusterTerm * clusterTerm * clusterTerm * entryValue;

            double HXYValue = marginalProbability[ actualSideLoop1 ] * marginalProbability[ actualSideLoop2 ];
            if( HXYValue>0 )
            {
                double HXYValueLog = log( HXYValue );
                HXY1 += entryValue * HXYValueLog;
                HXY2 += HXYValue * HXYValueLog;
            }
        }

        correlationStdDeviation += (actualSideLoop1-correlationMean) * (actualSideLoop1-correlationMean) * sideEntryValueSum;
    }

    HXY1 = -HXY1;
    HXY2 = -HXY2;

    descriptors[ CV_GLCMDESC_CORRELATIONINFO1 ] = ( HXY - HXY1 ) / ( correlationMean );
    descriptors[ CV_GLCMDESC_CORRELATIONINFO2 ] = sqrt( 1.0 - exp( -2.0 * (HXY2 - HXY ) ) );

    correlationStdDeviation = sqrt( correlationStdDeviation );

    descriptors[ CV_GLCMDESC_CORRELATION ] = correlationProductTerm / (correlationStdDeviation*correlationStdDeviation );

    delete [] marginalProbability;
}


CV_IMPL double cvGetGLCMDescriptor( CvGLCM* GLCM, int step, int descriptor )
{
    double value = DBL_MAX;

    CV_FUNCNAME( "cvGetGLCMDescriptor" );

    __BEGIN__;

    if( !GLCM )
        CV_ERROR( CV_StsNullPtr, "" );

    if( !(GLCM->descriptors) )
        CV_ERROR( CV_StsNullPtr, "" );

    if( (unsigned)step >= (unsigned)(GLCM->numMatrices))
        CV_ERROR( CV_StsOutOfRange, "step is not in 0 .. GLCM->numMatrices - 1" );

    if( (unsigned)descriptor >= (unsigned)(GLCM->numDescriptors))
        CV_ERROR( CV_StsOutOfRange, "descriptor is not in 0 .. GLCM->numDescriptors - 1" );

    value = GLCM->descriptors[step][descriptor];

    __END__;

    return value;
}


CV_IMPL void
cvGetGLCMDescriptorStatistics( CvGLCM* GLCM, int descriptor,
                               double* _average, double* _standardDeviation )
{
    CV_FUNCNAME( "cvGetGLCMDescriptorStatistics" );

    if( _average )
        *_average = DBL_MAX;

    if( _standardDeviation )
        *_standardDeviation = DBL_MAX;

    __BEGIN__;

    int matrixLoop, numMatrices;
    double average = 0, squareSum = 0;

    if( !GLCM )
        CV_ERROR( CV_StsNullPtr, "" );

    if( !(GLCM->descriptors))
        CV_ERROR( CV_StsNullPtr, "Descriptors are not calculated" );

    if( (unsigned)descriptor >= (unsigned)(GLCM->numDescriptors) )
        CV_ERROR( CV_StsOutOfRange, "Descriptor index is out of range" );

    numMatrices = GLCM->numMatrices;

    for( matrixLoop = 0; matrixLoop < numMatrices; matrixLoop++ )
    {
        double temp = GLCM->descriptors[ matrixLoop ][ descriptor ];
        average += temp;
        squareSum += temp*temp;
    }

    average /= numMatrices;

    if( _average )
        *_average = average;

    if( _standardDeviation )
        *_standardDeviation = sqrt( (squareSum - average*average*numMatrices)/(numMatrices-1));

    __END__;
}


CV_IMPL IplImage*
cvCreateGLCMImage( CvGLCM* GLCM, int step )
{
    IplImage* dest = 0;

    CV_FUNCNAME( "cvCreateGLCMImage" );

    __BEGIN__;

    float* destData;
    int sideLoop1, sideLoop2;

    if( !GLCM )
        CV_ERROR( CV_StsNullPtr, "" );

    if( !(GLCM->matrices) )
        CV_ERROR( CV_StsNullPtr, "Matrices are not allocated" );

    if( (unsigned)step >= (unsigned)(GLCM->numMatrices) )
        CV_ERROR( CV_StsOutOfRange, "The step index is out of range" );

    dest = cvCreateImage( cvSize( GLCM->matrixSideLength, GLCM->matrixSideLength ), IPL_DEPTH_32F, 1 );
    destData = (float*)(dest->imageData);

    for( sideLoop1 = 0; sideLoop1 < GLCM->matrixSideLength;
                        sideLoop1++, (float*&)destData += dest->widthStep )
    {
        for( sideLoop2=0; sideLoop2 < GLCM->matrixSideLength; sideLoop2++ )
        {
            double matrixValue = GLCM->matrices[step][sideLoop1][sideLoop2];
            destData[ sideLoop2 ] = (float)matrixValue;
        }
    }

    __END__;

    if( cvGetErrStatus() < 0 )
        cvReleaseImage( &dest );

    return dest;
}

