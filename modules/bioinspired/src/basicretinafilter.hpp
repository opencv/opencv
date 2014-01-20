/*#******************************************************************************
** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
**
** By downloading, copying, installing or using the software you agree to this license.
** If you do not agree to this license, do not download, install,
** copy or use the software.
**
**
** bioinspired : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
** Use: extract still images & image sequences features, from contours details to motion spatio-temporal features, etc. for high level visual scene analysis. Also contribute to image enhancement/compression such as tone mapping.
**
** Maintainers : Listic lab (code author current affiliation & applications) and Gipsa Lab (original research origins & applications)
**
**  Creation - enhancement process 2007-2011
**      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
**
** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
** Refer to the following research paper for more information:
** Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
** This work have been carried out thanks to Jeanny Herault who's research and great discussions are the basis of all this work, please take a look at his book:
** Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
**
** The retina filter includes the research contributions of phd/research collegues from which code has been redrawn by the author :
** _take a look at the retinacolor.hpp module to discover Brice Chaix de Lavarene color mosaicing/demosaicing and the reference paper:
** ====> B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007
** _take a look at imagelogpolprojection.hpp to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions.
** ====> more informations in the above cited Jeanny Heraults's book.
**
**                          License Agreement
**               For Open Source Computer Vision Library
**
** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
**
**               For Human Visual System tools (bioinspired)
** Copyright (C) 2007-2011, LISTIC Lab, Annecy le Vieux and GIPSA Lab, Grenoble, France, all rights reserved.
**
** Third party copyrights are property of their respective owners.
**
** Redistribution and use in source and binary forms, with or without modification,
** are permitted provided that the following conditions are met:
**
** * Redistributions of source code must retain the above copyright notice,
**    this list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
**    this list of conditions and the following disclaimer in the documentation
**    and/or other materials provided with the distribution.
**
** * The name of the copyright holders may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
**
** This software is provided by the copyright holders and contributors "as is" and
** any express or implied warranties, including, but not limited to, the implied
** warranties of merchantability and fitness for a particular purpose are disclaimed.
** In no event shall the Intel Corporation or contributors be liable for any direct,
** indirect, incidental, special, exemplary, or consequential damages
** (including, but not limited to, procurement of substitute goods or services;
** loss of use, data, or profits; or business interruption) however caused
** and on any theory of liability, whether in contract, strict liability,
** or tort (including negligence or otherwise) arising in any way out of
** the use of this software, even if advised of the possibility of such damage.
*******************************************************************************/

#ifndef BASICRETINAELEMENT_HPP_
#define BASICRETINAELEMENT_HPP_

#include <cstring>


/**
* @class BasicRetinaFilter
* @brief Brief overview, this class provides tools for low level image processing:
* --> this class is able to perform:
* -> first order Low pass optimized filtering
* -> local luminance adaptation (able to correct back light problems and contrast enhancement)
* -> progressive low pass filter filtering (higher filtering on the borders than on the center)
* -> image data between 0 and 255 resampling with different options, linear rescaling, sigmoide)
*
* NOTE : initially the retina model was based on double format scalar values but
* a good memory/precision compromise is float...
* also the double format precision does not make so much sense from a biological point of view (neurons value coding is not so precise)
*
* TYPICAL USE:
*
* // create object at a specified picture size
* BasicRetinaFilter *_photoreceptorsPrefilter;
* _photoreceptorsPrefilter =new BasicRetinaFilter(sizeRows, sizeWindows);
*
* // init gain, spatial and temporal parameters:
* _photoreceptorsPrefilter->setCoefficientsTable(gain,temporalConstant, spatialConstant);
*
* // during program execution, call the filter for local luminance correction or low pass filtering for an input picture called "FrameBuffer":
* _photoreceptorsPrefilter->runFilter_LocalAdapdation(FrameBuffer);
* // or (Low pass first order filter)
* _photoreceptorsPrefilter->runFilter_LPfilter(FrameBuffer);
* // get output frame and its size:
* const unsigned int output_nbRows=_photoreceptorsPrefilter->getNBrows();
* const unsigned int output_nbColumns=_photoreceptorsPrefilter->getNBcolumns();
* const double *outputFrame=_photoreceptorsPrefilter->getOutput();
*
* // at the end of the program, destroy object:
* delete _photoreceptorsPrefilter;

* @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
* Creation date 2007
* synthesis of the work described in Alexandre BENOIT thesis: "Le systeme visuel humain au secours de la vision par ordinateur"
*/

#include <iostream>
#include "templatebuffer.hpp"

//#define __BASIC_RETINA_ELEMENT_DEBUG

namespace cv
{
namespace bioinspired
{
    class BasicRetinaFilter
    {
    public:

        /**
        * constructor of the base bio-inspired toolbox, parameters are only linked to imae input size and number of filtering capabilities of the object
        * @param NBrows: number of rows of the input image
        * @param NBcolumns: number of columns of the input image
        * @param parametersListSize: specifies the number of parameters set (each parameters set represents a specific low pass spatio-temporal filter)
        * @param useProgressiveFilter: specifies if the filter has irreguar (progressive) filtering capabilities (this can be activated later using setProgressiveFilterConstants_xxx methods)
        */
        BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize=1, const bool useProgressiveFilter=false);

        /**
        * standrad destructore
        */
        ~BasicRetinaFilter();

        /**
        * function which clears the output buffer of the object
        */
        inline void clearOutputBuffer() { _filterOutput = 0; }

        /**
        * function which clears the secondary buffer of the object
        */
        inline void clearSecondaryBuffer() { _localBuffer = 0; }

        /**
        * function which clears the output and the secondary buffer of the object
        */
        inline void clearAllBuffers() { clearOutputBuffer(); clearSecondaryBuffer(); }

        /**
        * resize basic retina filter object (resize all allocated buffers
        * @param NBrows: the new height size
        * @param NBcolumns: the new width size
        */
        void resize(const unsigned int NBrows, const unsigned int NBcolumns);

        /**
        * forbiden method inherited from parent std::valarray
        * prefer not to use this method since the filter matrix become vectors
        */
        void resize(const unsigned int) { std::cerr<<"error, not accessible method"<<std::endl; }

        /**
        *  low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
        * @param inputFrame: the input image to be processed
        * @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
        * @return the processed image, the output is reachable later by using function getOutput()
        */
        const std::valarray<float> &runFilter_LPfilter(const std::valarray<float> &inputFrame, const unsigned int filterIndex=0); // run the LP filter for a new frame input and save result in _filterOutput

        /**
        * low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
        * @param inputFrame: the input image to be processed
        * @param outputFrame: the output buffer in which the result is writed
        * @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
        */
        void runFilter_LPfilter(const std::valarray<float> &inputFrame, std::valarray<float> &outputFrame, const unsigned int filterIndex=0); // run LP filter on a specific output adress

        /**
        *  low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
        * @param inputOutputFrame: the input image to be processed on which the result is rewrited
        * @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
        */
        void runFilter_LPfilter_Autonomous(std::valarray<float> &inputOutputFrame, const unsigned int filterIndex=0);// run LP filter on the input data and rewrite it

        /**
        *  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
        * @param inputOutputFrame: the input image to be processed
        * @param localLuminance: an image which represents the local luminance of the inputFrame parameter, in general, it is its low pass spatial filtering
        * @return the processed image, the output is reachable later by using function getOutput()
        */
        const std::valarray<float> &runFilter_LocalAdapdation(const std::valarray<float> &inputOutputFrame, const std::valarray<float> &localLuminance);// run local adaptation filter and save result in _filterOutput

        /**
        *  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
        * @param inputFrame: the input image to be processed
        * @param localLuminance: an image which represents the local luminance of the inputFrame parameter, in general, it is its low pass spatial filtering
        * @param outputFrame: the output buffer in which the result is writed
        */
        void runFilter_LocalAdapdation(const std::valarray<float> &inputFrame, const std::valarray<float> &localLuminance, std::valarray<float> &outputFrame); // run local adaptation filter on a specific output adress

        /**
        *  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
        * @param inputFrame: the input image to be processed
        * @return the processed image, the output is reachable later by using function getOutput()
        */
        const std::valarray<float> &runFilter_LocalAdapdation_autonomous(const std::valarray<float> &inputFrame);// run local adaptation filter and save result in _filterOutput

        /**
        *  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
        * @param inputFrame: the input image to be processed
        * @param outputFrame: the output buffer in which the result is writen
        */
        void runFilter_LocalAdapdation_autonomous(const std::valarray<float> &inputFrame, std::valarray<float> &outputFrame); // run local adaptation filter on a specific output adress

        /**
        * run low pass filtering with progressive parameters (models the retina log sampling of the photoreceptors and its low pass filtering effect consequence: more powerfull low pass filtering effect on the corners)
        * @param inputFrame: the input image to be processed
        * @param filterIndex: the index which specifies the parameter set that should be used for the filtering
        * @return the processed image, the output is reachable later by using function getOutput() if outputFrame is NULL
        */
        inline void runProgressiveFilter(std::valarray<float> &inputFrame, const unsigned int filterIndex=0) { _spatiotemporalLPfilter_Irregular(&inputFrame[0], filterIndex); }

        /**
        * run low pass filtering with progressive parameters (models the retina log sampling of the photoreceptors and its low pass filtering effect consequence: more powerfull low pass filtering effect on the corners)
        * @param inputFrame: the input image to be processed
        * @param outputFrame: the output buffer in which the result is writen
        * @param filterIndex: the index which specifies the parameter set that should be used for the filtering
        */
        inline void runProgressiveFilter(const std::valarray<float> &inputFrame,
            std::valarray<float> &outputFrame,
            const unsigned int filterIndex=0)
        {_spatiotemporalLPfilter_Irregular(get_data(inputFrame), &outputFrame[0], filterIndex); }

        /**
        * first order spatio-temporal low pass filter setup function
        * @param beta: gain of the filter (generally set to zero)
        * @param tau: time constant of the filter (unit is frame for video processing)
        * @param k: spatial constant of the filter (unit is pixels)
        * @param filterIndex: the index which specifies the parameter set that should be used for the filtering
        */
        void setLPfilterParameters(const float beta, const float tau, const float k, const unsigned int filterIndex=0); // change the parameters of the filter

        /**
        * first order spatio-temporal low pass filter setup function
        * @param beta: gain of the filter (generally set to zero)
        * @param tau: time constant of the filter (unit is frame for video processing)
        * @param alpha0: spatial constant of the filter (unit is pixels) on the border of the image
        * @param filterIndex: the index which specifies the parameter set that should be used for the filtering
        */
        void setProgressiveFilterConstants_CentredAccuracy(const float beta, const float tau, const float alpha0, const unsigned int filterIndex=0);

        /**
        * first order spatio-temporal low pass filter setup function
        * @param beta: gain of the filter (generally set to zero)
        * @param tau: time constant of the filter (unit is frame for video processing)
        * @param alpha0: spatial constant of the filter (unit is pixels) on the border of the image
        * @param accuracyMap an image (float format) which values range is between 0 and 1, where 0 means, apply no filtering and 1 means apply the filtering as specified in the parameters set, intermediate values allow to smooth variations of the filtering strenght
        * @param filterIndex: the index which specifies the parameter set that should be used for the filtering
        */
        void setProgressiveFilterConstants_CustomAccuracy(const float beta, const float tau, const float alpha0, const std::valarray<float> &accuracyMap, const unsigned int filterIndex=0);

        /**
        * local luminance adaptation setup, this function should be applied for normal local adaptation (not for tone mapping operation)
        * @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
        * @param maxInputValue: the maximum amplitude value measured after local adaptation processing (c.f. function runFilter_LocalAdapdation & runFilter_LocalAdapdation_autonomous)
        * @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
        */
        void setV0CompressionParameter(const float v0, const float maxInputValue, const float)
        {
            _v0=v0*maxInputValue;
            _localLuminanceFactor=v0;
            _localLuminanceAddon=maxInputValue*(1.0f-v0);
            _maxInputValue=maxInputValue;
        }

        /**
        * update local luminance adaptation setup, initial maxInputValue is kept. This function should be applied for normal local adaptation (not for tone mapping operation)
        * @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
        * @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
        */
        void setV0CompressionParameter(const float v0, const float meanLuminance) { this->setV0CompressionParameter(v0, _maxInputValue, meanLuminance); }

        /**
        * local luminance adaptation setup, this function should be applied for normal local adaptation (not for tone mapping operation)
        * @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
        */
        void setV0CompressionParameter(const float v0)
        {
            _v0=v0*_maxInputValue;
            _localLuminanceFactor=v0;
            _localLuminanceAddon=_maxInputValue*(1.0f-v0);
        }

        /**
        * local luminance adaptation setup, this function should be applied for local adaptation applied to tone mapping operation
        * @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
        * @param maxInputValue: the maximum amplitude value measured after local adaptation processing (c.f. function runFilter_LocalAdapdation & runFilter_LocalAdapdation_autonomous)
        * @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
        */
        void setV0CompressionParameterToneMapping(const float v0, const float maxInputValue, const float meanLuminance=128.0f)
        {
            _v0=v0*maxInputValue;
            _localLuminanceFactor=1.0f;
            _localLuminanceAddon=meanLuminance*v0;
            _maxInputValue=maxInputValue;
        }

        /**
        * update compression parameters while keeping v0 parameter value
        * @param meanLuminance the input frame mean luminance
        */
        inline void updateCompressionParameter(const float meanLuminance)
        {
            _localLuminanceFactor=1;
            _localLuminanceAddon=meanLuminance*_v0;
        }

        /**
        * @return the v0 compression parameter used to compute the local adaptation
        */
        float getV0CompressionParameter() { return _v0/_maxInputValue; }

        /**
        * @return the output result of the object
        */
        inline const std::valarray<float> &getOutput() const { return _filterOutput; }

        /**
        * @return number of rows of the filter
        */
        inline unsigned int getNBrows() { return _filterOutput.getNBrows(); }

        /**
        * @return number of columns of the filter
        */
        inline unsigned int getNBcolumns() { return _filterOutput.getNBcolumns(); }

        /**
        * @return number of pixels of the filter
        */
        inline unsigned int getNBpixels() { return _filterOutput.getNBpixels(); }

        /**
        * force filter output to be normalized between 0 and maxValue
        * @param maxValue: the maximum output value that is required
        */
        inline void normalizeGrayOutput_0_maxOutputValue(const float maxValue) { _filterOutput.normalizeGrayOutput_0_maxOutputValue(maxValue); }

        /**
        * force filter output to be normalized around 0 and rescaled with a sigmoide effect (extrem values saturation)
        * @param maxValue: the maximum output value that is required
        */
        inline void normalizeGrayOutputCentredSigmoide() { _filterOutput.normalizeGrayOutputCentredSigmoide(); }

        /**
        * force filter output to be normalized : data centering and std normalisation
        * @param maxValue: the maximum output value that is required
        */
        inline void centerReductImageLuminance() { _filterOutput.centerReductImageLuminance(); }

        /**
        * @return the maximum input buffer value
        */
        inline float getMaxInputValue() { return _maxInputValue; }

        /**
        * @return the maximum input buffer value
        */
        inline void setMaxInputValue(const float newMaxInputValue) { this->_maxInputValue=newMaxInputValue; }

    protected:

        /////////////////////////
        // data buffers
        TemplateBuffer<float> _filterOutput; // primary buffer (contains processing outputs)
        std::valarray<float> _localBuffer; // local secondary buffer
        /////////////////////////
        // PARAMETERS
        unsigned int _halfNBrows;
        unsigned int _halfNBcolumns;

        // parameters buffers
        std::valarray <float>_filteringCoeficientsTable;
        std::valarray <float>_progressiveSpatialConstant;// pointer to a local table containing local spatial constant (allocated with the object)
        std::valarray <float>_progressiveGain;// pointer to a local table containing local spatial constant (allocated with the object)

        // local adaptation filtering parameters
        float _v0; //value used for local luminance adaptation function
        float _maxInputValue;
        float _meanInputValue;
        float _localLuminanceFactor;
        float _localLuminanceAddon;

        // protected data related to standard low pass filters parameters
        float _a;
        float _tau;
        float _gain;

        /////////////////////////
        // FILTERS METHODS

        // Basic low pass spation temporal low pass filter used by each retina filters
        void _spatiotemporalLPfilter(const float *inputFrame, float *LPfilterOutput, const unsigned int coefTableOffset=0);
        float _squaringSpatiotemporalLPfilter(const float *inputFrame, float *outputFrame, const unsigned int filterIndex=0);

        // LP filter with an irregular spatial filtering

        // -> rewrites the input buffer
        void _spatiotemporalLPfilter_Irregular(float *inputOutputFrame, const unsigned int filterIndex=0);
        // writes the output on another buffer
        void _spatiotemporalLPfilter_Irregular(const float *inputFrame, float *outputFrame, const unsigned int filterIndex=0);
        // LP filter that squares the input and computes the output ONLY on the areas where the integrationAreas map are TRUE
        void _localSquaringSpatioTemporalLPfilter(const float *inputFrame, float *LPfilterOutput, const unsigned int *integrationAreas, const unsigned int filterIndex=0);

        // local luminance adaptation of the input in regard of localLuminance buffer
        void _localLuminanceAdaptation(const float *inputFrame, const float *localLuminance, float *outputFrame, const bool updateLuminanceMean=true);
        // local luminance adaptation of the input in regard of localLuminance buffer, the input is rewrited and becomes the output
        void _localLuminanceAdaptation(float *inputOutputFrame, const float *localLuminance);
        // local adaptation applied on a range of values which can be positive and negative
        void _localLuminanceAdaptationPosNegValues(const float *inputFrame, const float *localLuminance, float *outputFrame);


        //////////////////////////////////////////////////////////////
        // 1D directional filters used for the 2D low pass filtering

        // 1D filters with image input
        void _horizontalCausalFilter_addInput(const float *inputFrame, float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
        // 1D filters  with image input that is squared in the function // parallelized with TBB
        void _squaringHorizontalCausalFilter(const float *inputFrame, float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
        //  vertical anticausal filter that returns the mean value of its result
        float _verticalAnticausalFilter_returnMeanValue(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);

        // most simple functions: only perform 1D filtering with output=input (no add on)
        void _horizontalCausalFilter(float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
        void _horizontalAnticausalFilter(float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);     // parallelized with TBB
        void _verticalCausalFilter(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);     // parallelized with TBB
        void _verticalAnticausalFilter(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);

        // perform 1D filtering with output with varrying spatial coefficient
        void _horizontalCausalFilter_Irregular(float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
        void _horizontalCausalFilter_Irregular_addInput(const float *inputFrame, float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
        void _horizontalAnticausalFilter_Irregular(float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const float *spatialConstantBuffer);   // parallelized with TBB
        void _verticalCausalFilter_Irregular(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const float *spatialConstantBuffer);   // parallelized with TBB
        void _verticalAnticausalFilter_Irregular_multGain(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);


        // 1D filters in which the output is multiplied by _gain
        void _verticalAnticausalFilter_multGain(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd); // this functions affects _gain at the output // parallelized with TBB
        void _horizontalAnticausalFilter_multGain(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd); // this functions affects _gain at the output

        // LP filter on specific parts of the picture instead of all the image
        // same functions (some of them) but take a binary flag to allow integration, false flag means, 0 at the output...
        void _local_squaringHorizontalCausalFilter(const float *inputFrame, float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas);
        void _local_horizontalAnticausalFilter(float *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas);
        void _local_verticalCausalFilter(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas);
        void _local_verticalAnticausalFilter_multGain(float *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas); // this functions affects _gain at the output

#ifdef MAKE_PARALLEL
        /******************************************************
        ** IF some parallelizing thread methods are available, then, main loops are parallelized using these functors
        ** ==> main idea paralellise main filters loops, then, only the most used methods are parallelized... TODO : increase the number of parallelised methods as necessary
        ** ==> functors names = Parallel_$$$ where $$$= the name of the serial method that is parallelised
        ** ==> functors constructors can differ from the parameters used with their related serial functions
        */

#define _DEBUG_TBB // define DEBUG_TBB in order to display additionnal data on stdout
        class Parallel_horizontalAnticausalFilter: public cv::ParallelLoopBody
        {
        private:
            float *outputFrame;
            unsigned int IDrowEnd, nbColumns;
            float filterParam_a;
        public:
            // constructor which takes the input image pointer reference reference and limits
            Parallel_horizontalAnticausalFilter(float *bufferToProcess, const unsigned int idEnd, const unsigned int nbCols, const float a )
                :outputFrame(bufferToProcess), IDrowEnd(idEnd), nbColumns(nbCols), filterParam_a(a)
            {
#ifdef DEBUG_TBB
                std::cout<<"Parallel_horizontalAnticausalFilter::Parallel_horizontalAnticausalFilter :"
                    <<"\n\t idEnd="<<IDrowEnd
                    <<"\n\t nbCols="<<nbColumns
                    <<"\n\t filterParam="<<filterParam_a
                    <<std::endl;
#endif
            }

            virtual void operator()( const Range& r ) const {

#ifdef DEBUG_TBB
                std::cout<<"Parallel_horizontalAnticausalFilter::operator() :"
                    <<"\n\t range size="<<r.size()
                    <<"\n\t first index="<<r.start
                    //<<"\n\t last index="<<filterParam
                    <<std::endl;
#endif
                for (int IDrow=r.start; IDrow!=r.end; ++IDrow)
                {
                    register float* outputPTR=outputFrame+(IDrowEnd-IDrow)*(nbColumns)-1;
                    register float result=0;
                    for (unsigned int index=0; index<nbColumns; ++index)
                    {
                        result = *(outputPTR)+  filterParam_a* result;
                        *(outputPTR--) = result;
                    }
                }
            }
        };

        class Parallel_horizontalCausalFilter_addInput: public cv::ParallelLoopBody
        {
        private:
            const float *inputFrame;
            float *outputFrame;
            unsigned int IDrowStart, nbColumns;
            float filterParam_a, filterParam_tau;
        public:
            Parallel_horizontalCausalFilter_addInput(const float *bufferToAddAsInputProcess, float *bufferToProcess, const unsigned int idStart, const unsigned int nbCols,  const float a,  const float tau)
                :inputFrame(bufferToAddAsInputProcess), outputFrame(bufferToProcess), IDrowStart(idStart), nbColumns(nbCols), filterParam_a(a), filterParam_tau(tau){}

            virtual void operator()( const Range& r ) const {
                for (int IDrow=r.start; IDrow!=r.end; ++IDrow)
                {
                    register float* outputPTR=outputFrame+(IDrowStart+IDrow)*nbColumns;
                    register const float* inputPTR=inputFrame+(IDrowStart+IDrow)*nbColumns;
                    register float result=0;
                    for (unsigned int index=0; index<nbColumns; ++index)
                    {
                        result = *(inputPTR++) + filterParam_tau**(outputPTR)+  filterParam_a* result;
                        *(outputPTR++) = result;
                    }
                }
            }
        };

        class Parallel_verticalCausalFilter: public cv::ParallelLoopBody
        {
        private:
            float *outputFrame;
            unsigned int nbRows, nbColumns;
            float filterParam_a;
        public:
            Parallel_verticalCausalFilter(float *bufferToProcess, const unsigned int nbRws, const unsigned int nbCols, const float a )
                :outputFrame(bufferToProcess), nbRows(nbRws), nbColumns(nbCols), filterParam_a(a){}

            virtual void operator()( const Range& r ) const {
                for (int IDcolumn=r.start; IDcolumn!=r.end; ++IDcolumn)
                {
                    register float result=0;
                    register float *outputPTR=outputFrame+IDcolumn;

                    for (unsigned int index=0; index<nbRows; ++index)
                    {
                        result = *(outputPTR) + filterParam_a * result;
                        *(outputPTR) = result;
                        outputPTR+=nbColumns;

                    }
                }
            }
        };

        class Parallel_verticalAnticausalFilter_multGain: public cv::ParallelLoopBody
        {
        private:
            float *outputFrame;
            unsigned int nbRows, nbColumns;
            float filterParam_a, filterParam_gain;
        public:
            Parallel_verticalAnticausalFilter_multGain(float *bufferToProcess, const unsigned int nbRws, const unsigned int nbCols, const float a, const float  gain)
                :outputFrame(bufferToProcess), nbRows(nbRws), nbColumns(nbCols), filterParam_a(a), filterParam_gain(gain){}

            virtual void operator()( const Range& r ) const {
                float* offset=outputFrame+nbColumns*nbRows-nbColumns;
                for (int IDcolumn=r.start; IDcolumn!=r.end; ++IDcolumn)
                {
                    register float result=0;
                    register float *outputPTR=offset+IDcolumn;

                    for (unsigned int index=0; index<nbRows; ++index)
                    {
                        result = *(outputPTR) + filterParam_a * result;
                        *(outputPTR) = filterParam_gain*result;
                        outputPTR-=nbColumns;

                    }
                }
            }
        };

        class Parallel_localAdaptation: public cv::ParallelLoopBody
        {
        private:
            const float *localLuminance, *inputFrame;
            float *outputFrame;
            float localLuminanceFactor, localLuminanceAddon, maxInputValue;
        public:
            Parallel_localAdaptation(const float *localLum, const float *inputImg, float *bufferToProcess, const float localLuminanceFact, const float localLuminanceAdd, const float maxInputVal)
                :localLuminance(localLum), inputFrame(inputImg),outputFrame(bufferToProcess), localLuminanceFactor(localLuminanceFact), localLuminanceAddon(localLuminanceAdd), maxInputValue(maxInputVal) {}

            virtual void operator()( const Range& r ) const {
                const float *localLuminancePTR=localLuminance+r.start;
                const float *inputFramePTR=inputFrame+r.start;
                float *outputFramePTR=outputFrame+r.start;
                for (register int IDpixel=r.start ; IDpixel!=r.end ; ++IDpixel, ++inputFramePTR, ++outputFramePTR)
                {
                    float X0=*(localLuminancePTR++)*localLuminanceFactor+localLuminanceAddon;
                    // TODO : the following line can lead to a divide by zero ! A small offset is added, take care if the offset is too large in case of High Dynamic Range images which can use very small values...
                    *(outputFramePTR) = (maxInputValue+X0)**inputFramePTR/(*inputFramePTR +X0+0.00000000001f);
                    //std::cout<<"BasicRetinaFilter::inputFrame[IDpixel]=%f, X0=%f, outputFrame[IDpixel]=%f\n", inputFrame[IDpixel], X0, outputFrame[IDpixel]);
                }
            }
        };

        //////////////////////////////////////////
        /// Specific filtering methods which manage non const spatial filtering parameter (used By retinacolor and LogProjectors)
        class Parallel_horizontalAnticausalFilter_Irregular: public cv::ParallelLoopBody
        {
        private:
            float *outputFrame;
            const float *spatialConstantBuffer;
            unsigned int IDrowEnd, nbColumns;
        public:
            Parallel_horizontalAnticausalFilter_Irregular(float *bufferToProcess, const float *spatialConst, const unsigned int idEnd, const unsigned int nbCols)
                :outputFrame(bufferToProcess), spatialConstantBuffer(spatialConst), IDrowEnd(idEnd), nbColumns(nbCols){}

            virtual void operator()( const Range& r ) const {

                for (int IDrow=r.start; IDrow!=r.end; ++IDrow)
                {
                    register float* outputPTR=outputFrame+(IDrowEnd-IDrow)*(nbColumns)-1;
                    register const float* spatialConstantPTR=spatialConstantBuffer+(IDrowEnd-IDrow)*(nbColumns)-1;
                    register float result=0;
                    for (unsigned int index=0; index<nbColumns; ++index)
                    {
                        result = *(outputPTR)+  *(spatialConstantPTR--)* result;
                        *(outputPTR--) = result;
                    }
                }
            }
        };

        class Parallel_verticalCausalFilter_Irregular: public cv::ParallelLoopBody
        {
        private:
            float *outputFrame;
            const float *spatialConstantBuffer;
            unsigned int nbRows, nbColumns;
        public:
            Parallel_verticalCausalFilter_Irregular(float *bufferToProcess, const float *spatialConst, const unsigned int nbRws, const unsigned int nbCols)
                :outputFrame(bufferToProcess), spatialConstantBuffer(spatialConst), nbRows(nbRws), nbColumns(nbCols){}

            virtual void operator()( const Range& r ) const {
                for (int IDcolumn=r.start; IDcolumn!=r.end; ++IDcolumn)
                {
                    register float result=0;
                    register float *outputPTR=outputFrame+IDcolumn;
                    register const float* spatialConstantPTR=spatialConstantBuffer+IDcolumn;
                    for (unsigned int index=0; index<nbRows; ++index)
                    {
                        result = *(outputPTR) +  *(spatialConstantPTR) * result;
                        *(outputPTR) = result;
                        outputPTR+=nbColumns;
                        spatialConstantPTR+=nbColumns;
                    }
                }
            }
        };

#endif

    };

}// end of namespace bioinspired
}// end of namespace cv
#endif
