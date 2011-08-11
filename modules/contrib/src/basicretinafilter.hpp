/*#******************************************************************************
** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
** 
** By downloading, copying, installing or using the software you agree to this license.
** If you do not agree to this license, do not download, install,
** copy or use the software.
** 
** 
** HVStools : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
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
**               For Human Visual System tools (hvstools)
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

//using namespace std;
namespace cv
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
	inline void clearOutputBuffer(){_filterOutput=0;};

	/**
	* function which clears the secondary buffer of the object
	*/
	inline void clearSecondaryBuffer(){_localBuffer=0;};

	/**
	* function which clears the output and the secondary buffer of the object
	*/
	inline void clearAllBuffers(){clearOutputBuffer();clearSecondaryBuffer();};

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
	void resize(const unsigned int NBpixels){std::cerr<<"error, not accessible method"<<std::endl;};

	/**
	*  low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
	* @param inputFrame: the input image to be processed
	* @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
	* @return the processed image, the output is reachable later by using function getOutput()
	*/
	const std::valarray<double> &runFilter_LPfilter(const std::valarray<double> &inputFrame, const unsigned int filterIndex=0); // run the LP filter for a new frame input and save result in _filterOutput

	/**
	* low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
	* @param inputFrame: the input image to be processed
	* @param outputFrame: the output buffer in which the result is writed
	* @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
	*/
	void runFilter_LPfilter(const std::valarray<double> &inputFrame, std::valarray<double> &outputFrame, const unsigned int filterIndex=0); // run LP filter on a specific output adress

	/**
	*  low pass filter call and run (models the homogeneous cells network at the retina level, for example horizontal cells or photoreceptors)
	* @param inputOutputFrame: the input image to be processed on which the result is rewrited
	* @param filterIndex: the offset which specifies the parameter set that should be used for the filtering
	*/
	void runFilter_LPfilter_Autonomous(std::valarray<double> &inputOutputFrame, const unsigned int filterIndex=0);// run LP filter on the input data and rewrite it

	/**
	*  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
	* @param inputOutputFrame: the input image to be processed
	* @param localLuminance: an image which represents the local luminance of the inputFrame parameter, in general, it is its low pass spatial filtering
	* @return the processed image, the output is reachable later by using function getOutput()
	*/
	const std::valarray<double> &runFilter_LocalAdapdation(const std::valarray<double> &inputOutputFrame, const std::valarray<double> &localLuminance);// run local adaptation filter and save result in _filterOutput

	/**
	*  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
	* @param inputFrame: the input image to be processed
	* @param localLuminance: an image which represents the local luminance of the inputFrame parameter, in general, it is its low pass spatial filtering
	* @param outputFrame: the output buffer in which the result is writed
	*/
	void runFilter_LocalAdapdation(const std::valarray<double> &inputFrame, const std::valarray<double> &localLuminance, std::valarray<double> &outputFrame); // run local adaptation filter on a specific output adress

	/**
	*  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
	* @param inputFrame: the input image to be processed
	* @return the processed image, the output is reachable later by using function getOutput()
	*/
	const std::valarray<double> &runFilter_LocalAdapdation_autonomous(const std::valarray<double> &inputFrame);// run local adaptation filter and save result in _filterOutput

	/**
	*  local luminance adaptation call and run (contrast enhancement property of the photoreceptors)
	* @param inputFrame: the input image to be processed
	* @param outputFrame: the output buffer in which the result is writen
	*/
	void runFilter_LocalAdapdation_autonomous(const std::valarray<double> &inputFrame, std::valarray<double> &outputFrame); // run local adaptation filter on a specific output adress

	/**
	* run low pass filtering with progressive parameters (models the retina log sampling of the photoreceptors and its low pass filtering effect consequence: more powerfull low pass filtering effect on the corners)
	* @param inputFrame: the input image to be processed
	* @param filterIndex: the index which specifies the parameter set that should be used for the filtering
	* @return the processed image, the output is reachable later by using function getOutput() if outputFrame is NULL
	*/
	inline void runProgressiveFilter(std::valarray<double> &inputFrame, const unsigned int filterIndex=0){_spatiotemporalLPfilter_Irregular(&inputFrame[0], filterIndex);};

	/**
	* run low pass filtering with progressive parameters (models the retina log sampling of the photoreceptors and its low pass filtering effect consequence: more powerfull low pass filtering effect on the corners)
	* @param inputFrame: the input image to be processed
	* @param outputFrame: the output buffer in which the result is writen
	* @param filterIndex: the index which specifies the parameter set that should be used for the filtering
	*/
	inline void runProgressiveFilter(const std::valarray<double> &inputFrame, std::valarray<double> &outputFrame, const unsigned int filterIndex=0){_spatiotemporalLPfilter_Irregular(&inputFrame[0], &outputFrame[0], filterIndex);};

	/**
	* first order spatio-temporal low pass filter setup function
	* @param beta: gain of the filter (generally set to zero)
	* @param tau: time constant of the filter (unit is frame for video processing)
	* @param k: spatial constant of the filter (unit is pixels)
	* @param filterIndex: the index which specifies the parameter set that should be used for the filtering
	*/
	void setLPfilterParameters(const double beta, const double tau, const double k, const unsigned int filterIndex=0); // change the parameters of the filter

	/**
	* first order spatio-temporal low pass filter setup function
	* @param beta: gain of the filter (generally set to zero)
	* @param tau: time constant of the filter (unit is frame for video processing)
	* @param alpha0: spatial constant of the filter (unit is pixels) on the border of the image
	* @param filterIndex: the index which specifies the parameter set that should be used for the filtering
	*/
	void setProgressiveFilterConstants_CentredAccuracy(const double beta, const double tau, const double alpha0, const unsigned int filterIndex=0);

	/**
	* first order spatio-temporal low pass filter setup function
	* @param beta: gain of the filter (generally set to zero)
	* @param tau: time constant of the filter (unit is frame for video processing)
	* @param alpha0: spatial constant of the filter (unit is pixels) on the border of the image
	* @param accuracyMap an image (double format) which values range is between 0 and 1, where 0 means, apply no filtering and 1 means apply the filtering as specified in the parameters set, intermediate values allow to smooth variations of the filtering strenght
	* @param filterIndex: the index which specifies the parameter set that should be used for the filtering
	*/
	void setProgressiveFilterConstants_CustomAccuracy(const double beta, const double tau, const double alpha0, const std::valarray<double> &accuracyMap, const unsigned int filterIndex=0);

	/**
	* local luminance adaptation setup, this function should be applied for normal local adaptation (not for tone mapping operation)
	* @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
	* @param maxInputValue: the maximum amplitude value measured after local adaptation processing (c.f. function runFilter_LocalAdapdation & runFilter_LocalAdapdation_autonomous)
	* @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
	*/
	void setV0CompressionParameter(const double v0, const double maxInputValue, const double meanLuminance){ _v0=v0*maxInputValue; _localLuminanceFactor=v0; _localLuminanceAddon=maxInputValue*(1.0-v0); _maxInputValue=maxInputValue;};

	/**
	* update local luminance adaptation setup, initial maxInputValue is kept. This function should be applied for normal local adaptation (not for tone mapping operation)
	* @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
	* @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
	*/
	void setV0CompressionParameter(const double v0, const double meanLuminance){ this->setV0CompressionParameter(v0, _maxInputValue, meanLuminance);};

	/**
	* local luminance adaptation setup, this function should be applied for normal local adaptation (not for tone mapping operation)
	* @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
	*/
	void setV0CompressionParameter(const double v0){ _v0=v0*_maxInputValue; _localLuminanceFactor=v0; _localLuminanceAddon=_maxInputValue*(1.0-v0);};

	/**
	* local luminance adaptation setup, this function should be applied for local adaptation applied to tone mapping operation
	* @param v0: compression effect for the local luminance adaptation processing, set a value between 0.6 and 0.9 for best results, a high value yields to a high compression effect
	* @param maxInputValue: the maximum amplitude value measured after local adaptation processing (c.f. function runFilter_LocalAdapdation & runFilter_LocalAdapdation_autonomous)
	* @param meanLuminance: the a priori meann luminance of the input data (should be 128 for 8bits images but can vary greatly in case of High Dynamic Range Images (HDRI)
	*/
	void setV0CompressionParameterToneMapping(const double v0, const double maxInputValue, const double meanLuminance=128.0){ _v0=v0*maxInputValue; _localLuminanceFactor=1; _localLuminanceAddon=meanLuminance*_v0; _maxInputValue=maxInputValue;};

	/**
	* update compression parameters while keeping v0 parameter value
	* @param meanLuminance the input frame mean luminance
	*/
	inline void updateCompressionParameter(const double meanLuminance){_localLuminanceFactor=1; _localLuminanceAddon=meanLuminance*_v0;};

	/**
	* @return the v0 compression parameter used to compute the local adaptation
	*/
	const double getV0CompressionParameter(){ return _v0/_maxInputValue;};

	/**
	* @return the output result of the object
	*/
	inline const std::valarray<double> &getOutput() const {return _filterOutput;};

	/**
	* @return number of rows of the filter
	*/
	inline const unsigned int getNBrows(){return _filterOutput.getNBrows();};

	/**
	* @return number of columns of the filter
	*/
	inline const unsigned int getNBcolumns(){return _filterOutput.getNBcolumns();};

	/**
	* @return number of pixels of the filter
	*/
	inline const unsigned int getNBpixels(){return _filterOutput.getNBpixels();};

	/**
	* force filter output to be normalized between 0 and maxValue
	* @param maxValue: the maximum output value that is required
	*/
	inline void normalizeGrayOutput_0_maxOutputValue(const double maxValue){_filterOutput.normalizeGrayOutput_0_maxOutputValue(maxValue);};

	/**
	* force filter output to be normalized around 0 and rescaled with a sigmoide effect (extrem values saturation)
	* @param maxValue: the maximum output value that is required
	*/
	inline void normalizeGrayOutputCentredSigmoide(){_filterOutput.normalizeGrayOutputCentredSigmoide();};

	/**
	* force filter output to be normalized : data centering and std normalisation
	* @param maxValue: the maximum output value that is required
	*/
	inline void centerReductImageLuminance(){_filterOutput.centerReductImageLuminance();};

	/**
	* @return the maximum input buffer value
	*/
	inline const double getMaxInputValue(){return this->_maxInputValue;};

	/**
	* @return the maximum input buffer value
	*/
	inline void setMaxInputValue(const double newMaxInputValue){this->_maxInputValue=newMaxInputValue;};

protected:

	/////////////////////////
	// data buffers
	TemplateBuffer<double> _filterOutput; // primary buffer (contains processing outputs)
	std::valarray<double> _localBuffer; // local secondary buffer
	/////////////////////////
	// PARAMETERS
	unsigned int _halfNBrows;
	unsigned int _halfNBcolumns;

	// parameters buffers
	std::valarray <double>_filteringCoeficientsTable;
	std::valarray <double>_progressiveSpatialConstant;// pointer to a local table containing local spatial constant (allocated with the object)
	std::valarray <double>_progressiveGain;// pointer to a local table containing local spatial constant (allocated with the object)

	// local adaptation filtering parameters
	double _v0; //value used for local luminance adaptation function
	double _maxInputValue;
	double _meanInputValue;
	double _localLuminanceFactor;
	double _localLuminanceAddon;

	// protected data related to standard low pass filters parameters
	double _a;
	double _tau;
	double _gain;

	/////////////////////////
	// FILTERS METHODS

	// Basic low pass spation temporal low pass filter used by each retina filters
	void _spatiotemporalLPfilter(const double *inputFrame, double *LPfilterOutput, const unsigned int coefTableOffset=0);
	const double _squaringSpatiotemporalLPfilter(const double *inputFrame, double *outputFrame, const unsigned int filterIndex=0);

	// LP filter with an irregular spatial filtering

	// -> rewrites the input buffer
	void _spatiotemporalLPfilter_Irregular(double *inputOutputFrame, const unsigned int filterIndex=0);
	// writes the output on another buffer
	void _spatiotemporalLPfilter_Irregular(const double *inputFrame, double *outputFrame, const unsigned int filterIndex=0);
	// LP filter that squares the input and computes the output ONLY on the areas where the integrationAreas map are TRUE
	void _localSquaringSpatioTemporalLPfilter(const double *inputFrame, double *LPfilterOutput, const unsigned int *integrationAreas, const unsigned int filterIndex=0);

	// local luminance adaptation of the input in regard of localLuminance buffer
	void _localLuminanceAdaptation(const double *inputFrame, const double *localLuminance, double *outputFrame);
	// local luminance adaptation of the input in regard of localLuminance buffer, the input is rewrited and becomes the output
	void _localLuminanceAdaptation(double *inputOutputFrame, const double *localLuminance);
	// local adaptation applied on a range of values which can be positive and negative
	void _localLuminanceAdaptationPosNegValues(const double *inputFrame, const double *localLuminance, double *outputFrame);


	//////////////////////////////////////////////////////////////
	// 1D directional filters used for the 2D low pass filtering

	// 1D filters with image input
	void _horizontalCausalFilter_addInput(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	// 1D filters  with image input that is squared in the function
	void _squaringHorizontalCausalFilter(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	//  vertical anticausal filter that returns the mean value of its result
	const double _verticalAnticausalFilter_returnMeanValue(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);

	// most simple functions: only perform 1D filtering with output=input (no add on)
	void _horizontalCausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	void _horizontalAnticausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	void _verticalCausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);
	void _verticalAnticausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);

	// perform 1D filtering with output with varrying spatial coefficient
	void _horizontalCausalFilter_Irregular(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	void _horizontalCausalFilter_Irregular_addInput(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	void _horizontalAnticausalFilter_Irregular(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd);
	void _verticalCausalFilter_Irregular(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);
	void _verticalAnticausalFilter_Irregular_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd);


	// 1D filters in which the output is multiplied by _gain
	void _verticalAnticausalFilter_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd); // this functions affects _gain at the output
	void _horizontalAnticausalFilter_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd); // this functions affects _gain at the output

	// LP filter on specific parts of the picture instead of all the image
	// same functions (some of them) but take a binary flag to allow integration, false flag means, 0 at the output...
	void _local_squaringHorizontalCausalFilter(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas);
	void _local_horizontalAnticausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas);
	void _local_verticalCausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas);
	void _local_verticalAnticausalFilter_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas); // this functions affects _gain at the output

};

}
#endif


