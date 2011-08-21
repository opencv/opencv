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

/**
* @class RetinaColor a color multilexing/demultiplexing (demosaicing) based on a human vision inspiration. Different mosaicing strategies can be used, included random sampling !
* => please take a look at the nice and efficient demosaicing strategy introduced by B.Chaix de Lavarene, take a look at the cited paper for more mathematical details
* @brief Retina color sampling model which allows classical bayer sampling, random and potentially several other method ! Low color errors on corners !
* -> Based on the research of:
*		.Brice Chaix Lavarene (chaix@lis.inpg.fr)
*		.Jeanny Herault (herault@lis.inpg.fr)
*		.David Alleyson (david.alleyson@upmf-grenoble.fr)
*      .collaboration: alexandre benoit (benoit.alexandre.vision@gmail.com or benoit@lis.inpg.fr)
* Please cite: B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007
* @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC / Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
* Creation date 2007
*/

#ifndef RETINACOLOR_HPP_
#define RETINACOLOR_HPP_

#include "basicretinafilter.hpp"

//#define __RETINACOLORDEBUG //define RETINACOLORDEBUG in order to display debug data

namespace cv
{

class RetinaColor: public BasicRetinaFilter
{
public:
	/**
	* @typedef which allows to select the type of photoreceptors color sampling
	*/

	/**
	* constructor of the retina color processing model
	* @param NBrows: number of rows of the input image
	* @param NBcolumns: number of columns of the input image
	* @param samplingMethod: the chosen color sampling method
	*/
	RetinaColor(const unsigned int NBrows, const unsigned int NBcolumns, const RETINA_COLORSAMPLINGMETHOD samplingMethod=RETINA_COLOR_DIAGONAL);

	/**
	* standard destructor
	*/
	virtual ~RetinaColor();

	/**
	* function that clears all buffers of the object
	*/
	void clearAllBuffers();

	/**
	* resize retina color filter object (resize all allocated buffers)
	* @param NBrows: the new height size
	* @param NBcolumns: the new width size
	*/
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);


	/**
	* color multiplexing function: a demultiplexed RGB frame of size M*N*3 is transformed into a multiplexed M*N*1 pixels frame where each pixel is either Red, or Green or Blue
	* @param inputRGBFrame: the input RGB frame to be processed
	* @return, nothing but the multiplexed frame is available by the use of the getMultiplexedFrame() function
	*/
	inline void runColorMultiplexing(const std::valarray<float> &inputRGBFrame){runColorMultiplexing(inputRGBFrame, *_multiplexedFrame);};

	/**
	* color multiplexing function: a demultipleed RGB frame of size M*N*3 is transformed into a multiplexed M*N*1 pixels frame where each pixel is either Red, or Green or Blue if using RGB images
	* @param demultiplexedInputFrame: the demultiplexed input frame to be processed of size M*N*3
	* @param multiplexedFrame: the resulting multiplexed frame
	*/
	void runColorMultiplexing(const std::valarray<float> &demultiplexedInputFrame, std::valarray<float> &multiplexedFrame);

	/**
	* color demultiplexing function: a multiplexed frame of size M*N*1 pixels is transformed into a RGB demultiplexed M*N*3 pixels frame
	* @param multiplexedColorFrame: the input multiplexed frame to be processed
	* @param adaptiveFiltering: specifies if an adaptive filtering has to be perform rather than standard filtering (adaptive filtering allows a better rendering)
	* @param maxInputValue: the maximum input data value (should be 255 for 8 bits images but it can change in the case of High Dynamic Range Images (HDRI)
	* @return, nothing but the output demultiplexed frame is available by the use of the getDemultiplexedColorFrame() function, also use getLuminance() and getChrominance() in order to retreive either luminance or chrominance
	*/
	void runColorDemultiplexing(const std::valarray<float> &multiplexedColorFrame, const bool adaptiveFiltering=false, const float maxInputValue=255.0);

	/**
	* activate color saturation as the final step of the color demultiplexing process
	* -> this saturation is a sigmoide function applied to each channel of the demultiplexed image.
	* @param saturateColors: boolean that activates color saturation (if true) or desactivate (if false)
	* @param colorSaturationValue: the saturation factor
	* */
	void setColorSaturation(const bool saturateColors=true, const float colorSaturationValue=4.0){_saturateColors=saturateColors; _colorSaturationValue=colorSaturationValue;};

	/**
	* set parameters of the low pass spatio-temporal filter used to retreive the low chrominance
	* @param beta: gain of the filter (generally set to zero)
	* @param tau: time constant of the filter (unit is frame for video processing), typically 0 when considering static processing, 1 or more if a temporal smoothing effect is required
	* @param k: spatial constant of the filter (unit is pixels), typical value is 2.5
	*/
	void setChrominanceLPfilterParameters(const float beta, const float tau, const float k){setLPfilterParameters(beta, tau, k);};

	/**
	* apply to the retina color output the Krauskopf transformation which leads to an opponent color system: output colorspace if Acr1cr2 if input of the retina was LMS color space
	* @param result: the input buffer to fill with the transformed colorspace retina output
	* @return true if process ended successfully
	*/
	const bool applyKrauskopfLMS2Acr1cr2Transform(std::valarray<float> &result);

	/**
	* apply to the retina color output the CIE Lab color transformation
	* @param result: the input buffer to fill with the transformed colorspace retina output
	* @return true if process ended successfully
	*/
	const bool applyLMS2LabTransform(std::valarray<float> &result);

	/**
	* @return the multiplexed frame result (use this after function runColorMultiplexing)
	*/
	inline const std::valarray<float> &getMultiplexedFrame() const {return *_multiplexedFrame;};

	/**
	* @return the demultiplexed frame result (use this after function runColorDemultiplexing)
	*/
	inline const std::valarray<float> &getDemultiplexedColorFrame() const {return _demultiplexedColorFrame;};

	/**
	* @return the luminance of the processed frame (use this after function runColorDemultiplexing)
	*/
	inline const std::valarray<float> &getLuminance() const {return *_luminance;};

	/**
	* @return the chrominance of the processed frame (use this after function runColorDemultiplexing)
	*/
	inline const std::valarray<float> &getChrominance() const {return _chrominance;};

	/**
	* standard 0 to 255 image clipping function appled to RGB images (of size M*N*3 pixels)
	* @param inputOutputBuffer: the image to be normalized (rewrites the input), if no parameter, then, the built in buffer reachable by getOutput() function is normalized
	* @param maxOutputValue: the maximum value allowed at the output (values superior to it would be clipped
	*/
	void clipRGBOutput_0_maxInputValue(float *inputOutputBuffer, const float maxOutputValue=255.0);

	/**
	* standard 0 to 255 image normalization function appled to RGB images (of size M*N*3 pixels)
	* @param maxOutputValue: the maximum value allowed at the output (values superior to it would be clipped
	*/
	void normalizeRGBOutput_0_maxOutputValue(const float maxOutputValue=255.0);

	/**
	* return the color sampling map: a Nrows*Mcolumns image in which each pixel value is the ofsset adress which gives the adress of the sampled pixel on an Nrows*Mcolumns*3 color image ordered by layers: layer1, layer2, layer3
	*/
	inline const std::valarray<unsigned int> &getSamplingMap() const {return _colorSampling;};

	/**
	* function used (to bypass processing) to manually set the color output
	* @param demultiplexedImage: the color image (luminance+chrominance) which has to be written in the object buffer
	*/
	inline void setDemultiplexedColorFrame(const std::valarray<float> &demultiplexedImage){_demultiplexedColorFrame=demultiplexedImage;};

protected:

	// private functions
	RETINA_COLORSAMPLINGMETHOD _samplingMethod;
	bool _saturateColors;
	float _colorSaturationValue;
	// links to parent buffers (more convienient names
	TemplateBuffer<float> *_luminance;
	std::valarray<float> *_multiplexedFrame;
	// instance buffers
	std::valarray<unsigned int> _colorSampling; // table (size (_nbRows*_nbColumns) which specifies the color of each pixel
	std::valarray<float> _RGBmosaic;
	std::valarray<float> _tempMultiplexedFrame;
	std::valarray<float> _demultiplexedTempBuffer;
	std::valarray<float> _demultiplexedColorFrame;
	std::valarray<float> _chrominance;
	std::valarray<float> _colorLocalDensity;// buffer which contains the local density of the R, G and B photoreceptors for a normalization use
	std::valarray<float> _imageGradient;

	// variables
	float _pR, _pG, _pB; // probabilities of color R, G and B
	bool _objectInit;

	// protected functions
	void _initColorSampling();
	void _interpolateImageDemultiplexedImage(float *inputOutputBuffer);
	void _interpolateSingleChannelImage111(float *inputOutputBuffer);
	void _interpolateBayerRGBchannels(float *inputOutputBuffer);
	void _applyRIFfilter(const float *sourceBuffer, float *destinationBuffer);
	void _getNormalizedContoursImage(const float *inputFrame, float *outputFrame);
	// -> special adaptive filters dedicated to low pass filtering on the chrominance (skeeps filtering on the edges)
	void _adaptiveSpatialLPfilter(const float *inputFrame,  float *outputFrame);
	void _adaptiveHorizontalCausalFilter_addInput(const float *inputFrame, float *outputFrame, const unsigned int IDrowStart, const unsigned int IDrowEnd);
	void _adaptiveHorizontalAnticausalFilter(float *outputFrame, const unsigned int IDrowStart, const unsigned int IDrowEnd);
	void _adaptiveVerticalCausalFilter(float *outputFrame, const unsigned int IDcolumnStart, const unsigned int IDcolumnEnd);
	void _adaptiveVerticalAnticausalFilter_multGain(float *outputFrame, const unsigned int IDcolumnStart, const unsigned int IDcolumnEnd);
	void _computeGradient(const float *luminance);
	void _normalizeOutputs_0_maxOutputValue(void);

	// color space transform
	void _applyImageColorSpaceConversion(const std::valarray<float> &inputFrame, std::valarray<float> &outputFrame, const float *transformTable);

};
}

#endif /*RETINACOLOR_HPP_*/


