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

#ifndef ParvoRetinaFilter_H_
#define ParvoRetinaFilter_H_

/**
* @class ParvoRetinaFilter
* @brief class which describes the OPL retina model and the Inner Plexiform Layer parvocellular channel of the retina:
* -> performs a contours extraction with powerfull local data enhancement as at the retina level
* -> spectrum whitening occurs at the OPL (Outer Plexiform Layer) of the retina: corrects the 1/f spectrum tendancy of natural images
* ---> enhances details with mid spatial frequencies, attenuates low spatial frequencies (luminance), attenuates high temporal frequencies and high spatial frequencies, etc.
*
* TYPICAL USE:
*
* // create object at a specified picture size
* ParvoRetinaFilter *contoursExtractor;
* contoursExtractor =new ParvoRetinaFilter(frameSizeRows, frameSizeColumns);
*
* // init gain, spatial and temporal parameters:
* contoursExtractor->setCoefficientsTable(0, 0.7, 1, 0, 7, 1);
*
* // during program execution, call the filter for contours extraction for an input picture called "FrameBuffer":
* contoursExtractor->runfilter(FrameBuffer);
*
* // get the output frame, check in the class description below for more outputs:
* const float *contours=contoursExtractor->getParvoONminusOFF();
*
* // at the end of the program, destroy object:
* delete contoursExtractor;

* @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
* Creation date 2007
* Based on Alexandre BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"
*
*/

#include "basicretinafilter.hpp"


//#define _OPL_RETINA_ELEMENT_DEBUG

namespace cv
{
//retina classes that derivate from the Basic Retrina class
class ParvoRetinaFilter: public BasicRetinaFilter
{

public:
	/**
	* constructor parameters are only linked to image input size
	* @param NBrows: number of rows of the input image
	* @param NBcolumns: number of columns of the input image
	*/
	ParvoRetinaFilter(const unsigned int NBrows=480, const unsigned int NBcolumns=640);

	/**
	* standard desctructor
	*/
	virtual ~ParvoRetinaFilter();

	/**
	* resize method, keeps initial parameters, all buffers are flushed
	* @param NBrows: number of rows of the input image
	* @param NBcolumns: number of columns of the input image
	*/
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	* function that clears all buffers of the object
	*/
	void clearAllBuffers();

	/**
	* setup the OPL and IPL parvo channels
	* @param beta1: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, the amplitude is boosted but it should only be used for values rescaling... if needed
	* @param tau1: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
	* @param k1: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
	* @param beta2: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
	* @param tau2: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
	* @param k2: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
	*/
	void setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2);

	/**
	* setup more precisely the low pass filter used for the ganglion cells low pass filtering (used for local luminance adaptation)
	* @param tau: time constant of the filter (unit is frame for video processing)
	* @param k: spatial constant of the filter (unit is pixels)
	*/
	void setGanglionCellsLocalAdaptationLPfilterParameters(const float tau, const float k){BasicRetinaFilter::setLPfilterParameters(0, tau, k, 2);}; // change the parameters of the filter


	/**
	* launch filter that runs the OPL spatiotemporal filtering and optionally finalizes IPL Pagno filter (model of the Parvocellular channel of the Inner Plexiform Layer of the retina)
	* @param inputFrame: the input image to be processed, this can be the direct gray level input frame, but a better efficacy is expected if the input is preliminary processed by the photoreceptors local adaptation possible to acheive with the help of a BasicRetinaFilter object
	* @param useParvoOutput: set true if the final IPL filtering step has to be computed (local contrast enhancement)
	* @return the processed Parvocellular channel output (updated only if useParvoOutput is true)
	* @details: in any case, after this function call, photoreceptors and horizontal cells output are updated, use getPhotoreceptorsLPfilteringOutput() and getHorizontalCellsOutput() to get them
	* also, bipolar cells output are accessible (difference between photoreceptors and horizontal cells, ON output has positive values, OFF ouput has negative values), use the following access methods: getBipolarCellsON() and getBipolarCellsOFF()if useParvoOutput is true,
	* if useParvoOutput is true, the complete Parvocellular channel is computed, more outputs are updated and can be accessed threw: getParvoON(), getParvoOFF() and their difference with getOutput()
	*/
	const std::valarray<float> &runFilter(const std::valarray<float> &inputFrame, const bool useParvoOutput=true); // output return is _parvocellularOutputONminusOFF

	/**
	* @return the output of the photoreceptors filtering step (high cut frequency spatio-temporal low pass filter)
	*/
	inline const std::valarray<float> &getPhotoreceptorsLPfilteringOutput() const {return _photoreceptorsOutput;};

	/**
	* @return the output of the photoreceptors filtering step (low cut frequency spatio-temporal low pass filter)
	*/
	inline const std::valarray<float> &getHorizontalCellsOutput() const { return _horizontalCellsOutput;};

	/**
	* @return the output Parvocellular ON channel of the retina model
	*/
	inline const std::valarray<float> &getParvoON() const {return _parvocellularOutputON;};

	/**
	* @return the output Parvocellular OFF channel of the retina model
	*/
	inline const std::valarray<float> &getParvoOFF() const {return _parvocellularOutputOFF;};

	/**
	* @return the output of the Bipolar cells of the ON channel of the retina model same as function getParvoON() but without luminance local adaptation
	*/
	inline const std::valarray<float> &getBipolarCellsON() const {return _bipolarCellsOutputON;};

	/**
	* @return the output of the Bipolar cells of the OFF channel of the retina model same as function getParvoON() but without luminance local adaptation
	*/
	inline const std::valarray<float> &getBipolarCellsOFF() const {return _bipolarCellsOutputOFF;};

	/**
	* @return the photoreceptors's temporal constant
	*/
	inline const float getPhotoreceptorsTemporalConstant(){return this->_filteringCoeficientsTable[2];};

	/**
	* @return the horizontal cells' temporal constant
	*/
	inline const float getHcellsTemporalConstant(){return this->_filteringCoeficientsTable[5];};

private:
	// template buffers
	std::valarray <float>_photoreceptorsOutput;
	std::valarray <float>_horizontalCellsOutput;
	std::valarray <float>_parvocellularOutputON;
	std::valarray <float>_parvocellularOutputOFF;
	std::valarray <float>_bipolarCellsOutputON;
	std::valarray <float>_bipolarCellsOutputOFF;
	std::valarray <float>_localAdaptationOFF;
	std::valarray <float> *_localAdaptationON;
	TemplateBuffer<float> *_parvocellularOutputONminusOFF;
	// private functions
	void _OPL_OnOffWaysComputing();

};
}
#endif

