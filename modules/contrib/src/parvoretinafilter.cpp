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

#include "precomp.hpp"

#include "parvoretinafilter.hpp"

// @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/

#include <iostream>
#include <cmath>

namespace cv
{
//////////////////////////////////////////////////////////
//                 OPL RETINA FILTER
//////////////////////////////////////////////////////////

// Constructor and Desctructor of the OPL retina filter

ParvoRetinaFilter::ParvoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
:BasicRetinaFilter(NBrows, NBcolumns, 3),
 _photoreceptorsOutput(NBrows*NBcolumns),
 _horizontalCellsOutput(NBrows*NBcolumns),
 _parvocellularOutputON(NBrows*NBcolumns),
 _parvocellularOutputOFF(NBrows*NBcolumns),
 _bipolarCellsOutputON(NBrows*NBcolumns),
 _bipolarCellsOutputOFF(NBrows*NBcolumns),
 _localAdaptationOFF(NBrows*NBcolumns)
{
	// link to the required local parent adaptation buffers
	_localAdaptationON=&_localBuffer;
	_parvocellularOutputONminusOFF=&_filterOutput;
	// (*_localAdaptationON)=&_localBuffer;
	// (*_parvocellularOutputONminusOFF)=&(BasicRetinaFilter::TemplateBuffer);

	// init: set all the values to 0
	clearAllBuffers();


#ifdef OPL_RETINA_ELEMENT_DEBUG
	std::cout<<"ParvoRetinaFilter::Init OPL retina filter at specified frame size OK\n"<<std::endl;
#endif

}

ParvoRetinaFilter::~ParvoRetinaFilter()
{

#ifdef OPL_RETINA_ELEMENT_DEBUG
	std::cout<<"ParvoRetinaFilter::Delete OPL retina filter OK"<<std::endl;
#endif
}

////////////////////////////////////
// functions of the PARVO filter
////////////////////////////////////

// function that clears all buffers of the object
void ParvoRetinaFilter::clearAllBuffers()
{
	BasicRetinaFilter::clearAllBuffers();
	_photoreceptorsOutput=0;
	_horizontalCellsOutput=0;
	_parvocellularOutputON=0;
	_parvocellularOutputOFF=0;
	_bipolarCellsOutputON=0;
	_bipolarCellsOutputOFF=0;
	_localAdaptationOFF=0;
}

/**
* resize parvo retina filter object (resize all allocated buffers
* @param NBrows: the new height size
* @param NBcolumns: the new width size
*/
void ParvoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	BasicRetinaFilter::resize(NBrows, NBcolumns);
	_photoreceptorsOutput.resize(NBrows*NBcolumns);
	_horizontalCellsOutput.resize(NBrows*NBcolumns);
	_parvocellularOutputON.resize(NBrows*NBcolumns);
	_parvocellularOutputOFF.resize(NBrows*NBcolumns);
	_bipolarCellsOutputON.resize(NBrows*NBcolumns);
	_bipolarCellsOutputOFF.resize(NBrows*NBcolumns);
	_localAdaptationOFF.resize(NBrows*NBcolumns);

	// link to the required local parent adaptation buffers
	_localAdaptationON=&_localBuffer;
	_parvocellularOutputONminusOFF=&_filterOutput;

	// clean buffers
	clearAllBuffers();
}

// change the parameters of the filter
void ParvoRetinaFilter::setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2)
{
	// init photoreceptors low pass filter
	setLPfilterParameters(beta1, tau1, k1);
	// init horizontal cells low pass filter
	setLPfilterParameters(beta2, tau2, k2, 1);
	// init parasol ganglion cells low pass filter (default parameters)
	setLPfilterParameters(0, tau1, k1, 2);

}

// update/set size of the frames

// run filter for a new frame input
// output return is (*_parvocellularOutputONminusOFF)
const std::valarray<float> &ParvoRetinaFilter::runFilter(const std::valarray<float> &inputFrame, const bool useParvoOutput)
{
	_spatiotemporalLPfilter(get_data(inputFrame), &_photoreceptorsOutput[0]);
	_spatiotemporalLPfilter(&_photoreceptorsOutput[0], &_horizontalCellsOutput[0], 1);
	_OPL_OnOffWaysComputing();

	if (useParvoOutput)
	{
		// local adaptation processes on ON and OFF ways
		_spatiotemporalLPfilter(&_bipolarCellsOutputON[0], &(*_localAdaptationON)[0], 2);
		_localLuminanceAdaptation(&_parvocellularOutputON[0], &(*_localAdaptationON)[0]);

		_spatiotemporalLPfilter(&_bipolarCellsOutputOFF[0], &_localAdaptationOFF[0], 2);
		_localLuminanceAdaptation(&_parvocellularOutputOFF[0], &_localAdaptationOFF[0]);

		//// Final loop that computes the main output of this filter
		//
		//// loop that makes the difference between photoreceptor cells output and horizontal cells
		//// positive part goes on the ON way, negative pat goes on the OFF way
		register float *parvocellularOutputONminusOFF_PTR=&(*_parvocellularOutputONminusOFF)[0];
		register float *parvocellularOutputON_PTR=&_parvocellularOutputON[0];
		register float *parvocellularOutputOFF_PTR=&_parvocellularOutputOFF[0];

		for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel)
			*(parvocellularOutputONminusOFF_PTR++)= (*(parvocellularOutputON_PTR++)-*(parvocellularOutputOFF_PTR++));
	}
	return (*_parvocellularOutputONminusOFF);
}

void ParvoRetinaFilter::_OPL_OnOffWaysComputing()
{
	// loop that makes the difference between photoreceptor cells output and horizontal cells
	// positive part goes on the ON way, negative pat goes on the OFF way
	register float *photoreceptorsOutput_PTR= &_photoreceptorsOutput[0];
	register float *horizontalCellsOutput_PTR= &_horizontalCellsOutput[0];
	register float *bipolarCellsON_PTR = &_bipolarCellsOutputON[0];
	register float *bipolarCellsOFF_PTR = &_bipolarCellsOutputOFF[0];
	register float *parvocellularOutputON_PTR= &_parvocellularOutputON[0];
	register float *parvocellularOutputOFF_PTR= &_parvocellularOutputOFF[0];

	// compute bipolar cells response equal to photoreceptors minus horizontal cells response
	// and copy the result on parvo cellular outputs... keeping time before their local contrast adaptation for final result
	for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel)
	{
		float pixelDifference = *(photoreceptorsOutput_PTR++) -*(horizontalCellsOutput_PTR++);
		// test condition to allow write pixelDifference in ON or OFF buffer and 0 in the over
		float isPositive=(float) (pixelDifference>0.0f);

		// ON and OFF channels writing step
		*(parvocellularOutputON_PTR++)=*(bipolarCellsON_PTR++) = isPositive*pixelDifference;
		*(parvocellularOutputOFF_PTR++)=*(bipolarCellsOFF_PTR++)= (isPositive-1.0f)*pixelDifference;
	}
}
}

