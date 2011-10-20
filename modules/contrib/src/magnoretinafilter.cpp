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

#include <iostream>

#include "magnoretinafilter.hpp"

#include <cmath>

namespace cv
{
// Constructor and Desctructor of the OPL retina filter
MagnoRetinaFilter::MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
:BasicRetinaFilter(NBrows, NBcolumns, 2),
 _previousInput_ON(NBrows*NBcolumns),
 _previousInput_OFF(NBrows*NBcolumns),
 _amacrinCellsTempOutput_ON(NBrows*NBcolumns),
 _amacrinCellsTempOutput_OFF(NBrows*NBcolumns),
 _magnoXOutputON(NBrows*NBcolumns),
 _magnoXOutputOFF(NBrows*NBcolumns),
 _localProcessBufferON(NBrows*NBcolumns),
 _localProcessBufferOFF(NBrows*NBcolumns)
{
	_magnoYOutput=&_filterOutput;
	_magnoYsaturated=&_localBuffer;


	clearAllBuffers();

#ifdef IPL_RETINA_ELEMENT_DEBUG
	std::cout<<"MagnoRetinaFilter::Init IPL retina filter at specified frame size OK"<<std::endl;
#endif
}

MagnoRetinaFilter::~MagnoRetinaFilter()
{
#ifdef IPL_RETINA_ELEMENT_DEBUG
	std::cout<<"MagnoRetinaFilter::Delete IPL retina filter OK"<<std::endl;
#endif
}

// function that clears all buffers of the object
void MagnoRetinaFilter::clearAllBuffers()
{
	BasicRetinaFilter::clearAllBuffers();
	_previousInput_ON=0;
	_previousInput_OFF=0;
	_amacrinCellsTempOutput_ON=0;
	_amacrinCellsTempOutput_OFF=0;
	_magnoXOutputON=0;
	_magnoXOutputOFF=0;
	_localProcessBufferON=0;
	_localProcessBufferOFF=0;

}

/**
* resize retina magno filter object (resize all allocated buffers
* @param NBrows: the new height size
* @param NBcolumns: the new width size
*/
void MagnoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	BasicRetinaFilter::resize(NBrows, NBcolumns);
	_previousInput_ON.resize(NBrows*NBcolumns);
	_previousInput_OFF.resize(NBrows*NBcolumns);
	_amacrinCellsTempOutput_ON.resize(NBrows*NBcolumns);
	_amacrinCellsTempOutput_OFF.resize(NBrows*NBcolumns);
	_magnoXOutputON.resize(NBrows*NBcolumns);
	_magnoXOutputOFF.resize(NBrows*NBcolumns);
	_localProcessBufferON.resize(NBrows*NBcolumns);
	_localProcessBufferOFF.resize(NBrows*NBcolumns);

	// to be sure, relink buffers
	_magnoYOutput=&_filterOutput;
	_magnoYsaturated=&_localBuffer;

	// reset all buffers
	clearAllBuffers();
}

void MagnoRetinaFilter::setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau, const float localAdaptIntegration_k )
{
	_temporalCoefficient=(float)exp(-1.0f/amacrinCellsTemporalCutFrequency);
	// the first set of parameters is dedicated to the low pass filtering property of the ganglion cells
	BasicRetinaFilter::setLPfilterParameters(parasolCells_beta, parasolCells_tau, parasolCells_k, 0);
	// the second set of parameters is dedicated to the ganglion cells output intergartion for their local adaptation property
	BasicRetinaFilter::setLPfilterParameters(0, localAdaptIntegration_tau, localAdaptIntegration_k, 1);
}

void MagnoRetinaFilter::_amacrineCellsComputing(const float *OPL_ON, const float *OPL_OFF)
{
	register const float *OPL_ON_PTR=OPL_ON;
	register const float *OPL_OFF_PTR=OPL_OFF;
	register float *previousInput_ON_PTR= &_previousInput_ON[0];
	register float *previousInput_OFF_PTR= &_previousInput_OFF[0];
	register float *amacrinCellsTempOutput_ON_PTR= &_amacrinCellsTempOutput_ON[0];
	register float *amacrinCellsTempOutput_OFF_PTR= &_amacrinCellsTempOutput_OFF[0];

	for (unsigned int IDpixel=0 ; IDpixel<this->getNBpixels(); ++IDpixel)
	{

		/* Compute ON and OFF amacrin cells high pass temporal filter */
		float magnoXonPixelResult = _temporalCoefficient*(*amacrinCellsTempOutput_ON_PTR+ *OPL_ON_PTR-*previousInput_ON_PTR);
		*(amacrinCellsTempOutput_ON_PTR++)=((float)(magnoXonPixelResult>0))*magnoXonPixelResult;

		float magnoXoffPixelResult = _temporalCoefficient*(*amacrinCellsTempOutput_OFF_PTR+ *OPL_OFF_PTR-*previousInput_OFF_PTR);
		*(amacrinCellsTempOutput_OFF_PTR++)=((float)(magnoXoffPixelResult>0))*magnoXoffPixelResult;

		/* prepare next loop */
		*(previousInput_ON_PTR++)=*(OPL_ON_PTR++);
		*(previousInput_OFF_PTR++)=*(OPL_OFF_PTR++);

	}
}

// launch filter that runs all the IPL filter
const std::valarray<float> &MagnoRetinaFilter::runFilter(const std::valarray<float> &OPL_ON, const std::valarray<float> &OPL_OFF)
{
	// Compute the high pass temporal filter
	_amacrineCellsComputing(get_data(OPL_ON), get_data(OPL_OFF));

	// apply low pass filtering on ON and OFF ways after temporal high pass filtering
	_spatiotemporalLPfilter(&_amacrinCellsTempOutput_ON[0], &_magnoXOutputON[0], 0);
	_spatiotemporalLPfilter(&_amacrinCellsTempOutput_OFF[0], &_magnoXOutputOFF[0], 0);

	// local adaptation of the ganglion cells to the local contrast of the moving contours
	_spatiotemporalLPfilter(&_magnoXOutputON[0], &_localProcessBufferON[0], 1);
	_localLuminanceAdaptation(&_magnoXOutputON[0], &_localProcessBufferON[0]);
	_spatiotemporalLPfilter(&_magnoXOutputOFF[0], &_localProcessBufferOFF[0], 1);
	_localLuminanceAdaptation(&_magnoXOutputOFF[0], &_localProcessBufferOFF[0]);

	/* Compute MagnoY */
	register float *magnoYOutput= &(*_magnoYOutput)[0];
	register float *magnoXOutputON_PTR= &_magnoXOutputON[0];
	register float *magnoXOutputOFF_PTR= &_magnoXOutputOFF[0];
	for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel)
		*(magnoYOutput++)=*(magnoXOutputON_PTR++)+*(magnoXOutputOFF_PTR++);

	return (*_magnoYOutput);
}
}


