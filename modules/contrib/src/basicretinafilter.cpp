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
#include <cstdlib>
#include "basicretinafilter.hpp"
#include <cmath>


namespace cv
{

// @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr Gipsa-Lab, France: www.gipsa-lab.inpg.fr/

//////////////////////////////////////////////////////////
//                 BASIC RETINA FILTER
//////////////////////////////////////////////////////////

// Constructor and Desctructor of the basic retina filter
BasicRetinaFilter::BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize, const bool useProgressiveFilter)
:_filterOutput(NBrows, NBcolumns),
 _localBuffer(NBrows*NBcolumns),
 _filteringCoeficientsTable(3*parametersListSize),
 _progressiveSpatialConstant(0),// pointer to a local table containing local spatial constant (allocated with the object)
 _progressiveGain(0)
{
#ifdef T_BASIC_RETINA_ELEMENT_DEBUG
	std::cout<<"BasicRetinaFilter::BasicRetinaFilter: new filter, size="<<NBrows<<", "<<NBcolumns<<std::endl;
#endif
	_halfNBrows=_filterOutput.getNBrows()/2;
	_halfNBcolumns=_filterOutput.getNBcolumns()/2;

	if (useProgressiveFilter)
	{
#ifdef T_BASIC_RETINA_ELEMENT_DEBUG
		std::cout<<"BasicRetinaFilter::BasicRetinaFilter: _progressiveSpatialConstant_Tbuffer"<<std::endl;
#endif
		_progressiveSpatialConstant.resize(_filterOutput.size());
#ifdef T_BASIC_RETINA_ELEMENT_DEBUG
		std::cout<<"BasicRetinaFilter::BasicRetinaFilter: new _progressiveGain_Tbuffer"<<NBrows<<", "<<NBcolumns<<std::endl;
#endif
		_progressiveGain.resize(_filterOutput.size());
	}
#ifdef T_BASIC_RETINA_ELEMENT_DEBUG
	std::cout<<"BasicRetinaFilter::BasicRetinaFilter: new filter, size="<<NBrows<<", "<<NBcolumns<<std::endl;
#endif

	// set default values
	_maxInputValue=256.0;

	// reset all buffers
	clearAllBuffers();

#ifdef T_BASIC_RETINA_ELEMENT_DEBUG
	std::cout<<"BasicRetinaFilter::Init BasicRetinaElement at specified frame size OK, size="<<this->size()<<std::endl;
#endif

}

BasicRetinaFilter::~BasicRetinaFilter()
{

#ifdef BASIC_RETINA_ELEMENT_DEBUG
	std::cout<<"BasicRetinaFilter::BasicRetinaElement Deleted OK"<<std::endl;
#endif

}

////////////////////////////////////
// functions of the basic filter
////////////////////////////////////


// resize all allocated buffers
void BasicRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{

	std::cout<<"BasicRetinaFilter::resize( "<<NBrows<<", "<<NBcolumns<<")"<<std::endl;

	// resizing buffers
	_filterOutput.resizeBuffer(NBrows, NBcolumns);

	// updating variables
	_halfNBrows=_filterOutput.getNBrows()/2;
	_halfNBcolumns=_filterOutput.getNBcolumns()/2;

	_localBuffer.resize(_filterOutput.size());
	// in case of spatial adapted filter
	if (_progressiveSpatialConstant.size()>0)
	{
		_progressiveSpatialConstant.resize(_filterOutput.size());
		_progressiveGain.resize(_filterOutput.size());
	}
	// reset buffers
	clearAllBuffers();
}

// Change coefficients table
void BasicRetinaFilter::setLPfilterParameters(const double beta, const double tau, const double desired_k, const unsigned int filterIndex)
{
	double _beta = beta+tau;
	double k=desired_k;
	// check if the spatial constant is correct (avoid 0 value to avoid division by 0)
	if (desired_k<=0)
	{
		k=0.001;
		std::cerr<<"BasicRetinaFilter::spatial constant of the low pass filter must be superior to zero !!! correcting parameter setting to 0,001"<<std::endl;
	}

	double _alpha = k*k;
	double _mu = 0.8;
	unsigned int tableOffset=filterIndex*3;
	if (k<=0)
	{
		std::cerr<<"BasicRetinaFilter::spatial filtering coefficient must be superior to zero, correcting value to 0.01"<<std::endl;
		_alpha=0.0001;
	}

	double _temp =  (1.0+_beta)/(2.0*_mu*_alpha);
	double _a = _filteringCoeficientsTable[tableOffset] = 1.0 + _temp - sqrt( (1.0+_temp)*(1.0+_temp) - 1.0);
	_filteringCoeficientsTable[1+tableOffset]=(1.0-_a)*(1.0-_a)*(1.0-_a)*(1.0-_a)/(1.0+_beta);
	_filteringCoeficientsTable[2+tableOffset] =tau;

	//std::cout<<"BasicRetinaFilter::normal:"<<(1.0-_a)*(1.0-_a)*(1.0-_a)*(1.0-_a)/(1.0+_beta)<<" -> old:"<<(1-_a)*(1-_a)*(1-_a)*(1-_a)/(1+_beta)<<std::endl;

	//std::cout<<"BasicRetinaFilter::_a="<<_a<<", gain="<<_filteringCoeficientsTable[1+tableOffset]<<", tau="<<tau<<std::endl;
}

void BasicRetinaFilter::setProgressiveFilterConstants_CentredAccuracy(const double beta, const double tau, const double alpha0, const unsigned int filterIndex)
{
	// check if dedicated buffers are already allocated, if not create them
	if (_progressiveSpatialConstant.size()!=_filterOutput.size())
	{
		_progressiveSpatialConstant.resize(_filterOutput.size());
		_progressiveGain.resize(_filterOutput.size());
	}

	double _beta = beta+tau;
	double _mu=0.8;
	if (alpha0<=0)
	{
		std::cerr<<"BasicRetinaFilter::spatial filtering coefficient must be superior to zero, correcting value to 0.01"<<std::endl;
		//alpha0=0.0001;
	}

	unsigned int tableOffset=filterIndex*3;

	double _alpha=0.8;
	double _temp =  (1.0+_beta)/(2.0*_mu*_alpha);
	double _a=_filteringCoeficientsTable[tableOffset] = 1.0 + _temp - sqrt( (1.0+_temp)*(1.0+_temp) - 1.0);
	_filteringCoeficientsTable[tableOffset+1]=(1.0-_a)*(1.0-_a)*(1.0-_a)*(1.0-_a)/(1.0+_beta);
	_filteringCoeficientsTable[tableOffset+2] =tau;

	double commonFactor=alpha0/sqrt((double)(_halfNBcolumns*_halfNBcolumns+_halfNBrows*_halfNBrows)+1.0);
	//memset(_progressiveSpatialConstant, 255, _filterOutput.getNBpixels());
	for (unsigned int idColumn=0;idColumn<_halfNBcolumns; ++idColumn)
		for (unsigned int idRow=0;idRow<_halfNBrows; ++idRow)
		{
			// computing local spatial constant
			double localSpatialConstantValue=commonFactor*sqrt((double)(idColumn*idColumn)+(double)(idRow*idRow));
			if (localSpatialConstantValue>1)
				localSpatialConstantValue=1;

			_progressiveSpatialConstant[_halfNBcolumns-1+idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1+idRow)]=localSpatialConstantValue;
			_progressiveSpatialConstant[_halfNBcolumns-1-idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1+idRow)]=localSpatialConstantValue;
			_progressiveSpatialConstant[_halfNBcolumns-1+idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1-idRow)]=localSpatialConstantValue;
			_progressiveSpatialConstant[_halfNBcolumns-1-idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1-idRow)]=localSpatialConstantValue;

			// computing local gain
			double localGain=(1-localSpatialConstantValue)*(1-localSpatialConstantValue)*(1-localSpatialConstantValue)*(1-localSpatialConstantValue)/(1+_beta);
			_progressiveGain[_halfNBcolumns-1+idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1+idRow)]=localGain;
			_progressiveGain[_halfNBcolumns-1-idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1+idRow)]=localGain;
			_progressiveGain[_halfNBcolumns-1+idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1-idRow)]=localGain;
			_progressiveGain[_halfNBcolumns-1-idColumn+_filterOutput.getNBcolumns()*(_halfNBrows-1-idRow)]=localGain;

			//std::cout<<commonFactor<<", "<<sqrt((_halfNBcolumns-1-idColumn)+(_halfNBrows-idRow-1))<<", "<<(_halfNBcolumns-1-idColumn)<<", "<<(_halfNBrows-idRow-1)<<", "<<localSpatialConstantValue<<std::endl;
		}
}

void BasicRetinaFilter::setProgressiveFilterConstants_CustomAccuracy(const double beta, const double tau, const double k, const std::valarray<double> &accuracyMap, const unsigned int filterIndex)
{

	if (accuracyMap.size()!=_filterOutput.size())
	{
		std::cerr<<"BasicRetinaFilter::setProgressiveFilterConstants_CustomAccuracy: error: input accuracy map does not match filter size, init skept"<<std::endl;
		return ;
	}

	// check if dedicated buffers are already allocated, if not create them
	if (_progressiveSpatialConstant.size()!=_filterOutput.size())
	{
		_progressiveSpatialConstant.resize(accuracyMap.size());
		_progressiveGain.resize(accuracyMap.size());
	}

	double _beta = beta+tau;
	double _alpha=k*k;
	double _mu=0.8;
	if (k<=0)
	{
		std::cerr<<"BasicRetinaFilter::spatial filtering coefficient must be superior to zero, correcting value to 0.01"<<std::endl;
		//alpha0=0.0001;
	}
	unsigned int tableOffset=filterIndex*3;
	double _temp =  (1.0+_beta)/(2.0*_mu*_alpha);
	double _a=_filteringCoeficientsTable[tableOffset] = 1.0 + _temp - sqrt( (1.0+_temp)*(1.0+_temp) - 1.0);
	_filteringCoeficientsTable[tableOffset+1]=(1.0-_a)*(1.0-_a)*(1.0-_a)*(1.0-_a)/(1.0+_beta);
	_filteringCoeficientsTable[tableOffset+2] =tau;

	//memset(_progressiveSpatialConstant, 255, _filterOutput.getNBpixels());
	for (unsigned int idColumn=0;idColumn<_filterOutput.getNBcolumns(); ++idColumn)
		for (unsigned int idRow=0;idRow<_filterOutput.getNBrows(); ++idRow)
		{
			// computing local spatial constant
			unsigned int index=idColumn+idRow*_filterOutput.getNBcolumns();
			double localSpatialConstantValue=_a*accuracyMap[index];
			if (localSpatialConstantValue>1)
				localSpatialConstantValue=1;

			_progressiveSpatialConstant[index]=localSpatialConstantValue;

			// computing local gain
			double localGain=(1-localSpatialConstantValue)*(1-localSpatialConstantValue)*(1-localSpatialConstantValue)*(1-localSpatialConstantValue)/(1+_beta);
			_progressiveGain[index]=localGain;

			//std::cout<<commonFactor<<", "<<sqrt((_halfNBcolumns-1-idColumn)+(_halfNBrows-idRow-1))<<", "<<(_halfNBcolumns-1-idColumn)<<", "<<(_halfNBrows-idRow-1)<<", "<<localSpatialConstantValue<<std::endl;
		}
}

///////////////////////////////////////////////////////////////////////
/// Local luminance adaptation functions
// run local adaptation filter and save result in _filterOutput
const std::valarray<double> &BasicRetinaFilter::runFilter_LocalAdapdation(const std::valarray<double> &inputFrame, const std::valarray<double> &localLuminance)
{
	_localLuminanceAdaptation(&inputFrame[0], &localLuminance[0], &_filterOutput[0]);
	return _filterOutput;
}
// run local adaptation filter at a specific output adress
void BasicRetinaFilter::runFilter_LocalAdapdation(const std::valarray<double> &inputFrame, const std::valarray<double> &localLuminance, std::valarray<double> &outputFrame)
{
	_localLuminanceAdaptation(&inputFrame[0], &localLuminance[0], &outputFrame[0]);
}
// run local adaptation filter and save result in _filterOutput with autonomous low pass filtering before adaptation
const std::valarray<double> &BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const std::valarray<double> &inputFrame)
{
	_spatiotemporalLPfilter(&inputFrame[0], &_filterOutput[0]);
	_localLuminanceAdaptation(&inputFrame[0], &_filterOutput[0], &_filterOutput[0]);
	return _filterOutput;
}
// run local adaptation filter at a specific output adress with autonomous low pass filtering before adaptation
void BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const std::valarray<double> &inputFrame, std::valarray<double> &outputFrame)
{
	_spatiotemporalLPfilter(&inputFrame[0], &_filterOutput[0]);
	_localLuminanceAdaptation(&inputFrame[0], &_filterOutput[0], &outputFrame[0]);
}
// local luminance adaptation of the input in regard of localLuminance buffer
void BasicRetinaFilter::_localLuminanceAdaptation(const double *inputFrame, const double *localLuminance, double *outputFrame)
{
	double meanLuminance=0;
	const double *luminancePTR=inputFrame;
	for (unsigned int i=0;i<_filterOutput.getNBpixels();++i)
		meanLuminance+=*(luminancePTR++);
	meanLuminance/=_filterOutput.getNBpixels();
	//double tempMeanValue=meanLuminance+_meanInputValue*_tau;

	updateCompressionParameter(meanLuminance);
	//std::cout<<meanLuminance<<std::endl;
	const double *localLuminancePTR=localLuminance;
	const double *inputFramePTR=inputFrame;
	double *outputFramePTR=outputFrame;
	for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel, ++inputFramePTR)
	{
		double X0=*(localLuminancePTR++)*_localLuminanceFactor+_localLuminanceAddon;
		*(outputFramePTR++) = (_maxInputValue+X0)**inputFramePTR/(*inputFramePTR +X0);
		//std::cout<<"BasicRetinaFilter::inputFrame[IDpixel]=%f, X0=%f, outputFrame[IDpixel]=%f\n", inputFrame[IDpixel], X0, outputFrame[IDpixel]);
	}
}

// local adaptation applied on a range of values which can be positive and negative
void BasicRetinaFilter::_localLuminanceAdaptationPosNegValues(const double *inputFrame, const double *localLuminance, double *outputFrame)
{
	const double *localLuminancePTR=localLuminance;
	const double *inputFramePTR=inputFrame;
	double *outputFramePTR=outputFrame;
	double factor=_maxInputValue*2/CV_PI;
	for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel, ++inputFramePTR)
	{
		double X0=*(localLuminancePTR++)*_localLuminanceFactor+_localLuminanceAddon;
		*(outputFramePTR++) = factor*atan(*inputFramePTR/X0);//(_maxInputValue+X0)**inputFramePTR/(*inputFramePTR +X0);
		//std::cout<<"BasicRetinaFilter::inputFrame[IDpixel]=%f, X0=%f, outputFrame[IDpixel]=%f\n", inputFrame[IDpixel], X0, outputFrame[IDpixel]);
	}
}

// local luminance adaptation of the input in regard of localLuminance buffer, the input is rewrited and becomes the output
void BasicRetinaFilter::_localLuminanceAdaptation(double *inputOutputFrame, const double *localLuminance)
{
	/*double meanLuminance=0;
    const double *luminancePTR=inputOutputFrame;
    for (unsigned int i=0;i<_filterOutput.getNBpixels();++i)
      meanLuminance+=*(luminancePTR++);
    meanLuminance/=_filterOutput.getNBpixels();
    //double tempMeanValue=meanLuminance+_meanInputValue*_tau;

    updateCompressionParameter(meanLuminance);
	 */
	const double *localLuminancePTR=localLuminance;
	double *inputOutputFramePTR=inputOutputFrame;

	for (register unsigned int IDpixel=0 ; IDpixel<_filterOutput.getNBpixels() ; ++IDpixel, ++inputOutputFramePTR)
	{
		double X0=*(localLuminancePTR++)*_localLuminanceFactor+_localLuminanceAddon;
		*(inputOutputFramePTR) = (_maxInputValue+X0)**inputOutputFramePTR/(*inputOutputFramePTR +X0);
	}
}
///////////////////////////////////////////////////////////////////////
/// Spatio temporal Low Pass filter functions
// run LP filter and save result in the basic retina element buffer
const std::valarray<double> &BasicRetinaFilter::runFilter_LPfilter(const std::valarray<double> &inputFrame, const unsigned int filterIndex)
{
	_spatiotemporalLPfilter(&inputFrame[0], &_filterOutput[0], filterIndex);
	return _filterOutput;
}

// run LP filter for a new frame input and save result at a specific output adress
void BasicRetinaFilter::runFilter_LPfilter(const std::valarray<double> &inputFrame, std::valarray<double> &outputFrame, const unsigned int filterIndex)
{
	_spatiotemporalLPfilter(&inputFrame[0], &outputFrame[0], filterIndex);
}

// run LP filter on the input data and rewrite it
void BasicRetinaFilter::runFilter_LPfilter_Autonomous(std::valarray<double> &inputOutputFrame, const unsigned int filterIndex)
{
	unsigned int coefTableOffset=filterIndex*3;

	/**********/
	_a=_filteringCoeficientsTable[coefTableOffset];
	_gain=_filteringCoeficientsTable[1+coefTableOffset];
	_tau=_filteringCoeficientsTable[2+coefTableOffset];

	// launch the serie of 1D directional filters in order to compute the 2D low pass filter
	_horizontalCausalFilter(&inputOutputFrame[0], 0, _filterOutput.getNBrows());
	_horizontalAnticausalFilter(&inputOutputFrame[0], 0, _filterOutput.getNBrows());
	_verticalCausalFilter(&inputOutputFrame[0], 0, _filterOutput.getNBcolumns());
	_verticalAnticausalFilter_multGain(&inputOutputFrame[0], 0, _filterOutput.getNBcolumns());

}
// run LP filter for a new frame input and save result at a specific output adress
void BasicRetinaFilter::_spatiotemporalLPfilter(const double *inputFrame, double *outputFrame, const unsigned int filterIndex)
{
	unsigned int coefTableOffset=filterIndex*3;
	/**********/
	_a=_filteringCoeficientsTable[coefTableOffset];
	_gain=_filteringCoeficientsTable[1+coefTableOffset];
	_tau=_filteringCoeficientsTable[2+coefTableOffset];

	// launch the serie of 1D directional filters in order to compute the 2D low pass filter
	_horizontalCausalFilter_addInput(inputFrame, outputFrame, 0,_filterOutput.getNBrows());
	_horizontalAnticausalFilter(outputFrame, 0, _filterOutput.getNBrows());
	_verticalCausalFilter(outputFrame, 0, _filterOutput.getNBcolumns());
	_verticalAnticausalFilter_multGain(outputFrame, 0, _filterOutput.getNBcolumns());

}

// run SQUARING LP filter for a new frame input and save result at a specific output adress
const double BasicRetinaFilter::_squaringSpatiotemporalLPfilter(const double *inputFrame, double *outputFrame, const unsigned int filterIndex)
{
	unsigned int coefTableOffset=filterIndex*3;
	/**********/
	_a=_filteringCoeficientsTable[coefTableOffset];
	_gain=_filteringCoeficientsTable[1+coefTableOffset];
	_tau=_filteringCoeficientsTable[2+coefTableOffset];

	// launch the serie of 1D directional filters in order to compute the 2D low pass filter

	_squaringHorizontalCausalFilter(inputFrame, outputFrame, 0, _filterOutput.getNBrows());
	_horizontalAnticausalFilter(outputFrame, 0, _filterOutput.getNBrows());
	_verticalCausalFilter(outputFrame, 0, _filterOutput.getNBcolumns());
	return _verticalAnticausalFilter_returnMeanValue(outputFrame, 0, _filterOutput.getNBcolumns());
}

/////////////////////////////////////////////////
// standard version of the 1D low pass filters

//  horizontal causal filter which adds the input inside
void BasicRetinaFilter::_horizontalCausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{


	//#pragma omp parallel for
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double* outputPTR=outputFrame+(IDrowStart+IDrow)*_filterOutput.getNBcolumns();
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(outputPTR)+  _a* result;
			*(outputPTR++) = result;
		}
	}
}
//  horizontal causal filter which adds the input inside
void BasicRetinaFilter::_horizontalCausalFilter_addInput(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{
	//#pragma omp parallel for
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double* outputPTR=outputFrame+(IDrowStart+IDrow)*_filterOutput.getNBcolumns();
		register const double* inputPTR=inputFrame+(IDrowStart+IDrow)*_filterOutput.getNBcolumns();
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(inputPTR++) + _tau**(outputPTR)+  _a* result;
			*(outputPTR++) = result;
		}
	}

}

//  horizontal anticausal filter  (basic way, no add on)
void BasicRetinaFilter::_horizontalAnticausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{

	//#pragma omp parallel for
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double* outputPTR=outputFrame+(IDrowEnd-IDrow)*(_filterOutput.getNBcolumns())-1;
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(outputPTR)+  _a* result;
			*(outputPTR--) = result;
		}
	}


}
//  horizontal anticausal filter which multiplies the output by _gain
void BasicRetinaFilter::_horizontalAnticausalFilter_multGain(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{

	//#pragma omp parallel for
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double* outputPTR=outputFrame+(IDrowEnd-IDrow)*(_filterOutput.getNBcolumns())-1;
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(outputPTR)+  _a* result;
			*(outputPTR--) = _gain*result;
		}
	}
}

//  vertical anticausal filter
void BasicRetinaFilter::_verticalCausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	//#pragma omp parallel for
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=outputFrame+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + _a * result;
			*(outputPTR) = result;
			outputPTR+=_filterOutput.getNBcolumns();

		}
	}
}


//  vertical anticausal filter (basic way, no add on)
void BasicRetinaFilter::_verticalAnticausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	double* offset=outputFrame+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	//#pragma omp parallel for
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=offset+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + _a * result;
			*(outputPTR) = result;
			outputPTR-=_filterOutput.getNBcolumns();

		}
	}
}

//  vertical anticausal filter which multiplies the output by _gain
void BasicRetinaFilter::_verticalAnticausalFilter_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	double* offset=outputFrame+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	//#pragma omp parallel for
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=offset+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + _a * result;
			*(outputPTR) = _gain*result;
			outputPTR-=_filterOutput.getNBcolumns();

		}
	}

}

/////////////////////////////////////////
// specific modifications of 1D filters

// -> squaring horizontal causal filter
void BasicRetinaFilter::_squaringHorizontalCausalFilter(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{
	register double* outputPTR=outputFrame+IDrowStart*_filterOutput.getNBcolumns();
	register const double* inputPTR=inputFrame+IDrowStart*_filterOutput.getNBcolumns();
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(inputPTR)**(inputPTR) + _tau**(outputPTR)+  _a* result;
			*(outputPTR++) = result;
			++inputPTR;
		}
	}
}

//  vertical anticausal filter that returns the mean value of its result
const double BasicRetinaFilter::_verticalAnticausalFilter_returnMeanValue(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	register double meanValue=0;
	double* offset=outputFrame+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=offset+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + _a * result;
			*(outputPTR) = _gain*result;
			meanValue+=*(outputPTR);
			outputPTR-=_filterOutput.getNBcolumns();

		}
	}

	return meanValue/(double)_filterOutput.getNBpixels();
}

// LP filter with integration in specific areas (regarding true values of a binary parameters image)
void BasicRetinaFilter::_localSquaringSpatioTemporalLPfilter(const double *inputFrame, double *LPfilterOutput, const unsigned int *integrationAreas, const unsigned int filterIndex)
{
	unsigned int coefTableOffset=filterIndex*3;
	_a=_filteringCoeficientsTable[coefTableOffset+0];
	_gain=_filteringCoeficientsTable[coefTableOffset+1];
	_tau=_filteringCoeficientsTable[coefTableOffset+2];
	// launch the serie of 1D directional filters in order to compute the 2D low pass filter

	_local_squaringHorizontalCausalFilter(inputFrame, LPfilterOutput, 0, _filterOutput.getNBrows(), integrationAreas);
	_local_horizontalAnticausalFilter(LPfilterOutput, 0, _filterOutput.getNBrows(), integrationAreas);
	_local_verticalCausalFilter(LPfilterOutput, 0, _filterOutput.getNBcolumns(), integrationAreas);
	_local_verticalAnticausalFilter_multGain(LPfilterOutput, 0, _filterOutput.getNBcolumns(), integrationAreas);

}

// LP filter on specific parts of the picture instead of all the image
// same functions (some of them) but take a binary flag to allow integration, false flag means, no data change at the output...

// this function take an image in input and squares it befor computing
void BasicRetinaFilter::_local_squaringHorizontalCausalFilter(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas)
{
	register double* outputPTR=outputFrame+IDrowStart*_filterOutput.getNBcolumns();
	register const double* inputPTR=inputFrame+IDrowStart*_filterOutput.getNBcolumns();
	const unsigned int *integrationAreasPTR=integrationAreas;
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			if (*(integrationAreasPTR++))
				result = *(inputPTR)**(inputPTR) + _tau**(outputPTR)+  _a* result;
			else
				result=0;
			*(outputPTR++) = result;
			++inputPTR;

		}
	}
}

void BasicRetinaFilter::_local_horizontalAnticausalFilter(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd, const unsigned int *integrationAreas)
{

	register double* outputPTR=outputFrame+IDrowEnd*(_filterOutput.getNBcolumns())-1;
	const unsigned int *integrationAreasPTR=integrationAreas;

	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			if (*(integrationAreasPTR++))
				result = *(outputPTR)+  _a* result;
			else
				result=0;
			*(outputPTR--) = result;
		}
	}

}

void BasicRetinaFilter::_local_verticalCausalFilter(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas)
{
	const unsigned int *integrationAreasPTR=integrationAreas;

	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=outputFrame+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			if (*(integrationAreasPTR++))
				result = *(outputPTR)+  _a* result;
			else
				result=0;
			*(outputPTR) = result;
			outputPTR+=_filterOutput.getNBcolumns();

		}
	}
}
// this functions affects _gain at the output
void BasicRetinaFilter::_local_verticalAnticausalFilter_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd, const unsigned int *integrationAreas)
{
	const unsigned int *integrationAreasPTR=integrationAreas;
	double* offset=outputFrame+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();

	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=offset+IDcolumn;

		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			if (*(integrationAreasPTR++))
				result = *(outputPTR)+  _a* result;
			else
				result=0;
			*(outputPTR) = _gain*result;
			outputPTR-=_filterOutput.getNBcolumns();

		}
	}
}

////////////////////////////////////////////////////
// run LP filter for a new frame input and save result at a specific output adress
// -> USE IRREGULAR SPATIAL CONSTANT

// irregular filter computed from a buffer and rewrites it
void BasicRetinaFilter::_spatiotemporalLPfilter_Irregular(double *inputOutputFrame, const unsigned int filterIndex)
{
	if (_progressiveGain.size()==0)
	{
		std::cerr<<"BasicRetinaFilter::runProgressiveFilter: cannot perform filtering, no progressive filter settled up"<<std::endl;
		return;
	}
	unsigned int coefTableOffset=filterIndex*3;
	/**********/
	//_a=_filteringCoeficientsTable[coefTableOffset];
	_tau=_filteringCoeficientsTable[2+coefTableOffset];

	// launch the serie of 1D directional filters in order to compute the 2D low pass filter
	_horizontalCausalFilter_Irregular(inputOutputFrame, 0, (int)_filterOutput.getNBrows());
	_horizontalAnticausalFilter_Irregular(inputOutputFrame, 0, (int)_filterOutput.getNBrows());
	_verticalCausalFilter_Irregular(inputOutputFrame, 0, (int)_filterOutput.getNBcolumns());
	_verticalAnticausalFilter_Irregular_multGain(inputOutputFrame, 0, (int)_filterOutput.getNBcolumns());

}
// irregular filter computed from a buffer and puts result on another
void BasicRetinaFilter::_spatiotemporalLPfilter_Irregular(const double *inputFrame, double *outputFrame, const unsigned int filterIndex)
{
	if (_progressiveGain.size()==0)
	{
		std::cerr<<"BasicRetinaFilter::runProgressiveFilter: cannot perform filtering, no progressive filter settled up"<<std::endl;
		return;
	}
	unsigned int coefTableOffset=filterIndex*3;
	/**********/
	//_a=_filteringCoeficientsTable[coefTableOffset];
	_tau=_filteringCoeficientsTable[2+coefTableOffset];

	// launch the serie of 1D directional filters in order to compute the 2D low pass filter
	_horizontalCausalFilter_Irregular_addInput(inputFrame, outputFrame, 0, (int)_filterOutput.getNBrows());
	_horizontalAnticausalFilter_Irregular(outputFrame, 0, (int)_filterOutput.getNBrows());
	_verticalCausalFilter_Irregular(outputFrame, 0, (int)_filterOutput.getNBcolumns());
	_verticalAnticausalFilter_Irregular_multGain(outputFrame, 0, (int)_filterOutput.getNBcolumns());

}
// 1D filters with irregular spatial constant
//  horizontal causal filter wich runs on its input buffer
void BasicRetinaFilter::_horizontalCausalFilter_Irregular(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{
	register double* outputPTR=outputFrame+IDrowStart*_filterOutput.getNBcolumns();
	register const double* spatialConstantPTR=&_progressiveSpatialConstant[0]+IDrowStart*_filterOutput.getNBcolumns();
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(outputPTR)+  *(spatialConstantPTR++)* result;
			*(outputPTR++) = result;
		}
	}
}

// horizontal causal filter with add input
void BasicRetinaFilter::_horizontalCausalFilter_Irregular_addInput(const double *inputFrame, double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{
	register double* outputPTR=outputFrame+IDrowStart*_filterOutput.getNBcolumns();
	register const double* inputPTR=inputFrame+IDrowStart*_filterOutput.getNBcolumns();
	register const double* spatialConstantPTR=&_progressiveSpatialConstant[0]+IDrowStart*_filterOutput.getNBcolumns();
	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(inputPTR++) + _tau**(outputPTR)+  *(spatialConstantPTR++)* result;
			*(outputPTR++) = result;
		}
	}

}

//  horizontal anticausal filter  (basic way, no add on)
void BasicRetinaFilter::_horizontalAnticausalFilter_Irregular(double *outputFrame, unsigned int IDrowStart, unsigned int IDrowEnd)
{
	register double* outputPTR=outputFrame+IDrowEnd*(_filterOutput.getNBcolumns())-1;
	register const double* spatialConstantPTR=&_progressiveSpatialConstant[0]+IDrowEnd*(_filterOutput.getNBcolumns())-1;

	for (unsigned int IDrow=IDrowStart; IDrow<IDrowEnd; ++IDrow)
	{
		register double result=0;
		for (unsigned int index=0; index<_filterOutput.getNBcolumns(); ++index)
		{
			result = *(outputPTR)+  *(spatialConstantPTR--)* result;
			*(outputPTR--) = result;
		}
	}


}

//  vertical anticausal filter
void BasicRetinaFilter::_verticalCausalFilter_Irregular(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=outputFrame+IDcolumn;
		register const double *spatialConstantPTR=&_progressiveSpatialConstant[0]+IDcolumn;
		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + *(spatialConstantPTR) * result;
			*(outputPTR) = result;
			outputPTR+=_filterOutput.getNBcolumns();
			spatialConstantPTR+=_filterOutput.getNBcolumns();
		}
	}
}

//  vertical anticausal filter which multiplies the output by _gain
void BasicRetinaFilter::_verticalAnticausalFilter_Irregular_multGain(double *outputFrame, unsigned int IDcolumnStart, unsigned int IDcolumnEnd)
{
	double* outputOffset=outputFrame+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	const double* constantOffset=&_progressiveSpatialConstant[0]+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	const double* gainOffset=&_progressiveGain[0]+_filterOutput.getNBpixels()-_filterOutput.getNBcolumns();
	for (unsigned int IDcolumn=IDcolumnStart; IDcolumn<IDcolumnEnd; ++IDcolumn)
	{
		register double result=0;
		register double *outputPTR=outputOffset+IDcolumn;
		register const double *spatialConstantPTR=constantOffset+IDcolumn;
		register const double *progressiveGainPTR=gainOffset+IDcolumn;
		for (unsigned int index=0; index<_filterOutput.getNBrows(); ++index)
		{
			result = *(outputPTR) + *(spatialConstantPTR) * result;
			*(outputPTR) = *(progressiveGainPTR)*result;
			outputPTR-=_filterOutput.getNBcolumns();
			spatialConstantPTR-=_filterOutput.getNBcolumns();
			progressiveGainPTR-=_filterOutput.getNBcolumns();
		}
	}

}
}
