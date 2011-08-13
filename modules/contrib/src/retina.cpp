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

/*
 * Retina.cpp
 *
 *  Created on: Jul 19, 2011
 *      Author: Alexandre Benoit
 */
#include "precomp.hpp"
#include "retinafilter.hpp"
#include <iostream>

namespace cv
{
    
Retina::Retina(const std::string parametersSaveFile, const cv::Size inputSize)
{
	_retinaFilter = 0;
    _init(parametersSaveFile, inputSize, true, RETINA_COLOR_BAYER, false);
}

Retina::Retina(const std::string parametersSaveFile, const cv::Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    _retinaFilter = 0;
	_init(parametersSaveFile, inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
};
    
Retina::~Retina()
{
    delete _retinaFilter;
}

void Retina::setColorSaturation(const bool saturateColors, const double colorSaturationValue)
{
	_retinaFilter->setColorSaturation(saturateColors, colorSaturationValue);
}

void Retina::setup(std::string retinaParameterFile, const bool applyDefaultSetupOnFailure)
{
	// open specified parameters file
	std::cout<<"Retina::setup: setting up retina from parameter file : "<<retinaParameterFile<<std::endl;

	// very UGLY cases processing... to be updated...

	try
	{

		// rewriting a new parameter file...
		if (_parametersSaveFile.isOpened())
			_parametersSaveFile.release();
		_parametersSaveFile.open(_parametersSaveFileName, cv::FileStorage::WRITE);
		// opening retinaParameterFile in read mode
		cv::FileStorage fs(retinaParameterFile, cv::FileStorage::READ);
		// read parameters file if it exists or apply default setup if asked for
		if (!fs.isOpened())
		{
			std::cout<<"Retina::setup: provided parameters file could not be open... skeeping configuration"<<std::endl;
			return;
			// implicit else case : retinaParameterFile could be open (it exists at least)
		}

		// preparing parameter setup
		bool colorMode, normaliseOutput;
		double photoreceptorsLocalAdaptationSensitivity, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, hcellsTemporalConstant, hcellsSpatialConstant, ganglionCellsSensitivity;
		// OPL and Parvo init first
		cv::FileNode rootFn = fs.root(), currFn=rootFn["OPLandIPLparvo"];
		currFn["colorMode"]>>colorMode;
		currFn["normaliseOutput"]>>normaliseOutput;
		currFn["photoreceptorsLocalAdaptationSensitivity"]>>photoreceptorsLocalAdaptationSensitivity;
		currFn["photoreceptorsTemporalConstant"]>>photoreceptorsTemporalConstant;
		currFn["photoreceptorsSpatialConstant"]>>photoreceptorsSpatialConstant;
		currFn["horizontalCellsGain"]>>horizontalCellsGain;
		currFn["hcellsTemporalConstant"]>>hcellsTemporalConstant;
		currFn["hcellsSpatialConstant"]>>hcellsSpatialConstant;
		currFn["ganglionCellsSensitivity"]>>ganglionCellsSensitivity;
		setupOPLandIPLParvoChannel(colorMode, normaliseOutput, photoreceptorsLocalAdaptationSensitivity, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, hcellsTemporalConstant, hcellsSpatialConstant, ganglionCellsSensitivity);

		// init retina IPL magno setup
		currFn=rootFn["IPLmagno"];
		currFn["normaliseOutput"]>>normaliseOutput;
		double parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k;
		currFn["parasolCells_beta"]>>parasolCells_beta;
		currFn["parasolCells_tau"]>>parasolCells_tau;
		currFn["parasolCells_k"]>>parasolCells_k;
		currFn["amacrinCellsTemporalCutFrequency"]>>amacrinCellsTemporalCutFrequency;
		currFn["V0CompressionParameter"]>>V0CompressionParameter;
		currFn["localAdaptintegration_tau"]>>localAdaptintegration_tau;
		currFn["localAdaptintegration_k"]>>localAdaptintegration_k;

		setupIPLMagnoChannel(normaliseOutput, parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency,
				V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k);

	}catch(Exception &e)
	{
		std::cout<<"Retina::setup: resetting retina with default parameters"<<std::endl;
		if (applyDefaultSetupOnFailure)
		{
			setupOPLandIPLParvoChannel();
			setupIPLMagnoChannel();
		}
		std::cout<<"Retina::setup: wrong/unappropriate xml parameter file : error report :`n=>"<<e.what()<<std::endl;
		std::cout<<"=> keeping current parameters"<<std::endl;
	}
	_parametersSaveFile.release(); // close file after setup
	// report current configuration
	std::cout<<printSetup()<<std::endl;
}

const std::string Retina::printSetup()
{
	std::stringstream outmessage;


	try
	{
		cv::FileStorage parametersReader(_parametersSaveFileName, cv::FileStorage::READ);
		if (!parametersReader.isOpened())
		{
			outmessage<<"Retina is not already settled up";
		}
		else
		{
			// accessing xml parameters nodes
			cv::FileNode rootFn = parametersReader.root();
			cv::FileNode currFn=rootFn["OPLandIPLparvo"];

			// displaying OPL and IPL parvo setup
			outmessage<<"Current Retina instance setup :"
					<<"\nOPLandIPLparvo"<<"{"
					<< "\n==> colorMode : " << currFn["colorMode"].operator int()
					<< "\n==> normalizeParvoOutput :" << currFn["normaliseOutput"].operator int()
					<< "\n==> photoreceptorsLocalAdaptationSensitivity : " << currFn["photoreceptorsLocalAdaptationSensitivity"].operator float()
					<< "\n==> photoreceptorsTemporalConstant : " << currFn["photoreceptorsTemporalConstant"].operator float()
					<< "\n==> photoreceptorsSpatialConstant : " << currFn["photoreceptorsSpatialConstant"].operator float()
					<< "\n==> horizontalCellsGain : " << currFn["horizontalCellsGain"].operator float()
					<< "\n==> hcellsTemporalConstant : " << currFn["hcellsTemporalConstant"].operator float()
					<< "\n==> hcellsSpatialConstant : " << currFn["hcellsSpatialConstant"].operator float()
					<< "\n==> parvoGanglionCellsSensitivity : " << currFn["ganglionCellsSensitivity"].operator float()
					<<"}\n";

			// displaying IPL magno setup
			currFn=rootFn["IPLmagno"];
			outmessage<<"Current Retina instance setup :"
					<<"\nIPLmagno"<<"{"
					<< "\n==> normaliseOutput : " << currFn["normaliseOutput"].operator int()
					<< "\n==> parasolCells_beta : " << currFn["parasolCells_beta"].operator float()
					<< "\n==> parasolCells_tau : " << currFn["parasolCells_tau"].operator float()
					<< "\n==> parasolCells_k : " << currFn["parasolCells_k"].operator float()
					<< "\n==> amacrinCellsTemporalCutFrequency : " << currFn["amacrinCellsTemporalCutFrequency"].operator float()
					<< "\n==> V0CompressionParameter : " << currFn["V0CompressionParameter"].operator float()
					<< "\n==> localAdaptintegration_tau : " << currFn["localAdaptintegration_tau"].operator float()
					<< "\n==> localAdaptintegration_k : " << currFn["localAdaptintegration_k"].operator float()
					<<"}";
		}
	}catch(cv::Exception &e)
	{
		outmessage<<"Error reading parameters configuration file : "<<e.what()<<std::endl;
	}
	return outmessage.str();
}

void Retina::setupOPLandIPLParvoChannel(const bool colorMode, const bool normaliseOutput, const double photoreceptorsLocalAdaptationSensitivity, const double photoreceptorsTemporalConstant, const double photoreceptorsSpatialConstant, const double horizontalCellsGain, const double HcellsTemporalConstant, const double HcellsSpatialConstant, const double ganglionCellsSensitivity)
{
	// parameters setup (default setup)
	_retinaFilter->setColorMode(colorMode);
	_retinaFilter->setPhotoreceptorsLocalAdaptationSensitivity(photoreceptorsLocalAdaptationSensitivity);
	_retinaFilter->setOPLandParvoParameters(0, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, HcellsTemporalConstant, HcellsSpatialConstant, ganglionCellsSensitivity);
	_retinaFilter->setParvoGanglionCellsLocalAdaptationSensitivity(ganglionCellsSensitivity);
	_retinaFilter->activateNormalizeParvoOutput_0_maxOutputValue(normaliseOutput);

	// save parameters in the xml parameters tree... if parameters file is already open
	if (!_parametersSaveFile.isOpened())
		return;
	_parametersSaveFile<<"OPLandIPLparvo"<<"{";
	_parametersSaveFile << "colorMode" << colorMode;
	_parametersSaveFile << "normaliseOutput" << normaliseOutput;
	_parametersSaveFile << "photoreceptorsLocalAdaptationSensitivity" << photoreceptorsLocalAdaptationSensitivity;
	_parametersSaveFile << "photoreceptorsTemporalConstant" << photoreceptorsTemporalConstant;
	_parametersSaveFile << "photoreceptorsSpatialConstant" << photoreceptorsSpatialConstant;
	_parametersSaveFile << "horizontalCellsGain" << horizontalCellsGain;
	_parametersSaveFile << "hcellsTemporalConstant" << HcellsTemporalConstant;
	_parametersSaveFile << "hcellsSpatialConstant" << HcellsSpatialConstant;
	_parametersSaveFile << "ganglionCellsSensitivity" << ganglionCellsSensitivity;
	_parametersSaveFile << "}";
}

void Retina::setupIPLMagnoChannel(const bool normaliseOutput, const double parasolCells_beta, const double parasolCells_tau, const double parasolCells_k, const double amacrinCellsTemporalCutFrequency, const double V0CompressionParameter, const double localAdaptintegration_tau, const double localAdaptintegration_k)
{

	_retinaFilter->setMagnoCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k);
	_retinaFilter->activateNormalizeMagnoOutput_0_maxOutputValue(normaliseOutput);

	// save parameters in the xml parameters tree... if parameters file is already open
	if (!_parametersSaveFile.isOpened())
		return;
	_parametersSaveFile<<"IPLmagno"<<"{";
	_parametersSaveFile << "normaliseOutput" << normaliseOutput;
	_parametersSaveFile << "parasolCells_beta" << parasolCells_beta;
	_parametersSaveFile << "parasolCells_tau" << parasolCells_tau;
	_parametersSaveFile << "parasolCells_k" << parasolCells_k;
	_parametersSaveFile << "amacrinCellsTemporalCutFrequency" << amacrinCellsTemporalCutFrequency;
	_parametersSaveFile << "V0CompressionParameter" << V0CompressionParameter;
	_parametersSaveFile << "localAdaptintegration_tau" << localAdaptintegration_tau;
	_parametersSaveFile << "localAdaptintegration_k" << localAdaptintegration_k;
	_parametersSaveFile<<"}";

}

void Retina::run(const cv::Mat &inputMatToConvert)
{
	// first convert input image to the compatible format : std::valarray<double>
	const bool colorMode = _convertCvMat2ValarrayBuffer(inputMatToConvert, _inputBuffer);
	// process the retina
	if (!_retinaFilter->runFilter(_inputBuffer, colorMode, false, colorMode, false))
		throw cv::Exception(-1, "Retina cannot be applied, wrong input buffer size", "Retina::run", "Retina.h", 0);
}

void Retina::getParvo(cv::Mat &retinaOutput_parvo)
{
	if (_retinaFilter->getColorMode())
	{
		// reallocate output buffer (if necessary)
		_convertValarrayBuffer2cvMat(_retinaFilter->getColorOutput(), _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), true, retinaOutput_parvo);
	}else
	{
		// reallocate output buffer (if necessary)
		_convertValarrayBuffer2cvMat(_retinaFilter->getContours(), _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), false, retinaOutput_parvo);
	}
	//retinaOutput_parvo/=255.0;
}
void Retina::getMagno(cv::Mat &retinaOutput_magno)
{
	// reallocate output buffer (if necessary)
	_convertValarrayBuffer2cvMat(_retinaFilter->getMovingContours(), _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), false, retinaOutput_magno);
	//retinaOutput_magno/=255.0;
}


// private method called by constructirs
void Retina::_init(const std::string parametersSaveFile, const cv::Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
	_parametersSaveFileName = parametersSaveFile;

	// basic error check
	if (inputSize.height*inputSize.width <= 0)
		throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "Retina::setup", "Retina.h", 0);

	unsigned int nbPixels=inputSize.height*inputSize.width;
	// resize buffers if size does not match
	_inputBuffer.resize(nbPixels*3); // buffer supports gray images but also 3 channels color buffers... (larger is better...)

	// allocate the retina model
    delete _retinaFilter;
	_retinaFilter = new RetinaFilter(inputSize.height, inputSize.width, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);

	// prepare the parameter XML tree
	_parametersSaveFile.open(parametersSaveFile, cv::FileStorage::WRITE );

	_parametersSaveFile<<"InputSize"<<"{";
	_parametersSaveFile<<"height"<<inputSize.height;
	_parametersSaveFile<<"width"<<inputSize.width;
	_parametersSaveFile<<"}";

	// clear all retina buffers
	// apply default setup
	setupOPLandIPLParvoChannel();
	setupIPLMagnoChannel();

	// write current parameters to params file
	_parametersSaveFile.release();

	// init retina
	_retinaFilter->clearAllBuffers();

	// report current configuration
	std::cout<<printSetup()<<std::endl;
}

void Retina::_convertValarrayBuffer2cvMat(const std::valarray<double> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, cv::Mat &outBuffer)
{
	// fill output buffer with the valarray buffer
	const double *valarrayPTR=get_data(grayMatrixToConvert);
	if (!colorMode)
	{
		outBuffer.create(cv::Size(nbColumns, nbRows), CV_8U);
		for (unsigned int i=0;i<nbRows;++i)
		{
			for (unsigned int j=0;j<nbColumns;++j)
			{
				cv::Point2d pixel(j,i);
				outBuffer.at<unsigned char>(pixel)=(unsigned char)*(valarrayPTR++);
			}
		}
	}else
	{
		const unsigned int doubleNBpixels=_retinaFilter->getOutputNBpixels()*2;
		outBuffer.create(cv::Size(nbColumns, nbRows), CV_8UC3);
		for (unsigned int i=0;i<nbRows;++i)
		{
			for (unsigned int j=0;j<nbColumns;++j,++valarrayPTR)
			{
				cv::Point2d pixel(j,i);
				cv::Vec3b pixelValues;
				pixelValues[2]=(unsigned char)*(valarrayPTR);
				pixelValues[1]=(unsigned char)*(valarrayPTR+_retinaFilter->getOutputNBpixels());
				pixelValues[0]=(unsigned char)*(valarrayPTR+doubleNBpixels);

				outBuffer.at<cv::Vec3b>(pixel)=pixelValues;
			}
		}
	}
}


const bool Retina::_convertCvMat2ValarrayBuffer(const cv::Mat inputMatToConvert, std::valarray<double> &outputValarrayMatrix)
{
	// first check input consistency
	if (inputMatToConvert.empty())
		throw cv::Exception(-1, "Retina cannot be applied, input buffer is empty", "Retina::run", "Retina.h", 0);

	// retreive color mode from image input
	bool colorMode = inputMatToConvert.channels() >=3;

	// convert to double AND fill the valarray buffer
	const int dsttype = CV_64F; // output buffer is double format

	if (colorMode)
	{
		// create a cv::Mat table (for RGB planes)
		cv::Mat planes[] =
		{
				cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()*2]),
				cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()]),
				cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
		};
		// split color cv::Mat in 3 planes... it fills valarray directely
		cv::split(Mat_<double>(inputMatToConvert), planes);

	}else
	{
		// create a cv::Mat header for the valarray
		cv::Mat dst(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0]);
		inputMatToConvert.convertTo(dst, dsttype);
	}
	return colorMode;
}

void Retina::clearBuffers() {_retinaFilter->clearAllBuffers();}

} // end of namespace cv
