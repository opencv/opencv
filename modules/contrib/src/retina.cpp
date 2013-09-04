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

Retina::Retina(const cv::Size inputSz)
{
    _retinaFilter = 0;
    _init(inputSz, true, RETINA_COLOR_BAYER, false);
}

Retina::Retina(const cv::Size inputSz, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    _retinaFilter = 0;
    _init(inputSz, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
};

Retina::~Retina()
{
    if (_retinaFilter)
        delete _retinaFilter;
}

/**
* retreive retina input buffer size
*/
Size Retina::inputSize(){return cv::Size(_retinaFilter->getInputNBcolumns(), _retinaFilter->getInputNBrows());}

/**
* retreive retina output buffer size
*/
Size Retina::outputSize(){return cv::Size(_retinaFilter->getOutputNBcolumns(), _retinaFilter->getOutputNBrows());}


void Retina::setColorSaturation(const bool saturateColors, const float colorSaturationValue)
{
    _retinaFilter->setColorSaturation(saturateColors, colorSaturationValue);
}

struct Retina::RetinaParameters Retina::getParameters(){return _retinaParameters;}


void Retina::setup(std::string retinaParameterFile, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // opening retinaParameterFile in read mode
        cv::FileStorage fs(retinaParameterFile, cv::FileStorage::READ);
        setup(fs, applyDefaultSetupOnFailure);
    }catch(Exception &e)
    {
    std::cout<<"Retina::setup: wrong/unappropriate xml parameter file : error report :`n=>"<<e.what()<<std::endl;
    if (applyDefaultSetupOnFailure)
    {
            std::cout<<"Retina::setup: resetting retina with default parameters"<<std::endl;
        setupOPLandIPLParvoChannel();
        setupIPLMagnoChannel();
    }
        else
        {
        std::cout<<"=> keeping current parameters"<<std::endl;
        }
    }
}

void Retina::setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // read parameters file if it exists or apply default setup if asked for
        if (!fs.isOpened())
        {
            std::cout<<"Retina::setup: provided parameters file could not be open... skeeping configuration"<<std::endl;
            return;
            // implicit else case : retinaParameterFile could be open (it exists at least)
        }
                // OPL and Parvo init first... update at the same time the parameters structure and the retina core
        cv::FileNode rootFn = fs.root(), currFn=rootFn["OPLandIPLparvo"];
        currFn["colorMode"]>>_retinaParameters.OPLandIplParvo.colorMode;
        currFn["normaliseOutput"]>>_retinaParameters.OPLandIplParvo.normaliseOutput;
        currFn["photoreceptorsLocalAdaptationSensitivity"]>>_retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity;
        currFn["photoreceptorsTemporalConstant"]>>_retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant;
        currFn["photoreceptorsSpatialConstant"]>>_retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant;
        currFn["horizontalCellsGain"]>>_retinaParameters.OPLandIplParvo.horizontalCellsGain;
        currFn["hcellsTemporalConstant"]>>_retinaParameters.OPLandIplParvo.hcellsTemporalConstant;
        currFn["hcellsSpatialConstant"]>>_retinaParameters.OPLandIplParvo.hcellsSpatialConstant;
        currFn["ganglionCellsSensitivity"]>>_retinaParameters.OPLandIplParvo.ganglionCellsSensitivity;
        setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);

        // init retina IPL magno setup... update at the same time the parameters structure and the retina core
        currFn=rootFn["IPLmagno"];
        currFn["normaliseOutput"]>>_retinaParameters.IplMagno.normaliseOutput;
        currFn["parasolCells_beta"]>>_retinaParameters.IplMagno.parasolCells_beta;
        currFn["parasolCells_tau"]>>_retinaParameters.IplMagno.parasolCells_tau;
        currFn["parasolCells_k"]>>_retinaParameters.IplMagno.parasolCells_k;
        currFn["amacrinCellsTemporalCutFrequency"]>>_retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
        currFn["V0CompressionParameter"]>>_retinaParameters.IplMagno.V0CompressionParameter;
        currFn["localAdaptintegration_tau"]>>_retinaParameters.IplMagno.localAdaptintegration_tau;
        currFn["localAdaptintegration_k"]>>_retinaParameters.IplMagno.localAdaptintegration_k;

        setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency,_retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);

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

    // report current configuration
    std::cout<<printSetup()<<std::endl;
}

void Retina::setup(cv::Retina::RetinaParameters newConfiguration)
{
    // simply copy structures
    memcpy(&_retinaParameters, &newConfiguration, sizeof(cv::Retina::RetinaParameters));
    // apply setup
    setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
    setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency,_retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);


}

const std::string Retina::printSetup()
{
    std::stringstream outmessage;

    // displaying OPL and IPL parvo setup
    outmessage<<"Current Retina instance setup :"
            <<"\nOPLandIPLparvo"<<"{"
            << "\n==> colorMode : " << _retinaParameters.OPLandIplParvo.colorMode
            << "\n==> normalizeParvoOutput :" << _retinaParameters.OPLandIplParvo.normaliseOutput
            << "\n==> photoreceptorsLocalAdaptationSensitivity : " << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity
            << "\n==> photoreceptorsTemporalConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant
            << "\n==> photoreceptorsSpatialConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant
            << "\n==> horizontalCellsGain : " << _retinaParameters.OPLandIplParvo.horizontalCellsGain
            << "\n==> hcellsTemporalConstant : " << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant
            << "\n==> hcellsSpatialConstant : " << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant
            << "\n==> parvoGanglionCellsSensitivity : " << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity
            <<"}\n";

    // displaying IPL magno setup
    outmessage<<"Current Retina instance setup :"
            <<"\nIPLmagno"<<"{"
            << "\n==> normaliseOutput : " << _retinaParameters.IplMagno.normaliseOutput
            << "\n==> parasolCells_beta : " << _retinaParameters.IplMagno.parasolCells_beta
            << "\n==> parasolCells_tau : " << _retinaParameters.IplMagno.parasolCells_tau
            << "\n==> parasolCells_k : " << _retinaParameters.IplMagno.parasolCells_k
            << "\n==> amacrinCellsTemporalCutFrequency : " << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency
            << "\n==> V0CompressionParameter : " << _retinaParameters.IplMagno.V0CompressionParameter
            << "\n==> localAdaptintegration_tau : " << _retinaParameters.IplMagno.localAdaptintegration_tau
            << "\n==> localAdaptintegration_k : " << _retinaParameters.IplMagno.localAdaptintegration_k
            <<"}";
    return outmessage.str();
}

void Retina::write( std::string fs ) const
{
    FileStorage parametersSaveFile(fs, cv::FileStorage::WRITE );
    write(parametersSaveFile);
}

void Retina::write( FileStorage& fs ) const
{
    if (!fs.isOpened())
        return; // basic error case
    fs<<"OPLandIPLparvo"<<"{";
    fs << "colorMode" << _retinaParameters.OPLandIplParvo.colorMode;
    fs << "normaliseOutput" << _retinaParameters.OPLandIplParvo.normaliseOutput;
    fs << "photoreceptorsLocalAdaptationSensitivity" << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity;
    fs << "photoreceptorsTemporalConstant" << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant;
    fs << "photoreceptorsSpatialConstant" << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant;
    fs << "horizontalCellsGain" << _retinaParameters.OPLandIplParvo.horizontalCellsGain;
    fs << "hcellsTemporalConstant" << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant;
    fs << "hcellsSpatialConstant" << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant;
    fs << "ganglionCellsSensitivity" << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity;
    fs << "}";
    fs<<"IPLmagno"<<"{";
    fs << "normaliseOutput" << _retinaParameters.IplMagno.normaliseOutput;
    fs << "parasolCells_beta" << _retinaParameters.IplMagno.parasolCells_beta;
    fs << "parasolCells_tau" << _retinaParameters.IplMagno.parasolCells_tau;
    fs << "parasolCells_k" << _retinaParameters.IplMagno.parasolCells_k;
    fs << "amacrinCellsTemporalCutFrequency" << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
    fs << "V0CompressionParameter" << _retinaParameters.IplMagno.V0CompressionParameter;
    fs << "localAdaptintegration_tau" << _retinaParameters.IplMagno.localAdaptintegration_tau;
    fs << "localAdaptintegration_k" << _retinaParameters.IplMagno.localAdaptintegration_k;
    fs<<"}";
}

void Retina::setupOPLandIPLParvoChannel(const bool colorMode, const bool normaliseOutput, const float photoreceptorsLocalAdaptationSensitivity, const float photoreceptorsTemporalConstant, const float photoreceptorsSpatialConstant, const float horizontalCellsGain, const float HcellsTemporalConstant, const float HcellsSpatialConstant, const float ganglionCellsSensitivity)
{
    // retina core parameters setup
    _retinaFilter->setColorMode(colorMode);
    _retinaFilter->setPhotoreceptorsLocalAdaptationSensitivity(photoreceptorsLocalAdaptationSensitivity);
    _retinaFilter->setOPLandParvoParameters(0, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, HcellsTemporalConstant, HcellsSpatialConstant, ganglionCellsSensitivity);
    _retinaFilter->setParvoGanglionCellsLocalAdaptationSensitivity(ganglionCellsSensitivity);
    _retinaFilter->activateNormalizeParvoOutput_0_maxOutputValue(normaliseOutput);

        // update parameters struture

    _retinaParameters.OPLandIplParvo.colorMode = colorMode;
    _retinaParameters.OPLandIplParvo.normaliseOutput = normaliseOutput;
    _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity = photoreceptorsLocalAdaptationSensitivity;
    _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant = photoreceptorsTemporalConstant;
    _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant = photoreceptorsSpatialConstant;
    _retinaParameters.OPLandIplParvo.horizontalCellsGain = horizontalCellsGain;
    _retinaParameters.OPLandIplParvo.hcellsTemporalConstant = HcellsTemporalConstant;
    _retinaParameters.OPLandIplParvo.hcellsSpatialConstant = HcellsSpatialConstant;
    _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity = ganglionCellsSensitivity;

}

void Retina::setupIPLMagnoChannel(const bool normaliseOutput, const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
{

    _retinaFilter->setMagnoCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k);
    _retinaFilter->activateNormalizeMagnoOutput_0_maxOutputValue(normaliseOutput);

        // update parameters struture
    _retinaParameters.IplMagno.normaliseOutput = normaliseOutput;
    _retinaParameters.IplMagno.parasolCells_beta = parasolCells_beta;
    _retinaParameters.IplMagno.parasolCells_tau = parasolCells_tau;
    _retinaParameters.IplMagno.parasolCells_k = parasolCells_k;
    _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency = amacrinCellsTemporalCutFrequency;
    _retinaParameters.IplMagno.V0CompressionParameter = V0CompressionParameter;
    _retinaParameters.IplMagno.localAdaptintegration_tau = localAdaptintegration_tau;
    _retinaParameters.IplMagno.localAdaptintegration_k = localAdaptintegration_k;
}

void Retina::run(const cv::Mat &inputMatToConvert)
{
    // first convert input image to the compatible format : std::valarray<float>
    const bool colorMode = _convertCvMat2ValarrayBuffer(inputMatToConvert, _inputBuffer);
    // process the retina
    if (!_retinaFilter->runFilter(_inputBuffer, colorMode, false, _retinaParameters.OPLandIplParvo.colorMode && colorMode, false))
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

// original API level data accessors : copy buffers if size matches
void Retina::getMagno(std::valarray<float> &magnoOutputBufferCopy){if (magnoOutputBufferCopy.size()==_retinaFilter->getMovingContours().size()) magnoOutputBufferCopy = _retinaFilter->getMovingContours();}
void Retina::getParvo(std::valarray<float> &parvoOutputBufferCopy){if (parvoOutputBufferCopy.size()==_retinaFilter->getContours().size()) parvoOutputBufferCopy = _retinaFilter->getContours();}
// original API level data accessors : get buffers addresses...
const std::valarray<float> & Retina::getMagno() const {return _retinaFilter->getMovingContours();}
const std::valarray<float> & Retina::getParvo() const {if (_retinaFilter->getColorMode())return _retinaFilter->getColorOutput(); /* implicite else */return _retinaFilter->getContours();}

// private method called by constructirs
void Retina::_init(const cv::Size inputSz, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    // basic error check
    if (inputSz.height*inputSz.width <= 0)
        throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "Retina::setup", "Retina.h", 0);

    unsigned int nbPixels=inputSz.height*inputSz.width;
    // resize buffers if size does not match
    _inputBuffer.resize(nbPixels*3); // buffer supports gray images but also 3 channels color buffers... (larger is better...)

    // allocate the retina model
        if (_retinaFilter)
           delete _retinaFilter;
    _retinaFilter = new RetinaFilter(inputSz.height, inputSz.width, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);

    _retinaParameters.OPLandIplParvo.colorMode = colorMode;
    // prepare the default parameter XML file with default setup
        setup(_retinaParameters);

    // init retina
    _retinaFilter->clearAllBuffers();

    // report current configuration
    std::cout<<printSetup()<<std::endl;
}

void Retina::_convertValarrayBuffer2cvMat(const std::valarray<float> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, cv::Mat &outBuffer)
{
    // fill output buffer with the valarray buffer
    const float *valarrayPTR=get_data(grayMatrixToConvert);
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

bool Retina::_convertCvMat2ValarrayBuffer(const cv::Mat inputMatToConvert, std::valarray<float> &outputValarrayMatrix)
{
    // first check input consistency
    if (inputMatToConvert.empty())
        throw cv::Exception(-1, "Retina cannot be applied, input buffer is empty", "Retina::run", "Retina.h", 0);

    // retreive color mode from image input
    int imageNumberOfChannels = inputMatToConvert.channels();

        // convert to float AND fill the valarray buffer
    typedef float T; // define here the target pixel format, here, float
        const int dsttype = DataType<T>::depth; // output buffer is float format


    if(imageNumberOfChannels==4)
    {
    // create a cv::Mat table (for RGBA planes)
        cv::Mat planes[4] =
        {
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()*2]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
        };
        planes[3] = cv::Mat(inputMatToConvert.size(), dsttype);     // last channel (alpha) does not point on the valarray (not usefull in our case)
        // split color cv::Mat in 4 planes... it fills valarray directely
        cv::split(cv::Mat_<Vec<T, 4> >(inputMatToConvert), planes);
    }
    else if (imageNumberOfChannels==3)
    {
        // create a cv::Mat table (for RGB planes)
        cv::Mat planes[] =
        {
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()*2]),
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[_retinaFilter->getInputNBpixels()]),
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
        };
        // split color cv::Mat in 3 planes... it fills valarray directely
        cv::split(cv::Mat_<Vec<T, 3> >(inputMatToConvert), planes);
    }
    else if(imageNumberOfChannels==1)
    {
        // create a cv::Mat header for the valarray
        cv::Mat dst(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0]);
        inputMatToConvert.convertTo(dst, dsttype);
    }
        else
            CV_Error(CV_StsUnsupportedFormat, "input image must be single channel (gray levels), bgr format (color) or bgra (color with transparency which won't be considered");

    return imageNumberOfChannels>1; // return bool : false for gray level image processing, true for color mode
}

void Retina::clearBuffers() {_retinaFilter->clearAllBuffers();}

void Retina::activateMovingContoursProcessing(const bool activate){_retinaFilter->activateMovingContoursProcessing(activate);}

void Retina::activateContoursProcessing(const bool activate){_retinaFilter->activateContoursProcessing(activate);}

} // end of namespace cv
