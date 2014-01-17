/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "precomp.hpp"
#include "retina_ocl.hpp"
#include <iostream>
#include <sstream>

#ifdef HAVE_OPENCV_OCL

#include "opencl_kernels.hpp"

#define NOT_IMPLEMENTED CV_Error(cv::Error::StsNotImplemented, "Not implemented")

namespace cv
{
static ocl::ProgramEntry retina_kernel = ocl::bioinspired::retina_kernel;

namespace bioinspired
{
namespace ocl
{
using namespace cv::ocl;

class RetinaOCLImpl : public Retina
{
public:
    RetinaOCLImpl(Size getInputSize);
    RetinaOCLImpl(Size getInputSize, const bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrenght = 10.0);
    virtual ~RetinaOCLImpl();

    Size getInputSize();
    Size getOutputSize();

    void setup(String retinaParameterFile = "", const bool applyDefaultSetupOnFailure = true);
    void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure = true);
    void setup(RetinaParameters newParameters);

    RetinaOCLImpl::RetinaParameters getParameters();

    const String printSetup();
    virtual void write( String fs ) const;
    virtual void write( FileStorage& fs ) const;

    void setupOPLandIPLParvoChannel(const bool colorMode = true, const bool normaliseOutput = true, const float photoreceptorsLocalAdaptationSensitivity = 0.7, const float photoreceptorsTemporalConstant = 0.5, const float photoreceptorsSpatialConstant = 0.53, const float horizontalCellsGain = 0, const float HcellsTemporalConstant = 1, const float HcellsSpatialConstant = 7, const float ganglionCellsSensitivity = 0.7);
    void setupIPLMagnoChannel(const bool normaliseOutput = true, const float parasolCells_beta = 0, const float parasolCells_tau = 0, const float parasolCells_k = 7, const float amacrinCellsTemporalCutFrequency = 1.2, const float V0CompressionParameter = 0.95, const float localAdaptintegration_tau = 0, const float localAdaptintegration_k = 7);

    void run(InputArray inputImage);
    void getParvo(OutputArray retinaOutput_parvo);
    void getMagno(OutputArray retinaOutput_magno);

    void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0);
    void clearBuffers();
    void activateMovingContoursProcessing(const bool activate);
    void activateContoursProcessing(const bool activate);

    // unimplemented interfaces:
    void applyFastToneMapping(InputArray /*inputImage*/, OutputArray /*outputToneMappedImage*/) { NOT_IMPLEMENTED; }
    void getParvoRAW(OutputArray /*retinaOutput_parvo*/) { NOT_IMPLEMENTED; }
    void getMagnoRAW(OutputArray /*retinaOutput_magno*/) { NOT_IMPLEMENTED; }
    const Mat getMagnoRAW() const { NOT_IMPLEMENTED; return Mat(); }
    const Mat getParvoRAW() const { NOT_IMPLEMENTED; return Mat(); }

protected:
    RetinaParameters _retinaParameters;
    cv::ocl::oclMat _inputBuffer;
    RetinaFilter* _retinaFilter;
    bool convertToColorPlanes(const cv::ocl::oclMat& input, cv::ocl::oclMat &output);
    void convertToInterleaved(const cv::ocl::oclMat& input, bool colorMode, cv::ocl::oclMat &output);
    void _init(const Size getInputSize, const bool colorMode, int colorSamplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrenght = 10.0);
};

RetinaOCLImpl::RetinaOCLImpl(const cv::Size inputSz)
{
    _retinaFilter = 0;
    _init(inputSz, true, RETINA_COLOR_BAYER, false);
}

RetinaOCLImpl::RetinaOCLImpl(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    _retinaFilter = 0;
    _init(inputSz, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
}

RetinaOCLImpl::~RetinaOCLImpl()
{
    if (_retinaFilter)
    {
        delete _retinaFilter;
    }
}

/**
* retreive retina input buffer size
*/
Size RetinaOCLImpl::getInputSize()
{
    return cv::Size(_retinaFilter->getInputNBcolumns(), _retinaFilter->getInputNBrows());
}

/**
* retreive retina output buffer size
*/
Size RetinaOCLImpl::getOutputSize()
{
    return cv::Size(_retinaFilter->getOutputNBcolumns(), _retinaFilter->getOutputNBrows());
}


void RetinaOCLImpl::setColorSaturation(const bool saturateColors, const float colorSaturationValue)
{
    _retinaFilter->setColorSaturation(saturateColors, colorSaturationValue);
}

struct RetinaOCLImpl::RetinaParameters RetinaOCLImpl::getParameters()
{
    return _retinaParameters;
}


void RetinaOCLImpl::setup(String retinaParameterFile, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // opening retinaParameterFile in read mode
        cv::FileStorage fs(retinaParameterFile, cv::FileStorage::READ);
        setup(fs, applyDefaultSetupOnFailure);
    }
    catch(Exception &e)
    {
        std::cout << "RetinaOCLImpl::setup: wrong/unappropriate xml parameter file : error report :`n=>" << e.what() << std::endl;
        if (applyDefaultSetupOnFailure)
        {
            std::cout << "RetinaOCLImpl::setup: resetting retina with default parameters" << std::endl;
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        else
        {
            std::cout << "=> keeping current parameters" << std::endl;
        }
    }
}

void RetinaOCLImpl::setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // read parameters file if it exists or apply default setup if asked for
        if (!fs.isOpened())
        {
            std::cout << "RetinaOCLImpl::setup: provided parameters file could not be open... skeeping configuration" << std::endl;
            return;
            // implicit else case : retinaParameterFile could be open (it exists at least)
        }
        // OPL and Parvo init first... update at the same time the parameters structure and the retina core
        cv::FileNode rootFn = fs.root(), currFn = rootFn["OPLandIPLparvo"];
        currFn["colorMode"] >> _retinaParameters.OPLandIplParvo.colorMode;
        currFn["normaliseOutput"] >> _retinaParameters.OPLandIplParvo.normaliseOutput;
        currFn["photoreceptorsLocalAdaptationSensitivity"] >> _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity;
        currFn["photoreceptorsTemporalConstant"] >> _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant;
        currFn["photoreceptorsSpatialConstant"] >> _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant;
        currFn["horizontalCellsGain"] >> _retinaParameters.OPLandIplParvo.horizontalCellsGain;
        currFn["hcellsTemporalConstant"] >> _retinaParameters.OPLandIplParvo.hcellsTemporalConstant;
        currFn["hcellsSpatialConstant"] >> _retinaParameters.OPLandIplParvo.hcellsSpatialConstant;
        currFn["ganglionCellsSensitivity"] >> _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity;
        setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);

        // init retina IPL magno setup... update at the same time the parameters structure and the retina core
        currFn = rootFn["IPLmagno"];
        currFn["normaliseOutput"] >> _retinaParameters.IplMagno.normaliseOutput;
        currFn["parasolCells_beta"] >> _retinaParameters.IplMagno.parasolCells_beta;
        currFn["parasolCells_tau"] >> _retinaParameters.IplMagno.parasolCells_tau;
        currFn["parasolCells_k"] >> _retinaParameters.IplMagno.parasolCells_k;
        currFn["amacrinCellsTemporalCutFrequency"] >> _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
        currFn["V0CompressionParameter"] >> _retinaParameters.IplMagno.V0CompressionParameter;
        currFn["localAdaptintegration_tau"] >> _retinaParameters.IplMagno.localAdaptintegration_tau;
        currFn["localAdaptintegration_k"] >> _retinaParameters.IplMagno.localAdaptintegration_k;

        setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency, _retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);

    }
    catch(Exception &e)
    {
        std::cout << "RetinaOCLImpl::setup: resetting retina with default parameters" << std::endl;
        if (applyDefaultSetupOnFailure)
        {
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        std::cout << "RetinaOCLImpl::setup: wrong/unappropriate xml parameter file : error report :`n=>" << e.what() << std::endl;
        std::cout << "=> keeping current parameters" << std::endl;
    }
}

void RetinaOCLImpl::setup(cv::bioinspired::Retina::RetinaParameters newConfiguration)
{
    // simply copy structures
    memcpy(&_retinaParameters, &newConfiguration, sizeof(cv::bioinspired::Retina::RetinaParameters));
    // apply setup
    setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
    setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency, _retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);
}

const String RetinaOCLImpl::printSetup()
{
    std::stringstream outmessage;

    // displaying OPL and IPL parvo setup
    outmessage << "Current Retina instance setup :"
               << "\nOPLandIPLparvo" << "{"
               << "\n==> colorMode : " << _retinaParameters.OPLandIplParvo.colorMode
               << "\n==> normalizeParvoOutput :" << _retinaParameters.OPLandIplParvo.normaliseOutput
               << "\n==> photoreceptorsLocalAdaptationSensitivity : " << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity
               << "\n==> photoreceptorsTemporalConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant
               << "\n==> photoreceptorsSpatialConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant
               << "\n==> horizontalCellsGain : " << _retinaParameters.OPLandIplParvo.horizontalCellsGain
               << "\n==> hcellsTemporalConstant : " << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant
               << "\n==> hcellsSpatialConstant : " << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant
               << "\n==> parvoGanglionCellsSensitivity : " << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity
               << "}\n";

    // displaying IPL magno setup
    outmessage << "Current Retina instance setup :"
               << "\nIPLmagno" << "{"
               << "\n==> normaliseOutput : " << _retinaParameters.IplMagno.normaliseOutput
               << "\n==> parasolCells_beta : " << _retinaParameters.IplMagno.parasolCells_beta
               << "\n==> parasolCells_tau : " << _retinaParameters.IplMagno.parasolCells_tau
               << "\n==> parasolCells_k : " << _retinaParameters.IplMagno.parasolCells_k
               << "\n==> amacrinCellsTemporalCutFrequency : " << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency
               << "\n==> V0CompressionParameter : " << _retinaParameters.IplMagno.V0CompressionParameter
               << "\n==> localAdaptintegration_tau : " << _retinaParameters.IplMagno.localAdaptintegration_tau
               << "\n==> localAdaptintegration_k : " << _retinaParameters.IplMagno.localAdaptintegration_k
               << "}";
    return outmessage.str().c_str();
}

void RetinaOCLImpl::write( String fs ) const
{
    FileStorage parametersSaveFile(fs, cv::FileStorage::WRITE );
    write(parametersSaveFile);
}

void RetinaOCLImpl::write( FileStorage& fs ) const
{
    if (!fs.isOpened())
    {
        return;    // basic error case
    }
    fs << "OPLandIPLparvo" << "{";
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
    fs << "IPLmagno" << "{";
    fs << "normaliseOutput" << _retinaParameters.IplMagno.normaliseOutput;
    fs << "parasolCells_beta" << _retinaParameters.IplMagno.parasolCells_beta;
    fs << "parasolCells_tau" << _retinaParameters.IplMagno.parasolCells_tau;
    fs << "parasolCells_k" << _retinaParameters.IplMagno.parasolCells_k;
    fs << "amacrinCellsTemporalCutFrequency" << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency;
    fs << "V0CompressionParameter" << _retinaParameters.IplMagno.V0CompressionParameter;
    fs << "localAdaptintegration_tau" << _retinaParameters.IplMagno.localAdaptintegration_tau;
    fs << "localAdaptintegration_k" << _retinaParameters.IplMagno.localAdaptintegration_k;
    fs << "}";
}

void RetinaOCLImpl::setupOPLandIPLParvoChannel(const bool colorMode, const bool normaliseOutput, const float photoreceptorsLocalAdaptationSensitivity, const float photoreceptorsTemporalConstant, const float photoreceptorsSpatialConstant, const float horizontalCellsGain, const float HcellsTemporalConstant, const float HcellsSpatialConstant, const float ganglionCellsSensitivity)
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

void RetinaOCLImpl::setupIPLMagnoChannel(const bool normaliseOutput, const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
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

void RetinaOCLImpl::run(const InputArray input)
{
    oclMat &inputMatToConvert = getOclMatRef(input);
    bool colorMode = convertToColorPlanes(inputMatToConvert, _inputBuffer);
    // first convert input image to the compatible format : std::valarray<float>
    // process the retina
    if (!_retinaFilter->runFilter(_inputBuffer, colorMode, false, _retinaParameters.OPLandIplParvo.colorMode && colorMode, false))
    {
        throw cv::Exception(-1, "Retina cannot be applied, wrong input buffer size", "RetinaOCLImpl::run", "Retina.h", 0);
    }
}

void RetinaOCLImpl::getParvo(OutputArray output)
{
    oclMat &retinaOutput_parvo = getOclMatRef(output);
    if (_retinaFilter->getColorMode())
    {
        // reallocate output buffer (if necessary)
        convertToInterleaved(_retinaFilter->getColorOutput(), true, retinaOutput_parvo);
    }
    else
    {
        // reallocate output buffer (if necessary)
        convertToInterleaved(_retinaFilter->getContours(), false, retinaOutput_parvo);
    }
    //retinaOutput_parvo/=255.0;
}
void RetinaOCLImpl::getMagno(OutputArray output)
{
    oclMat &retinaOutput_magno = getOclMatRef(output);
    // reallocate output buffer (if necessary)
    convertToInterleaved(_retinaFilter->getMovingContours(), false, retinaOutput_magno);
    //retinaOutput_magno/=255.0;
}
// private method called by constructirs
void RetinaOCLImpl::_init(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    // basic error check
    if (inputSz.height*inputSz.width <= 0)
    {
        throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "RetinaOCLImpl::setup", "Retina.h", 0);
    }

    // allocate the retina model
    if (_retinaFilter)
    {
        delete _retinaFilter;
    }
    _retinaFilter = new RetinaFilter(inputSz.height, inputSz.width, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);

    // prepare the default parameter XML file with default setup
    setup(_retinaParameters);

    // init retina
    _retinaFilter->clearAllBuffers();
}

bool RetinaOCLImpl::convertToColorPlanes(const oclMat& input, oclMat &output)
{
    oclMat convert_input;
    input.convertTo(convert_input, CV_32F);
    if(convert_input.channels() == 3 || convert_input.channels() == 4)
    {
        ocl::ensureSizeIsEnough(int(_retinaFilter->getInputNBrows() * 4),
                                int(_retinaFilter->getInputNBcolumns()), CV_32FC1, output);
        oclMat channel_splits[4] =
        {
            output(Rect(Point(0, _retinaFilter->getInputNBrows() * 2), getInputSize())),
            output(Rect(Point(0, _retinaFilter->getInputNBrows()), getInputSize())),
            output(Rect(Point(0, 0), getInputSize())),
            output(Rect(Point(0, _retinaFilter->getInputNBrows() * 3), getInputSize()))
        };
        ocl::split(convert_input, channel_splits);
        return true;
    }
    else if(convert_input.channels() == 1)
    {
        convert_input.copyTo(output);
        return false;
    }
    else
    {
        CV_Error(-1, "Retina ocl only support 1, 3, 4 channel input");
        return false;
    }
}
void RetinaOCLImpl::convertToInterleaved(const oclMat& input, bool colorMode, oclMat &output)
{
    input.convertTo(output, CV_8U);
    if(colorMode)
    {
        int numOfSplits = input.rows / getInputSize().height;
        std::vector<oclMat> channel_splits(numOfSplits);
        for(int i = 0; i < static_cast<int>(channel_splits.size()); i ++)
        {
            channel_splits[i] =
                output(Rect(Point(0, _retinaFilter->getInputNBrows() * (numOfSplits - i - 1)), getInputSize()));
        }
        merge(channel_splits, output);
    }
    else
    {
        //...
    }
}

void RetinaOCLImpl::clearBuffers()
{
    _retinaFilter->clearAllBuffers();
}

void RetinaOCLImpl::activateMovingContoursProcessing(const bool activate)
{
    _retinaFilter->activateMovingContoursProcessing(activate);
}

void RetinaOCLImpl::activateContoursProcessing(const bool activate)
{
    _retinaFilter->activateContoursProcessing(activate);
}

///////////////////////////////////////
///////// BasicRetinaFilter ///////////
///////////////////////////////////////
BasicRetinaFilter::BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize, const bool)
    : _NBrows(NBrows), _NBcols(NBcolumns),
      _filterOutput(NBrows, NBcolumns, CV_32FC1),
      _localBuffer(NBrows, NBcolumns, CV_32FC1),
      _filteringCoeficientsTable(3 * parametersListSize)
{
    _halfNBrows = _filterOutput.rows / 2;
    _halfNBcolumns = _filterOutput.cols / 2;

    // set default values
    _maxInputValue = 256.0;

    // reset all buffers
    clearAllBuffers();
}

BasicRetinaFilter::~BasicRetinaFilter()
{
}

void BasicRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    // resizing buffers
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _filterOutput);

    // updating variables
    _halfNBrows = _filterOutput.rows / 2;
    _halfNBcolumns = _filterOutput.cols / 2;

    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localBuffer);
    // reset buffers
    clearAllBuffers();
}

void BasicRetinaFilter::setLPfilterParameters(const float beta, const float tau, const float desired_k, const unsigned int filterIndex)
{
    float _beta = beta + tau;
    float k = desired_k;
    // check if the spatial constant is correct (avoid 0 value to avoid division by 0)
    if (desired_k <= 0)
    {
        k = 0.001f;
        std::cerr << "BasicRetinaFilter::spatial constant of the low pass filter must be superior to zero !!! correcting parameter setting to 0,001" << std::endl;
    }

    float _alpha = k * k;
    float _mu = 0.8f;
    unsigned int tableOffset = filterIndex * 3;
    if (k <= 0)
    {
        std::cerr << "BasicRetinaFilter::spatial filtering coefficient must be superior to zero, correcting value to 0.01" << std::endl;
        _alpha = 0.0001f;
    }

    float _temp =  (1.0f + _beta) / (2.0f * _mu * _alpha);
    float a = _filteringCoeficientsTable[tableOffset] = 1.0f + _temp - (float)sqrt( (1.0f + _temp) * (1.0f + _temp) - 1.0f);
    _filteringCoeficientsTable[1 + tableOffset] = (1.0f - a) * (1.0f - a) * (1.0f - a) * (1.0f - a) / (1.0f + _beta);
    _filteringCoeficientsTable[2 + tableOffset] = tau;
}
const oclMat &BasicRetinaFilter::runFilter_LocalAdapdation(const oclMat &inputFrame, const oclMat &localLuminance)
{
    _localLuminanceAdaptation(inputFrame, localLuminance, _filterOutput);
    return _filterOutput;
}


void BasicRetinaFilter::runFilter_LocalAdapdation(const oclMat &inputFrame, const oclMat &localLuminance, oclMat &outputFrame)
{
    _localLuminanceAdaptation(inputFrame, localLuminance, outputFrame);
}

const oclMat &BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const oclMat &inputFrame)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput);
    _localLuminanceAdaptation(inputFrame, _filterOutput, _filterOutput);
    return _filterOutput;
}
void BasicRetinaFilter::runFilter_LocalAdapdation_autonomous(const oclMat &inputFrame, oclMat &outputFrame)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput);
    _localLuminanceAdaptation(inputFrame, _filterOutput, outputFrame);
}

void BasicRetinaFilter::_localLuminanceAdaptation(oclMat &inputOutputFrame, const oclMat &localLuminance)
{
    _localLuminanceAdaptation(inputOutputFrame, localLuminance, inputOutputFrame, false);
}

void BasicRetinaFilter::_localLuminanceAdaptation(const oclMat &inputFrame, const oclMat &localLuminance, oclMat &outputFrame, const bool updateLuminanceMean)
{
    if (updateLuminanceMean)
    {
        float meanLuminance = saturate_cast<float>(ocl::sum(inputFrame)[0]) / getNBpixels();
        updateCompressionParameter(meanLuminance);
    }
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, _NBrows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &localLuminance.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &inputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &_localLuminanceAddon));
    args.push_back(std::make_pair(sizeof(cl_float), &_localLuminanceFactor));
    args.push_back(std::make_pair(sizeof(cl_float), &_maxInputValue));
    openCLExecuteKernel(ctx, &retina_kernel, "localLuminanceAdaptation", globalSize, localSize, args, -1, -1);
}

const oclMat &BasicRetinaFilter::runFilter_LPfilter(const oclMat &inputFrame, const unsigned int filterIndex)
{
    _spatiotemporalLPfilter(inputFrame, _filterOutput, filterIndex);
    return _filterOutput;
}
void BasicRetinaFilter::runFilter_LPfilter(const oclMat &inputFrame, oclMat &outputFrame, const unsigned int filterIndex)
{
    _spatiotemporalLPfilter(inputFrame, outputFrame, filterIndex);
}

void BasicRetinaFilter::_spatiotemporalLPfilter(const oclMat &inputFrame, oclMat &LPfilterOutput, const unsigned int filterIndex)
{
    unsigned int coefTableOffset = filterIndex * 3;

    _a = _filteringCoeficientsTable[coefTableOffset];
    _gain = _filteringCoeficientsTable[1 + coefTableOffset];
    _tau = _filteringCoeficientsTable[2 + coefTableOffset];

    _horizontalCausalFilter_addInput(inputFrame, LPfilterOutput);
    _horizontalAnticausalFilter(LPfilterOutput);
    _verticalCausalFilter(LPfilterOutput);
    _verticalAnticausalFilter_multGain(LPfilterOutput);
}

void BasicRetinaFilter::_horizontalCausalFilter_addInput(const oclMat &inputFrame, oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &inputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_float), &_tau));
    args.push_back(std::make_pair(sizeof(cl_float), &_a));
    openCLExecuteKernel(ctx, &retina_kernel, "horizontalCausalFilter_addInput", globalSize, localSize, args, -1, -1);
}

void BasicRetinaFilter::_horizontalAnticausalFilter(oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_float), &_a));
    openCLExecuteKernel(ctx, &retina_kernel, "horizontalAnticausalFilter", globalSize, localSize, args, -1, -1);
}

void BasicRetinaFilter::_verticalCausalFilter(oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_float), &_a));
    openCLExecuteKernel(ctx, &retina_kernel, "verticalCausalFilter", globalSize, localSize, args, -1, -1);
}

void BasicRetinaFilter::_verticalAnticausalFilter_multGain(oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_float), &_a));
    args.push_back(std::make_pair(sizeof(cl_float), &_gain));
    openCLExecuteKernel(ctx, &retina_kernel, "verticalAnticausalFilter_multGain", globalSize, localSize, args, -1, -1);
}

void BasicRetinaFilter::_horizontalAnticausalFilter_Irregular(oclMat &outputFrame, const oclMat &spatialConstantBuffer)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {outputFrame.rows, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &spatialConstantBuffer.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_int),   &spatialConstantBuffer.offset));
    openCLExecuteKernel(ctx, &retina_kernel, "horizontalAnticausalFilter_Irregular", globalSize, localSize, args, -1, -1);
}

//  vertical anticausal filter
void BasicRetinaFilter::_verticalCausalFilter_Irregular(oclMat &outputFrame, const oclMat &spatialConstantBuffer)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {outputFrame.cols, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &spatialConstantBuffer.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_int),   &spatialConstantBuffer.offset));
    openCLExecuteKernel(ctx, &retina_kernel, "verticalCausalFilter_Irregular", globalSize, localSize, args, -1, -1);
}

void normalizeGrayOutput_0_maxOutputValue(oclMat &inputOutputBuffer, const float maxOutputValue)
{
    double min_val, max_val;
    ocl::minMax(inputOutputBuffer, &min_val, &max_val);
    float factor = maxOutputValue / static_cast<float>(max_val - min_val);
    float offset = - static_cast<float>(min_val) * factor;
    ocl::multiply(factor, inputOutputBuffer, inputOutputBuffer);
    ocl::add(inputOutputBuffer, offset, inputOutputBuffer);
}

void normalizeGrayOutputCentredSigmoide(const float meanValue, const float sensitivity, oclMat &in, oclMat &out, const float maxValue)
{
    if (sensitivity == 1.0f)
    {
        std::cerr << "TemplateBuffer::TemplateBuffer<type>::normalizeGrayOutputCentredSigmoide error: 2nd parameter (sensitivity) must not equal 0, copying original data..." << std::endl;
        in.copyTo(out);
        return;
    }

    float X0 = maxValue / (sensitivity - 1.0f);

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {in.cols, out.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    int elements_per_row = static_cast<int>(out.step / out.elemSize());

    args.push_back(std::make_pair(sizeof(cl_mem),   &in.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &out.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &in.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &in.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &meanValue));
    args.push_back(std::make_pair(sizeof(cl_float), &X0));
    openCLExecuteKernel(ctx, &retina_kernel, "normalizeGrayOutputCentredSigmoide", globalSize, localSize, args, -1, -1);
}

void normalizeGrayOutputNearZeroCentreredSigmoide(oclMat &inputPicture, oclMat &outputBuffer, const float sensitivity, const float maxOutputValue)
{
    float X0cube = sensitivity * sensitivity * sensitivity;

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {inputPicture.cols, inputPicture.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    int elements_per_row = static_cast<int>(inputPicture.step / inputPicture.elemSize());
    args.push_back(std::make_pair(sizeof(cl_mem),   &inputPicture.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &outputBuffer.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputPicture.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputPicture.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &maxOutputValue));
    args.push_back(std::make_pair(sizeof(cl_float), &X0cube));
    openCLExecuteKernel(ctx, &retina_kernel, "normalizeGrayOutputNearZeroCentreredSigmoide", globalSize, localSize, args, -1, -1);
}

void centerReductImageLuminance(oclMat &inputoutput)
{
    Scalar mean, stddev;
    cv::meanStdDev((Mat)inputoutput, mean, stddev);

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {inputoutput.cols, inputoutput.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    float f_mean = static_cast<float>(mean[0]);
    float f_stddev = static_cast<float>(stddev[0]);
    int elements_per_row = static_cast<int>(inputoutput.step / inputoutput.elemSize());
    args.push_back(std::make_pair(sizeof(cl_mem),   &inputoutput.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputoutput.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputoutput.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &f_mean));
    args.push_back(std::make_pair(sizeof(cl_float), &f_stddev));
    openCLExecuteKernel(ctx, &retina_kernel, "centerReductImageLuminance", globalSize, localSize, args, -1, -1);
}

///////////////////////////////////////
///////// ParvoRetinaFilter ///////////
///////////////////////////////////////
ParvoRetinaFilter::ParvoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
    : BasicRetinaFilter(NBrows, NBcolumns, 3),
      _photoreceptorsOutput(NBrows, NBcolumns, CV_32FC1),
      _horizontalCellsOutput(NBrows, NBcolumns, CV_32FC1),
      _parvocellularOutputON(NBrows, NBcolumns, CV_32FC1),
      _parvocellularOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _bipolarCellsOutputON(NBrows, NBcolumns, CV_32FC1),
      _bipolarCellsOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _localAdaptationOFF(NBrows, NBcolumns, CV_32FC1)
{
    // link to the required local parent adaptation buffers
    _localAdaptationON = _localBuffer;
    _parvocellularOutputONminusOFF = _filterOutput;

    // init: set all the values to 0
    clearAllBuffers();
}

ParvoRetinaFilter::~ParvoRetinaFilter()
{
}

void ParvoRetinaFilter::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _photoreceptorsOutput = 0;
    _horizontalCellsOutput = 0;
    _parvocellularOutputON = 0;
    _parvocellularOutputOFF = 0;
    _bipolarCellsOutputON = 0;
    _bipolarCellsOutputOFF = 0;
    _localAdaptationOFF = 0;
}
void ParvoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::resize(NBrows, NBcolumns);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _photoreceptorsOutput);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _horizontalCellsOutput);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _parvocellularOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _parvocellularOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _bipolarCellsOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _bipolarCellsOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localAdaptationOFF);

    // link to the required local parent adaptation buffers
    _localAdaptationON = _localBuffer;
    _parvocellularOutputONminusOFF = _filterOutput;

    // clean buffers
    clearAllBuffers();
}

void ParvoRetinaFilter::setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2)
{
    // init photoreceptors low pass filter
    setLPfilterParameters(beta1, tau1, k1);
    // init horizontal cells low pass filter
    setLPfilterParameters(beta2, tau2, k2, 1);
    // init parasol ganglion cells low pass filter (default parameters)
    setLPfilterParameters(0, tau1, k1, 2);

}
const oclMat &ParvoRetinaFilter::runFilter(const oclMat &inputFrame, const bool useParvoOutput)
{
    _spatiotemporalLPfilter(inputFrame, _photoreceptorsOutput);
    _spatiotemporalLPfilter(_photoreceptorsOutput, _horizontalCellsOutput, 1);
    _OPL_OnOffWaysComputing();

    if (useParvoOutput)
    {
        // local adaptation processes on ON and OFF ways
        _spatiotemporalLPfilter(_bipolarCellsOutputON, _localAdaptationON, 2);
        _localLuminanceAdaptation(_parvocellularOutputON, _localAdaptationON);
        _spatiotemporalLPfilter(_bipolarCellsOutputOFF, _localAdaptationOFF, 2);
        _localLuminanceAdaptation(_parvocellularOutputOFF, _localAdaptationOFF);
        ocl::subtract(_parvocellularOutputON, _parvocellularOutputOFF, _parvocellularOutputONminusOFF);
    }

    return _parvocellularOutputONminusOFF;
}
void ParvoRetinaFilter::_OPL_OnOffWaysComputing()
{
    int elements_per_row = static_cast<int>(_photoreceptorsOutput.step / _photoreceptorsOutput.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {(_photoreceptorsOutput.cols + 3) / 4, _photoreceptorsOutput.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &_photoreceptorsOutput.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_horizontalCellsOutput.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_bipolarCellsOutputON.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_bipolarCellsOutputOFF.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_parvocellularOutputON.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_parvocellularOutputOFF.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_photoreceptorsOutput.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_photoreceptorsOutput.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "OPL_OnOffWaysComputing", globalSize, localSize, args, -1, -1);
}

///////////////////////////////////////
//////////// MagnoFilter //////////////
///////////////////////////////////////
MagnoRetinaFilter::MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns)
    : BasicRetinaFilter(NBrows, NBcolumns, 2),
      _previousInput_ON(NBrows, NBcolumns, CV_32FC1),
      _previousInput_OFF(NBrows, NBcolumns, CV_32FC1),
      _amacrinCellsTempOutput_ON(NBrows, NBcolumns, CV_32FC1),
      _amacrinCellsTempOutput_OFF(NBrows, NBcolumns, CV_32FC1),
      _magnoXOutputON(NBrows, NBcolumns, CV_32FC1),
      _magnoXOutputOFF(NBrows, NBcolumns, CV_32FC1),
      _localProcessBufferON(NBrows, NBcolumns, CV_32FC1),
      _localProcessBufferOFF(NBrows, NBcolumns, CV_32FC1)
{
    _magnoYOutput = _filterOutput;
    _magnoYsaturated = _localBuffer;

    clearAllBuffers();
}

MagnoRetinaFilter::~MagnoRetinaFilter()
{
}
void MagnoRetinaFilter::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _previousInput_ON = 0;
    _previousInput_OFF = 0;
    _amacrinCellsTempOutput_ON = 0;
    _amacrinCellsTempOutput_OFF = 0;
    _magnoXOutputON = 0;
    _magnoXOutputOFF = 0;
    _localProcessBufferON = 0;
    _localProcessBufferOFF = 0;

}
void MagnoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::resize(NBrows, NBcolumns);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _previousInput_ON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _previousInput_OFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _amacrinCellsTempOutput_ON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _amacrinCellsTempOutput_OFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _magnoXOutputON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _magnoXOutputOFF);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localProcessBufferON);
    ensureSizeIsEnough(NBrows, NBcolumns, CV_32FC1, _localProcessBufferOFF);

    // to be sure, relink buffers
    _magnoYOutput = _filterOutput;
    _magnoYsaturated = _localBuffer;

    // reset all buffers
    clearAllBuffers();
}

void MagnoRetinaFilter::setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau, const float localAdaptIntegration_k )
{
    _temporalCoefficient = (float)std::exp(-1.0f / amacrinCellsTemporalCutFrequency);
    // the first set of parameters is dedicated to the low pass filtering property of the ganglion cells
    BasicRetinaFilter::setLPfilterParameters(parasolCells_beta, parasolCells_tau, parasolCells_k, 0);
    // the second set of parameters is dedicated to the ganglion cells output intergartion for their local adaptation property
    BasicRetinaFilter::setLPfilterParameters(0, localAdaptIntegration_tau, localAdaptIntegration_k, 1);
}

void MagnoRetinaFilter::_amacrineCellsComputing(
    const oclMat &OPL_ON,
    const oclMat &OPL_OFF
)
{
    int elements_per_row = static_cast<int>(OPL_ON.step / OPL_ON.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {OPL_ON.cols, OPL_ON.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &OPL_ON.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &OPL_OFF.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_previousInput_ON.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_previousInput_OFF.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_amacrinCellsTempOutput_ON.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &_amacrinCellsTempOutput_OFF.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &OPL_ON.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &OPL_ON.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &_temporalCoefficient));
    openCLExecuteKernel(ctx, &retina_kernel, "amacrineCellsComputing", globalSize, localSize, args, -1, -1);
}

const oclMat &MagnoRetinaFilter::runFilter(const oclMat &OPL_ON, const oclMat &OPL_OFF)
{
    // Compute the high pass temporal filter
    _amacrineCellsComputing(OPL_ON, OPL_OFF);

    // apply low pass filtering on ON and OFF ways after temporal high pass filtering
    _spatiotemporalLPfilter(_amacrinCellsTempOutput_ON, _magnoXOutputON, 0);
    _spatiotemporalLPfilter(_amacrinCellsTempOutput_OFF, _magnoXOutputOFF, 0);

    // local adaptation of the ganglion cells to the local contrast of the moving contours
    _spatiotemporalLPfilter(_magnoXOutputON, _localProcessBufferON, 1);
    _localLuminanceAdaptation(_magnoXOutputON, _localProcessBufferON);

    _spatiotemporalLPfilter(_magnoXOutputOFF, _localProcessBufferOFF, 1);
    _localLuminanceAdaptation(_magnoXOutputOFF, _localProcessBufferOFF);

    _magnoYOutput = _magnoXOutputON + _magnoXOutputOFF;

    return _magnoYOutput;
}

///////////////////////////////////////
//////////// RetinaColor //////////////
///////////////////////////////////////

// define an array of ROI headers of input x
#define MAKE_OCLMAT_SLICES(x, n) \
    oclMat x##_slices[n];\
    for(int _SLICE_INDEX_ = 0; _SLICE_INDEX_ < n; _SLICE_INDEX_ ++)\
    {\
        x##_slices[_SLICE_INDEX_] = x(getROI(_SLICE_INDEX_));\
    }

RetinaColor::RetinaColor(const unsigned int NBrows, const unsigned int NBcolumns, const int samplingMethod)
    : BasicRetinaFilter(NBrows, NBcolumns, 3),
      _RGBmosaic(NBrows * 3, NBcolumns, CV_32FC1),
      _tempMultiplexedFrame(NBrows, NBcolumns, CV_32FC1),
      _demultiplexedTempBuffer(NBrows * 3, NBcolumns, CV_32FC1),
      _demultiplexedColorFrame(NBrows * 3, NBcolumns, CV_32FC1),
      _chrominance(NBrows * 3, NBcolumns, CV_32FC1),
      _colorLocalDensity(NBrows * 3, NBcolumns, CV_32FC1),
      _imageGradient(NBrows * 3, NBcolumns, CV_32FC1)
{
    // link to parent buffers (let's recycle !)
    _luminance = _filterOutput;
    _multiplexedFrame = _localBuffer;

    _objectInit = false;
    _samplingMethod = samplingMethod;
    _saturateColors = false;
    _colorSaturationValue = 4.0;

    // set default spatio-temporal filter parameters
    setLPfilterParameters(0.0, 0.0, 1.5);
    setLPfilterParameters(0.0, 0.0, 10.5, 1);// for the low pass filter dedicated to contours energy extraction (demultiplexing process)
    setLPfilterParameters(0.f, 0.f, 0.9f, 2);

    // init default value on image Gradient
    _imageGradient = 0.57f;

    // init color sampling map
    _initColorSampling();

    // flush all buffers
    clearAllBuffers();
}

RetinaColor::~RetinaColor()
{

}

void RetinaColor::clearAllBuffers()
{
    BasicRetinaFilter::clearAllBuffers();
    _tempMultiplexedFrame = 0.f;
    _demultiplexedTempBuffer = 0.f;

    _demultiplexedColorFrame = 0.f;
    _chrominance = 0.f;
    _imageGradient = 0.57f;
}

void RetinaColor::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    BasicRetinaFilter::clearAllBuffers();
    ensureSizeIsEnough(NBrows,     NBcolumns, CV_32FC1, _tempMultiplexedFrame);
    ensureSizeIsEnough(NBrows * 2, NBcolumns, CV_32FC1, _imageGradient);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _RGBmosaic);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _demultiplexedTempBuffer);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _demultiplexedColorFrame);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _chrominance);
    ensureSizeIsEnough(NBrows * 3, NBcolumns, CV_32FC1, _colorLocalDensity);

    // link to parent buffers (let's recycle !)
    _luminance = _filterOutput;
    _multiplexedFrame = _localBuffer;

    // init color sampling map
    _initColorSampling();

    // clean buffers
    clearAllBuffers();
}

static void inverseValue(oclMat &input)
{
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {input.cols, input.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &input.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &input.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &input.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "inverseValue", globalSize, localSize, args, -1, -1);
}

void RetinaColor::_initColorSampling()
{
    CV_Assert(_samplingMethod == RETINA_COLOR_BAYER);
    _pR = _pB = 0.25;
    _pG = 0.5;
    // filling the mosaic buffer:
    _RGBmosaic = 0;
    Mat tmp_mat(_NBrows * 3, _NBcols, CV_32FC1);
    float * tmp_mat_ptr = tmp_mat.ptr<float>();
    tmp_mat.setTo(0);
    for (unsigned int index = 0 ; index < getNBpixels(); ++index)
    {
        tmp_mat_ptr[bayerSampleOffset(index)] = 1.0;
    }
    _RGBmosaic.upload(tmp_mat);
    // computing photoreceptors local density
    MAKE_OCLMAT_SLICES(_RGBmosaic, 3);
    MAKE_OCLMAT_SLICES(_colorLocalDensity, 3);
    _colorLocalDensity.setTo(0);
    _spatiotemporalLPfilter(_RGBmosaic_slices[0], _colorLocalDensity_slices[0]);
    _spatiotemporalLPfilter(_RGBmosaic_slices[1], _colorLocalDensity_slices[1]);
    _spatiotemporalLPfilter(_RGBmosaic_slices[2], _colorLocalDensity_slices[2]);

    //_colorLocalDensity = oclMat(_colorLocalDensity.size(), _colorLocalDensity.type(), 1.f) / _colorLocalDensity;
    inverseValue(_colorLocalDensity);

    _objectInit = true;
}

static void demultiplex(const oclMat &input, oclMat &ouput)
{
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {input.cols, input.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &input.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &ouput.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &input.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &input.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "runColorDemultiplexingBayer", globalSize, localSize, args, -1, -1);
}

static void normalizePhotoDensity(
    const oclMat &chroma,
    const oclMat &colorDensity,
    const oclMat &multiplex,
    oclMat &ocl_luma,
    oclMat &demultiplex,
    const float pG
)
{
    int elements_per_row = static_cast<int>(ocl_luma.step / ocl_luma.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {ocl_luma.cols, ocl_luma.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &chroma.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &colorDensity.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &multiplex.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &ocl_luma.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &demultiplex.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &ocl_luma.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &ocl_luma.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &pG));
    openCLExecuteKernel(ctx, &retina_kernel, "normalizePhotoDensity", globalSize, localSize, args, -1, -1);
}

static void substractResidual(
    oclMat &colorDemultiplex,
    float pR,
    float pG,
    float pB
)
{
    int elements_per_row = static_cast<int>(colorDemultiplex.step / colorDemultiplex.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    int rows = colorDemultiplex.rows / 3, cols = colorDemultiplex.cols;
    size_t globalSize[] = {cols, rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &colorDemultiplex.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &pR));
    args.push_back(std::make_pair(sizeof(cl_float), &pG));
    args.push_back(std::make_pair(sizeof(cl_float), &pB));
    openCLExecuteKernel(ctx, &retina_kernel, "substractResidual", globalSize, localSize, args, -1, -1);
}

static void demultiplexAssign(const oclMat& input, const oclMat& output)
{
    // only supports bayer
    int elements_per_row = static_cast<int>(input.step / input.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    int rows = input.rows / 3, cols = input.cols;
    size_t globalSize[] = {cols, rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &input.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &output.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "demultiplexAssign", globalSize, localSize, args, -1, -1);
}

void RetinaColor::runColorDemultiplexing(
    const oclMat &ocl_multiplexed_input,
    const bool adaptiveFiltering,
    const float maxInputValue
)
{
    MAKE_OCLMAT_SLICES(_demultiplexedTempBuffer, 3);
    MAKE_OCLMAT_SLICES(_chrominance, 3);
    MAKE_OCLMAT_SLICES(_RGBmosaic, 3);
    MAKE_OCLMAT_SLICES(_demultiplexedColorFrame, 3);
    MAKE_OCLMAT_SLICES(_colorLocalDensity, 3);

    _demultiplexedTempBuffer.setTo(0);
    demultiplex(ocl_multiplexed_input, _demultiplexedTempBuffer);

    // interpolate the demultiplexed frame depending on the color sampling method
    if (!adaptiveFiltering)
    {
        CV_Assert(adaptiveFiltering == false);
    }

    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[0], _chrominance_slices[0]);
    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[1], _chrominance_slices[1]);
    _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[2], _chrominance_slices[2]);

    if (!adaptiveFiltering)// compute the gradient on the luminance
    {
        // TODO: implement me!
        CV_Assert(adaptiveFiltering == false);
    }
    else
    {
        normalizePhotoDensity(_chrominance, _colorLocalDensity, ocl_multiplexed_input, _luminance, _demultiplexedTempBuffer, _pG);
        // compute the gradient of the luminance
        _computeGradient(_luminance, _imageGradient);

        _adaptiveSpatialLPfilter(_RGBmosaic_slices[0], _imageGradient, _chrominance_slices[0]);
        _adaptiveSpatialLPfilter(_RGBmosaic_slices[1], _imageGradient, _chrominance_slices[1]);
        _adaptiveSpatialLPfilter(_RGBmosaic_slices[2], _imageGradient, _chrominance_slices[2]);

        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[0], _imageGradient, _demultiplexedColorFrame_slices[0]);
        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[1], _imageGradient, _demultiplexedColorFrame_slices[1]);
        _adaptiveSpatialLPfilter(_demultiplexedTempBuffer_slices[2], _imageGradient, _demultiplexedColorFrame_slices[2]);

        _demultiplexedColorFrame /= _chrominance; // per element division
        substractResidual(_demultiplexedColorFrame, _pR, _pG, _pB);
        runColorMultiplexing(_demultiplexedColorFrame, _tempMultiplexedFrame);

        _demultiplexedTempBuffer.setTo(0);
        _luminance = ocl_multiplexed_input - _tempMultiplexedFrame;
        demultiplexAssign(_demultiplexedColorFrame, _demultiplexedTempBuffer);

        for(int i = 0; i < 3; i ++)
        {
            _spatiotemporalLPfilter(_demultiplexedTempBuffer_slices[i], _demultiplexedTempBuffer_slices[i]);
            _demultiplexedColorFrame_slices[i] = _demultiplexedTempBuffer_slices[i] * _colorLocalDensity_slices[i] + _luminance;
        }
    }
    // eliminate saturated colors by simple clipping values to the input range
    clipRGBOutput_0_maxInputValue(_demultiplexedColorFrame, maxInputValue);

    if (_saturateColors)
    {
        ocl::normalizeGrayOutputCentredSigmoide(128, maxInputValue, _demultiplexedColorFrame, _demultiplexedColorFrame);
    }
}
void RetinaColor::runColorMultiplexing(const oclMat &demultiplexedInputFrame, oclMat &multiplexedFrame)
{
    int elements_per_row = static_cast<int>(multiplexedFrame.step / multiplexedFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {multiplexedFrame.cols, multiplexedFrame.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &demultiplexedInputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &multiplexedFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &multiplexedFrame.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &multiplexedFrame.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "runColorMultiplexingBayer", globalSize, localSize, args, -1, -1);
}

void RetinaColor::clipRGBOutput_0_maxInputValue(oclMat &inputOutputBuffer, const float maxInputValue)
{
    // the kernel is equivalent to:
    //ocl::threshold(inputOutputBuffer, inputOutputBuffer, maxInputValue, maxInputValue, THRESH_TRUNC);
    //ocl::threshold(inputOutputBuffer, inputOutputBuffer, 0, 0, THRESH_TOZERO);
    int elements_per_row = static_cast<int>(inputOutputBuffer.step / inputOutputBuffer.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, inputOutputBuffer.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &inputOutputBuffer.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputOutputBuffer.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &maxInputValue));
    openCLExecuteKernel(ctx, &retina_kernel, "clipRGBOutput_0_maxInputValue", globalSize, localSize, args, -1, -1);
}

void RetinaColor::_adaptiveSpatialLPfilter(const oclMat &inputFrame, const oclMat &gradient, oclMat &outputFrame)
{
    /**********/
    _gain = (1 - 0.57f) * (1 - 0.57f) * (1 - 0.06f) * (1 - 0.06f);

    // launch the serie of 1D directional filters in order to compute the 2D low pass filter
    // -> horizontal filters work with the first layer of imageGradient
    _adaptiveHorizontalCausalFilter_addInput(inputFrame, gradient, outputFrame);
    _horizontalAnticausalFilter_Irregular(outputFrame, gradient);
    // -> horizontal filters work with the second layer of imageGradient
    _verticalCausalFilter_Irregular(outputFrame, gradient(getROI(1)));
    _adaptiveVerticalAnticausalFilter_multGain(gradient, outputFrame);
}

void RetinaColor::_adaptiveHorizontalCausalFilter_addInput(const oclMat &inputFrame, const oclMat &gradient, oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(inputFrame.step / inputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBrows, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &inputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &gradient.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &inputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_int),   &gradient.offset));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    openCLExecuteKernel(ctx, &retina_kernel, "adaptiveHorizontalCausalFilter_addInput", globalSize, localSize, args, -1, -1);
}

void RetinaColor::_adaptiveVerticalAnticausalFilter_multGain(const oclMat &gradient, oclMat &outputFrame)
{
    int elements_per_row = static_cast<int>(outputFrame.step / outputFrame.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, 1, 1};
    size_t localSize[]  = {256, 1, 1};

    int gradOffset = gradient.offset + static_cast<int>(gradient.step * _NBrows);

    args.push_back(std::make_pair(sizeof(cl_mem),   &gradient.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &outputFrame.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_int),   &gradOffset));
    args.push_back(std::make_pair(sizeof(cl_int),   &outputFrame.offset));
    args.push_back(std::make_pair(sizeof(cl_float), &_gain));
    openCLExecuteKernel(ctx, &retina_kernel, "adaptiveVerticalAnticausalFilter_multGain", globalSize, localSize, args, -1, -1);
}
void RetinaColor::_computeGradient(const oclMat &luminance, oclMat &gradient)
{
    int elements_per_row = static_cast<int>(luminance.step / luminance.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {_NBcols, _NBrows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &luminance.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &gradient.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBcols));
    args.push_back(std::make_pair(sizeof(cl_int),   &_NBrows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    openCLExecuteKernel(ctx, &retina_kernel, "computeGradient", globalSize, localSize, args, -1, -1);
}

///////////////////////////////////////
//////////// RetinaFilter /////////////
///////////////////////////////////////
RetinaFilter::RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode, const int samplingMethod, const bool useRetinaLogSampling, const double, const double)
    :
    _photoreceptorsPrefilter(sizeRows, sizeColumns, 4),
    _ParvoRetinaFilter(sizeRows, sizeColumns),
    _MagnoRetinaFilter(sizeRows, sizeColumns),
    _colorEngine(sizeRows, sizeColumns, samplingMethod)
{
    CV_Assert(!useRetinaLogSampling);

    // set default processing activities
    _useParvoOutput = true;
    _useMagnoOutput = true;

    _useColorMode = colorMode;

    // set default parameters
    setGlobalParameters();

    // stability controls values init
    _setInitPeriodCount();
    _globalTemporalConstant = 25;

    // reset all buffers
    clearAllBuffers();
}

RetinaFilter::~RetinaFilter()
{
}

void RetinaFilter::clearAllBuffers()
{
    _photoreceptorsPrefilter.clearAllBuffers();
    _ParvoRetinaFilter.clearAllBuffers();
    _MagnoRetinaFilter.clearAllBuffers();
    _colorEngine.clearAllBuffers();
    // stability controls value init
    _setInitPeriodCount();
}

void RetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
    unsigned int rows = NBrows, cols = NBcolumns;

    // resize optionnal member and adjust other modules size if required
    _photoreceptorsPrefilter.resize(rows, cols);
    _ParvoRetinaFilter.resize(rows, cols);
    _MagnoRetinaFilter.resize(rows, cols);
    _colorEngine.resize(rows, cols);

    // clean buffers
    clearAllBuffers();

}

void RetinaFilter::_setInitPeriodCount()
{
    // find out the maximum temporal constant value and apply a security factor
    // false value (obviously too long) but appropriate for simple use
    _globalTemporalConstant = (unsigned int)(_ParvoRetinaFilter.getPhotoreceptorsTemporalConstant() + _ParvoRetinaFilter.getHcellsTemporalConstant() + _MagnoRetinaFilter.getTemporalConstant());
    // reset frame counter
    _ellapsedFramesSinceLastReset = 0;
}

void RetinaFilter::setGlobalParameters(const float OPLspatialResponse1, const float OPLtemporalresponse1, const float OPLassymetryGain, const float OPLspatialResponse2, const float OPLtemporalresponse2, const float LPfilterSpatialResponse, const float LPfilterGain, const float LPfilterTemporalresponse, const float MovingContoursExtractorCoefficient, const bool normalizeParvoOutput_0_maxOutputValue, const bool normalizeMagnoOutput_0_maxOutputValue, const float maxOutputValue, const float maxInputValue, const float meanValue)
{
    _normalizeParvoOutput_0_maxOutputValue = normalizeParvoOutput_0_maxOutputValue;
    _normalizeMagnoOutput_0_maxOutputValue = normalizeMagnoOutput_0_maxOutputValue;
    _maxOutputValue = maxOutputValue;
    _photoreceptorsPrefilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
    _photoreceptorsPrefilter.setLPfilterParameters(0, 0, 10, 3); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
    _ParvoRetinaFilter.setOPLandParvoFiltersParameters(0, OPLtemporalresponse1, OPLspatialResponse1, OPLassymetryGain, OPLtemporalresponse2, OPLspatialResponse2);
    _ParvoRetinaFilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
    _MagnoRetinaFilter.setCoefficientsTable(LPfilterGain, LPfilterTemporalresponse, LPfilterSpatialResponse, MovingContoursExtractorCoefficient, 0, 2.0f * LPfilterSpatialResponse);
    _MagnoRetinaFilter.setV0CompressionParameter(0.7f, maxInputValue, meanValue);

    // stability controls value init
    _setInitPeriodCount();
}

bool RetinaFilter::checkInput(const oclMat &input, const bool)
{
    BasicRetinaFilter *inputTarget = &_photoreceptorsPrefilter;

    bool test = (input.rows == static_cast<int>(inputTarget->getNBrows())
                 || input.rows == static_cast<int>(inputTarget->getNBrows()) * 3
                 || input.rows == static_cast<int>(inputTarget->getNBrows()) * 4)
                && input.cols == static_cast<int>(inputTarget->getNBcolumns());
    if (!test)
    {
        std::cerr << "RetinaFilter::checkInput: input buffer does not match retina buffer size, conversion aborted" << std::endl;
        return false;
    }

    return true;
}

// main function that runs the filter for a given input frame
bool RetinaFilter::runFilter(const oclMat &imageInput, const bool useAdaptiveFiltering, const bool processRetinaParvoMagnoMapping, const bool useColorMode, const bool inputIsColorMultiplexed)
{
    // preliminary check
    bool processSuccess = true;
    if (!checkInput(imageInput, useColorMode))
    {
        return false;
    }

    // run the color multiplexing if needed and compute each suub filter of the retina:
    // -> local adaptation
    // -> contours OPL extraction
    // -> moving contours extraction

    // stability controls value update
    ++_ellapsedFramesSinceLastReset;

    _useColorMode = useColorMode;

    oclMat selectedPhotoreceptorsLocalAdaptationInput = imageInput;
    oclMat selectedPhotoreceptorsColorInput = imageInput;

    //********** Following is input data specific photoreceptors processing
    if (useColorMode && (!inputIsColorMultiplexed)) // not multiplexed color input case
    {
        _colorEngine.runColorMultiplexing(selectedPhotoreceptorsColorInput);
        selectedPhotoreceptorsLocalAdaptationInput = _colorEngine.getMultiplexedFrame();
    }
    //********** Following is generic Retina processing

    // photoreceptors local adaptation
    _photoreceptorsPrefilter.runFilter_LocalAdapdation(selectedPhotoreceptorsLocalAdaptationInput, _ParvoRetinaFilter.getHorizontalCellsOutput());

    // run parvo filter
    _ParvoRetinaFilter.runFilter(_photoreceptorsPrefilter.getOutput(), _useParvoOutput);

    if (_useParvoOutput)
    {
        _ParvoRetinaFilter.normalizeGrayOutputCentredSigmoide(); // models the saturation of the cells, usefull for visualisation of the ON-OFF Parvo Output, Bipolar cells outputs do not change !!!
        _ParvoRetinaFilter.centerReductImageLuminance(); // best for further spectrum analysis

        if (_normalizeParvoOutput_0_maxOutputValue)
        {
            _ParvoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
        }
    }

    if (_useParvoOutput && _useMagnoOutput)
    {
        _MagnoRetinaFilter.runFilter(_ParvoRetinaFilter.getBipolarCellsON(), _ParvoRetinaFilter.getBipolarCellsOFF());
        if (_normalizeMagnoOutput_0_maxOutputValue)
        {
            _MagnoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
        }
        _MagnoRetinaFilter.normalizeGrayOutputNearZeroCentreredSigmoide();
    }

    if (_useParvoOutput && _useMagnoOutput && processRetinaParvoMagnoMapping)
    {
        _processRetinaParvoMagnoMapping();
        if (_useColorMode)
        {
            _colorEngine.runColorDemultiplexing(_retinaParvoMagnoMappedFrame, useAdaptiveFiltering, _maxOutputValue);
        }
        return processSuccess;
    }

    if (_useParvoOutput && _useColorMode)
    {
        _colorEngine.runColorDemultiplexing(_ParvoRetinaFilter.getOutput(), useAdaptiveFiltering, _maxOutputValue);
    }
    return processSuccess;
}

const oclMat &RetinaFilter::getContours()
{
    if (_useColorMode)
    {
        return _colorEngine.getLuminance();
    }
    else
    {
        return _ParvoRetinaFilter.getOutput();
    }
}
void RetinaFilter::_processRetinaParvoMagnoMapping()
{
    oclMat parvo = _ParvoRetinaFilter.getOutput();
    oclMat magno = _MagnoRetinaFilter.getOutput();

    int halfRows = parvo.rows / 2;
    int halfCols = parvo.cols / 2;
    float minDistance = MIN(halfRows, halfCols) * 0.7f;

    int elements_per_row = static_cast<int>(parvo.step / parvo.elemSize());

    Context * ctx = Context::getContext();
    std::vector<std::pair<size_t, const void *> > args;
    size_t globalSize[] = {parvo.cols, parvo.rows, 1};
    size_t localSize[]  = {16, 16, 1};

    args.push_back(std::make_pair(sizeof(cl_mem),   &parvo.data));
    args.push_back(std::make_pair(sizeof(cl_mem),   &magno.data));
    args.push_back(std::make_pair(sizeof(cl_int),   &parvo.cols));
    args.push_back(std::make_pair(sizeof(cl_int),   &parvo.rows));
    args.push_back(std::make_pair(sizeof(cl_int),   &halfCols));
    args.push_back(std::make_pair(sizeof(cl_int),   &halfRows));
    args.push_back(std::make_pair(sizeof(cl_int),   &elements_per_row));
    args.push_back(std::make_pair(sizeof(cl_float), &minDistance));
    openCLExecuteKernel(ctx, &retina_kernel, "processRetinaParvoMagnoMapping", globalSize, localSize, args, -1, -1);
}
}  /* namespace ocl */

Ptr<Retina> createRetina_OCL(Size getInputSize){ return makePtr<ocl::RetinaOCLImpl>(getInputSize); }
Ptr<Retina> createRetina_OCL(Size getInputSize, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    return makePtr<ocl::RetinaOCLImpl>(getInputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
}

}  /* namespace bioinspired */
}  /* namespace cv */

#endif /* #ifdef HAVE_OPENCV_OCL */
