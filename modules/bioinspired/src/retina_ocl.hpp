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

#ifndef __OCL_RETINA_HPP__
#define __OCL_RETINA_HPP__

#include "precomp.hpp"

#ifdef HAVE_OPENCV_OCL

// please refer to c++ headers for API comments
namespace cv
{
namespace bioinspired
{
namespace ocl
{
void normalizeGrayOutputCentredSigmoide(const float meanValue, const float sensitivity, cv::ocl::oclMat &in, cv::ocl::oclMat &out, const float maxValue = 255.f);
void normalizeGrayOutput_0_maxOutputValue(cv::ocl::oclMat &inputOutputBuffer, const float maxOutputValue = 255.0);
void normalizeGrayOutputNearZeroCentreredSigmoide(cv::ocl::oclMat &inputPicture, cv::ocl::oclMat &outputBuffer, const float sensitivity = 40, const float maxOutputValue = 255.0f);
void centerReductImageLuminance(cv::ocl::oclMat &inputOutputBuffer);

class BasicRetinaFilter
{
public:
    BasicRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns, const unsigned int parametersListSize = 1, const bool useProgressiveFilter = false);
    ~BasicRetinaFilter();
    inline void clearOutputBuffer()
    {
        _filterOutput = 0;
    };
    inline void clearSecondaryBuffer()
    {
        _localBuffer = 0;
    };
    inline void clearAllBuffers()
    {
        clearOutputBuffer();
        clearSecondaryBuffer();
    };
    void  resize(const unsigned int NBrows, const unsigned int NBcolumns);
    const cv::ocl::oclMat &runFilter_LPfilter(const cv::ocl::oclMat &inputFrame, const unsigned int filterIndex = 0);
    void  runFilter_LPfilter(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame, const unsigned int filterIndex = 0);
    void  runFilter_LPfilter_Autonomous(cv::ocl::oclMat &inputOutputFrame, const unsigned int filterIndex = 0);
    const cv::ocl::oclMat &runFilter_LocalAdapdation(const cv::ocl::oclMat &inputOutputFrame, const cv::ocl::oclMat &localLuminance);
    void  runFilter_LocalAdapdation(const cv::ocl::oclMat &inputFrame, const cv::ocl::oclMat &localLuminance, cv::ocl::oclMat &outputFrame);
    const cv::ocl::oclMat &runFilter_LocalAdapdation_autonomous(const cv::ocl::oclMat &inputFrame);
    void  runFilter_LocalAdapdation_autonomous(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame);
    void  setLPfilterParameters(const float beta, const float tau, const float k, const unsigned int filterIndex = 0);
    inline void setV0CompressionParameter(const float v0, const float maxInputValue, const float)
    {
        _v0 = v0 * maxInputValue;
        _localLuminanceFactor = v0;
        _localLuminanceAddon = maxInputValue * (1.0f - v0);
        _maxInputValue = maxInputValue;
    };
    inline void setV0CompressionParameter(const float v0, const float meanLuminance)
    {
        this->setV0CompressionParameter(v0, _maxInputValue, meanLuminance);
    };
    inline void setV0CompressionParameter(const float v0)
    {
        _v0 = v0 * _maxInputValue;
        _localLuminanceFactor = v0;
        _localLuminanceAddon = _maxInputValue * (1.0f - v0);
    };
    inline void setV0CompressionParameterToneMapping(const float v0, const float maxInputValue, const float meanLuminance = 128.0f)
    {
        _v0 = v0 * maxInputValue;
        _localLuminanceFactor = 1.0f;
        _localLuminanceAddon = meanLuminance * _v0;
        _maxInputValue = maxInputValue;
    };
    inline void updateCompressionParameter(const float meanLuminance)
    {
        _localLuminanceFactor = 1;
        _localLuminanceAddon = meanLuminance * _v0;
    };
    inline float getV0CompressionParameter()
    {
        return _v0 / _maxInputValue;
    };
    inline const cv::ocl::oclMat &getOutput() const
    {
        return _filterOutput;
    };
    inline unsigned int getNBrows()
    {
        return _filterOutput.rows;
    };
    inline unsigned int getNBcolumns()
    {
        return _filterOutput.cols;
    };
    inline unsigned int getNBpixels()
    {
        return _filterOutput.size().area();
    };
    inline void normalizeGrayOutput_0_maxOutputValue(const float maxValue)
    {
        ocl::normalizeGrayOutput_0_maxOutputValue(_filterOutput, maxValue);
    };
    inline void normalizeGrayOutputCentredSigmoide()
    {
        ocl::normalizeGrayOutputCentredSigmoide(0.0, 2.0, _filterOutput, _filterOutput);
    };
    inline void centerReductImageLuminance()
    {
        ocl::centerReductImageLuminance(_filterOutput);
    };
    inline float getMaxInputValue()
    {
        return this->_maxInputValue;
    };
    inline void setMaxInputValue(const float newMaxInputValue)
    {
        this->_maxInputValue = newMaxInputValue;
    };

protected:

    int _NBrows;
    int _NBcols;
    unsigned int _halfNBrows;
    unsigned int _halfNBcolumns;

    cv::ocl::oclMat _filterOutput;
    cv::ocl::oclMat _localBuffer;

    std::valarray <float>_filteringCoeficientsTable;
    float _v0;
    float _maxInputValue;
    float _meanInputValue;
    float _localLuminanceFactor;
    float _localLuminanceAddon;

    float _a;
    float _tau;
    float _gain;

    void _spatiotemporalLPfilter(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &LPfilterOutput, const unsigned int coefTableOffset = 0);
    float _squaringSpatiotemporalLPfilter(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame, const unsigned int filterIndex = 0);
    void _spatiotemporalLPfilter_Irregular(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame, const unsigned int filterIndex = 0);
    void _localSquaringSpatioTemporalLPfilter(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &LPfilterOutput, const unsigned int *integrationAreas, const unsigned int filterIndex = 0);
    void _localLuminanceAdaptation(const cv::ocl::oclMat &inputFrame, const cv::ocl::oclMat &localLuminance, cv::ocl::oclMat &outputFrame, const bool updateLuminanceMean = true);
    void _localLuminanceAdaptation(cv::ocl::oclMat &inputOutputFrame, const cv::ocl::oclMat &localLuminance);
    void _localLuminanceAdaptationPosNegValues(const cv::ocl::oclMat &inputFrame, const cv::ocl::oclMat &localLuminance, float *outputFrame);
    void _horizontalCausalFilter_addInput(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame);
    void _horizontalAnticausalFilter(cv::ocl::oclMat &outputFrame);
    void _verticalCausalFilter(cv::ocl::oclMat &outputFrame);
    void _horizontalAnticausalFilter_Irregular(cv::ocl::oclMat &outputFrame, const cv::ocl::oclMat &spatialConstantBuffer);
    void _verticalCausalFilter_Irregular(cv::ocl::oclMat &outputFrame, const cv::ocl::oclMat &spatialConstantBuffer);
    void _verticalAnticausalFilter_multGain(cv::ocl::oclMat &outputFrame);
};

class MagnoRetinaFilter: public BasicRetinaFilter
{
public:
    MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns);
    virtual ~MagnoRetinaFilter();
    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    void setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau, const float localAdaptIntegration_k);

    const cv::ocl::oclMat &runFilter(const cv::ocl::oclMat &OPL_ON, const cv::ocl::oclMat &OPL_OFF);

    inline const cv::ocl::oclMat &getMagnoON() const
    {
        return _magnoXOutputON;
    };
    inline const cv::ocl::oclMat &getMagnoOFF() const
    {
        return _magnoXOutputOFF;
    };
    inline const cv::ocl::oclMat &getMagnoYsaturated() const
    {
        return _magnoYsaturated;
    };
    inline void normalizeGrayOutputNearZeroCentreredSigmoide()
    {
        ocl::normalizeGrayOutputNearZeroCentreredSigmoide(_magnoYOutput, _magnoYsaturated);
    };
    inline float getTemporalConstant()
    {
        return this->_filteringCoeficientsTable[2];
    };
private:
    cv::ocl::oclMat _previousInput_ON;
    cv::ocl::oclMat _previousInput_OFF;
    cv::ocl::oclMat _amacrinCellsTempOutput_ON;
    cv::ocl::oclMat _amacrinCellsTempOutput_OFF;
    cv::ocl::oclMat _magnoXOutputON;
    cv::ocl::oclMat _magnoXOutputOFF;
    cv::ocl::oclMat _localProcessBufferON;
    cv::ocl::oclMat _localProcessBufferOFF;
    cv::ocl::oclMat _magnoYOutput;
    cv::ocl::oclMat _magnoYsaturated;

    float _temporalCoefficient;
    void _amacrineCellsComputing(const cv::ocl::oclMat &OPL_ON,  const cv::ocl::oclMat &OPL_OFF);
};

class ParvoRetinaFilter: public BasicRetinaFilter
{
public:
    ParvoRetinaFilter(const unsigned int NBrows = 480, const unsigned int NBcolumns = 640);
    virtual ~ParvoRetinaFilter();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    void clearAllBuffers();
    void setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2);

    inline void setGanglionCellsLocalAdaptationLPfilterParameters(const float tau, const float k)
    {
        BasicRetinaFilter::setLPfilterParameters(0, tau, k, 2);
    };
    const cv::ocl::oclMat &runFilter(const cv::ocl::oclMat &inputFrame, const bool useParvoOutput = true);

    inline const cv::ocl::oclMat &getPhotoreceptorsLPfilteringOutput() const
    {
        return _photoreceptorsOutput;
    };

    inline const cv::ocl::oclMat &getHorizontalCellsOutput() const
    {
        return _horizontalCellsOutput;
    };

    inline const cv::ocl::oclMat &getParvoON() const
    {
        return _parvocellularOutputON;
    };

    inline const cv::ocl::oclMat &getParvoOFF() const
    {
        return _parvocellularOutputOFF;
    };

    inline const cv::ocl::oclMat &getBipolarCellsON() const
    {
        return _bipolarCellsOutputON;
    };

    inline const cv::ocl::oclMat &getBipolarCellsOFF() const
    {
        return _bipolarCellsOutputOFF;
    };

    inline float getPhotoreceptorsTemporalConstant()
    {
        return this->_filteringCoeficientsTable[2];
    };

    inline float getHcellsTemporalConstant()
    {
        return this->_filteringCoeficientsTable[5];
    };
private:
    cv::ocl::oclMat _photoreceptorsOutput;
    cv::ocl::oclMat _horizontalCellsOutput;
    cv::ocl::oclMat _parvocellularOutputON;
    cv::ocl::oclMat _parvocellularOutputOFF;
    cv::ocl::oclMat _bipolarCellsOutputON;
    cv::ocl::oclMat _bipolarCellsOutputOFF;
    cv::ocl::oclMat _localAdaptationOFF;
    cv::ocl::oclMat _localAdaptationON;
    cv::ocl::oclMat _parvocellularOutputONminusOFF;
    void _OPL_OnOffWaysComputing();
};
class RetinaColor: public BasicRetinaFilter
{
public:
    RetinaColor(const unsigned int NBrows, const unsigned int NBcolumns, const int samplingMethod = RETINA_COLOR_DIAGONAL);
    virtual ~RetinaColor();

    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    inline void runColorMultiplexing(const cv::ocl::oclMat &inputRGBFrame)
    {
        runColorMultiplexing(inputRGBFrame, _multiplexedFrame);
    };
    void runColorMultiplexing(const cv::ocl::oclMat &demultiplexedInputFrame, cv::ocl::oclMat &multiplexedFrame);
    void runColorDemultiplexing(const cv::ocl::oclMat &multiplexedColorFrame, const bool adaptiveFiltering = false, const float maxInputValue = 255.0);

    void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0)
    {
        _saturateColors = saturateColors;
        _colorSaturationValue = colorSaturationValue;
    };

    void setChrominanceLPfilterParameters(const float beta, const float tau, const float k)
    {
        setLPfilterParameters(beta, tau, k);
    };

    bool applyKrauskopfLMS2Acr1cr2Transform(cv::ocl::oclMat &result);
    bool applyLMS2LabTransform(cv::ocl::oclMat &result);
    inline const cv::ocl::oclMat &getMultiplexedFrame() const
    {
        return _multiplexedFrame;
    };

    inline const cv::ocl::oclMat &getDemultiplexedColorFrame() const
    {
        return _demultiplexedColorFrame;
    };

    inline const cv::ocl::oclMat &getLuminance() const
    {
        return _luminance;
    };
    inline const cv::ocl::oclMat &getChrominance() const
    {
        return _chrominance;
    };
    void clipRGBOutput_0_maxInputValue(cv::ocl::oclMat &inputOutputBuffer, const float maxOutputValue = 255.0);
    void normalizeRGBOutput_0_maxOutputValue(const float maxOutputValue = 255.0);
    inline void setDemultiplexedColorFrame(const cv::ocl::oclMat &demultiplexedImage)
    {
        _demultiplexedColorFrame = demultiplexedImage;
    };
protected:
    inline unsigned int bayerSampleOffset(unsigned int index)
    {
        return index + ((index / getNBcolumns()) % 2) * getNBpixels() + ((index % getNBcolumns()) % 2) * getNBpixels();
    }
    inline Rect getROI(int idx)
    {
        return Rect(0, idx * _NBrows, _NBcols, _NBrows);
    }
    int _samplingMethod;
    bool _saturateColors;
    float _colorSaturationValue;
    cv::ocl::oclMat _luminance;
    cv::ocl::oclMat _multiplexedFrame;
    cv::ocl::oclMat _RGBmosaic;
    cv::ocl::oclMat _tempMultiplexedFrame;
    cv::ocl::oclMat _demultiplexedTempBuffer;
    cv::ocl::oclMat _demultiplexedColorFrame;
    cv::ocl::oclMat _chrominance;
    cv::ocl::oclMat _colorLocalDensity;
    cv::ocl::oclMat _imageGradient;

    float _pR, _pG, _pB;
    bool _objectInit;

    void _initColorSampling();
    void _adaptiveSpatialLPfilter(const cv::ocl::oclMat &inputFrame, const cv::ocl::oclMat &gradient, cv::ocl::oclMat &outputFrame);
    void _adaptiveHorizontalCausalFilter_addInput(const cv::ocl::oclMat &inputFrame, const cv::ocl::oclMat &gradient, cv::ocl::oclMat &outputFrame);
    void _adaptiveVerticalAnticausalFilter_multGain(const cv::ocl::oclMat &gradient, cv::ocl::oclMat &outputFrame);
    void _computeGradient(const cv::ocl::oclMat &luminance, cv::ocl::oclMat &gradient);
    void _normalizeOutputs_0_maxOutputValue(void);
    void _applyImageColorSpaceConversion(const cv::ocl::oclMat &inputFrame, cv::ocl::oclMat &outputFrame, const float *transformTable);
};
class RetinaFilter
{
public:
    RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode = false, const int samplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrenght = 10.0);
    ~RetinaFilter();

    void clearAllBuffers();
    void resize(const unsigned int NBrows, const unsigned int NBcolumns);
    bool checkInput(const cv::ocl::oclMat &input, const bool colorMode);
    bool runFilter(const cv::ocl::oclMat &imageInput, const bool useAdaptiveFiltering = true, const bool processRetinaParvoMagnoMapping = false, const bool useColorMode = false, const bool inputIsColorMultiplexed = false);

    void setGlobalParameters(const float OPLspatialResponse1 = 0.7, const float OPLtemporalresponse1 = 1, const float OPLassymetryGain = 0, const float OPLspatialResponse2 = 5, const float OPLtemporalresponse2 = 1, const float LPfilterSpatialResponse = 5, const float LPfilterGain = 0, const float LPfilterTemporalresponse = 0, const float MovingContoursExtractorCoefficient = 5, const bool normalizeParvoOutput_0_maxOutputValue = false, const bool normalizeMagnoOutput_0_maxOutputValue = false, const float maxOutputValue = 255.0, const float maxInputValue = 255.0, const float meanValue = 128.0);

    inline void setPhotoreceptorsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _photoreceptorsPrefilter.setV0CompressionParameter(1 - V0CompressionParameter);
        _setInitPeriodCount();
    };

    inline void setParvoGanglionCellsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    };

    inline void setGanglionCellsLocalAdaptationLPfilterParameters(const float spatialResponse, const float temporalResponse)
    {
        _ParvoRetinaFilter.setGanglionCellsLocalAdaptationLPfilterParameters(temporalResponse, spatialResponse);
        _setInitPeriodCount();
    };

    inline void setMagnoGanglionCellsLocalAdaptationSensitivity(const float V0CompressionParameter)
    {
        _MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    };

    void setOPLandParvoParameters(const float beta1, const float tau1, const float k1, const float beta2, const float tau2, const float k2, const float V0CompressionParameter)
    {
        _ParvoRetinaFilter.setOPLandParvoFiltersParameters(beta1, tau1, k1, beta2, tau2, k2);
        _ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    };

    void setMagnoCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
    {
        _MagnoRetinaFilter.setCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, localAdaptintegration_tau, localAdaptintegration_k);
        _MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);
        _setInitPeriodCount();
    };

    inline void activateNormalizeParvoOutput_0_maxOutputValue(const bool normalizeParvoOutput_0_maxOutputValue)
    {
        _normalizeParvoOutput_0_maxOutputValue = normalizeParvoOutput_0_maxOutputValue;
    };

    inline void activateNormalizeMagnoOutput_0_maxOutputValue(const bool normalizeMagnoOutput_0_maxOutputValue)
    {
        _normalizeMagnoOutput_0_maxOutputValue = normalizeMagnoOutput_0_maxOutputValue;
    };

    inline void setMaxOutputValue(const float maxOutputValue)
    {
        _maxOutputValue = maxOutputValue;
    };

    void setColorMode(const bool desiredColorMode)
    {
        _useColorMode = desiredColorMode;
    };
    inline void setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0)
    {
        _colorEngine.setColorSaturation(saturateColors, colorSaturationValue);
    };
    inline const cv::ocl::oclMat &getLocalAdaptation() const
    {
        return _photoreceptorsPrefilter.getOutput();
    };
    inline const cv::ocl::oclMat &getPhotoreceptors() const
    {
        return _ParvoRetinaFilter.getPhotoreceptorsLPfilteringOutput();
    };

    inline const cv::ocl::oclMat &getHorizontalCells() const
    {
        return _ParvoRetinaFilter.getHorizontalCellsOutput();
    };
    inline bool areContoursProcessed()
    {
        return _useParvoOutput;
    };
    bool getParvoFoveaResponse(cv::ocl::oclMat &parvoFovealResponse);
    inline void activateContoursProcessing(const bool useParvoOutput)
    {
        _useParvoOutput = useParvoOutput;
    };

    const cv::ocl::oclMat &getContours();

    inline const cv::ocl::oclMat &getContoursON() const
    {
        return _ParvoRetinaFilter.getParvoON();
    };

    inline const cv::ocl::oclMat &getContoursOFF() const
    {
        return _ParvoRetinaFilter.getParvoOFF();
    };

    inline bool areMovingContoursProcessed()
    {
        return _useMagnoOutput;
    };

    inline void activateMovingContoursProcessing(const bool useMagnoOutput)
    {
        _useMagnoOutput = useMagnoOutput;
    };

    inline const cv::ocl::oclMat &getMovingContours() const
    {
        return _MagnoRetinaFilter.getOutput();
    };

    inline const cv::ocl::oclMat &getMovingContoursSaturated() const
    {
        return _MagnoRetinaFilter.getMagnoYsaturated();
    };

    inline const cv::ocl::oclMat &getMovingContoursON() const
    {
        return _MagnoRetinaFilter.getMagnoON();
    };

    inline const cv::ocl::oclMat &getMovingContoursOFF() const
    {
        return _MagnoRetinaFilter.getMagnoOFF();
    };

    inline const cv::ocl::oclMat &getRetinaParvoMagnoMappedOutput() const
    {
        return _retinaParvoMagnoMappedFrame;
    };

    inline const cv::ocl::oclMat &getParvoContoursChannel() const
    {
        return _colorEngine.getLuminance();
    };

    inline const cv::ocl::oclMat &getParvoChrominance() const
    {
        return _colorEngine.getChrominance();
    };
    inline const cv::ocl::oclMat &getColorOutput() const
    {
        return _colorEngine.getDemultiplexedColorFrame();
    };

    inline bool isColorMode()
    {
        return _useColorMode;
    };
    bool getColorMode()
    {
        return _useColorMode;
    };

    inline bool isInitTransitionDone()
    {
        if (_ellapsedFramesSinceLastReset < _globalTemporalConstant)
        {
            return false;
        }
        return true;
    };
    inline float getRetinaSamplingBackProjection(const float projectedRadiusLength)
    {
        return projectedRadiusLength;
    };

    inline unsigned int getInputNBrows()
    {
        return _photoreceptorsPrefilter.getNBrows();
    };

    inline unsigned int getInputNBcolumns()
    {
        return _photoreceptorsPrefilter.getNBcolumns();
    };

    inline unsigned int getInputNBpixels()
    {
        return _photoreceptorsPrefilter.getNBpixels();
    };

    inline unsigned int getOutputNBrows()
    {
        return _photoreceptorsPrefilter.getNBrows();
    };

    inline unsigned int getOutputNBcolumns()
    {
        return _photoreceptorsPrefilter.getNBcolumns();
    };

    inline unsigned int getOutputNBpixels()
    {
        return _photoreceptorsPrefilter.getNBpixels();
    };
private:
    bool _useParvoOutput;
    bool _useMagnoOutput;

    unsigned int _ellapsedFramesSinceLastReset;
    unsigned int _globalTemporalConstant;

    cv::ocl::oclMat _retinaParvoMagnoMappedFrame;
    BasicRetinaFilter _photoreceptorsPrefilter;
    ParvoRetinaFilter _ParvoRetinaFilter;
    MagnoRetinaFilter _MagnoRetinaFilter;
    RetinaColor       _colorEngine;

    bool _useMinimalMemoryForToneMappingONLY;
    bool _normalizeParvoOutput_0_maxOutputValue;
    bool _normalizeMagnoOutput_0_maxOutputValue;
    float _maxOutputValue;
    bool _useColorMode;

    void _setInitPeriodCount();
    void _processRetinaParvoMagnoMapping();
    void _runGrayToneMapping(const cv::ocl::oclMat &grayImageInput, cv::ocl::oclMat &grayImageOutput , const float PhotoreceptorsCompression = 0.6, const float ganglionCellsCompression = 0.6);
};

}  /* namespace ocl */
}  /* namespace bioinspired */
}  /* namespace cv */

#endif  /* HAVE_OPENCV_OCL */
#endif  /* __OCL_RETINA_HPP__ */
