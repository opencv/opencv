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

#include "retinafilter.hpp"

// @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/

#include <iostream>
#include <cmath>

namespace cv
{
    // standard constructor without any log sampling of the input frame
    RetinaFilter::RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode, const RETINA_COLORSAMPLINGMETHOD samplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
        :
    _retinaParvoMagnoMappedFrame(0),
        _retinaParvoMagnoMapCoefTable(0),
        _photoreceptorsPrefilter((1-(int)useRetinaLogSampling)*sizeRows+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor), (1-(int)useRetinaLogSampling)*sizeColumns+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor), 4),
        _ParvoRetinaFilter((1-(int)useRetinaLogSampling)*sizeRows+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor), (1-(int)useRetinaLogSampling)*sizeColumns+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor)),
        _MagnoRetinaFilter((1-(int)useRetinaLogSampling)*sizeRows+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor), (1-(int)useRetinaLogSampling)*sizeColumns+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor)),
        _colorEngine((1-(int)useRetinaLogSampling)*sizeRows+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor), (1-(int)useRetinaLogSampling)*sizeColumns+useRetinaLogSampling*ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor), samplingMethod),
        // configure retina photoreceptors log sampling... if necessary
        _photoreceptorsLogSampling(NULL)
    {

#ifdef RETINADEBUG
        std::cout<<"RetinaFilter::size( "<<_photoreceptorsPrefilter.getNBrows()<<", "<<_photoreceptorsPrefilter.getNBcolumns()<<")"<<" =? "<<_photoreceptorsPrefilter.getNBpixels()<<std::endl;
#endif
        if (useRetinaLogSampling)
        {
            _photoreceptorsLogSampling = new ImageLogPolProjection(sizeRows, sizeColumns, ImageLogPolProjection::RETINALOGPROJECTION, true);
            if (!_photoreceptorsLogSampling->initProjection(reductionFactor, samplingStrenght))
            {
                std::cerr<<"RetinaFilter::Problem initializing photoreceptors log sampling, could not setup retina filter"<<std::endl;
                delete _photoreceptorsLogSampling;
                _photoreceptorsLogSampling=NULL;
            }
            else
            {
#ifdef RETINADEBUG
                std::cout<<"_photoreceptorsLogSampling::size( "<<_photoreceptorsLogSampling->getNBrows()<<", "<<_photoreceptorsLogSampling->getNBcolumns()<<")"<<" =? "<<_photoreceptorsLogSampling->getNBpixels()<<std::endl;
#endif
            }
        }

        // set default processing activities
        _useParvoOutput=true;
        _useMagnoOutput=true;

        _useColorMode=colorMode;

        // create hybrid output and related coefficient table
        _createHybridTable();

        // set default parameters
        setGlobalParameters();

        // stability controls values init
        _setInitPeriodCount();
        _globalTemporalConstant=25;

        // reset all buffers
        clearAllBuffers();


        //  std::cout<<"RetinaFilter::size( "<<this->getNBrows()<<", "<<this->getNBcolumns()<<")"<<_filterOutput.size()<<" =? "<<_filterOutput.getNBpixels()<<std::endl;

    }

    // destructor
    RetinaFilter::~RetinaFilter()
    {
        if (_photoreceptorsLogSampling!=NULL)
            delete _photoreceptorsLogSampling;
    }

    // function that clears all buffers of the object
    void RetinaFilter::clearAllBuffers()
    {
        _photoreceptorsPrefilter.clearAllBuffers();
        _ParvoRetinaFilter.clearAllBuffers();
        _MagnoRetinaFilter.clearAllBuffers();
        _colorEngine.clearAllBuffers();
        if (_photoreceptorsLogSampling!=NULL)
            _photoreceptorsLogSampling->clearAllBuffers();
        // stability controls value init
        _setInitPeriodCount();
    }

    /**
    * resize retina filter object (resize all allocated buffers
    * @param NBrows: the new height size
    * @param NBcolumns: the new width size
    */
    void RetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
    {
        unsigned int rows=NBrows, cols=NBcolumns;

        // resize optionnal member and adjust other modules size if required
        if (_photoreceptorsLogSampling)
        {
            _photoreceptorsLogSampling->resize(NBrows, NBcolumns);
            rows=_photoreceptorsLogSampling->getOutputNBrows();
            cols=_photoreceptorsLogSampling->getOutputNBcolumns();
        }

        _photoreceptorsPrefilter.resize(rows, cols);
        _ParvoRetinaFilter.resize(rows, cols);
        _MagnoRetinaFilter.resize(rows, cols);
        _colorEngine.resize(rows, cols);

        // reset parvo magno mapping
        _createHybridTable();

        // clean buffers
        clearAllBuffers();

    }

    // stability controls value init
    void RetinaFilter::_setInitPeriodCount()
    {

        // find out the maximum temporal constant value and apply a security factor
        // false value (obviously too long) but appropriate for simple use
        _globalTemporalConstant=(unsigned int)(_ParvoRetinaFilter.getPhotoreceptorsTemporalConstant()+_ParvoRetinaFilter.getHcellsTemporalConstant()+_MagnoRetinaFilter.getTemporalConstant());
        // reset frame counter
        _ellapsedFramesSinceLastReset=0;
    }

    void RetinaFilter::_createHybridTable()
    {
        // create hybrid output and related coefficient table
        _retinaParvoMagnoMappedFrame.resize(_photoreceptorsPrefilter.getNBpixels());

        _retinaParvoMagnoMapCoefTable.resize(_photoreceptorsPrefilter.getNBpixels()*2);

        // fill _hybridParvoMagnoCoefTable
        int i, j, halfRows=_photoreceptorsPrefilter.getNBrows()/2, halfColumns=_photoreceptorsPrefilter.getNBcolumns()/2;
        float *hybridParvoMagnoCoefTablePTR= &_retinaParvoMagnoMapCoefTable[0];
        float minDistance=MIN(halfRows, halfColumns)*0.7f;
        for (i=0;i<(int)_photoreceptorsPrefilter.getNBrows();++i)
        {
            for (j=0;j<(int)_photoreceptorsPrefilter.getNBcolumns();++j)
            {
                float distanceToCenter=sqrt(((float)(i-halfRows)*(i-halfRows)+(j-halfColumns)*(j-halfColumns)));
                if (distanceToCenter<minDistance)
                {
                    float a=*(hybridParvoMagnoCoefTablePTR++)=0.5f+0.5f*(float)cos(CV_PI*distanceToCenter/minDistance);
                    *(hybridParvoMagnoCoefTablePTR++)=1.f-a;
                }else
                {
                    *(hybridParvoMagnoCoefTablePTR++)=0.f;
                    *(hybridParvoMagnoCoefTablePTR++)=1.f;
                }
            }
        }
    }

    // setup parameters function and global data filling
    void RetinaFilter::setGlobalParameters(const float OPLspatialResponse1, const float OPLtemporalresponse1, const float OPLassymetryGain, const float OPLspatialResponse2, const float OPLtemporalresponse2, const float LPfilterSpatialResponse, const float LPfilterGain, const float LPfilterTemporalresponse, const float MovingContoursExtractorCoefficient, const bool normalizeParvoOutput_0_maxOutputValue, const bool normalizeMagnoOutput_0_maxOutputValue, const float maxOutputValue, const float maxInputValue, const float meanValue)
    {
        _normalizeParvoOutput_0_maxOutputValue=normalizeParvoOutput_0_maxOutputValue;
        _normalizeMagnoOutput_0_maxOutputValue=normalizeMagnoOutput_0_maxOutputValue;
        _maxOutputValue=maxOutputValue;
        _photoreceptorsPrefilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
        _photoreceptorsPrefilter.setLPfilterParameters(10, 0, 1.5, 1); // keeps low pass filter with high cut frequency in memory (usefull for the tone mapping function)
        _photoreceptorsPrefilter.setLPfilterParameters(10, 0, 3.0, 2); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
        _photoreceptorsPrefilter.setLPfilterParameters(0, 0, 10, 3); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
        //this->setV0CompressionParameter(0.6, maxInputValue, meanValue); // keeps log compression sensitivity parameter (usefull for the tone mapping function)
        _ParvoRetinaFilter.setOPLandParvoFiltersParameters(0,OPLtemporalresponse1, OPLspatialResponse1, OPLassymetryGain, OPLtemporalresponse2, OPLspatialResponse2);
        _ParvoRetinaFilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
        _MagnoRetinaFilter.setCoefficientsTable(LPfilterGain, LPfilterTemporalresponse, LPfilterSpatialResponse, MovingContoursExtractorCoefficient, 0, 2.0f*LPfilterSpatialResponse);
        _MagnoRetinaFilter.setV0CompressionParameter(0.7f, maxInputValue, meanValue);

        // stability controls value init
        _setInitPeriodCount();
    }

    bool RetinaFilter::checkInput(const std::valarray<float> &input, const bool)
    {

        BasicRetinaFilter *inputTarget=&_photoreceptorsPrefilter;
        if (_photoreceptorsLogSampling)
            inputTarget=_photoreceptorsLogSampling;

        bool test=input.size()==inputTarget->getNBpixels() || input.size()==(inputTarget->getNBpixels()*3) ;
        if (!test)
        {
            std::cerr<<"RetinaFilter::checkInput: input buffer does not match retina buffer size, conversion aborted"<<std::endl;
            std::cout<<"RetinaFilter::checkInput: input size="<<input.size()<<" / "<<"retina size="<<inputTarget->getNBpixels()<<std::endl;
            return false;
        }

        return true;
    }

    // main function that runs the filter for a given input frame
    bool RetinaFilter::runFilter(const std::valarray<float> &imageInput, const bool useAdaptiveFiltering, const bool processRetinaParvoMagnoMapping, const bool useColorMode, const bool inputIsColorMultiplexed)
    {
        // preliminary check
        bool processSuccess=true;
        if (!checkInput(imageInput, useColorMode))
            return false;

        // run the color multiplexing if needed and compute each suub filter of the retina:
        // -> local adaptation
        // -> contours OPL extraction
        // -> moving contours extraction

        // stability controls value update
        ++_ellapsedFramesSinceLastReset;

        _useColorMode=useColorMode;

        /* pointer to the appropriate input data after,
        * by default, if graylevel mode, the input is processed,
        * if color or something else must be considered, specific preprocessing are applied
        */

        const std::valarray<float> *selectedPhotoreceptorsLocalAdaptationInput= &imageInput;
        const std::valarray<float> *selectedPhotoreceptorsColorInput=&imageInput;

        //********** Following is input data specific photoreceptors processing
        if (_photoreceptorsLogSampling)
        {
            _photoreceptorsLogSampling->runProjection(imageInput, useColorMode);
            selectedPhotoreceptorsColorInput=selectedPhotoreceptorsLocalAdaptationInput=&(_photoreceptorsLogSampling->getSampledFrame());
        }

        if (useColorMode&& (!inputIsColorMultiplexed)) // not multiplexed color input case
        {
            _colorEngine.runColorMultiplexing(*selectedPhotoreceptorsColorInput);
            selectedPhotoreceptorsLocalAdaptationInput=&(_colorEngine.getMultiplexedFrame());
        }

        //********** Following is generic Retina processing

        // photoreceptors local adaptation
        _photoreceptorsPrefilter.runFilter_LocalAdapdation(*selectedPhotoreceptorsLocalAdaptationInput, _ParvoRetinaFilter.getHorizontalCellsOutput());
        // safety pixel values checks
        //_photoreceptorsPrefilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);

        // run parvo filter
        _ParvoRetinaFilter.runFilter(_photoreceptorsPrefilter.getOutput(), _useParvoOutput);

        if (_useParvoOutput)
        {
            _ParvoRetinaFilter.normalizeGrayOutputCentredSigmoide(); // models the saturation of the cells, usefull for visualisation of the ON-OFF Parvo Output, Bipolar cells outputs do not change !!!
            _ParvoRetinaFilter.centerReductImageLuminance(); // best for further spectrum analysis

            if (_normalizeParvoOutput_0_maxOutputValue)
                _ParvoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
        }

        if (_useParvoOutput&&_useMagnoOutput)
        {
            _MagnoRetinaFilter.runFilter(_ParvoRetinaFilter.getBipolarCellsON(), _ParvoRetinaFilter.getBipolarCellsOFF());
            if (_normalizeMagnoOutput_0_maxOutputValue)
            {
                _MagnoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
            }
            _MagnoRetinaFilter.normalizeGrayOutputNearZeroCentreredSigmoide();
        }

        if (_useParvoOutput&&_useMagnoOutput&&processRetinaParvoMagnoMapping)
        {
            _processRetinaParvoMagnoMapping();
            if (_useColorMode)
                _colorEngine.runColorDemultiplexing(_retinaParvoMagnoMappedFrame, useAdaptiveFiltering, _maxOutputValue);//_ColorEngine->getMultiplexedFrame());//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());

            return processSuccess;
        }

        if (_useParvoOutput&&_useColorMode)
        {
            _colorEngine.runColorDemultiplexing(_ParvoRetinaFilter.getOutput(), useAdaptiveFiltering, _maxOutputValue);//_ColorEngine->getMultiplexedFrame());//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());
            // compute A Cr1 Cr2 to LMS color space conversion
            //if (true)
            //  _applyImageColorSpaceConversion(_ColorEngine->getChrominance(), lmsTempBuffer.Buffer(), _LMStoACr1Cr2);
        }

        return processSuccess;
    }

    const std::valarray<float> &RetinaFilter::getContours()
    {
        if (_useColorMode)
            return _colorEngine.getLuminance();
        else
            return _ParvoRetinaFilter.getOutput();
    }

    // run the initilized retina filter in order to perform gray image tone mapping, after this call all retina outputs are updated
    void RetinaFilter::runGrayToneMapping(const std::valarray<float> &grayImageInput, std::valarray<float> &grayImageOutput, const float PhotoreceptorsCompression, const float ganglionCellsCompression)
    {
        // preliminary check
        if (!checkInput(grayImageInput, false))
            return;

        this->_runGrayToneMapping(grayImageInput, grayImageOutput, PhotoreceptorsCompression, ganglionCellsCompression);
    }

    // run the initilized retina filter in order to perform gray image tone mapping, after this call all retina outputs are updated
    void RetinaFilter::_runGrayToneMapping(const std::valarray<float> &grayImageInput, std::valarray<float> &grayImageOutput, const float PhotoreceptorsCompression, const float ganglionCellsCompression)
    {
        // stability controls value update
        ++_ellapsedFramesSinceLastReset;

        std::valarray<float> temp2(grayImageInput.size());

        // apply tone mapping on the multiplexed image
        // -> photoreceptors local adaptation (large area adaptation)
        _photoreceptorsPrefilter.runFilter_LPfilter(grayImageInput, grayImageOutput, 2); // compute low pass filtering modeling the horizontal cells filtering to acess local luminance
        _photoreceptorsPrefilter.setV0CompressionParameterToneMapping(PhotoreceptorsCompression, grayImageOutput.sum()/(float)_photoreceptorsPrefilter.getNBpixels());
        _photoreceptorsPrefilter.runFilter_LocalAdapdation(grayImageInput, grayImageOutput, temp2); // adapt contrast to local luminance

        // high pass filter
        //_spatiotemporalLPfilter(_localBuffer, _filterOutput, 2); // compute low pass filtering (high cut frequency (remove spatio-temporal noise)

        //for (unsigned int i=0;i<_NBpixels;++i)
        //  _localBuffer[i]-= _filterOutput[i]/2.0;

        // -> ganglion cells local adaptation (short area adaptation)
        _photoreceptorsPrefilter.runFilter_LPfilter(temp2, grayImageOutput, 1); // compute low pass filtering (high cut frequency (remove spatio-temporal noise)
        _photoreceptorsPrefilter.setV0CompressionParameterToneMapping(ganglionCellsCompression, temp2.max(), temp2.sum()/(float)_photoreceptorsPrefilter.getNBpixels());
        _photoreceptorsPrefilter.runFilter_LocalAdapdation(temp2, grayImageOutput, grayImageOutput); // adapt contrast to local luminance

    }
    // run the initilized retina filter in order to perform color tone mapping, after this call all retina outputs are updated
    void RetinaFilter::runRGBToneMapping(const std::valarray<float> &RGBimageInput, std::valarray<float> &RGBimageOutput, const bool useAdaptiveFiltering, const float PhotoreceptorsCompression, const float ganglionCellsCompression)
    {
        // preliminary check
        if (!checkInput(RGBimageInput, true))
            return;

        // multiplex the image with the color sampling method specified in the constructor
        _colorEngine.runColorMultiplexing(RGBimageInput);

        // apply tone mapping on the multiplexed image
        _runGrayToneMapping(_colorEngine.getMultiplexedFrame(), RGBimageOutput, PhotoreceptorsCompression, ganglionCellsCompression);

        // demultiplex tone maped image
        _colorEngine.runColorDemultiplexing(RGBimageOutput, useAdaptiveFiltering, _photoreceptorsPrefilter.getMaxInputValue());//_ColorEngine->getMultiplexedFrame());//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());

        // rescaling result between 0 and 255
        _colorEngine.normalizeRGBOutput_0_maxOutputValue(255.0);

        // return the result
        RGBimageOutput=_colorEngine.getDemultiplexedColorFrame();
    }

    void RetinaFilter::runLMSToneMapping(const std::valarray<float> &, std::valarray<float> &, const bool, const float, const float)
    {
        std::cerr<<"not working, sorry"<<std::endl;

        /*  // preliminary check
        const std::valarray<float> &bufferInput=checkInput(LMSimageInput, true);
        if (!bufferInput)
        return NULL;

        if (!_useColorMode)
        std::cerr<<"RetinaFilter::Can not call tone mapping oeration if the retina filter was created for gray scale images"<<std::endl;

        // create a temporary buffer of size nrows, Mcolumns, 3 layers
        std::valarray<float> lmsTempBuffer(LMSimageInput);
        std::cout<<"RetinaFilter::--->min LMS value="<<lmsTempBuffer.min()<<std::endl;

        // setup local adaptation parameter at the photoreceptors level
        setV0CompressionParameter(PhotoreceptorsCompression, _maxInputValue);
        // get the local energy of each color channel
        // ->L
        _spatiotemporalLPfilter(LMSimageInput, _filterOutput, 1);
        setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
        _localLuminanceAdaptation(LMSimageInput, _filterOutput, lmsTempBuffer.Buffer());
        // ->M
        _spatiotemporalLPfilter(LMSimageInput+_NBpixels, _filterOutput, 1);
        setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
        _localLuminanceAdaptation(LMSimageInput+_NBpixels, _filterOutput, lmsTempBuffer.Buffer()+_NBpixels);
        // ->S
        _spatiotemporalLPfilter(LMSimageInput+_NBpixels*2, _filterOutput, 1);
        setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
        _localLuminanceAdaptation(LMSimageInput+_NBpixels*2, _filterOutput, lmsTempBuffer.Buffer()+_NBpixels*2);

        // eliminate negative values
        for (unsigned int i=0;i<lmsTempBuffer.size();++i)
        if (lmsTempBuffer.Buffer()[i]<0)
        lmsTempBuffer.Buffer()[i]=0;
        std::cout<<"RetinaFilter::->min LMS value="<<lmsTempBuffer.min()<<std::endl;

        // compute LMS to A Cr1 Cr2 color space conversion
        _applyImageColorSpaceConversion(lmsTempBuffer.Buffer(), lmsTempBuffer.Buffer(), _LMStoACr1Cr2);

        TemplateBuffer <float> acr1cr2TempBuffer(_NBrows, _NBcolumns, 3);
        memcpy(acr1cr2TempBuffer.Buffer(), lmsTempBuffer.Buffer(), sizeof(float)*_NBpixels*3);

        // compute A Cr1 Cr2 to LMS color space conversion
        _applyImageColorSpaceConversion(acr1cr2TempBuffer.Buffer(), lmsTempBuffer.Buffer(), _ACr1Cr2toLMS);

        // eliminate negative values
        for (unsigned int i=0;i<lmsTempBuffer.size();++i)
        if (lmsTempBuffer.Buffer()[i]<0)
        lmsTempBuffer.Buffer()[i]=0;

        // rewrite output to the appropriate buffer
        _colorEngine->setDemultiplexedColorFrame(lmsTempBuffer.Buffer());
        */
    }

    // return image with center Parvo and peripheral Magno channels
    void RetinaFilter::_processRetinaParvoMagnoMapping()
    {
        register float *hybridParvoMagnoPTR= &_retinaParvoMagnoMappedFrame[0];
        register const float *parvoOutputPTR= get_data(_ParvoRetinaFilter.getOutput());
        register const float *magnoXOutputPTR= get_data(_MagnoRetinaFilter.getOutput());
        register float *hybridParvoMagnoCoefTablePTR= &_retinaParvoMagnoMapCoefTable[0];

        for (unsigned int i=0 ; i<_photoreceptorsPrefilter.getNBpixels() ; ++i, hybridParvoMagnoCoefTablePTR+=2)
        {
            float hybridValue=*(parvoOutputPTR++)**(hybridParvoMagnoCoefTablePTR)+*(magnoXOutputPTR++)**(hybridParvoMagnoCoefTablePTR+1);
            *(hybridParvoMagnoPTR++)=hybridValue;
        }

        TemplateBuffer<float>::normalizeGrayOutput_0_maxOutputValue(&_retinaParvoMagnoMappedFrame[0], _photoreceptorsPrefilter.getNBpixels());

    }

    bool RetinaFilter::getParvoFoveaResponse(std::valarray<float> &parvoFovealResponse)
    {
        if (!_useParvoOutput)
            return false;
        if (parvoFovealResponse.size() != _ParvoRetinaFilter.getNBpixels())
            return false;

        register const float *parvoOutputPTR= get_data(_ParvoRetinaFilter.getOutput());
        register float *fovealParvoResponsePTR= &parvoFovealResponse[0];
        register float *hybridParvoMagnoCoefTablePTR= &_retinaParvoMagnoMapCoefTable[0];

        for (unsigned int i=0 ; i<_photoreceptorsPrefilter.getNBpixels() ; ++i, hybridParvoMagnoCoefTablePTR+=2)
        {
            *(fovealParvoResponsePTR++)=*(parvoOutputPTR++)**(hybridParvoMagnoCoefTablePTR);
        }

        return true;
    }

    // method to retrieve the parafoveal magnocellular pathway response (no energy motion in fovea)
    bool RetinaFilter::getMagnoParaFoveaResponse(std::valarray<float> &magnoParafovealResponse)
    {
        if (!_useMagnoOutput)
            return false;
        if (magnoParafovealResponse.size() != _MagnoRetinaFilter.getNBpixels())
            return false;

        register const float *magnoXOutputPTR= get_data(_MagnoRetinaFilter.getOutput());
        register float *parafovealMagnoResponsePTR=&magnoParafovealResponse[0];
        register float *hybridParvoMagnoCoefTablePTR=&_retinaParvoMagnoMapCoefTable[0]+1;

        for (unsigned int i=0 ; i<_photoreceptorsPrefilter.getNBpixels() ; ++i, hybridParvoMagnoCoefTablePTR+=2)
        {
            *(parafovealMagnoResponsePTR++)=*(magnoXOutputPTR++)**(hybridParvoMagnoCoefTablePTR);
        }

        return true;
    }
}
