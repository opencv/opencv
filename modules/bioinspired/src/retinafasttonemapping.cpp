
/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license.
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 **
 ** bioinspired : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
 **
 ** Maintainers : Listic lab (code author current affiliation & applications) and Gipsa Lab (original research origins & applications)
 **
 **  Creation - enhancement process 2007-2013
 **      Author: Alexandre Benoit (benoit.alexandre.vision@gmail.com), LISTIC lab, Annecy le vieux, France
 **
 ** Theses algorithm have been developped by Alexandre BENOIT since his thesis with Alice Caplier at Gipsa-Lab (www.gipsa-lab.inpg.fr) and the research he pursues at LISTIC Lab (www.listic.univ-savoie.fr).
 ** Refer to the following research paper for more information:
 ** Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
 ** This work have been carried out thanks to Jeanny Herault who's research and great discussions are the basis of all this work, please take a look at his book:
 ** Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
 **
 **
 ** This class is based on image processing tools of the author and already used within the Retina class (this is the same code as method retina::applyFastToneMapping, but in an independent class, it is ligth from a memory requirement point of view). It implements an adaptation of the efficient tone mapping algorithm propose by David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
 ** -> Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816
 **
 **
 **                          License Agreement
 **               For Open Source Computer Vision Library
 **
 ** Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 ** Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
 **
 **               For Human Visual System tools (bioinspired)
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
 * retinafasttonemapping.cpp
 *
 *  Created on: May 26, 2013
 *      Author: Alexandre Benoit
 */

#include "precomp.hpp"
#include "basicretinafilter.hpp"
#include "retinacolor.hpp"
#include <cstdio>
#include <sstream>
#include <valarray>

namespace cv
{
namespace bioinspired
{
/**
 * @class RetinaFastToneMappingImpl a wrapper class which allows the tone mapping algorithm of Meylan&al(2007) to be used with OpenCV.
 * This algorithm is already implemented in thre Retina class (retina::applyFastToneMapping) but used it does not require all the retina model to be allocated. This allows a light memory use for low memory devices (smartphones, etc.
 * As a summary, these are the model properties:
 * => 2 stages of local luminance adaptation with a different local neighborhood for each.
 * => first stage models the retina photorecetors local luminance adaptation
 * => second stage models th ganglion cells local information adaptation
 * => compared to the initial publication, this class uses spatio-temporal low pass filters instead of spatial only filters.
 * ====> this can help noise robustness and temporal stability for video sequence use cases.
 * for more information, read to the following papers :
 *  Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
 * regarding spatio-temporal filter and the bigger retina model :
 * Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
 */

class RetinaFastToneMappingImpl : public RetinaFastToneMapping
{
public:
    /**
     * constructor
     * @param imageInput: the size of the images to process
     */
    RetinaFastToneMappingImpl(Size imageInput)
    {
        unsigned int nbPixels=imageInput.height*imageInput.width;

        // basic error check
        if (nbPixels <= 0)
        throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "RetinaImpl::setup", "retinafasttonemapping.cpp", 0);

        // resize buffers
        _inputBuffer.resize(nbPixels*3); // buffer supports gray images but also 3 channels color buffers... (larger is better...)
        _imageOutput.resize(nbPixels*3);
        _temp2.resize(nbPixels);
        // allocate the main filter with 2 setup sets properties (one for each low pass filter
        _multiuseFilter = makePtr<BasicRetinaFilter>(imageInput.height, imageInput.width, 2);
        // allocate the color manager (multiplexer/demultiplexer
        _colorEngine = makePtr<RetinaColor>(imageInput.height, imageInput.width);
        // setup filter behaviors with default values
        setup();
    }

    /**
     * basic destructor
     */
    virtual ~RetinaFastToneMappingImpl() { }

    /**
     * method that applies a luminance correction (initially High Dynamic Range (HDR) tone mapping) using only the 2 local adaptation stages of the retina parvocellular channel : photoreceptors level and ganlion cells level. Spatio temporal filtering is applied but limited to temporal smoothing and eventually high frequencies attenuation. This is a lighter method than the one available using the regular retina::run method. It is then faster but it does not include complete temporal filtering nor retina spectral whitening. Then, it can have a more limited effect on images with a very high dynamic range. This is an adptation of the original still image HDR tone mapping algorithm of David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
    * -> Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816
     @param inputImage the input image to process RGB or gray levels
     @param outputToneMappedImage the output tone mapped image
     */
    virtual void applyFastToneMapping(InputArray inputImage, OutputArray outputToneMappedImage)
    {
        // first convert input image to the compatible format :
        const bool colorMode = _convertCvMat2ValarrayBuffer(inputImage.getMat(), _inputBuffer);

        // process tone mapping
        if (colorMode)
        {
            _runRGBToneMapping(_inputBuffer, _imageOutput, true);
            _convertValarrayBuffer2cvMat(_imageOutput, _multiuseFilter->getNBrows(), _multiuseFilter->getNBcolumns(), true, outputToneMappedImage);
        }
        else
        {
            _runGrayToneMapping(_inputBuffer, _imageOutput);
            _convertValarrayBuffer2cvMat(_imageOutput, _multiuseFilter->getNBrows(), _multiuseFilter->getNBcolumns(), false, outputToneMappedImage);
        }

    }

    /**
     * setup method that updates tone mapping behaviors by adjusing the local luminance computation area
     * @param photoreceptorsNeighborhoodRadius the first stage local adaptation area
     * @param ganglioncellsNeighborhoodRadius the second stage local adaptation area
     * @param meanLuminanceModulatorK the factor applied to modulate the meanLuminance information (default is 1, see reference paper)
     */
    virtual void setup(const float photoreceptorsNeighborhoodRadius=3.f, const float ganglioncellsNeighborhoodRadius=1.f, const float meanLuminanceModulatorK=1.f)
    {
        // setup the spatio-temporal properties of each filter
        _meanLuminanceModulatorK = meanLuminanceModulatorK;
        _multiuseFilter->setV0CompressionParameter(1.f, 255.f, 128.f);
        _multiuseFilter->setLPfilterParameters(0.f, 0.f, photoreceptorsNeighborhoodRadius, 1);
        _multiuseFilter->setLPfilterParameters(0.f, 0.f, ganglioncellsNeighborhoodRadius, 2);
    }

private:
    // a filter able to perform local adaptation and low pass spatio-temporal filtering
    cv::Ptr <BasicRetinaFilter> _multiuseFilter;
    cv::Ptr <RetinaColor> _colorEngine;

    //!< buffer used to convert input cv::Mat to internal retina buffers format (valarrays)
    std::valarray<float> _inputBuffer;
    std::valarray<float> _imageOutput;
    std::valarray<float> _temp2;
    float _meanLuminanceModulatorK;


void _convertValarrayBuffer2cvMat(const std::valarray<float> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, OutputArray outBuffer)
{
    // fill output buffer with the valarray buffer
    const float *valarrayPTR=get_data(grayMatrixToConvert);
    if (!colorMode)
    {
        outBuffer.create(cv::Size(nbColumns, nbRows), CV_8U);
        Mat outMat = outBuffer.getMat();
        for (unsigned int i=0;i<nbRows;++i)
        {
            for (unsigned int j=0;j<nbColumns;++j)
            {
                cv::Point2d pixel(j,i);
                outMat.at<unsigned char>(pixel)=(unsigned char)*(valarrayPTR++);
            }
        }
    }
    else
    {
        const unsigned int nbPixels=nbColumns*nbRows;
        const unsigned int doubleNBpixels=nbColumns*nbRows*2;
        outBuffer.create(cv::Size(nbColumns, nbRows), CV_8UC3);
        Mat outMat = outBuffer.getMat();
        for (unsigned int i=0;i<nbRows;++i)
        {
            for (unsigned int j=0;j<nbColumns;++j,++valarrayPTR)
            {
                cv::Point2d pixel(j,i);
                cv::Vec3b pixelValues;
                pixelValues[2]=(unsigned char)*(valarrayPTR);
                pixelValues[1]=(unsigned char)*(valarrayPTR+nbPixels);
                pixelValues[0]=(unsigned char)*(valarrayPTR+doubleNBpixels);

                outMat.at<cv::Vec3b>(pixel)=pixelValues;
            }
        }
    }
}

bool _convertCvMat2ValarrayBuffer(InputArray inputMat, std::valarray<float> &outputValarrayMatrix)
{
    const Mat inputMatToConvert=inputMat.getMat();
    // first check input consistency
    if (inputMatToConvert.empty())
        throw cv::Exception(-1, "RetinaImpl cannot be applied, input buffer is empty", "RetinaImpl::run", "RetinaImpl.h", 0);

    // retreive color mode from image input
    int imageNumberOfChannels = inputMatToConvert.channels();

        // convert to float AND fill the valarray buffer
    typedef float T; // define here the target pixel format, here, float
    const int dsttype = DataType<T>::depth; // output buffer is float format

    const unsigned int nbPixels=inputMat.getMat().rows*inputMat.getMat().cols;
    const unsigned int doubleNBpixels=inputMat.getMat().rows*inputMat.getMat().cols*2;

    if(imageNumberOfChannels==4)
    {
    // create a cv::Mat table (for RGBA planes)
        cv::Mat planes[4] =
        {
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[doubleNBpixels]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[nbPixels]),
            cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[0])
        };
        planes[3] = cv::Mat(inputMatToConvert.size(), dsttype);     // last channel (alpha) does not point on the valarray (not usefull in our case)
        // split color cv::Mat in 4 planes... it fills valarray directely
        cv::split(Mat_<Vec<T, 4> >(inputMatToConvert), planes);
    }
    else if (imageNumberOfChannels==3)
    {
        // create a cv::Mat table (for RGB planes)
        cv::Mat planes[] =
        {
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[doubleNBpixels]),
        cv::Mat(inputMatToConvert.size(), dsttype, &outputValarrayMatrix[nbPixels]),
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
        CV_Error(Error::StsUnsupportedFormat, "input image must be single channel (gray levels), bgr format (color) or bgra (color with transparency which won't be considered");

    return imageNumberOfChannels>1; // return bool : false for gray level image processing, true for color mode
}


// run the initilized retina filter in order to perform gray image tone mapping, after this call all retina outputs are updated
void _runGrayToneMapping(const std::valarray<float> &grayImageInput, std::valarray<float> &grayImageOutput)
{
     // apply tone mapping on the multiplexed image
    // -> photoreceptors local adaptation (large area adaptation)
    _multiuseFilter->runFilter_LPfilter(grayImageInput, grayImageOutput, 0); // compute low pass filtering modeling the horizontal cells filtering to acess local luminance
    _multiuseFilter->setV0CompressionParameterToneMapping(1.f, grayImageOutput.max(), _meanLuminanceModulatorK*grayImageOutput.sum()/(float)_multiuseFilter->getNBpixels());
    _multiuseFilter->runFilter_LocalAdapdation(grayImageInput, grayImageOutput, _temp2); // adapt contrast to local luminance

    // -> ganglion cells local adaptation (short area adaptation)
    _multiuseFilter->runFilter_LPfilter(_temp2, grayImageOutput, 1); // compute low pass filtering (high cut frequency (remove spatio-temporal noise)
    _multiuseFilter->setV0CompressionParameterToneMapping(1.f, _temp2.max(), _meanLuminanceModulatorK*grayImageOutput.sum()/(float)_multiuseFilter->getNBpixels());
    _multiuseFilter->runFilter_LocalAdapdation(_temp2, grayImageOutput, grayImageOutput); // adapt contrast to local luminance

}

// run the initilized retina filter in order to perform color tone mapping, after this call all retina outputs are updated
void _runRGBToneMapping(const std::valarray<float> &RGBimageInput, std::valarray<float> &RGBimageOutput, const bool useAdaptiveFiltering)
{
    // multiplex the image with the color sampling method specified in the constructor
    _colorEngine->runColorMultiplexing(RGBimageInput);

    // apply tone mapping on the multiplexed image
    _runGrayToneMapping(_colorEngine->getMultiplexedFrame(), RGBimageOutput);

    // demultiplex tone maped image
    _colorEngine->runColorDemultiplexing(RGBimageOutput, useAdaptiveFiltering, _multiuseFilter->getMaxInputValue());//_ColorEngine->getMultiplexedFrame());//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());

    // rescaling result between 0 and 255
    _colorEngine->normalizeRGBOutput_0_maxOutputValue(255.0);

    // return the result
    RGBimageOutput=_colorEngine->getDemultiplexedColorFrame();
}

};

CV_EXPORTS Ptr<RetinaFastToneMapping> createRetinaFastToneMapping(Size inputSize)
{
    return makePtr<RetinaFastToneMappingImpl>(inputSize);
}

}// end of namespace bioinspired
}// end of namespace cv
