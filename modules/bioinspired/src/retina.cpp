/*#******************************************************************************
 ** IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 **
 ** By downloading, copying, installing or using the software you agree to this license.
 ** If you do not agree to this license, do not download, install,
 ** copy or use the software.
 **
 **
 ** bioinspired : interfaces allowing OpenCV users to integrate Human Vision System models. Presented models originate from Jeanny Herault's original research and have been reused and adapted by the author&collaborators for computed vision applications since his thesis with Alice Caplier at Gipsa-Lab.
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
 * Retina.cpp
 *
 *  Created on: Jul 19, 2011
 *      Author: Alexandre Benoit
 */
#include "precomp.hpp"
#include "retinafilter.hpp"
#include <cstdio>
#include <sstream>
#include <valarray>

namespace cv
{
namespace bioinspired
{

class RetinaImpl : public Retina
{
public:
    /**
     * Main constructor with most commun use setup : create an instance of color ready retina model
     * @param inputSize : the input frame size
     */
    RetinaImpl(Size inputSize);

    /**
     * Complete Retina filter constructor which allows all basic structural parameters definition
         * @param inputSize : the input frame size
     * @param colorMode : the chosen processing mode : with or without color processing
     * @param colorSamplingMethod: specifies which kind of color sampling will be used
     * @param useRetinaLogSampling: activate retina log sampling, if true, the 2 following parameters can be used
     * @param reductionFactor: only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
     * @param samplingStrenght: only usefull if param useRetinaLogSampling=true, specifies the strenght of the log scale that is applied
     */
    RetinaImpl(Size inputSize, const bool colorMode, int colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);

    virtual ~RetinaImpl();
    /**
        * retreive retina input buffer size
        */
        Size getInputSize();

    /**
        * retreive retina output buffer size
        */
        Size getOutputSize();

    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param retinaParameterFile : the parameters filename
         * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    void setup(String retinaParameterFile="", const bool applyDefaultSetupOnFailure=true);


    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param fs : the open Filestorage which contains retina parameters
         * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
        void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure=true);

    /**
     * try to open an XML retina parameters file to adjust current retina instance setup
     * => if the xml file does not exist, then default setup is applied
     * => warning, Exceptions are thrown if read XML file is not valid
     * @param newParameters : a parameters structures updated with the new target configuration
         * @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
     */
    void setup(Retina::RetinaParameters newParameters);

    /**
    * @return the current parameters setup
    */
    struct Retina::RetinaParameters getParameters();

    /**
     * parameters setup display method
     * @return a string which contains formatted parameters information
     */
    const String printSetup();

    /**
     * write xml/yml formated parameters information
     * @rparam fs : the filename of the xml file that will be open and writen with formatted parameters information
     */
    virtual void write( String fs ) const;


    /**
     * write xml/yml formated parameters information
     * @param fs : a cv::Filestorage object ready to be filled
         */
    virtual void write( FileStorage& fs ) const;

    /**
     * setup the OPL and IPL parvo channels (see biologocal model)
     * OPL is referred as Outer Plexiform Layer of the retina, it allows the spatio-temporal filtering which withens the spectrum and reduces spatio-temporal noise while attenuating global luminance (low frequency energy)
     * IPL parvo is the OPL next processing stage, it refers to Inner Plexiform layer of the retina, it allows high contours sensitivity in foveal vision.
     * for more informations, please have a look at the paper Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
     * @param colorMode : specifies if (true) color is processed of not (false) to then processing gray level image
     * @param normaliseOutput : specifies if (true) output is rescaled between 0 and 255 of not (false)
     * @param photoreceptorsLocalAdaptationSensitivity: the photoreceptors sensitivity renage is 0-1 (more log compression effect when value increases)
     * @param photoreceptorsTemporalConstant: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
     * @param photoreceptorsSpatialConstant: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
     * @param horizontalCellsGain: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
     * @param HcellsTemporalConstant: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
     * @param HcellsSpatialConstant: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
     * @param ganglionCellsSensitivity: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 230
     */
    void setupOPLandIPLParvoChannel(const bool colorMode=true, const bool normaliseOutput = true, const float photoreceptorsLocalAdaptationSensitivity=0.7, const float photoreceptorsTemporalConstant=0.5, const float photoreceptorsSpatialConstant=0.53, const float horizontalCellsGain=0, const float HcellsTemporalConstant=1, const float HcellsSpatialConstant=7, const float ganglionCellsSensitivity=0.7);

    /**
     * set parameters values for the Inner Plexiform Layer (IPL) magnocellular channel
     * this channel processes signals outpint from OPL processing stage in peripheral vision, it allows motion information enhancement. It is decorrelated from the details channel. See reference paper for more details.
     * @param normaliseOutput : specifies if (true) output is rescaled between 0 and 255 of not (false)
     * @param parasolCells_beta: the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
     * @param parasolCells_tau: the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
     * @param parasolCells_k: the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
     * @param amacrinCellsTemporalCutFrequency: the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, tipicall value is 5
     * @param V0CompressionParameter: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 200
     * @param localAdaptintegration_tau: specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
     * @param localAdaptintegration_k: specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
     */
    void setupIPLMagnoChannel(const bool normaliseOutput = true, const float parasolCells_beta=0, const float parasolCells_tau=0, const float parasolCells_k=7, const float amacrinCellsTemporalCutFrequency=1.2, const float V0CompressionParameter=0.95, const float localAdaptintegration_tau=0, const float localAdaptintegration_k=7);

    /**
     * method which allows retina to be applied on an input image, after run, encapsulated retina module is ready to deliver its outputs using dedicated acccessors, see getParvo and getMagno methods
     * @param inputImage : the input cv::Mat image to be processed, can be gray level or BGR coded in any format (from 8bit to 16bits)
     */
    void run(InputArray inputImage);

    /**
     * method that applies a luminance correction (initially High Dynamic Range (HDR) tone mapping) using only the 2 local adaptation stages of the retina parvo channel : photoreceptors level and ganlion cells level. Spatio temporal filtering is applied but limited to temporal smoothing and eventually high frequencies attenuation. This is a lighter method than the one available using the regular run method. It is then faster but it does not include complete temporal filtering nor retina spectral whitening. This is an adptation of the original still image HDR tone mapping algorithm of David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
    * -> Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816
     @param inputImage the input image to process RGB or gray levels
     @param outputToneMappedImage the output tone mapped image
     */
    void applyFastToneMapping(InputArray inputImage, OutputArray outputToneMappedImage);

    /**
     * accessor of the details channel of the retina (models foveal vision)
     * @param retinaOutput_parvo : the output buffer (reallocated if necessary), this output is rescaled for standard 8bits image processing use in OpenCV
     */
    void getParvo(OutputArray retinaOutput_parvo);

    /**
     * accessor of the details channel of the retina (models foveal vision)
     * @param retinaOutput_parvo : a cv::Mat header filled with the internal parvo buffer of the retina module. This output is the original retina filter model output, without any quantification or rescaling
     */
    void getParvoRAW(OutputArray retinaOutput_parvo);

    /**
     * accessor of the motion channel of the retina (models peripheral vision)
     * @param retinaOutput_magno : the output buffer (reallocated if necessary), this output is rescaled for standard 8bits image processing use in OpenCV
     */
    void getMagno(OutputArray retinaOutput_magno);

    /**
     * accessor of the motion channel of the retina (models peripheral vision)
     * @param retinaOutput_magno : a cv::Mat header filled with the internal retina magno buffer of the retina module. This output is the original retina filter model output, without any quantification or rescaling
     */
    void getMagnoRAW(OutputArray retinaOutput_magno);

    // original API level data accessors : get buffers addresses from a Mat header, similar to getParvoRAW and getMagnoRAW...
    const Mat getMagnoRAW() const;
    const Mat getParvoRAW() const;

    /**
     * activate color saturation as the final step of the color demultiplexing process
     * -> this saturation is a sigmoide function applied to each channel of the demultiplexed image.
     * @param saturateColors: boolean that activates color saturation (if true) or desactivate (if false)
     * @param colorSaturationValue: the saturation factor
     */
    void setColorSaturation(const bool saturateColors=true, const float colorSaturationValue=4.0);

    /**
     * clear all retina buffers (equivalent to opening the eyes after a long period of eye close ;o)
     */
    void clearBuffers();

        /**
        * Activate/desactivate the Magnocellular pathway processing (motion information extraction), by default, it is activated
        * @param activate: true if Magnocellular output should be activated, false if not
        */
        void activateMovingContoursProcessing(const bool activate);

        /**
        * Activate/desactivate the Parvocellular pathway processing (contours information extraction), by default, it is activated
        * @param activate: true if Parvocellular (contours information extraction) output should be activated, false if not
        */
        void activateContoursProcessing(const bool activate);
private:

    // Parameteres setup members
    RetinaParameters _retinaParameters; // structure of parameters

    // Retina model related modules
    std::valarray<float> _inputBuffer; //!< buffer used to convert input cv::Mat to internal retina buffers format (valarrays)

    // pointer to retina model
    RetinaFilter* _retinaFilter; //!< the pointer to the retina module, allocated with instance construction

    //! private method called by constructors, gathers their parameters and use them in a unified way
    void _init(const Size inputSize, const bool colorMode, int colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);

    /**
     * exports a valarray buffer outing from bioinspired objects to a cv::Mat in CV_8UC1 (gray level picture) or CV_8UC3 (color) format
     * @param grayMatrixToConvert the valarray to export to OpenCV
     * @param nbRows : the number of rows of the valarray flatten matrix
     * @param nbColumns : the number of rows of the valarray flatten matrix
     * @param colorMode : a flag which mentions if matrix is color (true) or graylevel (false)
     * @param outBuffer : the output matrix which is reallocated to satisfy Retina output buffer dimensions
     */
    void _convertValarrayBuffer2cvMat(const std::valarray<float> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, OutputArray outBuffer);

    /**
     * convert a cv::Mat to a valarray buffer in float format
     * @param inputMatToConvert : the OpenCV cv::Mat that has to be converted to gray or RGB valarray buffer that will be processed by the retina model
     * @param outputValarrayMatrix : the output valarray
     * @return the input image color mode (color=true, gray levels=false)
     */
    bool _convertCvMat2ValarrayBuffer(InputArray inputMatToConvert, std::valarray<float> &outputValarrayMatrix);


};

// smart pointers allocation :
Ptr<Retina> createRetina(Size inputSize){ return makePtr<RetinaImpl>(inputSize); }
Ptr<Retina> createRetina(Size inputSize, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght){
    return makePtr<RetinaImpl>(inputSize, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
}


// RetinaImpl code
RetinaImpl::RetinaImpl(const cv::Size inputSz)
{
    _retinaFilter = 0;
    _init(inputSz, true, RETINA_COLOR_BAYER, false);
}

RetinaImpl::RetinaImpl(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    _retinaFilter = 0;
    _init(inputSz, colorMode, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrenght);
};

RetinaImpl::~RetinaImpl()
{
    if (_retinaFilter)
        delete _retinaFilter;
}

/**
* retreive retina input buffer size
*/
Size RetinaImpl::getInputSize(){return cv::Size(_retinaFilter->getInputNBcolumns(), _retinaFilter->getInputNBrows());}

/**
* retreive retina output buffer size
*/
Size RetinaImpl::getOutputSize(){return cv::Size(_retinaFilter->getOutputNBcolumns(), _retinaFilter->getOutputNBrows());}


void RetinaImpl::setColorSaturation(const bool saturateColors, const float colorSaturationValue)
{
    _retinaFilter->setColorSaturation(saturateColors, colorSaturationValue);
}

struct Retina::RetinaParameters RetinaImpl::getParameters(){return _retinaParameters;}

void RetinaImpl::setup(String retinaParameterFile, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // opening retinaParameterFile in read mode
        cv::FileStorage fs(retinaParameterFile, cv::FileStorage::READ);
        setup(fs, applyDefaultSetupOnFailure);
    }
    catch(Exception &e)
    {
        printf("Retina::setup: wrong/unappropriate xml parameter file : error report :`n=>%s\n", e.what());
        if (applyDefaultSetupOnFailure)
        {
            printf("Retina::setup: resetting retina with default parameters\n");
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        else
        {
            printf("=> keeping current parameters\n");
        }
    }
}

void RetinaImpl::setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure)
{
    try
    {
        // read parameters file if it exists or apply default setup if asked for
        if (!fs.isOpened())
        {
            printf("Retina::setup: provided parameters file could not be open... skeeping configuration\n");
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
        printf("RetinaImpl::setup: resetting retina with default parameters\n");
        if (applyDefaultSetupOnFailure)
        {
            setupOPLandIPLParvoChannel();
            setupIPLMagnoChannel();
        }
        printf("Retina::setup: wrong/unappropriate xml parameter file : error report :`n=>%s\n", e.what());
        printf("=> keeping current parameters\n");
    }

    // report current configuration
    printf("%s\n", printSetup().c_str());
}

void RetinaImpl::setup(Retina::RetinaParameters newConfiguration)
{
    // simply copy structures
    memcpy(&_retinaParameters, &newConfiguration, sizeof(Retina::RetinaParameters));
    // apply setup
    setupOPLandIPLParvoChannel(_retinaParameters.OPLandIplParvo.colorMode, _retinaParameters.OPLandIplParvo.normaliseOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant, _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant, _retinaParameters.OPLandIplParvo.horizontalCellsGain, _retinaParameters.OPLandIplParvo.hcellsTemporalConstant, _retinaParameters.OPLandIplParvo.hcellsSpatialConstant, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
    setupIPLMagnoChannel(_retinaParameters.IplMagno.normaliseOutput, _retinaParameters.IplMagno.parasolCells_beta, _retinaParameters.IplMagno.parasolCells_tau, _retinaParameters.IplMagno.parasolCells_k, _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency,_retinaParameters.IplMagno.V0CompressionParameter, _retinaParameters.IplMagno.localAdaptintegration_tau, _retinaParameters.IplMagno.localAdaptintegration_k);

}

const String RetinaImpl::printSetup()
{
    std::stringstream outmessage;

    // displaying OPL and IPL parvo setup
    outmessage<<"Current Retina instance setup :"
            <<"\nOPLandIPLparvo"<<"{"
            << "\n\t colorMode : " << _retinaParameters.OPLandIplParvo.colorMode
            << "\n\t normalizeParvoOutput :" << _retinaParameters.OPLandIplParvo.normaliseOutput
            << "\n\t photoreceptorsLocalAdaptationSensitivity : " << _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity
            << "\n\t photoreceptorsTemporalConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsTemporalConstant
            << "\n\t photoreceptorsSpatialConstant : " << _retinaParameters.OPLandIplParvo.photoreceptorsSpatialConstant
            << "\n\t horizontalCellsGain : " << _retinaParameters.OPLandIplParvo.horizontalCellsGain
            << "\n\t hcellsTemporalConstant : " << _retinaParameters.OPLandIplParvo.hcellsTemporalConstant
            << "\n\t hcellsSpatialConstant : " << _retinaParameters.OPLandIplParvo.hcellsSpatialConstant
            << "\n\t parvoGanglionCellsSensitivity : " << _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity
            <<"}\n";

    // displaying IPL magno setup
    outmessage<<"Current Retina instance setup :"
            <<"\nIPLmagno"<<"{"
            << "\n\t normaliseOutput : " << _retinaParameters.IplMagno.normaliseOutput
            << "\n\t parasolCells_beta : " << _retinaParameters.IplMagno.parasolCells_beta
            << "\n\t parasolCells_tau : " << _retinaParameters.IplMagno.parasolCells_tau
            << "\n\t parasolCells_k : " << _retinaParameters.IplMagno.parasolCells_k
            << "\n\t amacrinCellsTemporalCutFrequency : " << _retinaParameters.IplMagno.amacrinCellsTemporalCutFrequency
            << "\n\t V0CompressionParameter : " << _retinaParameters.IplMagno.V0CompressionParameter
            << "\n\t localAdaptintegration_tau : " << _retinaParameters.IplMagno.localAdaptintegration_tau
            << "\n\t localAdaptintegration_k : " << _retinaParameters.IplMagno.localAdaptintegration_k
            <<"}";
    return outmessage.str().c_str();
}

void RetinaImpl::write( String fs ) const
{
    FileStorage parametersSaveFile(fs, cv::FileStorage::WRITE );
    write(parametersSaveFile);
}

void RetinaImpl::write( FileStorage& fs ) const
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

void RetinaImpl::setupOPLandIPLParvoChannel(const bool colorMode, const bool normaliseOutput, const float photoreceptorsLocalAdaptationSensitivity, const float photoreceptorsTemporalConstant, const float photoreceptorsSpatialConstant, const float horizontalCellsGain, const float HcellsTemporalConstant, const float HcellsSpatialConstant, const float ganglionCellsSensitivity)
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

void RetinaImpl::setupIPLMagnoChannel(const bool normaliseOutput, const float parasolCells_beta, const float parasolCells_tau, const float parasolCells_k, const float amacrinCellsTemporalCutFrequency, const float V0CompressionParameter, const float localAdaptintegration_tau, const float localAdaptintegration_k)
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

void RetinaImpl::run(InputArray inputMatToConvert)
{
    // first convert input image to the compatible format : std::valarray<float>
    const bool colorMode = _convertCvMat2ValarrayBuffer(inputMatToConvert.getMat(), _inputBuffer);
    // process the retina
    if (!_retinaFilter->runFilter(_inputBuffer, colorMode, false, _retinaParameters.OPLandIplParvo.colorMode && colorMode, false))
        throw cv::Exception(-1, "RetinaImpl cannot be applied, wrong input buffer size", "RetinaImpl::run", "RetinaImpl.h", 0);
}

void RetinaImpl::applyFastToneMapping(InputArray inputImage, OutputArray outputToneMappedImage)
{
    // first convert input image to the compatible format :
    const bool colorMode = _convertCvMat2ValarrayBuffer(inputImage.getMat(), _inputBuffer);
    const unsigned int nbPixels=_retinaFilter->getOutputNBrows()*_retinaFilter->getOutputNBcolumns();

    // process tone mapping
    if (colorMode)
    {
        std::valarray<float> imageOutput(nbPixels*3);
        _retinaFilter->runRGBToneMapping(_inputBuffer, imageOutput, true, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
        _convertValarrayBuffer2cvMat(imageOutput, _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), true, outputToneMappedImage);
    }else
    {
        std::valarray<float> imageOutput(nbPixels);
        _retinaFilter->runGrayToneMapping(_inputBuffer, imageOutput, _retinaParameters.OPLandIplParvo.photoreceptorsLocalAdaptationSensitivity, _retinaParameters.OPLandIplParvo.ganglionCellsSensitivity);
        _convertValarrayBuffer2cvMat(imageOutput, _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), false, outputToneMappedImage);
    }

}

void RetinaImpl::getParvo(OutputArray retinaOutput_parvo)
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
void RetinaImpl::getMagno(OutputArray retinaOutput_magno)
{
    // reallocate output buffer (if necessary)
    _convertValarrayBuffer2cvMat(_retinaFilter->getMovingContours(), _retinaFilter->getOutputNBrows(), _retinaFilter->getOutputNBcolumns(), false, retinaOutput_magno);
    //retinaOutput_magno/=255.0;
}

// original API level data accessors : copy buffers if size matches, reallocate if required
void RetinaImpl::getMagnoRAW(OutputArray magnoOutputBufferCopy){
    // get magno channel header
    const cv::Mat magnoChannel=cv::Mat(getMagnoRAW());
    // copy data
    magnoChannel.copyTo(magnoOutputBufferCopy);
}

void RetinaImpl::getParvoRAW(OutputArray parvoOutputBufferCopy){
    // get parvo channel header
    const cv::Mat parvoChannel=cv::Mat(getMagnoRAW());
    // copy data
    parvoChannel.copyTo(parvoOutputBufferCopy);
}

// original API level data accessors : get buffers addresses...
const Mat RetinaImpl::getMagnoRAW() const {
    // create a cv::Mat header for the valarray
    return Mat((int)_retinaFilter->getMovingContours().size(),1, CV_32F, (void*)get_data(_retinaFilter->getMovingContours()));

}

const Mat RetinaImpl::getParvoRAW() const {
    if (_retinaFilter->getColorMode()) // check if color mode is enabled
    {
        // create a cv::Mat table (for RGB planes as a single vector)
        return Mat((int)_retinaFilter->getColorOutput().size(), 1, CV_32F, (void*)get_data(_retinaFilter->getColorOutput()));
    }
    // otherwise, output is gray level
    // create a cv::Mat header for the valarray
    return Mat((int)_retinaFilter->getContours().size(), 1, CV_32F, (void*)get_data(_retinaFilter->getContours()));
}

// private method called by constructirs
void RetinaImpl::_init(const cv::Size inputSz, const bool colorMode, int colorSamplingMethod, const bool useRetinaLogSampling, const double reductionFactor, const double samplingStrenght)
{
    // basic error check
    if (inputSz.height*inputSz.width <= 0)
        throw cv::Exception(-1, "Bad retina size setup : size height and with must be superior to zero", "RetinaImpl::setup", "Retina.cpp", 0);

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
    printf("%s\n", printSetup().c_str());
}

void RetinaImpl::_convertValarrayBuffer2cvMat(const std::valarray<float> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, OutputArray outBuffer)
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
    }else
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

bool RetinaImpl::_convertCvMat2ValarrayBuffer(InputArray inputMat, std::valarray<float> &outputValarrayMatrix)
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

void RetinaImpl::clearBuffers() {_retinaFilter->clearAllBuffers();}

void RetinaImpl::activateMovingContoursProcessing(const bool activate){_retinaFilter->activateMovingContoursProcessing(activate);}

void RetinaImpl::activateContoursProcessing(const bool activate){_retinaFilter->activateContoursProcessing(activate);}

}// end of namespace bioinspired
}// end of namespace cv
