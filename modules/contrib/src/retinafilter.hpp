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

/**
* @class  RetinaFilter
* @brief class which describes the retina model developped at the LIS/GIPSA-LAB www.gipsa-lab.inpg.fr:
* -> performs a contours and moving contours extraction with powerfull local data enhancement as at the retina level
* Based on Alexandre BENOIT thesis: "Le systeme visuel humain au secours de la vision par ordinateur"
*
* => various optimisations and enhancements added after 2007 such as tone mapping capabilities, see reference paper cited in the licence and :
* Benoit A.,Alleysson D., Herault J., Le Callet P. (2009),  "Spatio-Temporal Tone Mapping Operator based on a Retina model", Computational Color Imaging Workshop (CCIW09),pp 12-22, Saint Etienne, France
*
* TYPICAL USE:
*
* // create object at a specified picture size
* Retina *retina;
* retina =new Retina(frameSizeRows, frameSizeColumns, RGBmode);
*
* // init gain, spatial and temporal parameters:
* retina->setParameters(0.7, 1, 0, 7, 1, 5, 0, 0, 3 , true);
*
* // during program execution, call the filter for local luminance correction,  contours extraction, moving contours extraction from an input picture called "FrameBuffer":
* retina->runfilter(FrameBuffer);
*
* // get the different output frames, check in the class description below for more outputs:
* const std::valarray<double> correctedLuminance=retina->getLocalAdaptation();
* const std::valarray<double> contours=retina->getContours();
* const std::valarray<double> movingContours=retina->getMovingContours();
*
* // at the end of the program, destroy object:
* delete retina;
*
* @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC / Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
* Creation date 2007
*/

#ifndef RETINACLASSES_H_
#define RETINACLASSES_H_

#include "basicretinafilter.hpp"
#include "parvoretinafilter.hpp"
#include "magnoretinafilter.hpp"

// optional includes (depending on the related publications)
#include "imagelogpolprojection.hpp"

#include "retinacolor.hpp"

//#define __RETINADEBUG // define RETINADEBUG to display debug data
namespace cv
{

// retina class that process the 3 outputs of the retina filtering stages
class RetinaFilter//: public BasicRetinaFilter
{
public:

	/**
	* constructor of the retina filter model with log sampling of the input frame (models the photoreceptors log sampling (central high resolution fovea and lower precision borders))
	* @param sizeRows: number of rows of the input image
	* @param sizeColumns: number of columns of the input image
	* @param colorMode: specifies if the retina works with color (true) of stays in grayscale processing (false), can be adjusted online by the use of setColorMode method
	* @param samplingMethod: specifies which kind of color sampling will be used
	* @param useRetinaLogSampling: activate retina log sampling, if true, the 2 following parameters can be used
	* @param reductionFactor: only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
	* @param samplingStrenght: only usefull if param useRetinaLogSampling=true, specifies the strenght of the log scale that is applied
	*/
	RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns, const bool colorMode=false, const RETINA_COLORSAMPLINGMETHOD samplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);

	/**
	* standard destructor
	*/
	~RetinaFilter();

	/**
	* function that clears all buffers of the object
	*/
	void clearAllBuffers();

	/**
	* resize retina parvo filter object (resize all allocated buffers)
	* @param NBrows: the new height size
	* @param NBcolumns: the new width size
	*/
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	* Input buffer checker: allows to check if the passed image buffer corresponds to retina filter expectations
	* @param input: the input image buffer
	* @param colorMode: specifiy if the input should be considered by the retina as colored of not
	* @return false if not compatible or it returns true if OK
	*/
	const bool checkInput(const std::valarray<double> &input, const bool colorMode);

	/**
	* run the initilized retina filter, after this call all retina outputs are updated
	* @param imageInput: image input buffer, can be grayscale or RGB image respecting the size specified at the constructor level
	* @param useAdaptiveFiltering: set true if you want to use adaptive color demultilexing (solve some color artefact problems), see RetinaColor for citation references
	* @param processRetinaParvoMagnoMapping: tels if the main outputs takes into account the mapping of the Parvo and Magno channels on the retina (centred parvo (fovea) and magno outside (parafovea))
	* @param useColorMode: color information is used if true, warning, if input is only gray level, a buffer overflow error will occur
	-> note that if color mode is activated and processRetinaParvoMagnoMapping==true, then the demultiplexed color frame (accessible throw getColorOutput() will be a color contours frame in the fovea and gray level moving contours outside
	@param inputIsColorMultiplexed: set trus if the input data is a multiplexed color image (using Bayer sampling for example), the color sampling method must correspond to the RETINA_COLORSAMPLINGMETHOD passed at constructor!
	* @return true if process ran well, false in case of failure
	*/
	const bool runFilter(const std::valarray<double> &imageInput, const bool useAdaptiveFiltering=true, const bool processRetinaParvoMagnoMapping=false, const bool useColorMode=false, const bool inputIsColorMultiplexed=false);

	/**
	* run the initilized retina filter in order to perform color tone mapping applied on an RGB image, after this call the color output of the retina is updated (use function getColorOutput() to grab it)
	* the algorithm is based on David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
	* -> Meylan L., Alleysson D., and S�sstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N� 9, September, 1st, 2007, pp. 2807-2816
	* get the resulting gray frame by calling function getParvoColor()
	* @param grayImageInput: RGB image input buffer respecting the size specified at the constructor level
	* @param PhotoreceptorsCompression: sets the log compression parameters applied at the photoreceptors level (enhance luminance in dark areas)
	* @param ganglionCellsCompression: sets the log compression applied at the gnaglion cells output (enhance contrast)
	*/
	void runGrayToneMapping(const std::valarray<double> &grayImageInput, std::valarray<double> &grayImageOutput, const double PhotoreceptorsCompression=0.6, const double ganglionCellsCompression=0.6);

	/**
	* run the initilized retina filter in order to perform color tone mapping applied on an RGB image, after this call the color output of the retina is updated (use function getColorOutput() to grab it)
	* the algorithm is based on David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
	* -> Meylan L., Alleysson D., and S�sstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N� 9, September, 1st, 2007, pp. 2807-2816
	* get the resulting RGB frame by calling function getParvoColor()
	* @param RGBimageInput: RGB image input buffer respecting the size specified at the constructor level
	* @param useAdaptiveFiltering: set true if you want to use adaptive color demultilexing (solve some color artefact problems), see RetinaColor for citation references
	* @param PhotoreceptorsCompression: sets the log compression parameters applied at the photoreceptors level (enhance luminance in dark areas)
	* @param ganglionCellsCompression: sets the log compression applied at the ganglion cells output (enhance contrast)
	*/
	void runRGBToneMapping(const std::valarray<double> &RGBimageInput, std::valarray<double> &imageOutput, const bool useAdaptiveFiltering, const double PhotoreceptorsCompression=0.6, const double ganglionCellsCompression=0.6);

	/**
	* run the initilized retina filter in order to perform color tone mapping applied on an RGB image, after this call the color output of the retina is updated (use function getColorOutput() to grab it)
	* get the resulting RGB frame by calling function getParvoColor()
	* @param LMSimageInput: RGB image input buffer respecting the size specified at the constructor level
	* @param useAdaptiveFiltering: set true if you want to use adaptive color demultilexing (solve some color artefact problems), see RetinaColor for citation references
	* @param PhotoreceptorsCompression: sets the log compression parameters applied at the photoreceptors level (enhance luminance in dark areas)
	* @param ganglionCellsCompression: sets the log compression applied at the gnaglion cells output (enhance contrast)
	*/
	void runLMSToneMapping(const std::valarray<double> &LMSimageInput, std::valarray<double> &imageOutput, const bool useAdaptiveFiltering, const double PhotoreceptorsCompression=0.6, const double ganglionCellsCompression=0.6);

	/**
	* set up function of the retina filter: all the retina is initialized at this step, some specific parameters are set by default, use setOPLandParvoCoefficientsTable() and setMagnoCoefficientsTable in order to setup the retina with more options
	* @param OPLspatialResponse1: (equal to k1 in setOPLandParvoCoefficientsTable() function) the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
	* @param OPLtemporalresponse1: (equal to tau1 in setOPLandParvoCoefficientsTable() function) the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
	* @param OPLassymetryGain: (equal to beta2 in setOPLandParvoCoefficientsTable() function) gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
	* @param OPLspatialResponse2: (equal to k2 in setOPLandParvoCoefficientsTable() function) the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel
	* @param OPLtemporalresponse2: (equal to tau2 in setOPLandParvoCoefficientsTable() function) the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
	* @param LPfilterSpatialResponse: (equal to parasolCells_k in setMagnoCoefficientsTable() function) the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
	* @param LPfilterGain: (equal to parasolCells_beta in setMagnoCoefficientsTable() function) the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
	* @param LPfilterTemporalresponse: (equal to parasolCells_tau in setMagnoCoefficientsTable() function) the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
	* @param MovingContoursExtractorCoefficient: (equal to amacrinCellsTemporalCutFrequency in setMagnoCoefficientsTable() function)the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, tipicall value is 5
	* @param normalizeParvoOutput_0_maxOutputValue: specifies if the Parvo cellular output should be normalized between 0 and maxOutputValue (true) or not (false) in order to remain at a null mean value, true value is recommended for visualisation
	* @param normalizeMagnoOutput_0_maxOutputValue: specifies if the Magno cellular output should be normalized between 0 and maxOutputValue (true) or not (false), setting true may be hazardous because it can enhace the noise response when nothing is moving
	* @param maxOutputValue: the maximum amplitude value of the normalized outputs (generally 255 for 8bit per channel pictures)
	* @param maxInputValue: the maximum pixel value of the input picture (generally 255 for 8bit per channel pictures), specify it in other case (for example High Dynamic Range Images)
	* @param meanValue: the global mean value of the input data usefull for local adaptation setup
	*/
	void setGlobalParameters(const double OPLspatialResponse1=0.7, const double OPLtemporalresponse1=1, const double OPLassymetryGain=0, const double OPLspatialResponse2=5, const double OPLtemporalresponse2=1, const double LPfilterSpatialResponse=5, const double LPfilterGain=0, const double LPfilterTemporalresponse=0, const double MovingContoursExtractorCoefficient=5, const bool normalizeParvoOutput_0_maxOutputValue=false, const bool normalizeMagnoOutput_0_maxOutputValue=false, const double maxOutputValue=255.0, const double maxInputValue=255.0, const double meanValue=128.0);

	/**
	* setup the local luminance adaptation capability
	* @param V0CompressionParameter: the compression strengh of the photoreceptors local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 160
	*/
	inline void setPhotoreceptorsLocalAdaptationSensitivity(const double V0CompressionParameter){_photoreceptorsPrefilter.setV0CompressionParameter(V0CompressionParameter);_setInitPeriodCount();};

	/**
	* setup the local luminance adaptation capability
	* @param V0CompressionParameter: the compression strengh of the parvocellular pathway (details) local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 160
	*/
	inline void setParvoGanglionCellsLocalAdaptationSensitivity(const double V0CompressionParameter){_ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);_setInitPeriodCount();};

	/**
	* setup the local luminance adaptation area of integration
	* @param spatialResponse: the spatial constant of the low pass filter applied on the bipolar cells output in order to compute local contrast mean values
	* @param temporalResponse: the spatial constant of the low pass filter applied on the bipolar cells output in order to compute local contrast mean values (generally set to zero: immediate response)
	*/
	inline void setGanglionCellsLocalAdaptationLPfilterParameters(const double spatialResponse, const double temporalResponse){_ParvoRetinaFilter.setGanglionCellsLocalAdaptationLPfilterParameters(temporalResponse, spatialResponse);_setInitPeriodCount();};

	/**
	* setup the local luminance adaptation capability
	* @param V0CompressionParameter: the compression strengh of the magnocellular pathway (motion) local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 160
	*/
	inline void setMagnoGanglionCellsLocalAdaptationSensitivity(const double V0CompressionParameter){_MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);_setInitPeriodCount();};

	/**
	* setup the OPL and IPL parvo channels
	* @param beta1: gain of the horizontal cells network, if 0, then the mean value of the output is zero (default value), if the parameter is near 1, the amplitude is boosted but it should only be used for values rescaling... if needed
	* @param tau1: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
	* @param k1: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
	* @param beta2: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
	* @param tau2: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
	* @param k2: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
	* @param V0CompressionParameter: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 230
	*/
	void setOPLandParvoParameters(const double beta1, const double tau1, const double k1, const double beta2, const double tau2, const double k2, const double V0CompressionParameter){_ParvoRetinaFilter.setOPLandParvoFiltersParameters(beta1, tau1, k1, beta2, tau2, k2);_ParvoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);_setInitPeriodCount();};

	/**
	* set parameters values for the Inner Plexiform Layer (IPL) magnocellular channel
	* @param parasolCells_beta: the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
	* @param parasolCells_tau: the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
	* @param parasolCells_k: the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
	* @param amacrinCellsTemporalCutFrequency: the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, tipicall value is 5
	* @param V0CompressionParameter: the compression strengh of the ganglion cells local adaptation output, set a value between 160 and 250 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 200
	* @param localAdaptintegration_tau: specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
	* @param localAdaptintegration_k: specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
	*/
	void setMagnoCoefficientsTable(const double parasolCells_beta, const double parasolCells_tau, const double parasolCells_k, const double amacrinCellsTemporalCutFrequency, const double V0CompressionParameter, const double localAdaptintegration_tau, const double localAdaptintegration_k){_MagnoRetinaFilter.setCoefficientsTable(parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, localAdaptintegration_tau, localAdaptintegration_k);_MagnoRetinaFilter.setV0CompressionParameter(V0CompressionParameter);_setInitPeriodCount();};

	/**
	* set if the parvo output should be or not normalized between 0 and 255 (for display purpose generally)
	* @param normalizeParvoOutput_0_maxOutputValue: true if normalization should be done
	*/
	inline void activateNormalizeParvoOutput_0_maxOutputValue(const bool normalizeParvoOutput_0_maxOutputValue){_normalizeParvoOutput_0_maxOutputValue=normalizeParvoOutput_0_maxOutputValue;};

	/**
	* set if the magno output should be or not normalized between 0 and 255 (for display purpose generally), take care, if nothing is moving, then, the noise will be enanced !!!
	* @param normalizeMagnoOutput_0_maxOutputValue: true if normalization should be done
	*/
	inline void activateNormalizeMagnoOutput_0_maxOutputValue(const bool normalizeMagnoOutput_0_maxOutputValue){_normalizeMagnoOutput_0_maxOutputValue=normalizeMagnoOutput_0_maxOutputValue;};

	/**
	* setup the maximum amplitude value of the normalized outputs (generally 255 for 8bit per channel pictures)
	* @param maxOutputValue: maximum amplitude value of the normalized outputs (generally 255 for 8bit per channel pictures)
	*/
	inline void setMaxOutputValue(const double maxOutputValue){_maxOutputValue=maxOutputValue;};

	/**
	* sets the color mode of the frame grabber
	* @param desiredColorMode: true if the user needs color information, false for graylevels
	*/
	void setColorMode(const bool desiredColorMode){_useColorMode=desiredColorMode;};

	/**
	* activate color saturation as the final step of the color demultiplexing process
	* -> this saturation is a sigmoide function applied to each channel of the demultiplexed image.
	* @param saturateColors: boolean that activates color saturation (if true) or desactivate (if false)
	* @param colorSaturationValue: the saturation factor
	* */
	inline void setColorSaturation(const bool saturateColors=true, const double colorSaturationValue=4.0){_colorEngine.setColorSaturation(saturateColors, colorSaturationValue);};

	/////////////////////////////////////////////////////////////////
	// function that retrieve the main retina outputs, one by one, or all in a structure

	/**
	* @return the input image sampled by the photoreceptors spatial sampling
	*/
	inline const std::valarray<double> &getPhotoreceptorsSampledFrame() const {if (_photoreceptorsLogSampling)return _photoreceptorsLogSampling->getSampledFrame();};

	/**
	* @return photoreceptors output, locally adapted luminance only, no high frequency spatio-temporal noise reduction at the next retina processing stages, use getPhotoreceptors method to get complete photoreceptors output
	*/
	inline const std::valarray<double> &getLocalAdaptation() const {return _photoreceptorsPrefilter.getOutput();};

	/**
	* @return photoreceptors output: locally adapted luminance and high frequency spatio-temporal noise reduction, high luminance is a little saturated at this stage, but this is corrected naturally at the next retina processing stages
	*/
	inline const std::valarray<double> &getPhotoreceptors() const {return _ParvoRetinaFilter.getPhotoreceptorsLPfilteringOutput();};

	/**
	* @return the local luminance of the processed frame (it is the horizontal cells output)
	*/
	inline const std::valarray<double> &getHorizontalCells() const {return _ParvoRetinaFilter.getHorizontalCellsOutput();};

	///////// CONTOURS part, PARVOCELLULAR RETINA PATHWAY
	/**
	* @return true if Parvocellular output is activated, false if not
	*/
	inline const bool areContoursProcessed(){return _useParvoOutput;};

	/**
	*  method to retrieve the foveal parvocellular pathway response (no details energy in parafovea)
	* @param parvoParafovealResponse: buffer that will be filled with the response of the magnocellular pathway in the parafoveal area
	* @return true if process succeeded (if buffer exists, is its size matches retina size, if magno channel is activated and if mapping is initialized
	*/
	const bool getParvoFoveaResponse(std::valarray<double> &parvoFovealResponse);

	/**
	* @param useParvoOutput: true if Parvocellular output should be activated, false if not
	*/
	inline void activateContoursProcessing(const bool useParvoOutput){_useParvoOutput=useParvoOutput;};

	/**
	* @return the parvocellular contours information (details), should be used at the fovea level
	*/
	const std::valarray<double> &getContours(); // Parvocellular output

	/**
	* @return the parvocellular contours ON information (details), should be used at the fovea level
	*/
	inline const std::valarray<double> &getContoursON() const {return _ParvoRetinaFilter.getParvoON();};// Parvocellular ON output

	/**
	* @return the parvocellular contours OFF information (details), should be used at the fovea level
	*/
	inline const std::valarray<double> &getContoursOFF() const {return _ParvoRetinaFilter.getParvoOFF();};// Parvocellular OFF output

	///////// MOVING CONTOURS part, MAGNOCELLULAR RETINA PATHWAY
	/**
	* @return true if Magnocellular output is activated, false if not
	*/
	inline const bool areMovingContoursProcessed(){return _useMagnoOutput;};

	/**
	*  method to retrieve the parafoveal magnocellular pathway response (no motion energy in fovea)
	* @param magnoParafovealResponse: buffer that will be filled with the response of the magnocellular pathway in the parafoveal area
	* @return true if process succeeded (if buffer exists, is its size matches retina size, if magno channel is activated and if mapping is initialized
	*/
	const bool getMagnoParaFoveaResponse(std::valarray<double> &magnoParafovealResponse);

	/**
	* @param useMagnoOutput: true if Magnoocellular output should be activated, false if not
	*/
	inline void activateMovingContoursProcessing(const bool useMagnoOutput){_useMagnoOutput=useMagnoOutput;};

	/**
	* @return the magnocellular moving contours information (motion), should be used at the parafovea level without post-processing
	*/
	inline const std::valarray<double> &getMovingContours() const {return _MagnoRetinaFilter.getOutput();};// Magnocellular output

	/**
	* @return the magnocellular moving contours information (motion), should be used at the parafovea level with assymetric sigmoide post-processing which saturates motion information
	*/
	inline const std::valarray<double> &getMovingContoursSaturated() const {return _MagnoRetinaFilter.getMagnoYsaturated();};// Saturated Magnocellular output

	/**
	* @return the magnocellular moving contours ON information (motion), should be used at the parafovea level without post-processing
	*/
	inline const std::valarray<double> &getMovingContoursON() const {return _MagnoRetinaFilter.getMagnoON();};// Magnocellular ON output

	/**
	* @return the magnocellular moving contours OFF information (motion), should be used at the parafovea level without post-processing
	*/
	inline const std::valarray<double> &getMovingContoursOFF() const {return _MagnoRetinaFilter.getMagnoOFF();};// Magnocellular OFF output

	/**
	* @return a gray level image with center Parvo and peripheral Magno X channels, WARNING, the result will be ok if you called previously fucntion runFilter(imageInput, processRetinaParvoMagnoMapping=true);
	*    -> will be accessible even if color mode is activated (but the image is color sampled so quality is poor), but get the same thing but in color by the use of function getParvoColor()
	*/
	inline const std::valarray<double> &getRetinaParvoMagnoMappedOutput() const {return _retinaParvoMagnoMappedFrame;};// return image with center Parvo and peripheral Magno channels

	/**
	* color processing dedicated functions
	* @return the parvo channel (contours, details) of the processed frame, grayscale output
	*/
	inline const std::valarray<double> &getParvoContoursChannel() const {return _colorEngine.getLuminance();};

	/**
	* color processing dedicated functions
	* @return the chrominance of the processed frame (same colorspace as the input output, usually RGB)
	*/
	inline const std::valarray<double> &getParvoChrominance() const {return _colorEngine.getChrominance();}; // only retreive chrominance

	/**
	* color processing dedicated functions
	* @return the parvo + chrominance channels of the processed frame (same colorspace as the input output, usually RGB)
	*/
	inline const std::valarray<double> &getColorOutput() const {return _colorEngine.getDemultiplexedColorFrame();};// retrieve luminance+chrominance

	/**
	* apply to the retina color output the Krauskopf transformation which leads to an opponent color system: output colorspace if Acr1cr2 if input of the retina was LMS color space
	* @param result: the input buffer to fill with the transformed colorspace retina output
	* @return true if process ended successfully
	*/
	inline const bool applyKrauskopfLMS2Acr1cr2Transform(std::valarray<double> &result){return _colorEngine.applyKrauskopfLMS2Acr1cr2Transform(result);};

	/**
	* apply to the retina color output the Krauskopf transformation which leads to an opponent color system: output colorspace if Acr1cr2 if input of the retina was LMS color space
	* @param result: the input buffer to fill with the transformed colorspace retina output
	* @return true if process ended successfully
	*/
	inline const bool applyLMS2LabTransform(std::valarray<double> &result){return _colorEngine.applyLMS2LabTransform(result);};

	/**
	* color processing dedicated functions
	* @return the retina initialized mode, true if color mode (RGB), false if grayscale
	*/
	inline const bool isColorMode(){return _useColorMode;}; // return true if RGB mode, false if gray level mode

	/**
	* @return the irregular low pass filter ouput at the photoreceptors level
	*/
	inline const std::valarray<double> &getIrregularLPfilteredInputFrame() const {return _photoreceptorsLogSampling->getIrregularLPfilteredInputFrame();};

	/**
	* @return true if color mode is activated, false if gray levels processing
	*/
	const bool getColorMode(){return _useColorMode;};

	/**
	*
	* @return true if a sufficient number of processed frames has been done since the last parameters update in order to get the stable state (r�gime permanent)
	*/
	inline const bool isInitTransitionDone(){if (_ellapsedFramesSinceLastReset<_globalTemporalConstant)return false; return true;};

	/**
	* find a distance in the image input space when the distance is known in the retina log sampled space...read again if it is not clear enough....sorry, i should sleep
	* @param projectedRadiusLength: the distance to image center in the retina log sampled space
	* @return the distance to image center in the input image space
	*/
	inline const double getRetinaSamplingBackProjection(const double projectedRadiusLength){if (_photoreceptorsLogSampling)return _photoreceptorsLogSampling->getOriginalRadiusLength(projectedRadiusLength);else return projectedRadiusLength;};

	/////////////////:
	// retina dimensions getters

	/**
	* @return number of rows of the filter
	*/
	inline const unsigned int getInputNBrows(){if (_photoreceptorsLogSampling) return _photoreceptorsLogSampling->getNBrows();else return _photoreceptorsPrefilter.getNBrows();};

	/**
	* @return number of columns of the filter
	*/
	inline const unsigned int getInputNBcolumns(){if (_photoreceptorsLogSampling) return _photoreceptorsLogSampling->getNBcolumns();else return _photoreceptorsPrefilter.getNBcolumns();};

	/**
	* @return number of pixels of the filter
	*/
	inline const unsigned int getInputNBpixels(){if (_photoreceptorsLogSampling) return _photoreceptorsLogSampling->getNBpixels();else return _photoreceptorsPrefilter.getNBpixels();};

	/**
	* @return the height of the frame output
	*/
	inline const unsigned int getOutputNBrows(){return _photoreceptorsPrefilter.getNBrows();};

	/**
	* @return the width of the frame output
	*/
	inline const unsigned int getOutputNBcolumns(){return _photoreceptorsPrefilter.getNBcolumns();};

	/**
	* @return the numbers of output pixels (width*height) of the images used by the object
	*/
	inline const unsigned int getOutputNBpixels(){return _photoreceptorsPrefilter.getNBpixels();};


private:

	// processing activation flags
	bool _useParvoOutput;
	bool _useMagnoOutput;


	// filter stability controls
	unsigned int _ellapsedFramesSinceLastReset;
	unsigned int _globalTemporalConstant;

	// private template buffers and related access pointers
	std::valarray<double> _retinaParvoMagnoMappedFrame;
	std::valarray<double> _retinaParvoMagnoMapCoefTable;
	// private objects of the class
	BasicRetinaFilter _photoreceptorsPrefilter;
	ParvoRetinaFilter _ParvoRetinaFilter;
	MagnoRetinaFilter _MagnoRetinaFilter;
	RetinaColor       _colorEngine;
	ImageLogPolProjection *_photoreceptorsLogSampling;

	bool _useMinimalMemoryForToneMappingONLY;

	bool _normalizeParvoOutput_0_maxOutputValue;
	bool _normalizeMagnoOutput_0_maxOutputValue;
	double _maxOutputValue;
	bool _useColorMode;



	// private functions
	void _setInitPeriodCount();
	void _createHybridTable();
	void _processRetinaParvoMagnoMapping();
	void _runGrayToneMapping(const std::valarray<double> &grayImageInput, std::valarray<double> &grayImageOutput ,const double PhotoreceptorsCompression=0.6, const double ganglionCellsCompression=0.6);


};

}
#endif /*RETINACLASSES_H_*/




