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

#ifndef __OPENCV_CONTRIB_RETINA_HPP__
#define __OPENCV_CONTRIB_RETINA_HPP__

/*
 * Retina.hpp
 *
 *  Created on: Jul 19, 2011
 *      Author: Alexandre Benoit
 */

#include "opencv2/core/core.hpp" // for all OpenCV core functionalities access, including cv::Exception support
#include <valarray>

namespace cv
{
    
    enum RETINA_COLORSAMPLINGMETHOD
    {
        RETINA_COLOR_RANDOM, /// each pixel position is either R, G or B in a random choice
        RETINA_COLOR_DIAGONAL,/// color sampling is RGBRGBRGB..., line 2 BRGBRGBRG..., line 3, GBRGBRGBR...
        RETINA_COLOR_BAYER/// standard bayer sampling
    };
    
    class RetinaFilter;
    
    /**
     * @brief a wrapper class which allows the use of the Gipsa/Listic Labs retina model
     * @class Retina object is a wrapper class which allows the Gipsa/Listic Labs model to be used.
     * This retina model allows spatio-temporal image processing (applied on still images, video sequences).
     * As a summary, these are the retina model properties:
     * => It applies a spectral whithening (mid-frequency details enhancement)
     * => high frequency spatio-temporal noise reduction
     * => low frequency luminance to be reduced (luminance range compression)
     * => local logarithmic luminance compression allows details to be enhanced in low light conditions
     *
     * for more information, reer to the following papers :
     * Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
     * Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.
     */
    class CV_EXPORTS Retina {
        
    public:
        
        /**
         * Main constructor with most commun use setup : create an instance of color ready retina model
         * @param inputSize : the input frame size
         */
        Retina(const std::string parametersSaveFile, Size inputSize);
        
        /**
         * Complete Retina filter constructor which allows all basic structural parameters definition
         * @param inputSize : the input frame size
         * @param colorMode : the chosen processing mode : with or without color processing
         * @param samplingMethod: specifies which kind of color sampling will be used
         * @param useRetinaLogSampling: activate retina log sampling, if true, the 2 following parameters can be used
         * @param reductionFactor: only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
         * @param samplingStrenght: only usefull if param useRetinaLogSampling=true, specifies the strenght of the log scale that is applied
         */
        Retina(const std::string parametersSaveFile, Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);
        
        virtual ~Retina();
        
        /**
         * try to open an XML retina parameters file to adjust current retina instance setup
         * => if the xml file does not exist, then default setup is applied
         * => warning, Exceptions are thrown if read XML file is not valid
         * @param retinaParameterFile : the parameters filename
         */
        void setup(std::string retinaParameterFile="", const bool applyDefaultSetupOnFailure=true);
        
        /**
         * parameters setup display method
         * @return a string which contains formatted parameters information
         */
        const std::string printSetup();
        
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
        void setupOPLandIPLParvoChannel(const bool colorMode=true, const bool normaliseOutput = true, const double photoreceptorsLocalAdaptationSensitivity=0.7, const double photoreceptorsTemporalConstant=0.5, const double photoreceptorsSpatialConstant=0.53, const double horizontalCellsGain=0, const double HcellsTemporalConstant=1, const double HcellsSpatialConstant=7, const double ganglionCellsSensitivity=0.7);
        
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
        void setupIPLMagnoChannel(const bool normaliseOutput = true, const double parasolCells_beta=0, const double parasolCells_tau=0, const double parasolCells_k=7, const double amacrinCellsTemporalCutFrequency=1.2, const double V0CompressionParameter=0.95, const double localAdaptintegration_tau=0, const double localAdaptintegration_k=7);
        /**
         * method which allows retina to be applied on an input image
         * @param
         * 		/// encapsulated retina module is ready to deliver its outputs using dedicated acccessors, see getParvo and getMagno methods
         *
         */
        void run(const Mat &inputImage);
        /**
         * accessor of the details channel of the retina (models foveal vision)
         * @param retinaOutput_parvo : the output buffer (reallocated if necessary)
         */
        void getParvo(Mat &retinaOutput_parvo);
        
        /**
         * accessor of the motion channel of the retina (models peripheral vision)
         * @param retinaOutput_magno : the output buffer (reallocated if necessary)
         */
        void getMagno(Mat &retinaOutput_magno);
        
        void clearBuffers();
        
    protected:
        //// Parameteres setup members
        // parameters file ... saved on instance delete
        FileStorage _parametersSaveFile;
        std::string _parametersSaveFileName;
        //// Retina model related modules
        // buffer that ensure library cross-compatibility
        std::valarray<double> _inputBuffer;
        
        // pointer to retina model
        RetinaFilter* _retinaFilter;
        
        /**
         * exports a valarray buffer outing from HVStools objects to a cv::Mat in CV_8UC1 (gray level picture) or CV_8UC3 (color) format
         * @param grayMatrixToConvert the valarray to export to OpenCV
         * @param nbRows : the number of rows of the valarray flatten matrix
         * @param nbColumns : the number of rows of the valarray flatten matrix
         * @param colorMode : a flag which mentions if matrix is color (true) or graylevel (false)
         * @param outBuffer : the output matrix which is reallocated to satisfy Retina output buffer dimensions
         */
        void _convertValarrayGrayBuffer2cvMat(const std::valarray<double> &grayMatrixToConvert, const unsigned int nbRows, const unsigned int nbColumns, const bool colorMode, Mat &outBuffer);
        
        // private method called by constructirs
        void _init(const std::string parametersSaveFile, Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);
        
        
    };
    
}
#endif /* __OPENCV_CONTRIB_RETINA_HPP__ */

