
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
 **
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

#ifndef __OPENCV_BIOINSPIRED_RETINAFASTTONEMAPPING_HPP__
#define __OPENCV_BIOINSPIRED_RETINAFASTTONEMAPPING_HPP__

/*
 * retinafasttonemapping.hpp
 *
 *  Created on: May 26, 2013
 *      Author: Alexandre Benoit
 */

#include "opencv2/core.hpp" // for all OpenCV core functionalities access, including cv::Exception support

namespace cv{
namespace bioinspired{

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
class CV_EXPORTS RetinaFastToneMapping : public Algorithm
{
public:

    /**
     * method that applies a luminance correction (initially High Dynamic Range (HDR) tone mapping) using only the 2 local adaptation stages of the retina parvocellular channel : photoreceptors level and ganlion cells level. Spatio temporal filtering is applied but limited to temporal smoothing and eventually high frequencies attenuation. This is a lighter method than the one available using the regular retina::run method. It is then faster but it does not include complete temporal filtering nor retina spectral whitening. Then, it can have a more limited effect on images with a very high dynamic range. This is an adptation of the original still image HDR tone mapping algorithm of David Alleyson, Sabine Susstruck and Laurence Meylan's work, please cite:
    * -> Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816
     @param inputImage the input image to process RGB or gray levels
     @param outputToneMappedImage the output tone mapped image
     */
    virtual void applyFastToneMapping(InputArray inputImage, OutputArray outputToneMappedImage)=0;

    /**
     * setup method that updates tone mapping behaviors by adjusing the local luminance computation area
     * @param photoreceptorsNeighborhoodRadius the first stage local adaptation area
     * @param ganglioncellsNeighborhoodRadius the second stage local adaptation area
     * @param meanLuminanceModulatorK the factor applied to modulate the meanLuminance information (default is 1, see reference paper)
     */
    virtual void setup(const float photoreceptorsNeighborhoodRadius=3.f, const float ganglioncellsNeighborhoodRadius=1.f, const float meanLuminanceModulatorK=1.f)=0;
};

CV_EXPORTS Ptr<RetinaFastToneMapping> createRetinaFastToneMapping(Size inputSize);

}
}
#endif /* __OPENCV_BIOINSPIRED_RETINAFASTTONEMAPPING_HPP__ */
