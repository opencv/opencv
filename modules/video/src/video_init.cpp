/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
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
#include "opencv2/video/video.hpp"

namespace cv
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(BackgroundSubtractorMOG, "BackgroundSubtractor.MOG",
    obj.info()->addParam(obj, "history", obj.history);
    obj.info()->addParam(obj, "nmixtures", obj.nmixtures);
    obj.info()->addParam(obj, "backgroundRatio", obj.backgroundRatio);
    obj.info()->addParam(obj, "noiseSigma", obj.noiseSigma))

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(BackgroundSubtractorMOG2, "BackgroundSubtractor.MOG2",
    obj.info()->addParam(obj, "history", obj.history);
    obj.info()->addParam(obj, "nmixtures", obj.nmixtures);
    obj.info()->addParam(obj, "varThreshold", obj.varThreshold);
    obj.info()->addParam(obj, "detectShadows", obj.bShadowDetection);
    obj.info()->addParam(obj, "backgroundRatio", obj.backgroundRatio);
    obj.info()->addParam(obj, "varThresholdGen", obj.varThresholdGen);
    obj.info()->addParam(obj, "fVarInit", obj.fVarInit);
    obj.info()->addParam(obj, "fVarMin", obj.fVarMin);
    obj.info()->addParam(obj, "fVarMax", obj.fVarMax);
    obj.info()->addParam(obj, "fCT", obj.fCT);
    obj.info()->addParam(obj, "nShadowDetection", obj.nShadowDetection);
    obj.info()->addParam(obj, "fTau", obj.fTau))

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(BackgroundSubtractorGMG, "BackgroundSubtractor.GMG",
                  obj.info()->addParam(obj, "maxFeatures", obj.maxFeatures,false,0,0,
                                       "Maximum number of features to store in histogram. Harsh enforcement of sparsity constraint.");
                  obj.info()->addParam(obj, "learningRate", obj.learningRate,false,0,0,
                                       "Adaptation rate of histogram. Close to 1, slow adaptation. Close to 0, fast adaptation, features forgotten quickly.");
                  obj.info()->addParam(obj, "initializationFrames", obj.numInitializationFrames,false,0,0,
                                       "Number of frames to use to initialize histograms of pixels.");
                  obj.info()->addParam(obj, "quantizationLevels", obj.quantizationLevels,false,0,0,
                                       "Number of discrete colors to be used in histograms. Up-front quantization.");
                  obj.info()->addParam(obj, "backgroundPrior", obj.backgroundPrior,false,0,0,
                                       "Prior probability that each individual pixel is a background pixel.");
                  obj.info()->addParam(obj, "smoothingRadius", obj.smoothingRadius,false,0,0,
                                       "Radius of smoothing kernel to filter noise from FG mask image.");
                  obj.info()->addParam(obj, "decisionThreshold", obj.decisionThreshold,false,0,0,
                                       "Threshold for FG decision rule. Pixel is FG if posterior probability exceeds threshold.");
                  obj.info()->addParam(obj, "updateBackgroundModel", obj.updateBackgroundModel,false,0,0,
                                       "Perform background model update."))

bool initModule_video(void)
{
    bool all = true;
    all &= !BackgroundSubtractorMOG_info_auto.name().empty();
    all &= !BackgroundSubtractorMOG2_info_auto.name().empty();
    all &= !BackgroundSubtractorGMG_info_auto.name().empty();

    return all;
}

}
