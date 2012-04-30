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

namespace cv
{
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////

static Algorithm* createMOG()
{
    return new BackgroundSubtractorMOG;
}

static AlgorithmInfo& mog_info()
{
    static AlgorithmInfo mog_info_var("BackgroundSubtractor.MOG", createMOG);
    return mog_info_var;
}

static AlgorithmInfo& mog_info_auto = mog_info();

AlgorithmInfo* BackgroundSubtractorMOG::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        BackgroundSubtractorMOG obj;
        
        mog_info().addParam(obj, "history", obj.history);
        mog_info().addParam(obj, "nmixtures", obj.nmixtures);
        mog_info().addParam(obj, "backgroundRatio", obj.backgroundRatio);
        mog_info().addParam(obj, "noiseSigma", obj.noiseSigma);
        
        initialized = true;
    }
    return &mog_info();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

static Algorithm* createMOG2()
{
    return new BackgroundSubtractorMOG2;
}

static AlgorithmInfo& mog2_info()
{
    static AlgorithmInfo mog2_info_var("BackgroundSubtractor.MOG2", createMOG2);
    return mog2_info_var;
}

static AlgorithmInfo& mog2_info_auto = mog2_info();

AlgorithmInfo* BackgroundSubtractorMOG2::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        BackgroundSubtractorMOG2 obj;
        
        mog2_info().addParam(obj, "history", obj.history);
        mog2_info().addParam(obj, "varThreshold", obj.varThreshold);
        mog2_info().addParam(obj, "detectShadows", obj.bShadowDetection);
        
        initialized = true;
    }
    return &mog2_info();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////    

bool initModule_video(void)
{
    Ptr<Algorithm> mog = createMOG(), mog2 = createMOG2();
    return mog->info() != 0 && mog2->info() != 0;
}
    
}
