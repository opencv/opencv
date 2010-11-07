/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "cvtest.h"

CvTS test_system("cv");

const char* blacklist[] =
{
    //"_3d-fundam",                     //ticket 484
    
    "calibrate-camera-artificial",    //ticket 472
    //"chessboard-detector",            //ticket 569
    //"chessboard-subpixel",            //ticket 473
    //"color-luv",                      //ticket 502

    //"filter-generic",
    //"flann_saved",                    //ticket 618

    "inpaint",                        //ticket 570

    //"mhi-global",                     //ticket 457
    //"morph-ex",                       //ticket 612
    //"MSER",                           //ticket 437
    //"operations",                     //ticket 613
    //"optflow-estimate-rigid",         //ticket 433
    //"posit",                          //ticket 430

    //"segmentation-pyramid",           //ticket 464
    //"shape-minarearect",              //ticket 436
    //"stereogc",                       //ticket 439
    //"subdiv",                         //ticket 454

    //"track-camshift",                 //ticket 483

    //"warp-affine",                    //ticket 572
    //"warp-perspective",               //ticket 575
    //"warp-remap",                     //ticket 576
    "warp-resize",                    //ticket 429
    //"warp-undistort",                 //ticket 577

    //"hist-backproj",                  //ticket 579
    //"projectPoints-c",                //ticket 652
    0
};

int main(int argC,char *argV[])
{
    return test_system.run( argC, argV, blacklist );
}

/* End of file. */
