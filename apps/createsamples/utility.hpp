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

#ifndef __CREATESAMPLES_UTILITY_HPP__
#define __CREATESAMPLES_UTILITY_HPP__

#define CV_VERBOSE  1

/*
 * cvCreateTrainingSamples
 *
 * Create training samples applying random distortions to sample image and
 * store them in .vec file
 *
 * filename        - .vec file name
 * imgfilename     - sample image file name
 * bgcolor         - background color for sample image
 * bgthreshold     - background color threshold. Pixels those colors are in range
 *   [bgcolor-bgthreshold, bgcolor+bgthreshold] are considered as transparent
 * bgfilename      - background description file name. If not NULL samples
 *   will be put on arbitrary background
 * count           - desired number of samples
 * invert          - if not 0 sample foreground pixels will be inverted
 *   if invert == CV_RANDOM_INVERT then samples will be inverted randomly
 * maxintensitydev - desired max intensity deviation of foreground samples pixels
 * maxxangle       - max rotation angles
 * maxyangle
 * maxzangle
 * showsamples     - if not 0 samples will be shown
 * winwidth        - desired samples width
 * winheight       - desired samples height
 */
#define CV_RANDOM_INVERT 0x7FFFFFFF

void cvCreateTrainingSamples( const char* filename,
                              const char* imgfilename, int bgcolor, int bgthreshold,
                              const char* bgfilename, int count,
                              int invert = 0, int maxintensitydev = 40,
                              double maxxangle = 1.1,
                              double maxyangle = 1.1,
                              double maxzangle = 0.5,
                              int showsamples = 0,
                              int winwidth = 24, int winheight = 24 );

void cvCreateTestSamples( const char* infoname,
                          const char* imgfilename, int bgcolor, int bgthreshold,
                          const char* bgfilename, int count,
                          int invert, int maxintensitydev,
                          double maxxangle, double maxyangle, double maxzangle,
                          int showsamples,
                          int winwidth, int winheight );

/*
 * cvCreateTrainingSamplesFromInfo
 *
 * Create training samples from a set of marked up images and store them into .vec file
 * infoname    - file in which marked up image descriptions are stored
 * num         - desired number of samples
 * showsamples - if not 0 samples will be shown
 * winwidth    - sample width
 * winheight   - sample height
 *
 * Return number of successfully created samples
 */
int cvCreateTrainingSamplesFromInfo( const char* infoname, const char* vecfilename,
                                     int num,
                                     int showsamples,
                                     int winwidth, int winheight );

/*
 * cvShowVecSamples
 *
 * Shows samples stored in .vec file
 *
 * filename
 *   .vec file name
 * winwidth
 *   sample width
 * winheight
 *   sample height
 * scale
 *   the scale each sample is adjusted to
 */
void cvShowVecSamples( const char* filename, int winwidth, int winheight, double scale );

#endif //__CREATESAMPLES_UTILITY_HPP__
