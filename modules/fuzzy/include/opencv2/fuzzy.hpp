/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_FUZZY_H__
#define __OPENCV_FUZZY_H__

#include "opencv2/core.hpp"

namespace cv
{

namespace ft
{
    enum
    {
        LINEAR = 1,
        SINUS = 2
    };

    enum
    {
        ONE_STEP = 1,
        MULTI_STEP = 2,
        ITERATIVE = 3
    };

    /** @brief Computes components of the array using direct F0-transform.
    @param matrix Input 1-channel array.
    @param kernel Kernel used for processing. Function **createKernel** can be used.
    @param components Output 32-bit array for the components.
    @param mask Mask can be used for unwanted area marking.

    The function computes components using predefined kernel and mask.

    @note
        F-transform technique is described in paper @cite Perf:FT.
     */
    CV_EXPORTS void FT02D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components, cv::InputArray mask);

    /** @brief Computes components of the array using direct F0-transform.
    @param matrix Input 1-channel array.
    @param kernel Kernel used for processing. Function **createKernel** can be used.
    @param components Output 32-bit array for the components.

    The function computes components using predefined kernel.

    @note
        F-transform technique is described in paper @cite Perf:FT.
     */
    CV_EXPORTS void FT02D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components);

    /** @brief Computes inverse F0-transfrom.
    @param components Input 32-bit array for the components.
    @param kernel Kernel used for processing. Function **createKernel** can be used.
    @param output Output 32-bit array.
    @param width Width of the output array.
    @param height Height of the output array.

    @note
        F-transform technique is described in paper @cite Perf:FT.
     */
    CV_EXPORTS void FT02D_inverseFT(cv::InputArray components, cv::InputArray kernel, cv::OutputArray output, int width, int height);

    /** @brief Computes F0-transfrom and inverse F0-transfrom at once.
    @param image Input image.
    @param kernel Kernel used for processing. Function **createKernel** can be used.
    @param output Output 32-bit array.
    @param mask Mask used for unwanted area marking.

    This function computes F-transfrom and inverse F-transfotm in one step. It is fully sufficient and optimized for **cv::Mat**.
    */
    CV_EXPORTS void FT02D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask);

    /** @brief Computes F0-transfrom and inverse F0-transfrom at once and return state.
    @param image Input image.
    @param kernel Kernel used for processing. Function **createKernel** can be used.
    @param imageOutput Output 32-bit array.
    @param mask Mask used for unwanted area marking.
    @param maskOutput Mask after one iteration.
    @param firstStop If **true** function returns -1 when first problem appears. In case of **false**, the process is completed and summation of all problems returned.

    This function computes iteration of F-transfrom and inverse F-transfotm and handle image and mask change. The function is used in *inpaint* function.
    */
    CV_EXPORTS int FT02D_iteration(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &imageOutput, const cv::Mat &mask, cv::Mat &maskOutput, bool firstStop = true);

    /** @brief Creates kernel from basic functions.
    @param A Basic function used in axis **x**.
    @param B Basic function used in axis **y**.
    @param kernel Final 32-b kernel derived from **A** and **B**.
    @param chn Number of kernel channels.

    The function creates kernel usable for latter fuzzy image processing.
    */
    CV_EXPORTS void createKernel(cv::InputArray A, cv::InputArray B, cv::OutputArray kernel, const int chn = 1);

    /** @brief Creates kernel from general functions.
    @param function Function type could be one of the following:
        -   **LINEAR** Linear basic function.
    @param radius Radius of the basic function.
    @param kernel Final 32-b kernel.
    @param chn Number of kernel channels.

    The function creates kernel from predefined functions.
    */
    CV_EXPORTS void createKernel(int function, int radius, cv::OutputArray kernel, const int chn = 1);

    /** @brief Image inpainting
    @param image Input image.
    @param mask Mask used for unwanted area marking.
    @param output Output 32-bit image.
    @param radius Radius of the basic function.
    @param function Function type could be one of the following:
        -   **LINEAR** Linear basic function.
    @param algorithm Algorithm could be one of the following:
        -   **ONE_STEP** One step algorithm.
        -   **MULTI_STEP** Algorithm automaticaly increasing radius of the basic function.
        -   **ITERATIVE** Iterative algorithm running in more steps using partial computations.

    This function provides inpainting technique based on the fuzzy mathematic.

    @note
        The algorithms are described in paper @cite Perf:rec.
    */
    CV_EXPORTS void inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius = 2, int function = ft::LINEAR, int algorithm = ft::ONE_STEP);

    /** @brief Image filtering
    @param image Input image.
    @param kernel Final 32-b kernel.
    @param output Output 32-bit image.

    Filtering of the input image by means of F-transform.
    */
    CV_EXPORTS void filter(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output);
}

}

#endif // __OPENCV_FUZZY_H__
