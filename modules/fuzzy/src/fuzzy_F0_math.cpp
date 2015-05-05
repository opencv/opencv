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

#include "precomp.hpp"

using namespace cv;

void ft::FT02D_components(InputArray matrix, InputArray kernel, OutputArray components, InputArray mask)
{
    Mat matrixMat = matrix.getMat();
    Mat kernelMat = kernel.getMat();
    Mat maskMat = mask.getMat();

    CV_Assert(matrixMat.channels() == 1 && kernelMat.channels() == 1 && maskMat.channels() == 1);

    int radiusX = (kernelMat.cols - 1) / 2;
    int radiusY = (kernelMat.rows - 1) / 2;
    int An = matrixMat.cols / radiusX + 1;
    int Bn = matrixMat.rows / radiusY + 1;

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrixMat, matrixPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(maskMat, maskPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));

    components.create(Bn, An, CV_32F);
    Mat componentsMat = components.getMat();

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernelMat.cols, kernelMat.rows);

            Mat roiImage(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernelMat.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiImage, kernelMasked, numerator, 1, CV_32F);

            componentsMat.row(o).col(i) = sum(numerator) / sum(kernelMasked);
        }
    }
}

void ft::FT02D_components(InputArray matrix, InputArray kernel, OutputArray components)
{
    Mat mask = Mat::ones(matrix.size(), CV_8U);

    ft::FT02D_components(matrix, kernel, components, mask);
}

void ft::FT02D_inverseFT(InputArray components, InputArray kernel, OutputArray output, int width, int height)
{
    Mat componentsMat = components.getMat();
    Mat kernelMat = kernel.getMat();

    CV_Assert(componentsMat.channels() == 1 && kernelMat.channels() == 1);

    int radiusX = (kernelMat.cols - 1) / 2;
    int radiusY = (kernelMat.rows - 1) / 2;
    int paddedOutputWidth = radiusX + width + kernelMat.cols;
    int paddedOutputHeight = radiusY + height + kernelMat.rows;

    output.create(height, width, CV_32F);

    Mat outputZeroes(paddedOutputHeight, paddedOutputWidth, CV_32F, Scalar(0));

    for (int i = 0; i < componentsMat.cols; i++)
    {
        for (int o = 0; o < componentsMat.rows; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernelMat.cols, kernelMat.rows);

            Mat roiOutput(outputZeroes, area);
            roiOutput += kernelMat.mul(componentsMat.at<float>(o,i));
        }
    }

    outputZeroes(Rect(radiusX, radiusY, width, height)).copyTo(output);
}

void ft::FT02D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask)
{
    CV_Assert(image.channels() == kernel.channels());

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = image.cols / radiusX + 1;
    int Bn = image.rows / radiusY + 1;
    int outputWidthPadded = radiusX + image.cols + kernel.cols;
    int outputHeightPadded = radiusY + image.rows + kernel.rows;

    Mat imagePadded;
    Mat maskPadded;

    output = Mat::zeros(outputHeightPadded, outputWidthPadded, CV_MAKETYPE(CV_32F, image.channels()));

    copyMakeBorder(image, imagePadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiImage(imagePadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiImage, kernelMasked, numerator, 1, CV_32F);

            Scalar component;
            divide(sum(numerator), sum(kernelMasked), component, 1, CV_32F);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(output, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    output = output(Rect(radiusX, radiusY, image.cols, image.rows));
}

int ft::FT02D_iteration(const Mat &image, const Mat &kernel, Mat &imageOutput, const Mat &mask, Mat &maskOutput, bool firstStop)
{
    CV_Assert(image.channels() == kernel.channels() && mask.channels() == 1);

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = image.cols / radiusX + 1;
    int Bn = image.rows / radiusY + 1;
    int outputWidthPadded = radiusX + image.cols + kernel.cols;
    int outputHeightPadded = radiusY + image.rows + kernel.rows;
    int undefinedComponents = 0;

    Mat imagePadded;
    Mat maskPadded;

    imageOutput = Mat::zeros(outputHeightPadded, outputWidthPadded, CV_MAKETYPE(CV_32F, image.channels()));
    maskOutput = Mat::ones(outputHeightPadded, outputWidthPadded, CV_8UC1);

    copyMakeBorder(image, imagePadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiImage(imagePadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiImage, kernelMasked, numerator, 1, CV_32F);

            Scalar denominator = sum(kernelMasked);

            if (denominator[0] == 0)
            {
                if (firstStop)
                {
                    imageOutput = imageOutput(Rect(radiusX, radiusY, image.cols, image.rows));
                    maskOutput = maskPadded(Rect(radiusX, radiusY, image.cols, image.rows));

                    return -1;
                }
                else
                {
                    undefinedComponents++;

                    Mat roiMaskOutput(maskOutput, Rect(centerX - radiusX + 1, centerY - radiusY + 1, kernel.cols - 2, kernel.rows - 2));
                    roiMaskOutput = 0;

                    continue;
                }
            }

            Scalar component;
            divide(sum(numerator), denominator, component, 1, CV_32F);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiImageOutput(imageOutput, area);
            add(roiImageOutput, inverse, roiImageOutput);
        }
    }

    imageOutput = imageOutput(Rect(radiusX, radiusY, image.cols, image.rows));
    maskOutput = maskOutput(Rect(radiusX, radiusY, image.cols, image.rows));

    return undefinedComponents;
}
