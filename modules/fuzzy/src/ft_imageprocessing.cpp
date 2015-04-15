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
// Copyright (C) 2015, Pavel Vlasanek, all rights reserved.
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

using namespace cv;

void ft::createKernel(InputArray _A, InputArray _B, OutputArray _kernel, const int chn)
{
    Mat A = _A.getMat();
    Mat B = _B.getMat();
    Mat kernelOneChannel = A * B;
    std::vector<Mat> channels;

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(kernelOneChannel);
    }

    merge(channels, _kernel);
}

void ft::createKernel(int function, int radius, cv::OutputArray _kernel, const int chn)
{
    int basicFunctionWidth = 2 * radius + 1;
    Mat kernelOneChannel;
    Mat A(1, basicFunctionWidth, CV_32F, 0.0f);
    std::vector<Mat> channels;

    A.at<float>(0, radius) = 1;

    if (function == ft::LINEAR)
    {
        float a = 1.0f / radius;

        for (int i = 1; i < radius; i++)
        {
            float previous = A.at<float>(0, i - 1);
            float current =  previous + a;

            A.at<float>(0, i) = current;
            A.at<float>(0, (2 * radius) - i) = current;
        }

        mulTransposed(A, kernelOneChannel, true);
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(kernelOneChannel);
    }

    merge(channels, _kernel);
}

void ft::FT02D_components(InputArray _matrix, InputArray _kernel, OutputArray _components, InputArray _mask)
{
    Mat matrix = _matrix.getMat();
    Mat kernel = _kernel.getMat();
    Mat mask = _mask.getMat();

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = matrix.cols / radiusX + 1;
    int Bn = matrix.rows / radiusY + 1;
    int type = matrix.type();

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrix, matrixPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));

    _components.create(Bn, An, type);
    Mat components = _components.getMat();

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiImage(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator = roiImage.mul(kernelMasked);
            Scalar component;
            divide(sum(numerator), sum(kernelMasked), component);

            components.row(o).col(i) = component;
        }
    }
}

void ft::FT02D_components(InputArray _matrix, InputArray _kernel, OutputArray _components)
{
    Mat mask = Mat::ones(_matrix.size(), CV_8U);

    ft::FT02D_components(_matrix, _kernel, _components, mask);
}

void ft::FT02D_inverseFT(InputArray _components, InputArray _kernel, OutputArray _output, int width, int height, int type)
{
    // Only for 1chn yet because of the float!

    Mat components = _components.getMat();
    Mat kernel = _kernel.getMat();

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int paddedOutputWidth = radiusX + width + kernel.cols;
    int paddedOutputHeight = radiusY + height + kernel.rows;

    _output.create(height, width, type);

    Mat outputZeroes(paddedOutputHeight, paddedOutputWidth, type, Scalar(0));

    for (int i = 0; i < components.cols; i++)
    {
        for (int o = 0; o < components.rows; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiOutput(outputZeroes, area);
            float value = components.at<float>(o,i);

            add(roiOutput, kernel.mul(value), roiOutput);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, width, height)).copyTo(_output);
}

void ft::inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius, int function, int algorithm)
{
    Mat kernel;
    ft::createKernel(function, radius, kernel, image.channels());

    Mat processingOutput;
    ft::FT02D_process(image, kernel, processingOutput, mask);

    image.copyTo(processingOutput, mask);

    output = processingOutput;
}

void ft::filter(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output)
{
    Mat mask = Mat::ones(image.size(), CV_8U);

    ft::FT02D_process(image, kernel, output, mask);
}

int ft::FT02D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask)
{
    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = image.cols / radiusX + 1;
    int Bn = image.rows / radiusY + 1;
    int outputWidthPadded = radiusX + image.cols + kernel.cols;
    int outputHeightPadded = radiusY + image.rows + kernel.rows;

    Mat imagePadded;
    Mat maskPadded;

    output = Mat::zeros(outputHeightPadded, outputWidthPadded, image.type());

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

            Mat numerator = roiImage.mul(kernelMasked);

            Scalar component;
            divide(sum(numerator), sum(kernelMasked), component);

            Mat roiOutput(output, area);

            add(roiOutput, kernel.mul(component), roiOutput);
        }
    }

    output = output(Rect(radiusX, radiusY, image.cols, image.rows));

    return 0;
}
