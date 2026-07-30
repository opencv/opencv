// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "paddleocr_vl_preprocess.hpp"

#include <cmath>
#include <sstream>

namespace cv { namespace vlm {

using namespace cv::dnn;

void smartResize(int height, int width, int factor, int minPixels, int maxPixels,
                  int& outHeight, int& outWidth)
{
    if (height < factor)
    {
        width = cvRound((double)(width * factor) / height);
        height = factor;
    }
    if (width < factor)
    {
        height = cvRound((double)(height * factor) / width);
        width = factor;
    }
    CV_CheckLE((double)std::max(height, width) / std::min(height, width), 200.0,
               "vlm: absolute aspect ratio is too large");

    int hBar = cvRound((double)height / factor) * factor;
    int wBar = cvRound((double)width / factor) * factor;
    if ((int64_t)hBar * wBar > maxPixels)
    {
        double beta = std::sqrt((double)(height * width) / maxPixels);
        hBar = (int)(std::floor(height / beta / factor)) * factor;
        wBar = (int)(std::floor(width / beta / factor)) * factor;
    }
    else if ((int64_t)hBar * wBar < minPixels)
    {
        double beta = std::sqrt((double)minPixels / (height * width));
        hBar = (int)(std::ceil(height * beta / factor)) * factor;
        wBar = (int)(std::ceil(width * beta / factor)) * factor;
    }
    outHeight = hBar;
    outWidth = wBar;
}

Mat preprocessImage(const Mat& imageBgr, int patchSize, int mergeSize, int minPixels,
                     int maxPixels, const Vec3f& mean, const Vec3f& std_, float rescaleFactor,
                     int& gridH, int& gridW)
{
    int factor = patchSize * mergeSize;
    int resizedHeight, resizedWidth;
    smartResize(imageBgr.rows, imageBgr.cols, factor, minPixels, maxPixels,
                resizedHeight, resizedWidth);

    Mat resized;
    resize(imageBgr, resized, Size(resizedWidth, resizedHeight), 0, 0, INTER_CUBIC);
    Mat rgb;
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, rescaleFactor);

    std::vector<Mat> channels(3);
    split(rgb, channels);
    for (int c = 0; c < 3; c++)
        channels[c].convertTo(channels[c], -1, 1.0 / std_[c], -mean[c] / std_[c]);

    gridH = resizedHeight / patchSize;
    gridW = resizedWidth / patchSize;
    int numPatches = gridH * gridW;

    int sizes[] = {1, numPatches, 3, patchSize, patchSize};
    Mat pixelValues(5, sizes, CV_32F);

    int idx = 0;
    for (int h = 0; h < gridH; h++)
        for (int w = 0; w < gridW; w++)
        {
            Rect roi(w * patchSize, h * patchSize, patchSize, patchSize);
            for (int c = 0; c < 3; c++)
            {
                Mat dst(patchSize, patchSize, CV_32F, pixelValues.ptr<float>(0, idx, c));
                channels[c](roi).copyTo(dst);
            }
            idx++;
        }
    return pixelValues;
}

String buildPaddleOCRVLPrompt(const String& prompt, int imageTokenRepeats)
{
    std::ostringstream oss;
    oss << "<|begin_of_sentence|>User: <|IMAGE_START|>";
    for (int i = 0; i < imageTokenRepeats; i++)
        oss << "<|IMAGE_PLACEHOLDER|>";
    oss << "<|IMAGE_END|>" << prompt << "\nAssistant:\n";
    return oss.str();
}

}} // namespace cv::vlm
