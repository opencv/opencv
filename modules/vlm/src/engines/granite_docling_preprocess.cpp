// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "granite_docling_preprocess.hpp"
#include "preprocess_common.hpp"

#include <sstream>

namespace cv { namespace vlm {

using namespace cv::dnn;

void resizeAA(const Mat& src, Mat& dst, Size size)
{
    bool shrinking = (int64_t)size.width * size.height < (int64_t)src.cols * src.rows;
    resize(src, dst, size, 0, 0, shrinking ? INTER_AREA : INTER_LANCZOS4);
}

Mat tileImage(const Mat& imageBgr, int longestEdge, int tileSize,
              const Vec3f& mean, const Vec3f& std_, int& rowsOut, int& colsOut)
{
    int h0 = imageBgr.rows, w0 = imageBgr.cols;
    int newW, newH;
    if (w0 >= h0)
    {
        newW = longestEdge;
        newH = std::max(1, cvRound((double)longestEdge * h0 / w0));
    }
    else
    {
        newH = longestEdge;
        newW = std::max(1, cvRound((double)longestEdge * w0 / h0));
    }

    Mat resized;
    resizeAA(imageBgr, resized, Size(newW, newH));

    int rows = (newH + tileSize - 1) / tileSize;
    int cols = (newW + tileSize - 1) / tileSize;
    rowsOut = rows;
    colsOut = cols;

    Mat grid;
    resizeAA(resized, grid, Size(cols * tileSize, rows * tileSize));

    int numTiles = rows * cols + 1;
    int sizes[] = {1, numTiles, 3, tileSize, tileSize};
    Mat pixelValues(5, sizes, CV_32F);

    auto normalizeTile = [&](const Mat& tileBgr, int tileIdx)
    {
        Mat planes[3];
        for (int c = 0; c < 3; c++)
            planes[c] = Mat(tileSize, tileSize, CV_32F, pixelValues.ptr<float>(0, tileIdx, c));
        toNormalizedPlanes(tileBgr, 1.f / 255.f, mean, std_, planes);
    };

    // Serial: parallel_for_ measured 9.63 vs 10.06 ms, inside the run-to-run spread.
    for (int i = 0; i < rows * cols; i++)
    {
        Rect roi((i % cols) * tileSize, (i / cols) * tileSize, tileSize, tileSize);
        normalizeTile(grid(roi), i);
    }
    int idx = rows * cols;

    Mat thumbnail;
    resizeAA(resized, thumbnail, Size(tileSize, tileSize));
    normalizeTile(thumbnail, idx);

    return pixelValues;
}

String buildGraniteDoclingPrompt(int rows, int cols, int imageSeqLen, const String& userText)
{
    std::ostringstream imagePart;
    for (int h = 0; h < rows; h++)
    {
        for (int w = 0; w < cols; w++)
        {
            imagePart << "<fake_token_around_image><row_" << (h + 1) << "_col_" << (w + 1) << ">";
            for (int i = 0; i < imageSeqLen; i++)
                imagePart << "<image>";
        }
        imagePart << "\n";
    }
    imagePart << "\n<fake_token_around_image><global-img>";
    for (int i = 0; i < imageSeqLen; i++)
        imagePart << "<image>";
    imagePart << "<fake_token_around_image>";

    std::ostringstream full;
    full << "<|start_of_role|>user<|end_of_role|>" << imagePart.str() << userText
         << "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
    return full.str();
}

}} // namespace cv::vlm
