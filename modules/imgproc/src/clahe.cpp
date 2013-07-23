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
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
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
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

// ----------------------------------------------------------------------
// CLAHE

namespace
{
    class CLAHE_CalcLut_Body : public cv::ParallelLoopBody
    {
    public:
        CLAHE_CalcLut_Body(const cv::Mat& src, cv::Mat& lut, cv::Size tileSize, int tilesX, int tilesY, int clipLimit, float lutScale) :
            src_(src), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY), clipLimit_(clipLimit), lutScale_(lutScale)
        {
        }

        void operator ()(const cv::Range& range) const;

    private:
        cv::Mat src_;
        mutable cv::Mat lut_;

        cv::Size tileSize_;
        int tilesX_;
        int tilesY_;
        int clipLimit_;
        float lutScale_;
    };

    void CLAHE_CalcLut_Body::operator ()(const cv::Range& range) const
    {
        const int histSize = 256;

        uchar* tileLut = lut_.ptr(range.start);
        const size_t lut_step = lut_.step;

        for (int k = range.start; k < range.end; ++k, tileLut += lut_step)
        {
            const int ty = k / tilesX_;
            const int tx = k % tilesX_;

            // retrieve tile submatrix

            cv::Rect tileROI;
            tileROI.x = tx * tileSize_.width;
            tileROI.y = ty * tileSize_.height;
            tileROI.width = tileSize_.width;
            tileROI.height = tileSize_.height;

            const cv::Mat tile = src_(tileROI);

            // calc histogram

            int tileHist[histSize] = {0, };

            int height = tileROI.height;
            const size_t sstep = tile.step;
            for (const uchar* ptr = tile.ptr<uchar>(0); height--; ptr += sstep)
            {
                int x = 0;
                for (; x <= tileROI.width - 4; x += 4)
                {
                    int t0 = ptr[x], t1 = ptr[x+1];
                    tileHist[t0]++; tileHist[t1]++;
                    t0 = ptr[x+2]; t1 = ptr[x+3];
                    tileHist[t0]++; tileHist[t1]++;
                }

                for (; x < tileROI.width; ++x)
                    tileHist[ptr[x]]++;
            }

            // clip histogram

            if (clipLimit_ > 0)
            {
                // how many pixels were clipped
                int clipped = 0;
                for (int i = 0; i < histSize; ++i)
                {
                    if (tileHist[i] > clipLimit_)
                    {
                        clipped += tileHist[i] - clipLimit_;
                        tileHist[i] = clipLimit_;
                    }
                }

                // redistribute clipped pixels
                int redistBatch = clipped / histSize;
                int residual = clipped - redistBatch * histSize;

                for (int i = 0; i < histSize; ++i)
                    tileHist[i] += redistBatch;

                for (int i = 0; i < residual; ++i)
                    tileHist[i]++;
            }

            // calc Lut

            int sum = 0;
            for (int i = 0; i < histSize; ++i)
            {
                sum += tileHist[i];
                tileLut[i] = cv::saturate_cast<uchar>(sum * lutScale_);
            }
        }
    }

    class CLAHE_Interpolation_Body : public cv::ParallelLoopBody
    {
    public:
        CLAHE_Interpolation_Body(const cv::Mat& src, cv::Mat& dst, const cv::Mat& lut, cv::Size tileSize, int tilesX, int tilesY) :
            src_(src), dst_(dst), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY)
        {
        }

        void operator ()(const cv::Range& range) const;

    private:
        cv::Mat src_;
        mutable cv::Mat dst_;
        cv::Mat lut_;

        cv::Size tileSize_;
        int tilesX_;
        int tilesY_;
    };

    void CLAHE_Interpolation_Body::operator ()(const cv::Range& range) const
    {
        const size_t lut_step = lut_.step;

        for (int y = range.start; y < range.end; ++y)
        {
            const uchar* srcRow = src_.ptr<uchar>(y);
            uchar* dstRow = dst_.ptr<uchar>(y);

            const float tyf = (static_cast<float>(y) / tileSize_.height) - 0.5f;

            int ty1 = cvFloor(tyf);
            int ty2 = ty1 + 1;

            const float ya = tyf - ty1;

            ty1 = std::max(ty1, 0);
            ty2 = std::min(ty2, tilesY_ - 1);

            const uchar* lutPlane1 = lut_.ptr(ty1 * tilesX_);
            const uchar* lutPlane2 = lut_.ptr(ty2 * tilesX_);

            for (int x = 0; x < src_.cols; ++x)
            {
                const float txf = (static_cast<float>(x) / tileSize_.width) - 0.5f;

                int tx1 = cvFloor(txf);
                int tx2 = tx1 + 1;

                const float xa = txf - tx1;

                tx1 = std::max(tx1, 0);
                tx2 = std::min(tx2, tilesX_ - 1);

                const int srcVal = srcRow[x];

                const size_t ind1 = tx1 * lut_step + srcVal;
                const size_t ind2 = tx2 * lut_step + srcVal;

                float res = 0;

                res += lutPlane1[ind1] * ((1.0f - xa) * (1.0f - ya));
                res += lutPlane1[ind2] * ((xa) * (1.0f - ya));
                res += lutPlane2[ind1] * ((1.0f - xa) * (ya));
                res += lutPlane2[ind2] * ((xa) * (ya));

                dstRow[x] = cv::saturate_cast<uchar>(res);
            }
        }
    }

    class CLAHE_Impl : public cv::CLAHE
    {
    public:
        CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

        cv::AlgorithmInfo* info() const;

        void apply(cv::InputArray src, cv::OutputArray dst);

        void setClipLimit(double clipLimit);
        double getClipLimit() const;

        void setTilesGridSize(cv::Size tileGridSize);
        cv::Size getTilesGridSize() const;

        void collectGarbage();

    private:
        double clipLimit_;
        int tilesX_;
        int tilesY_;

        cv::Mat srcExt_;
        cv::Mat lut_;
    };

    CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
        clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
    {
    }

    CV_INIT_ALGORITHM(CLAHE_Impl, "CLAHE",
        obj.info()->addParam(obj, "clipLimit", obj.clipLimit_);
        obj.info()->addParam(obj, "tilesX", obj.tilesX_);
        obj.info()->addParam(obj, "tilesY", obj.tilesY_))

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
    {
        cv::Mat src = _src.getMat();

        CV_Assert( src.type() == CV_8UC1 );

        _dst.create( src.size(), src.type() );
        cv::Mat dst = _dst.getMat();

        const int histSize = 256;

        lut_.create(tilesX_ * tilesY_, histSize, CV_8UC1);

        cv::Size tileSize;
        cv::Mat srcForLut;

        if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0)
        {
            tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
            srcForLut = src;
        }
        else
        {
            cv::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101);

            tileSize = cv::Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
            srcForLut = srcExt_;
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

        int clipLimit = 0;
        if (clipLimit_ > 0.0)
        {
            clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
            clipLimit = std::max(clipLimit, 1);
        }

        CLAHE_CalcLut_Body calcLutBody(srcForLut, lut_, tileSize, tilesX_, tilesY_, clipLimit, lutScale);
        cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), calcLutBody);

        CLAHE_Interpolation_Body interpolationBody(src, dst, lut_, tileSize, tilesX_, tilesY_);
        cv::parallel_for_(cv::Range(0, src.rows), interpolationBody);
    }

    void CLAHE_Impl::setClipLimit(double clipLimit)
    {
        clipLimit_ = clipLimit;
    }

    double CLAHE_Impl::getClipLimit() const
    {
        return clipLimit_;
    }

    void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
    {
        tilesX_ = tileGridSize.width;
        tilesY_ = tileGridSize.height;
    }

    cv::Size CLAHE_Impl::getTilesGridSize() const
    {
        return cv::Size(tilesX_, tilesY_);
    }

    void CLAHE_Impl::collectGarbage()
    {
        srcExt_.release();
        lut_.release();
    }
}

cv::Ptr<cv::CLAHE> cv::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return new CLAHE_Impl(clipLimit, tileGridSize.width, tileGridSize.height);
}
