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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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

/*
OpenCV wrapper of reference implementation of
[1] KAZE Features. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison.
In European Conference on Computer Vision (ECCV), Fiorenze, Italy, October 2012
http://www.robesafe.com/personal/pablo.alcantarilla/papers/Alcantarilla12eccv.pdf
@author Eugene Khvedchenya <ekhvedchenya@gmail.com>
*/

#include "precomp.hpp"
#include "kaze/KAZEFeatures.h"

namespace cv
{

    class KAZE_Impl CV_FINAL : public KAZE
    {
    public:
        KAZE_Impl(bool _extended, bool _upright, float _threshold, int _octaves,
                   int _sublevels, int _diffusivity)
        : extended(_extended)
        , upright(_upright)
        , threshold(_threshold)
        , octaves(_octaves)
        , sublevels(_sublevels)
        , diffusivity(_diffusivity)
        {
        }

        virtual ~KAZE_Impl() CV_OVERRIDE {}

        void setExtended(bool extended_) CV_OVERRIDE { extended = extended_; }
        bool getExtended() const CV_OVERRIDE { return extended; }

        void setUpright(bool upright_) CV_OVERRIDE { upright = upright_; }
        bool getUpright() const CV_OVERRIDE { return upright; }

        void setThreshold(double threshold_) CV_OVERRIDE { threshold = (float)threshold_; }
        double getThreshold() const CV_OVERRIDE { return threshold; }

        void setNOctaves(int octaves_) CV_OVERRIDE { octaves = octaves_; }
        int getNOctaves() const CV_OVERRIDE { return octaves; }

        void setNOctaveLayers(int octaveLayers_) CV_OVERRIDE { sublevels = octaveLayers_; }
        int getNOctaveLayers() const CV_OVERRIDE { return sublevels; }

        void setDiffusivity(int diff_) CV_OVERRIDE { diffusivity = diff_; }
        int getDiffusivity() const CV_OVERRIDE { return diffusivity; }

        // returns the descriptor size in bytes
        int descriptorSize() const CV_OVERRIDE
        {
            return extended ? 128 : 64;
        }

        // returns the descriptor type
        int descriptorType() const CV_OVERRIDE
        {
            return CV_32F;
        }

        // returns the default norm type
        int defaultNorm() const CV_OVERRIDE
        {
            return NORM_L2;
        }

        void detectAndCompute(InputArray image, InputArray mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray descriptors,
                              bool useProvidedKeypoints) CV_OVERRIDE
        {
            CV_INSTRUMENT_REGION();

            cv::Mat img = image.getMat();
            if (img.channels() > 1)
                cvtColor(image, img, COLOR_BGR2GRAY);

            Mat img1_32;
            if ( img.depth() == CV_32F )
                img1_32 = img;
            else if ( img.depth() == CV_8U )
                img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
            else if ( img.depth() == CV_16U )
                img.convertTo(img1_32, CV_32F, 1.0 / 65535.0, 0);

            CV_Assert( ! img1_32.empty() );

            KAZEOptions options;
            options.img_width = img.cols;
            options.img_height = img.rows;
            options.extended = extended;
            options.upright = upright;
            options.dthreshold = threshold;
            options.omax = octaves;
            options.nsublevels = sublevels;
            options.diffusivity = diffusivity;

            KAZEFeatures impl(options);
            impl.Create_Nonlinear_Scale_Space(img1_32);

            if (!useProvidedKeypoints)
            {
                impl.Feature_Detection(keypoints);
            }

            if (!mask.empty())
            {
                cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
            }

            if( descriptors.needed() )
            {
                Mat desc;
                impl.Feature_Description(keypoints, desc);
                desc.copyTo(descriptors);

                CV_Assert((!desc.rows || desc.cols == descriptorSize()));
                CV_Assert((!desc.rows || (desc.type() == descriptorType())));
            }
        }

        void write(FileStorage& fs) const CV_OVERRIDE
        {
            writeFormat(fs);
            fs << "extended" << (int)extended;
            fs << "upright" << (int)upright;
            fs << "threshold" << threshold;
            fs << "octaves" << octaves;
            fs << "sublevels" << sublevels;
            fs << "diffusivity" << diffusivity;
        }

        void read(const FileNode& fn) CV_OVERRIDE
        {
            extended = (int)fn["extended"] != 0;
            upright = (int)fn["upright"] != 0;
            threshold = (float)fn["threshold"];
            octaves = (int)fn["octaves"];
            sublevels = (int)fn["sublevels"];
            diffusivity = (int)fn["diffusivity"];
        }

        bool extended;
        bool upright;
        float threshold;
        int octaves;
        int sublevels;
        int diffusivity;
    };

    Ptr<KAZE> KAZE::create(bool extended, bool upright,
                            float threshold,
                            int octaves, int sublevels,
                            int diffusivity)
    {
        return makePtr<KAZE_Impl>(extended, upright, threshold, octaves, sublevels, diffusivity);
    }

    String KAZE::getDefaultName() const
    {
        return (Feature2D::getDefaultName() + ".KAZE");
    }

}
