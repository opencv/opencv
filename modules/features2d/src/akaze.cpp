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
[1] Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces.
Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli.
In British Machine Vision Conference (BMVC), Bristol, UK, September 2013
http://www.robesafe.com/personal/pablo.alcantarilla/papers/Alcantarilla13bmvc.pdf
@author Eugene Khvedchenya <ekhvedchenya@gmail.com>
*/

#include "precomp.hpp"
#include "kaze/AKAZEFeatures.h"

#include <iostream>

namespace cv
{
    using namespace std;

    class AKAZE_Impl : public AKAZE
    {
    public:
        AKAZE_Impl(DescriptorType _descriptor_type, int _descriptor_size, int _descriptor_channels,
                 float _threshold, int _octaves, int _sublevels, KAZE::DiffusivityType _diffusivity, int _max_points)
        : descriptor(_descriptor_type)
        , descriptor_channels(_descriptor_channels)
        , descriptor_size(_descriptor_size)
        , threshold(_threshold)
        , octaves(_octaves)
        , sublevels(_sublevels)
        , diffusivity(_diffusivity)
        , max_points(_max_points)
        {
        }

        virtual ~AKAZE_Impl() CV_OVERRIDE
        {

        }

        void setDescriptorType(DescriptorType dtype) CV_OVERRIDE{ descriptor = dtype; }
        DescriptorType getDescriptorType() const CV_OVERRIDE{ return descriptor; }

        void setDescriptorSize(int dsize) CV_OVERRIDE { descriptor_size = dsize; }
        int getDescriptorSize() const CV_OVERRIDE { return descriptor_size; }

        void setDescriptorChannels(int dch) CV_OVERRIDE { descriptor_channels = dch; }
        int getDescriptorChannels() const CV_OVERRIDE { return descriptor_channels; }

        void setThreshold(double threshold_) CV_OVERRIDE { threshold = (float)threshold_; }
        double getThreshold() const CV_OVERRIDE { return threshold; }

        void setNOctaves(int octaves_) CV_OVERRIDE { octaves = octaves_; }
        int getNOctaves() const CV_OVERRIDE { return octaves; }

        void setNOctaveLayers(int octaveLayers_) CV_OVERRIDE { sublevels = octaveLayers_; }
        int getNOctaveLayers() const CV_OVERRIDE { return sublevels; }

        void setDiffusivity(KAZE::DiffusivityType diff_) CV_OVERRIDE{ diffusivity = diff_; }
        KAZE::DiffusivityType getDiffusivity() const CV_OVERRIDE{ return diffusivity; }

        void setMaxPoints(int max_points_) CV_OVERRIDE { max_points = max_points_; }
        int getMaxPoints() const CV_OVERRIDE { return max_points; }

        // returns the descriptor size in bytes
        int descriptorSize() const CV_OVERRIDE
        {
            switch (descriptor)
            {
            case DESCRIPTOR_KAZE:
            case DESCRIPTOR_KAZE_UPRIGHT:
                return 64;

            case DESCRIPTOR_MLDB:
            case DESCRIPTOR_MLDB_UPRIGHT:
                // We use the full length binary descriptor -> 486 bits
                if (descriptor_size == 0)
                {
                    int t = (6 + 36 + 120) * descriptor_channels;
                    return divUp(t, 8);
                }
                else
                {
                    // We use the random bit selection length binary descriptor
                    return divUp(descriptor_size, 8);
                }

            default:
                return -1;
            }
        }

        // returns the descriptor type
        int descriptorType() const CV_OVERRIDE
        {
            switch (descriptor)
            {
            case DESCRIPTOR_KAZE:
            case DESCRIPTOR_KAZE_UPRIGHT:
                    return CV_32F;

            case DESCRIPTOR_MLDB:
            case DESCRIPTOR_MLDB_UPRIGHT:
                    return CV_8U;

                default:
                    return -1;
            }
        }

        // returns the default norm type
        int defaultNorm() const CV_OVERRIDE
        {
            switch (descriptor)
            {
            case DESCRIPTOR_KAZE:
            case DESCRIPTOR_KAZE_UPRIGHT:
                return NORM_L2;

            case DESCRIPTOR_MLDB:
            case DESCRIPTOR_MLDB_UPRIGHT:
                return NORM_HAMMING;

            default:
                return -1;
            }
        }

        void detectAndCompute(InputArray image, InputArray mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray descriptors,
                              bool useProvidedKeypoints) CV_OVERRIDE
        {
            CV_INSTRUMENT_REGION();

            CV_Assert( ! image.empty() );

            AKAZEOptions options;
            options.descriptor = descriptor;
            options.descriptor_channels = descriptor_channels;
            options.descriptor_size = descriptor_size;
            options.img_width = image.cols();
            options.img_height = image.rows();
            options.dthreshold = threshold;
            options.omax = octaves;
            options.nsublevels = sublevels;
            options.diffusivity = diffusivity;

            AKAZEFeatures impl(options);
            impl.Create_Nonlinear_Scale_Space(image);

            if (!useProvidedKeypoints)
            {
                impl.Feature_Detection(keypoints);
            }

            if (!mask.empty())
            {
                KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
            }

            if (max_points > 0 && (int)keypoints.size() > max_points) {
                std::partial_sort(keypoints.begin(), keypoints.begin() + max_points, keypoints.end(),
                    [](const cv::KeyPoint& k1, const cv::KeyPoint& k2) {return k1.response > k2.response;});
                keypoints.erase(keypoints.begin() + max_points, keypoints.end());
            }

            if(descriptors.needed())
            {
                impl.Compute_Descriptors(keypoints, descriptors);

                CV_Assert((descriptors.empty() || descriptors.cols() == descriptorSize()));
                CV_Assert((descriptors.empty() || (descriptors.type() == descriptorType())));
            }
        }

        void write(FileStorage& fs) const CV_OVERRIDE
        {
            writeFormat(fs);
            fs << "name" << getDefaultName();
            fs << "descriptor" << descriptor;
            fs << "descriptor_channels" << descriptor_channels;
            fs << "descriptor_size" << descriptor_size;
            fs << "threshold" << threshold;
            fs << "octaves" << octaves;
            fs << "sublevels" << sublevels;
            fs << "diffusivity" << diffusivity;
            fs << "max_points" << max_points;
        }

        void read(const FileNode& fn) CV_OVERRIDE
        {
            // if node is empty, keep previous value
            if (!fn["descriptor"].empty())
                descriptor = static_cast<DescriptorType>((int)fn["descriptor"]);
            if (!fn["descriptor_channels"].empty())
                descriptor_channels = (int)fn["descriptor_channels"];
            if (!fn["descriptor_size"].empty())
                descriptor_size = (int)fn["descriptor_size"];
            if (!fn["threshold"].empty())
                threshold = (float)fn["threshold"];
            if (!fn["octaves"].empty())
                octaves = (int)fn["octaves"];
            if (!fn["sublevels"].empty())
                sublevels = (int)fn["sublevels"];
            if (!fn["diffusivity"].empty())
                diffusivity = static_cast<KAZE::DiffusivityType>((int)fn["diffusivity"]);
            if (!fn["max_points"].empty())
                max_points = (int)fn["max_points"];
        }

        DescriptorType descriptor;
        int descriptor_channels;
        int descriptor_size;
        float threshold;
        int octaves;
        int sublevels;
        KAZE::DiffusivityType diffusivity;
        int max_points;
    };

    Ptr<AKAZE> AKAZE::create(DescriptorType descriptor_type,
                             int descriptor_size, int descriptor_channels,
                             float threshold, int octaves,
                             int sublevels, KAZE::DiffusivityType diffusivity, int max_points)
    {
        return makePtr<AKAZE_Impl>(descriptor_type, descriptor_size, descriptor_channels,
                                   threshold, octaves, sublevels, diffusivity, max_points);
    }

    String AKAZE::getDefaultName() const
    {
        return (Feature2D::getDefaultName() + ".AKAZE");
    }

}
