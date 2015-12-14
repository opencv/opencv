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
        AKAZE_Impl(int _descriptor_type, int _descriptor_size, int _descriptor_channels,
                 float _threshold, int _octaves, int _sublevels, int _diffusivity)
        : descriptor(_descriptor_type)
        , descriptor_channels(_descriptor_channels)
        , descriptor_size(_descriptor_size)
        , threshold(_threshold)
        , octaves(_octaves)
        , sublevels(_sublevels)
        , diffusivity(_diffusivity)
        {
        }

        virtual ~AKAZE_Impl()
        {

        }

        void setDescriptorType(int dtype) { descriptor = dtype; }
        int getDescriptorType() const { return descriptor; }

        void setDescriptorSize(int dsize) { descriptor_size = dsize; }
        int getDescriptorSize() const { return descriptor_size; }

        void setDescriptorChannels(int dch) { descriptor_channels = dch; }
        int getDescriptorChannels() const { return descriptor_channels; }

        void setThreshold(double threshold_) { threshold = (float)threshold_; }
        double getThreshold() const { return threshold; }

        void setNOctaves(int octaves_) { octaves = octaves_; }
        int getNOctaves() const { return octaves; }

        void setNOctaveLayers(int octaveLayers_) { sublevels = octaveLayers_; }
        int getNOctaveLayers() const { return sublevels; }

        void setDiffusivity(int diff_) { diffusivity = diff_; }
        int getDiffusivity() const { return diffusivity; }

        // returns the descriptor size in bytes
        int descriptorSize() const
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
                    return (int)ceil(t / 8.);
                }
                else
                {
                    // We use the random bit selection length binary descriptor
                    return (int)ceil(descriptor_size / 8.);
                }

            default:
                return -1;
            }
        }

        // returns the descriptor type
        int descriptorType() const
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
        int defaultNorm() const
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
                              bool useProvidedKeypoints)
        {
            Mat img = image.getMat();
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

            AKAZEOptions options;
            options.descriptor = descriptor;
            options.descriptor_channels = descriptor_channels;
            options.descriptor_size = descriptor_size;
            options.img_width = img.cols;
            options.img_height = img.rows;
            options.dthreshold = threshold;
            options.omax = octaves;
            options.nsublevels = sublevels;
            options.diffusivity = diffusivity;

            AKAZEFeatures impl(options);
            impl.Create_Nonlinear_Scale_Space(img1_32);

            if (!useProvidedKeypoints)
            {
                impl.Feature_Detection(keypoints);
            }

            if (!mask.empty())
            {
                KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
            }

            if( descriptors.needed() )
            {
                Mat& desc = descriptors.getMatRef();
                impl.Compute_Descriptors(keypoints, desc);

                CV_Assert((!desc.rows || desc.cols == descriptorSize()));
                CV_Assert((!desc.rows || (desc.type() == descriptorType())));
            }
        }

        void write(FileStorage& fs) const
        {
            fs << "descriptor" << descriptor;
            fs << "descriptor_channels" << descriptor_channels;
            fs << "descriptor_size" << descriptor_size;
            fs << "threshold" << threshold;
            fs << "octaves" << octaves;
            fs << "sublevels" << sublevels;
            fs << "diffusivity" << diffusivity;
        }

        void read(const FileNode& fn)
        {
            descriptor = (int)fn["descriptor"];
            descriptor_channels = (int)fn["descriptor_channels"];
            descriptor_size = (int)fn["descriptor_size"];
            threshold = (float)fn["threshold"];
            octaves = (int)fn["octaves"];
            sublevels = (int)fn["sublevels"];
            diffusivity = (int)fn["diffusivity"];
        }

        int descriptor;
        int descriptor_channels;
        int descriptor_size;
        float threshold;
        int octaves;
        int sublevels;
        int diffusivity;
    };

    Ptr<AKAZE> AKAZE::create(int descriptor_type,
                             int descriptor_size, int descriptor_channels,
                             float threshold, int octaves,
                             int sublevels, int diffusivity)
    {
        return makePtr<AKAZE_Impl>(descriptor_type, descriptor_size, descriptor_channels,
                                   threshold, octaves, sublevels, diffusivity);
    }
}
