/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef __OPENCV_CUDEV_PTR2D_TEXTURE_HPP__
#define __OPENCV_CUDEV_PTR2D_TEXTURE_HPP__

#include <cstring>
#include "../common.hpp"
#include "glob.hpp"
#include "gpumat.hpp"
#include "traits.hpp"

namespace
{
    template <typename T> struct CvCudevTextureRef
    {
        typedef texture<T, cudaTextureType2D, cudaReadModeElementType> TexRef;

        static TexRef ref;

        __host__ static void bind(const cv::cudev::GlobPtrSz<T>& mat,
                                  bool normalizedCoords = false,
                                  cudaTextureFilterMode filterMode = cudaFilterModePoint,
                                  cudaTextureAddressMode addressMode = cudaAddressModeClamp)
        {
            ref.normalized = normalizedCoords;
            ref.filterMode = filterMode;
            ref.addressMode[0] = addressMode;
            ref.addressMode[1] = addressMode;
            ref.addressMode[2] = addressMode;

            cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();

            CV_CUDEV_SAFE_CALL( cudaBindTexture2D(0, &ref, mat.data, &desc, mat.cols, mat.rows, mat.step) );
        }

        __host__ static void unbind()
        {
            CV_CUDEV_SAFE_CALL( cudaUnbindTexture(ref) );
        }
    };

    template <typename T>
    typename CvCudevTextureRef<T>::TexRef CvCudevTextureRef<T>::ref;
}

namespace cv { namespace cudev {

template <typename T> struct TexturePtr
{
    typedef T     value_type;
    typedef float index_type;

    cudaTextureObject_t texObj;

    __device__ __forceinline__ T operator ()(float y, float x) const
    {
    #if CV_CUDEV_ARCH < 300
        // Use the texture reference
        return tex2D(CvCudevTextureRef<T>::ref, x, y);
    #else
        // Use the texture object
        return tex2D<T>(texObj, x, y);
    #endif
    }
};

template <typename T> struct Texture : TexturePtr<T>
{
    int rows, cols;
    bool cc30;

    __host__ explicit Texture(const GlobPtrSz<T>& mat,
                              bool normalizedCoords = false,
                              cudaTextureFilterMode filterMode = cudaFilterModePoint,
                              cudaTextureAddressMode addressMode = cudaAddressModeClamp)
    {
        cc30 = deviceSupports(FEATURE_SET_COMPUTE_30);

        rows = mat.rows;
        cols = mat.cols;

        if (cc30)
        {
            // Use the texture object
            cudaResourceDesc texRes;
            std::memset(&texRes, 0, sizeof(texRes));
            texRes.resType = cudaResourceTypePitch2D;
            texRes.res.pitch2D.devPtr = mat.data;
            texRes.res.pitch2D.height = mat.rows;
            texRes.res.pitch2D.width = mat.cols;
            texRes.res.pitch2D.pitchInBytes = mat.step;
            texRes.res.pitch2D.desc = cudaCreateChannelDesc<T>();

            cudaTextureDesc texDescr;
            std::memset(&texDescr, 0, sizeof(texDescr));
            texDescr.normalizedCoords = normalizedCoords;
            texDescr.filterMode = filterMode;
            texDescr.addressMode[0] = addressMode;
            texDescr.addressMode[1] = addressMode;
            texDescr.addressMode[2] = addressMode;
            texDescr.readMode = cudaReadModeElementType;

            CV_CUDEV_SAFE_CALL( cudaCreateTextureObject(&this->texObj, &texRes, &texDescr, 0) );
        }
        else
        {
            // Use the texture reference
            CvCudevTextureRef<T>::bind(mat, normalizedCoords, filterMode, addressMode);
        }
    }

    __host__ ~Texture()
    {
        if (cc30)
        {
            // Use the texture object
            cudaDestroyTextureObject(this->texObj);
        }
        else
        {
            // Use the texture reference
            CvCudevTextureRef<T>::unbind();
        }
    }
};

template <typename T> struct PtrTraits< Texture<T> > : PtrTraitsBase<Texture<T>, TexturePtr<T> >
{
};

}}

#endif
