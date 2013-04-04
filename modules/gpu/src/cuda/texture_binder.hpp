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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_GPU_TEXTURE_BINDER_HPP_
#define OPENCV_GPU_TEXTURE_BINDER_HPP_

#include "opencv2/gpu/devmem2d.hpp"
#include <safe_call.hpp>

namespace cv
{
  namespace gpu
  {
    class TextureBinder
    {
    public:
      template<class T, enum cudaTextureReadMode readMode>
      TextureBinder(const PtrStepSz<T>& arr, const struct texture<T, 2, readMode>& tex) : texref(&tex)
      {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
        cudaSafeCall( cudaBindTexture2D(0, tex, arr.data, desc, arr.cols, arr.rows, arr.step) );
      }

      template<class T, enum cudaTextureReadMode readMode>
      TextureBinder(const PtrSz<T>& arr, const struct texture<T, 1, readMode> &tex) : texref(&tex)
      {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
        cudaSafeCall( cudaBindTexture(0, tex, arr.data, desc, arr.size * arr.elemSize()) );
      }

      template<class A, class T, enum cudaTextureReadMode readMode>
      TextureBinder(const A& arr, const struct texture<T, 2, readMode>& tex, const cudaChannelFormatDesc& desc) : texref(&tex)
      {
        cudaSafeCall( cudaBindTexture2D(0, tex, arr.data, desc, arr.cols, arr.rows, arr.step) );
      }


      ~TextureBinder()
      {
        cudaSafeCall( cudaUnbindTexture(texref) );
      }
    private:
      const struct textureReference *texref;
    };
  }

  namespace device
  {
      using cv::gpu::TextureBinder;
  }
}

#endif /* OPENCV_GPU_TEXTURE_BINDER_HPP_*/
