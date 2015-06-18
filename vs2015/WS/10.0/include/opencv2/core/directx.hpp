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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// This software is provided by the copyright holders and contributors as is and
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

#ifndef __OPENCV_CORE_DIRECTX_HPP__
#define __OPENCV_CORE_DIRECTX_HPP__

#include "mat.hpp"
#include "ocl.hpp"

#if !defined(__d3d11_h__)
struct ID3D11Device;
struct ID3D11Texture2D;
#endif

#if !defined(__d3d10_h__)
struct ID3D10Device;
struct ID3D10Texture2D;
#endif

#if !defined(_D3D9_H_)
struct IDirect3DDevice9;
struct IDirect3DDevice9Ex;
struct IDirect3DSurface9;
#endif


namespace cv { namespace directx {

namespace ocl {
using namespace cv::ocl;

//! @addtogroup core_directx
//! @{

// TODO static functions in the Context class
CV_EXPORTS Context& initializeContextFromD3D11Device(ID3D11Device* pD3D11Device);
CV_EXPORTS Context& initializeContextFromD3D10Device(ID3D10Device* pD3D10Device);
CV_EXPORTS Context& initializeContextFromDirect3DDevice9Ex(IDirect3DDevice9Ex* pDirect3DDevice9Ex);
CV_EXPORTS Context& initializeContextFromDirect3DDevice9(IDirect3DDevice9* pDirect3DDevice9);

//! @}

} // namespace cv::directx::ocl

//! @addtogroup core_directx
//! @{

CV_EXPORTS void convertToD3D11Texture2D(InputArray src, ID3D11Texture2D* pD3D11Texture2D);
CV_EXPORTS void convertFromD3D11Texture2D(ID3D11Texture2D* pD3D11Texture2D, OutputArray dst);

CV_EXPORTS void convertToD3D10Texture2D(InputArray src, ID3D10Texture2D* pD3D10Texture2D);
CV_EXPORTS void convertFromD3D10Texture2D(ID3D10Texture2D* pD3D10Texture2D, OutputArray dst);

CV_EXPORTS void convertToDirect3DSurface9(InputArray src, IDirect3DSurface9* pDirect3DSurface9, void* surfaceSharedHandle = NULL);
CV_EXPORTS void convertFromDirect3DSurface9(IDirect3DSurface9* pDirect3DSurface9, OutputArray dst, void* surfaceSharedHandle = NULL);

// Get OpenCV type from DirectX type, return -1 if there is no equivalent
CV_EXPORTS int getTypeFromDXGI_FORMAT(const int iDXGI_FORMAT); // enum DXGI_FORMAT for D3D10/D3D11

// Get OpenCV type from DirectX type, return -1 if there is no equivalent
CV_EXPORTS int getTypeFromD3DFORMAT(const int iD3DFORMAT); // enum D3DTYPE for D3D9

//! @}

} } // namespace cv::directx

#endif // __OPENCV_CORE_DIRECTX_HPP__
