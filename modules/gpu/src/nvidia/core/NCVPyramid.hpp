/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009-2010, NVIDIA Corporation, all rights reserved.
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


#ifndef _ncvpyramid_hpp_
#define _ncvpyramid_hpp_

#include <memory>
#include <vector>
#include "NCV.hpp"

#ifdef _WIN32

template <class T>
class NCV_EXPORTS NCVMatrixStack
{
public:
    NCVMatrixStack() {this->_arr.clear();}
    ~NCVMatrixStack()
    {
        const Ncv32u nElem = this->_arr.size();
        for (Ncv32u i=0; i<nElem; i++)
        {
            pop_back();
        }
    }
    void push_back(NCVMatrix<T> *elem) {this->_arr.push_back(std::tr1::shared_ptr< NCVMatrix<T> >(elem));}
    void pop_back() {this->_arr.pop_back();}
    NCVMatrix<T> * operator [] (int i) const {return this->_arr[i].get();}
private:
    std::vector< std::tr1::shared_ptr< NCVMatrix<T> > > _arr;
};


template <class T>
class NCV_EXPORTS NCVImagePyramid
{
public:

    NCVImagePyramid(const NCVMatrix<T> &img,
                    Ncv8u nLayers,
                    INCVMemAllocator &alloc,
                    cudaStream_t cuStream);
    ~NCVImagePyramid();
    NcvBool isInitialized() const;
    NCVStatus getLayer(NCVMatrix<T> &outImg,
                       NcvSize32u outRoi,
                       NcvBool bTrilinear,
                       cudaStream_t cuStream) const;

private:

    NcvBool _isInitialized;
    const NCVMatrix<T> *layer0;
    NCVMatrixStack<T> pyramid;
    Ncv32u nLayers;
};

#endif //_WIN32

#endif //_ncvpyramid_hpp_
