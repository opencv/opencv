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

#ifndef _ncvtestsourceprovider_hpp_
#define _ncvtestsourceprovider_hpp_

#include <memory>

#include "NCV.hpp"
#include <opencv2/highgui/highgui.hpp>


template <class T>
class NCVTestSourceProvider
{
public:

    NCVTestSourceProvider(Ncv32u seed, T rangeLow, T rangeHigh, Ncv32u maxWidth, Ncv32u maxHeight)
        :
        bInit(false)
    {
        ncvAssertPrintReturn(rangeLow < rangeHigh, "NCVTestSourceProvider ctor:: Invalid range", );

        int devId;
        cudaDeviceProp devProp;
        ncvAssertPrintReturn(cudaSuccess == cudaGetDevice(&devId), "Error returned from cudaGetDevice", );
        ncvAssertPrintReturn(cudaSuccess == cudaGetDeviceProperties(&devProp, devId), "Error returned from cudaGetDeviceProperties", );

        //Ncv32u maxWpitch = alignUp(maxWidth * sizeof(T), devProp.textureAlignment);

        allocatorCPU.reset(new NCVMemNativeAllocator(NCVMemoryTypeHostPinned, static_cast<Ncv32u>(devProp.textureAlignment)));
        data.reset(new NCVMatrixAlloc<T>(*this->allocatorCPU.get(), maxWidth, maxHeight));
        ncvAssertPrintReturn(data.get()->isMemAllocated(), "NCVTestSourceProvider ctor:: Matrix not allocated", );

        this->dataWidth = maxWidth;
        this->dataHeight = maxHeight;

        srand(seed);

        for (Ncv32u i=0; i<maxHeight; i++)
        {
            for (Ncv32u j=0; j<data.get()->stride(); j++)
            {
                data.get()->ptr()[i * data.get()->stride() + j] =
                    (T)(((1.0 * rand()) / RAND_MAX) * (rangeHigh - rangeLow) + rangeLow);
            }
        }

        this->bInit = true;
    }

    NCVTestSourceProvider(std::string pgmFilename)
        :
        bInit(false)
    {
        ncvAssertPrintReturn(sizeof(T) == 1, "NCVTestSourceProvider ctor:: PGM constructor complies only with 8bit types", );

        cv::Mat image = cv::imread(pgmFilename);
        ncvAssertPrintReturn(!image.empty(), "NCVTestSourceProvider ctor:: PGM file error", );

        int devId;
        cudaDeviceProp devProp;
        ncvAssertPrintReturn(cudaSuccess == cudaGetDevice(&devId), "Error returned from cudaGetDevice", );
        ncvAssertPrintReturn(cudaSuccess == cudaGetDeviceProperties(&devProp, devId), "Error returned from cudaGetDeviceProperties", );

        allocatorCPU.reset(new NCVMemNativeAllocator(NCVMemoryTypeHostPinned, static_cast<Ncv32u>(devProp.textureAlignment)));
        data.reset(new NCVMatrixAlloc<T>(*this->allocatorCPU.get(), image.cols, image.rows));
        ncvAssertPrintReturn(data.get()->isMemAllocated(), "NCVTestSourceProvider ctor:: Matrix not allocated", );

        this->dataWidth = image.cols;
        this->dataHeight = image.rows;

        cv::Mat hdr(image.size(), CV_8UC1, data.get()->ptr(), data.get()->pitch());
        image.copyTo(hdr);

        this->bInit = true;
    }

    NcvBool fill(NCVMatrix<T> &dst)
    {
        ncvAssertReturn(this->isInit() &&
                        dst.memType() == allocatorCPU.get()->memType(), false);

        if (dst.width() == 0 || dst.height() == 0)
        {
            return true;
        }

        for (Ncv32u i=0; i<dst.height(); i++)
        {
            Ncv32u srcLine = i % this->dataHeight;

            Ncv32u srcFullChunks = dst.width() / this->dataWidth;
            for (Ncv32u j=0; j<srcFullChunks; j++)
            {
                memcpy(dst.ptr() + i * dst.stride() + j * this->dataWidth,
                    this->data.get()->ptr() + this->data.get()->stride() * srcLine,
                    this->dataWidth * sizeof(T));
            }

            Ncv32u srcLastChunk = dst.width() % this->dataWidth;
            memcpy(dst.ptr() + i * dst.stride() + srcFullChunks * this->dataWidth,
                this->data.get()->ptr() + this->data.get()->stride() * srcLine,
                srcLastChunk * sizeof(T));
        }

        return true;
    }

    NcvBool fill(NCVVector<T> &dst)
    {
        ncvAssertReturn(this->isInit() &&
                        dst.memType() == allocatorCPU.get()->memType(), false);

        if (dst.length() == 0)
        {
            return true;
        }

        Ncv32u srcLen = this->dataWidth * this->dataHeight;

        Ncv32u srcFullChunks = (Ncv32u)dst.length() / srcLen;
        for (Ncv32u j=0; j<srcFullChunks; j++)
        {
            memcpy(dst.ptr() + j * srcLen, this->data.get()->ptr(), srcLen * sizeof(T));
        }

        Ncv32u srcLastChunk = dst.length() % srcLen;
        memcpy(dst.ptr() + srcFullChunks * srcLen, this->data.get()->ptr(), srcLastChunk * sizeof(T));

        return true;
    }

    ~NCVTestSourceProvider()
    {
        data.reset();
        allocatorCPU.reset();
    }

private:

    NcvBool isInit(void)
    {
        return this->bInit;
    }

    NcvBool bInit;
    std::auto_ptr< INCVMemAllocator > allocatorCPU;
    std::auto_ptr< NCVMatrixAlloc<T> > data;
    Ncv32u dataWidth;
    Ncv32u dataHeight;
};

#endif // _ncvtestsourceprovider_hpp_
