/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual 
 * property and proprietary rights in and to this software and 
 * related documentation and any modifications thereto.  
 * Any use, reproduction, disclosure, or distribution of this 
 * software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 */
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

        allocatorCPU.reset(new NCVMemNativeAllocator(NCVMemoryTypeHostPinned, devProp.textureAlignment));
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

        allocatorCPU.reset(new NCVMemNativeAllocator(NCVMemoryTypeHostPinned, devProp.textureAlignment));
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
