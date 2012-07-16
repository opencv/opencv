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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#include "precomp.hpp"

#define ALIGN 32 
#define GPU_MATRIX_MALLOC_STEP(step) (((step) + ALIGN - 1) / ALIGN) * ALIGN

using namespace cv;
using namespace cv::ocl;
using namespace std;

////////////////////////////////////////////////////////////////////////
//////////////////////////////// oclMat ////////////////////////////////
////////////////////////////////////////////////////////////////////////

#if !defined (HAVE_OPENCL)

namespace cv
{
    namespace ocl
    {
        void oclMat::upload(const Mat& /*m*/)
        {
            throw_nogpu();
        }
        void oclMat::download(cv::Mat& /*m*/) const
        {
            throw_nogpu();
        }
        void oclMat::copyTo( oclMat& /*m*/ ) const
        {
            throw_nogpu();
        }
        void oclMat::copyTo( oclMat& /*m*/, const oclMat&/* mask */) const
        {
            throw_nogpu();
        }
        void oclMat::convertTo( oclMat& /*m*/, int /*rtype*/, double /*alpha*/, double /*beta*/ ) const
        {
            throw_nogpu();
        }
        oclMat &oclMat::operator = (const Scalar& /*s*/)
        {
            throw_nogpu();
            return *this;
        }
        oclMat &oclMat::setTo(const Scalar& /*s*/, const oclMat& /*mask*/)
        {
            throw_nogpu();
            return *this;
        }
        oclMat oclMat::reshape(int /*new_cn*/, int /*new_rows*/) const
        {
            throw_nogpu();
            return oclMat();
        }
        void oclMat::create(int /*_rows*/, int /*_cols*/, int /*_type*/)
        {
            throw_nogpu();
        }
        void oclMat::release()
        {
            throw_nogpu();
        }
    }
}

#else /* !defined (HAVE_OPENCL) */

//helper routines
namespace cv
{
    namespace ocl
    {
        ///////////////////////////OpenCL kernel strings///////////////////////////
        extern const char *operator_copyToM;
        extern const char *operator_convertTo;
        extern const char *operator_setTo;
        extern const char *operator_setToM;
    }
}

////////////////////////////////////////////////////////////////////////
// convert_C3C4
void convert_C3C4(const cl_mem &src, oclMat &dst, int srcStep)
{
    int dstStep = dst.step1() / dst.channels();
    Context *clCxt = dst.clCxt;
    string kernelName = "convertC3C4";

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.wholecols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.wholerows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&srcStep));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dstStep));

    size_t globalThreads[3] = {(dst.wholecols *dst.wholerows + 255) / 256 * 256, 1, 1};
    size_t localThreads[3] = {256, 1, 1};

    openCLExecuteKernel(clCxt, &convertC3C4, kernelName, globalThreads, localThreads, args, -1, dst.elemSize1() >> 1);
}
////////////////////////////////////////////////////////////////////////
// convert_C4C3
void convert_C4C3(const oclMat &src, cl_mem &dst, int dstStep)
{
    int srcStep = src.step1() / src.channels();
    Context *clCxt = src.clCxt;
    string kernelName = "convertC4C3";

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.wholecols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.wholerows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&srcStep));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dstStep));

    size_t globalThreads[3] = {(src.wholecols *src.wholerows + 255) / 256 * 256, 1, 1};
    size_t localThreads[3] = {256, 1, 1};

    openCLExecuteKernel(clCxt, &convertC3C4, kernelName, globalThreads, localThreads, args, -1, src.elemSize1() >> 1);
}

void cv::ocl::oclMat::upload(const Mat &m)
{
    CV_DbgAssert(!m.empty());
    Size wholeSize;
    Point ofs;
    m.locateROI(wholeSize, ofs);
    int type = m.type();
    //if(m.channels() == 3)
    //type = CV_MAKETYPE(m.depth(), 4);
    create(wholeSize, type);

    //if(m.channels() == 3)
    //{
    //int pitch = GPU_MATRIX_MALLOC_STEP(wholeSize.width * 3 * m.elemSize1());
    //int err;
    //cl_mem temp = clCreateBuffer(clCxt->clContext,CL_MEM_READ_WRITE,
    //pitch*wholeSize.height,0,&err);
    //CV_DbgAssert(err==0);

    //openCLMemcpy2D(clCxt,temp,pitch,m.datastart,m.step,wholeSize.width*m.elemSize(),wholeSize.height,clMemcpyHostToDevice);
    //convert_C3C4(temp, *this, pitch);
    //}
    //else
    openCLMemcpy2D(clCxt, data, step, m.datastart, m.step, wholeSize.width * elemSize(), wholeSize.height, clMemcpyHostToDevice);

    rows = m.rows;
    cols = m.cols;
    offset = ofs.y * step + ofs.x * elemSize();
    download_channels = m.channels();
}

void cv::ocl::oclMat::download(cv::Mat &m) const
{
    CV_DbgAssert(!this->empty());
    int t = type();
    //if(download_channels == 3)
    //t = CV_MAKETYPE(depth(), 3);
    m.create(wholerows, wholecols, t);

    //if(download_channels == 3)
    //{
    //int pitch = GPU_MATRIX_MALLOC_STEP(wholecols * 3 * m.elemSize1());
    //int err;
    //cl_mem temp = clCreateBuffer(clCxt->clContext,CL_MEM_READ_WRITE,
    //pitch*wholerows,0,&err);
    //CV_DbgAssert(err==0);

    //convert_C4C3(*this, temp, pitch/m.elemSize1());
    //openCLMemcpy2D(clCxt,m.data,m.step,temp,pitch,wholecols*m.elemSize(),wholerows,clMemcpyDeviceToHost);
    //}
    //else
    openCLMemcpy2D(clCxt, m.data, m.step, data, step, wholecols * elemSize(), wholerows, clMemcpyDeviceToHost);
    Size wholesize;
    Point ofs;
    locateROI(wholesize, ofs);
    m.adjustROI(-ofs.y, ofs.y + rows - wholerows, -ofs.x, ofs.x + cols - wholecols);
}

/////////////////////common//////////////////////////////////////
inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}
///////////////////////////////////////////////////////////////////////////
////////////////////////////////// CopyTo /////////////////////////////////
///////////////////////////////////////////////////////////////////////////
void copy_to_with_mask(const oclMat &src, oclMat &dst, const oclMat &mask, string kernelName)
{
    CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols &&
                  src.rows == dst.rows && src.cols == dst.cols);

    vector<pair<size_t , const void *> > args;

    int vector_lengths[4][7] = {{4, 4, 2, 2, 1, 1, 1},
        {2, 2, 1, 1, 1, 1, 1},
        {8, 8, 8, 8 , 4, 4, 4},      //vector length is undefined when channels = 3
        {1, 1, 1, 1, 1, 1, 1}
    };

    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3];

    int vector_length = vector_lengths[dst.channels() -1][dst.depth()];
    int offset_cols = divUp(dst.offset, dst.elemSize()) & (vector_length - 1);
    int cols = vector_length == 1 ? divUp(dst.cols, vector_length) : divUp(dst.cols + offset_cols, vector_length);

    globalThreads[0] = divUp(cols, localThreads[0]) * localThreads[0];
    globalThreads[1] = divUp(dst.rows, localThreads[1]) * localThreads[1];
    globalThreads[2] = 1;

    int dststep_in_pixel = dst.step / dst.elemSize(), dstoffset_in_pixel = dst.offset / dst.elemSize();
    int srcstep_in_pixel = src.step / src.elemSize(), srcoffset_in_pixel = src.offset / src.elemSize();

    args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&mask.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&srcstep_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&srcoffset_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dststep_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dstoffset_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.step ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.offset ));

    openCLExecuteKernel(dst.clCxt , &operator_copyToM, kernelName, globalThreads,
                        localThreads, args, dst.channels(), dst.depth());
}

void cv::ocl::oclMat::copyTo( oclMat &m ) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    openCLCopyBuffer2D(clCxt, m.data, m.step, m.offset,
                       data, step, cols * elemSize(), rows, offset, clMemcpyDeviceToDevice);
}

void cv::ocl::oclMat::copyTo( oclMat &mat, const oclMat &mask) const
{
    if (mask.empty())
    {
        copyTo(mat);
    }
    else
    {
        mat.create(size(), type());
        copy_to_with_mask(*this, mat, mask, "copy_to_with_mask");
    }
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// ConvertTo ////////////////////////////////
///////////////////////////////////////////////////////////////////////////
void convert_run(const oclMat &src, oclMat &dst, double alpha, double beta)
{
    string kernelName = "convert_to_S";
    stringstream idxStr;
    idxStr << src.depth();
    kernelName += idxStr.str();
    float alpha_f = alpha, beta_f = beta;
    CV_DbgAssert(src.rows == dst.rows && src.cols == dst.cols);
    vector<pair<size_t , const void *> > args;
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3];
    globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = 1;
    int dststep_in_pixel = dst.step / dst.elemSize(), dstoffset_in_pixel = dst.offset / dst.elemSize();
    int srcstep_in_pixel = src.step / src.elemSize(), srcoffset_in_pixel = src.offset / src.elemSize();
    if(dst.type() == CV_8UC1)
    {
        globalThreads[0] = ((dst.cols + 4) / 4 + localThreads[0]) / localThreads[0] * localThreads[0];
    }
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&srcstep_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&srcoffset_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dststep_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dstoffset_in_pixel ));
    args.push_back( make_pair( sizeof(cl_float) , (void *)&alpha_f ));
    args.push_back( make_pair( sizeof(cl_float) , (void *)&beta_f ));
    openCLExecuteKernel(dst.clCxt , &operator_convertTo, kernelName, globalThreads,
                        localThreads, args, dst.channels(), dst.depth());
}
void cv::ocl::oclMat::convertTo( oclMat &dst, int rtype, double alpha, double beta ) const
{
    //cout << "cv::ocl::oclMat::convertTo()" << endl;

    bool noScale = fabs(alpha - 1) < std::numeric_limits<double>::epsilon()
                   && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    //int scn = channels();
    int sdepth = depth(), ddepth = CV_MAT_DEPTH(rtype);
    if( sdepth == ddepth && noScale )
    {
        copyTo(dst);
        return;
    }

    oclMat temp;
    const oclMat *psrc = this;
    if( sdepth != ddepth && psrc == &dst )
        psrc = &(temp = *this);

    dst.create( size(), rtype );
    convert_run(*psrc, dst, alpha, beta);
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// setTo ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
oclMat &cv::ocl::oclMat::operator = (const Scalar &s)
{
    //cout << "cv::ocl::oclMat::=" << endl;
    setTo(s);
    return *this;
}
void set_to_withoutmask_run(const oclMat &dst, const Scalar &scalar, string kernelName)
{
    vector<pair<size_t , const void *> > args;
    cl_float4 val;
    val.s[0] = scalar.val[0];
    val.s[1] = scalar.val[1];
    val.s[2] = scalar.val[2];
    val.s[3] = scalar.val[3];
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3];
    globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = 1;
    int step_in_pixel = dst.step / dst.elemSize(), offset_in_pixel = dst.offset / dst.elemSize();
    if(dst.type() == CV_8UC1)
    {
        globalThreads[0] = ((dst.cols + 4) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    }
    args.push_back( make_pair( sizeof(cl_float4) , (void *)&val ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.cols ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.rows ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&offset_in_pixel));
    openCLExecuteKernel(dst.clCxt , &operator_setTo, kernelName, globalThreads,
                        localThreads, args, dst.channels(), dst.depth());
}

void set_to_withmask_run(const oclMat &dst, const Scalar &scalar, const oclMat &mask, string kernelName)
{
    CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols);
    vector<pair<size_t , const void *> > args;
    cl_float4 val;
    val.s[0] = scalar.val[0];
    val.s[1] = scalar.val[1];
    val.s[2] = scalar.val[2];
    val.s[3] = scalar.val[3];
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3];
    globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = 1;
    if(dst.type() == CV_8UC1)
    {
        globalThreads[0] = ((dst.cols + 4) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    }
    int step_in_pixel = dst.step / dst.elemSize(), offset_in_pixel = dst.offset / dst.elemSize();
    args.push_back( make_pair( sizeof(cl_float4) , (void *)&val ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.cols ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.rows ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&offset_in_pixel ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&mask.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.step ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.offset ));
    openCLExecuteKernel(dst.clCxt , &operator_setToM, kernelName, globalThreads,
                        localThreads, args, dst.channels(), dst.depth());
}

oclMat &cv::ocl::oclMat::setTo(const Scalar &scalar, const oclMat &mask)
{
    //cout << "cv::ocl::oclMat::setTo()" << endl;
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert( this->depth() >= 0 && this->depth() <= 6 );
    CV_DbgAssert( !this->empty());
    //cl_int status;
    //cl_mem mem;
    //mem = clCreateBuffer(this->clCxt->clContext,CL_MEM_READ_WRITE,
    //                   sizeof(double)*4,NULL,&status);
    //openCLVerifyCall(status);
    //double* s =  (double *)scalar.val;
    //openCLSafeCall(clEnqueueWriteBuffer(this->clCxt->clCmdQueue,
    //                   (cl_mem)mem,1,0,sizeof(double)*4,s,0,0,0));
    if (mask.empty())
    {
        set_to_withoutmask_run(*this, scalar, "set_to_without_mask");
    }
    else
    {
        set_to_withmask_run(*this, scalar, mask, "set_to_with_mask");
    }

    return *this;
}

oclMat cv::ocl::oclMat::reshape(int new_cn, int new_rows) const
{
	if( new_rows != 0 && new_rows != rows)
	{
		 CV_Error( CV_StsBadFunc,
            "oclMat's number of rows can not be changed for current version" );
	}

	oclMat hdr = *this;

    int cn = channels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(CV_StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            CV_Error(CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        CV_Error(CV_BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);

    return hdr;

}

void cv::ocl::oclMat::create(int _rows, int _cols, int _type)
{
    clCxt = Context::getContext();
    //cout << "cv::ocl::oclMat::create()." << endl;

    /* core logic */
    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;
        wholerows = _rows;
        wholecols = _cols;
        size_t esz = elemSize();

        void *dev_ptr;
        openCLMallocPitch(clCxt, &dev_ptr, &step, GPU_MATRIX_MALLOC_STEP(esz * cols), rows);
        //openCLMallocPitch(clCxt,&dev_ptr, &step, esz * cols, rows);

        if (esz *cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = (int64)step * rows;
        size_t nettosize = (size_t)_nettosize;

        datastart = data = (uchar *)dev_ptr;
        dataend = data + nettosize;

        refcount = (int *)fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

void cv::ocl::oclMat::release()
{
    //cout << "cv::ocl::oclMat::release()" << endl;
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        fastFree(refcount);
        openCLFree(datastart);
    }
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    offset = wholerows = wholecols = 0;
    refcount = 0;
}

#endif /* !defined (HAVE_OPENCL) */
