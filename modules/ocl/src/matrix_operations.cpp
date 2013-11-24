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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
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

#include "precomp.hpp"
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

#define ALIGN 32
#define GPU_MATRIX_MALLOC_STEP(step) (((step) + ALIGN - 1) / ALIGN) * ALIGN

// helper routines
namespace cv
{
    namespace ocl
    {
        extern DevMemType gDeviceMemType;
        extern DevMemRW gDeviceMemRW;
    }
}

////////////////////////////////////////////////////////////////////////
// convert_C3C4

static void convert_C3C4(const cl_mem &src, oclMat &dst)
{
    Context *clCxt = dst.clCxt;
    int pixel_end = dst.wholecols * dst.wholerows - 1;
    int dstStep_in_pixel = dst.step1() / dst.oclchannels();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D GENTYPE4=%s4", typeMap[dst.depth()]);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.wholecols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.wholerows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstStep_in_pixel));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&pixel_end));

    size_t globalThreads[3] = { divUp(dst.wholecols * dst.wholerows, 4), 1, 1 };
    size_t localThreads[3] = { 256, 1, 1 };

    openCLExecuteKernel(clCxt, &convertC3C4, "convertC3C4", globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

////////////////////////////////////////////////////////////////////////
// convert_C4C3

static void convert_C4C3(const oclMat &src, cl_mem &dst)
{
    int srcStep_in_pixel = src.step1() / src.oclchannels();
    int pixel_end = src.wholecols * src.wholerows - 1;
    Context *clCxt = src.clCxt;

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D GENTYPE4=%s4", typeMap[src.depth()]);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.wholecols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.wholerows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcStep_in_pixel));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&pixel_end));

    size_t globalThreads[3] = { divUp(src.wholecols * src.wholerows, 4), 1, 1};
    size_t localThreads[3] = { 256, 1, 1 };

    openCLExecuteKernel(clCxt, &convertC3C4, "convertC4C3", globalThreads, localThreads, args, -1, -1, buildOptions.c_str());
}

void cv::ocl::oclMat::upload(const Mat &m)
{
    if (!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE) && m.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_DbgAssert(!m.empty());
    Size wholeSize;
    Point ofs;
    m.locateROI(wholeSize, ofs);
    create(wholeSize, m.type());

    if (m.channels() == 3)
    {
        int pitch = wholeSize.width * 3 * m.elemSize1();
        int tail_padding = m.elemSize1() * 3072;
        int err;
        cl_mem temp = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE,
                                     (pitch * wholeSize.height + tail_padding - 1) / tail_padding * tail_padding, 0, &err);
        openCLVerifyCall(err);

        openCLMemcpy2D(clCxt, temp, pitch, m.datastart, m.step, wholeSize.width * m.elemSize(), wholeSize.height, clMemcpyHostToDevice, 3);
        convert_C3C4(temp, *this);
        openCLSafeCall(clReleaseMemObject(temp));
    }
    else
        openCLMemcpy2D(clCxt, data, step, m.datastart, m.step, wholeSize.width * elemSize(), wholeSize.height, clMemcpyHostToDevice);

    rows = m.rows;
    cols = m.cols;
    offset = ofs.y * step + ofs.x * elemSize();
}

cv::ocl::oclMat::operator cv::_InputArray()
{
    return _InputArray(cv::_InputArray::OCL_MAT, this);
}

cv::ocl::oclMat::operator cv::_OutputArray()
{
    return _OutputArray(cv::_InputArray::OCL_MAT, this);
}

cv::ocl::oclMat& cv::ocl::getOclMatRef(InputArray src)
{
    CV_Assert(src.kind() == cv::_InputArray::OCL_MAT);
    return *(oclMat*)src.getObj();
}

cv::ocl::oclMat& cv::ocl::getOclMatRef(OutputArray src)
{
    CV_Assert(src.kind() == cv::_InputArray::OCL_MAT);
    return *(oclMat*)src.getObj();
}

void cv::ocl::oclMat::download(cv::Mat &m) const
{
    CV_DbgAssert(!this->empty());
    m.create(wholerows, wholecols, type());

    if(m.channels() == 3)
    {
        int pitch = wholecols * 3 * m.elemSize1();
        int tail_padding = m.elemSize1() * 3072;
        int err;
        cl_mem temp = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(), CL_MEM_READ_WRITE,
                                     (pitch * wholerows + tail_padding - 1) / tail_padding * tail_padding, 0, &err);
        openCLVerifyCall(err);

        convert_C4C3(*this, temp);
        openCLMemcpy2D(clCxt, m.data, m.step, temp, pitch, wholecols * m.elemSize(), wholerows, clMemcpyDeviceToHost, 3);
        openCLSafeCall(clReleaseMemObject(temp));
    }
    else
    {
        openCLMemcpy2D(clCxt, m.data, m.step, data, step, wholecols * elemSize(), wholerows, clMemcpyDeviceToHost);
    }

    Size wholesize;
    Point ofs;
    locateROI(wholesize, ofs);
    m.adjustROI(-ofs.y, ofs.y + rows - wholerows, -ofs.x, ofs.x + cols - wholecols);
}

///////////////////////////////////////////////////////////////////////////
////////////////////////////////// CopyTo /////////////////////////////////
///////////////////////////////////////////////////////////////////////////
static void copy_to_with_mask(const oclMat &src, oclMat &dst, const oclMat &mask, String kernelName)
{
    CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols &&
                  src.rows == dst.rows && src.cols == dst.cols
                  && mask.type() == CV_8UC1);

    std::vector<std::pair<size_t , const void *> > args;

    String string_types[4][7] = {{"uchar", "char", "ushort", "short", "int", "float", "double"},
        {"uchar2", "char2", "ushort2", "short2", "int2", "float2", "double2"},
        {"uchar3", "char3", "ushort3", "short3", "int3", "float3", "double3"},
        {"uchar4", "char4", "ushort4", "short4", "int4", "float4", "double4"}
    };

    char compile_option[32];
    sprintf(compile_option, "-D GENTYPE=%s", string_types[dst.oclchannels() - 1][dst.depth()].c_str());
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    int dststep_in_pixel = dst.step / dst.elemSize(), dstoffset_in_pixel = dst.offset / dst.elemSize();
    int srcstep_in_pixel = src.step / src.elemSize(), srcoffset_in_pixel = src.offset / src.elemSize();

    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&mask.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&srcstep_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&srcoffset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dststep_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstoffset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.step ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.offset ));

    openCLExecuteKernel(dst.clCxt , &operator_copyToM, kernelName, globalThreads,
                        localThreads, args, -1, -1, compile_option);
}

void cv::ocl::oclMat::copyTo( oclMat &mat, const oclMat &mask) const
{
    if (mask.empty())
    {
        CV_DbgAssert(!this->empty());
        mat.create(size(), type());
        openCLCopyBuffer2D(clCxt, mat.data, mat.step, mat.offset,
                           data, step, cols * elemSize(), rows, offset);
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

static void convert_run(const oclMat &src, oclMat &dst, double alpha, double beta)
{
    String kernelName = "convert_to";
    float alpha_f = alpha, beta_f = beta;
    int sdepth = src.depth(), ddepth = dst.depth();
    int sstep1 = (int)src.step1(), dstep1 = (int)dst.step1();
    int cols1 = src.cols * src.oclchannels();

    char buildOptions[150], convertString[50];
    const char * typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    sprintf(convertString, "convert_%s_sat_rte", typeMap[ddepth]);
    sprintf(buildOptions, "-D srcT=%s -D dstT=%s -D convertToDstType=%s", typeMap[sdepth],
            typeMap[ddepth], CV_32F == ddepth || ddepth == CV_64F ? "" : convertString);

    CV_DbgAssert(src.rows == dst.rows && src.cols == dst.cols);
    std::vector<std::pair<size_t , const void *> > args;

    size_t localThreads[3] = { 16, 16, 1 };
    size_t globalThreads[3] = { divUp(cols1, localThreads[0]) * localThreads[0],
                                divUp(dst.rows, localThreads[1]) * localThreads[1], 1 };

    int doffset1 = dst.offset / dst.elemSize1();
    int soffset1 = src.offset / src.elemSize1();

    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols1 ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&sstep1 ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&soffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstep1 ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&doffset1 ));
    args.push_back( std::make_pair( sizeof(cl_float) , (void *)&alpha_f ));
    args.push_back( std::make_pair( sizeof(cl_float) , (void *)&beta_f ));

    openCLExecuteKernel(dst.clCxt , &operator_convertTo, kernelName, globalThreads,
                        localThreads, args, -1, -1, buildOptions);
}

void cv::ocl::oclMat::convertTo( oclMat &dst, int rtype, double alpha, double beta ) const
{
    if (!clCxt->supportsFeature(FEATURE_CL_DOUBLE) &&
            (depth() == CV_64F || dst.depth() == CV_64F))
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    bool noScale = fabs(alpha - 1) < std::numeric_limits<double>::epsilon()
                   && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

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
    setTo(s);
    return *this;
}

#ifdef CL_VERSION_1_2

template <typename CLT, typename PT>
static std::vector<uchar> cvt1(const cv::Scalar & s)
{
    std::vector<uchar> _buf(sizeof(CLT));
    CLT * const buf = reinterpret_cast<CLT *>(&_buf[0]);
    buf[0] = saturate_cast<PT>(s[0]);
    return _buf;
}

template <typename CLT, typename PT>
static std::vector<uchar> cvt2(const cv::Scalar & s)
{
    std::vector<uchar> _buf(sizeof(CLT));
    CLT * const buf = reinterpret_cast<CLT *>(&_buf[0]);
    buf->s[0] = saturate_cast<PT>(s[0]);
    buf->s[1] = saturate_cast<PT>(s[1]);
    return _buf;
}

template <typename CLT, typename PT>
static std::vector<uchar> cvt4(const cv::Scalar & s)
{
    std::vector<uchar> _buf(sizeof(CLT));
    CLT * const buf = reinterpret_cast<CLT *>(&_buf[0]);
    buf->s[0] = saturate_cast<PT>(s[0]);
    buf->s[1] = saturate_cast<PT>(s[1]);
    buf->s[2] = saturate_cast<PT>(s[2]);
    buf->s[3] = saturate_cast<PT>(s[3]);
    return _buf;
}

typedef std::vector<uchar> (*ConvertFunc)(const cv::Scalar & s);

static std::vector<uchar> scalarToCLVector(const cv::Scalar & s, int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int channels = CV_MAT_CN(type);

    static const ConvertFunc funcs[4][7] =
    {
        { cvt1<cl_uchar, uchar>, cvt1<cl_char, char>, cvt1<cl_ushort, ushort>, cvt1<cl_short, short>,
          cvt1<cl_int, int>, cvt1<cl_float, float>, cvt1<cl_double, double> },

        { cvt2<cl_uchar2, uchar>, cvt2<cl_char2, char>, cvt2<cl_ushort2, ushort>, cvt2<cl_short2, short>,
          cvt2<cl_int2, int>, cvt2<cl_float2, float>, cvt2<cl_double2, double> },

        { 0, 0, 0, 0, 0, 0, 0 },

        { cvt4<cl_uchar4, uchar>, cvt4<cl_char4, char>, cvt4<cl_ushort4, ushort>, cvt4<cl_short4, short>,
          cvt4<cl_int4, int>, cvt4<cl_float4, float>, cvt4<cl_double4, double> }
    };

    ConvertFunc func = funcs[channels - 1][depth];
    return func(s);
}

#endif

static void set_to_withoutmask_run(const oclMat &dst, const Scalar &scalar, String kernelName)
{
    std::vector<std::pair<size_t , const void *> > args;

    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };
    int step_in_pixel = dst.step / dst.elemSize(), offset_in_pixel = dst.offset / dst.elemSize();

    if (dst.type() == CV_8UC1)
        globalThreads[0] = ((dst.cols + 4) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char channelMap[] = { ' ', ' ', '2', '4', '4' };
    std::string buildOptions = format("-D GENTYPE=%s%c", typeMap[dst.depth()], channelMap[dst.channels()]);

    Mat mat(1, 1, dst.type(), scalar);

#ifdef CL_VERSION_1_2
    // this enables backwards portability to
    // run on OpenCL 1.1 platform if library binaries are compiled with OpenCL 1.2 support
    if (Context::getContext()->supportsFeature(FEATURE_CL_VER_1_2) && dst.isContinuous())
    {
        std::vector<uchar> p = ::scalarToCLVector(scalar, CV_MAKE_TYPE(dst.depth(), dst.oclchannels()));
        clEnqueueFillBuffer(getClCommandQueue(dst.clCxt),
                (cl_mem)dst.data, (void*)&p[0], p.size(),
                0, dst.step * dst.rows, 0, NULL, NULL);
    }
    else
#endif
    {
        oclMat m(mat);
        args.push_back( std::make_pair( sizeof(cl_mem) , (void*)&m.data ));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset_in_pixel ));

        openCLExecuteKernel(dst.clCxt , &operator_setTo, kernelName, globalThreads,
            localThreads, args, -1, -1, buildOptions.c_str());
    }
}

static void set_to_withmask_run(const oclMat &dst, const Scalar &scalar, const oclMat &mask, String kernelName)
{
    CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols);
    std::vector<std::pair<size_t , const void *> > args;
    size_t localThreads[3] = { 16, 16, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };
    int step_in_pixel = dst.step / dst.elemSize(), offset_in_pixel = dst.offset / dst.elemSize();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char channelMap[] = { ' ', ' ', '2', '4', '4' };
    std::string buildOptions = format("-D GENTYPE=%s%c", typeMap[dst.depth()], channelMap[dst.channels()]);

    oclMat m(Mat(1, 1, dst.type(), scalar));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&m.data ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&mask.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.step ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.offset ));
    openCLExecuteKernel(dst.clCxt , &operator_setToM, kernelName, globalThreads,
                        localThreads, args, -1, -1, buildOptions.c_str());
}

oclMat &cv::ocl::oclMat::setTo(const Scalar &scalar, const oclMat &mask)
{
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert( this->depth() >= 0 && this->depth() <= 6 );
    CV_DbgAssert( !this->empty());
    if (mask.empty())
    {
        set_to_withoutmask_run(*this, scalar, type() == CV_8UC1 ?
                                   "set_to_without_mask_C1_D0" : "set_to_without_mask");
    }
    else
        set_to_withmask_run(*this, scalar, mask, "set_to_with_mask");

    return *this;
}

oclMat cv::ocl::oclMat::reshape(int new_cn, int new_rows) const
{
    if( new_rows != 0 && new_rows != rows)
    {
        CV_Error( Error::StsBadFunc, "oclMat's number of rows can not be changed for current version" );
    }

    oclMat hdr = *this;

    int cn = oclchannels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;
    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(Error::BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(Error::StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;
        if (total_width * new_rows != total_size)
            CV_Error(Error::StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;
    if (new_width * new_cn != total_width)
        CV_Error(Error::BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.wholecols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);
    return hdr;

}

void cv::ocl::oclMat::createEx(Size size, int type,
                               DevMemRW rw_type, DevMemType mem_type)
{
    createEx(size.height, size.width, type, rw_type, mem_type);
}

void cv::ocl::oclMat::create(int _rows, int _cols, int _type)
{
    createEx(_rows, _cols, _type, gDeviceMemRW, gDeviceMemType);
}

void cv::ocl::oclMat::createEx(int _rows, int _cols, int _type,
                               DevMemRW rw_type, DevMemType mem_type)
{
    clCxt = Context::getContext();
    /* core logic */
    _type &= Mat::TYPE_MASK;
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
        openCLMallocPitchEx(clCxt, &dev_ptr, &step, GPU_MATRIX_MALLOC_STEP(esz * cols), rows, rw_type, mem_type);

        if (esz * cols == step)
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

oclMat& cv::ocl::oclMat::operator+=( const oclMat& m )
{
    add(*this, m, *this);
    return *this;
}

oclMat& cv::ocl::oclMat::operator-=( const oclMat& m )
{
    subtract(*this, m, *this);
    return *this;
}

oclMat& cv::ocl::oclMat::operator*=( const oclMat& m )
{
    multiply(*this, m, *this);
    return *this;
}

oclMat& cv::ocl::oclMat::operator/=( const oclMat& m )
{
    divide(*this, m, *this);
    return *this;
}
