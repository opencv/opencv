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

////////////////////////////////////////////////////////////////////////
//////////////////////////////// oclMat ////////////////////////////////
////////////////////////////////////////////////////////////////////////

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
        extern const char *convertC3C4;
        extern DevMemType gDeviceMemType;
        extern DevMemRW gDeviceMemRW;
    }
}


////////////////////////////////////////////////////////////////////////
// convert_C3C4
static void convert_C3C4(const cl_mem &src, oclMat &dst)
{
    int dstStep_in_pixel = dst.step1() / dst.oclchannels();
    int pixel_end = dst.wholecols * dst.wholerows - 1;
    Context *clCxt = dst.clCxt;
    String kernelName = "convertC3C4";
    char compile_option[32];
    switch(dst.depth())
    {
    case 0:
        sprintf(compile_option, "-D GENTYPE4=uchar4");
        break;
    case 1:
        sprintf(compile_option, "-D GENTYPE4=char4");
        break;
    case 2:
        sprintf(compile_option, "-D GENTYPE4=ushort4");
        break;
    case 3:
        sprintf(compile_option, "-D GENTYPE4=short4");
        break;
    case 4:
        sprintf(compile_option, "-D GENTYPE4=int4");
        break;
    case 5:
        sprintf(compile_option, "-D GENTYPE4=float4");
        break;
    case 6:
        sprintf(compile_option, "-D GENTYPE4=double4");
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unknown depth");
    }
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.wholecols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.wholerows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstStep_in_pixel));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&pixel_end));

    size_t globalThreads[3] = {((dst.wholecols * dst.wholerows + 3) / 4 + 255) / 256 * 256, 1, 1};
    size_t localThreads[3] = {256, 1, 1};

    openCLExecuteKernel(clCxt, &convertC3C4, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
}
////////////////////////////////////////////////////////////////////////
// convert_C4C3
static void convert_C4C3(const oclMat &src, cl_mem &dst)
{
    int srcStep_in_pixel = src.step1() / src.oclchannels();
    int pixel_end = src.wholecols * src.wholerows - 1;
    Context *clCxt = src.clCxt;
    String kernelName = "convertC4C3";
    char compile_option[32];
    switch(src.depth())
    {
    case 0:
        sprintf(compile_option, "-D GENTYPE4=uchar4");
        break;
    case 1:
        sprintf(compile_option, "-D GENTYPE4=char4");
        break;
    case 2:
        sprintf(compile_option, "-D GENTYPE4=ushort4");
        break;
    case 3:
        sprintf(compile_option, "-D GENTYPE4=short4");
        break;
    case 4:
        sprintf(compile_option, "-D GENTYPE4=int4");
        break;
    case 5:
        sprintf(compile_option, "-D GENTYPE4=float4");
        break;
    case 6:
        sprintf(compile_option, "-D GENTYPE4=double4");
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unknown depth");
    }

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.wholecols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.wholerows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcStep_in_pixel));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&pixel_end));

    size_t globalThreads[3] = {((src.wholecols * src.wholerows + 3) / 4 + 255) / 256 * 256, 1, 1};
    size_t localThreads[3] = {256, 1, 1};

    openCLExecuteKernel(clCxt, &convertC3C4, kernelName, globalThreads, localThreads, args, -1, -1, compile_option);
}

void cv::ocl::oclMat::upload(const Mat &m)
{
    CV_DbgAssert(!m.empty());
    Size wholeSize;
    Point ofs;
    m.locateROI(wholeSize, ofs);
    if(m.channels() == 3)
    {
        create(wholeSize, m.type());
        int pitch = wholeSize.width * 3 * m.elemSize1();
        int tail_padding = m.elemSize1() * 3072;
        int err;
        cl_mem temp;
        if(gDeviceMemType!=DEVICE_MEM_UHP && gDeviceMemType!=DEVICE_MEM_CHP){
            temp = clCreateBuffer((cl_context)clCxt->oclContext(), CL_MEM_READ_WRITE,
                                  (pitch * wholeSize.height + tail_padding - 1) / tail_padding * tail_padding, 0, &err);
            openCLVerifyCall(err);
            openCLMemcpy2D(clCxt, temp, pitch, m.datastart, m.step,
                           wholeSize.width * m.elemSize(), wholeSize.height, clMemcpyHostToDevice, 3);
        }
        else{
            temp = clCreateBuffer((cl_context)clCxt->oclContext(), CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                                  (pitch * wholeSize.height + tail_padding - 1) / tail_padding * tail_padding, m.datastart, &err);
            openCLVerifyCall(err);
        }


        convert_C3C4(temp, *this);
        openCLSafeCall(clReleaseMemObject(temp));
    }
    else
    {
        // try to use host ptr
        createEx(wholeSize, m.type(), gDeviceMemRW, gDeviceMemType, m.datastart);
        if(gDeviceMemType!=DEVICE_MEM_UHP && gDeviceMemType!=DEVICE_MEM_CHP)
            openCLMemcpy2D(clCxt, data, step, m.datastart, m.step,
                           wholeSize.width * elemSize(), wholeSize.height, clMemcpyHostToDevice);
    }

    rows = m.rows;
    cols = m.cols;
    offset = ofs.y * step + ofs.x * elemSize();
}

cv::ocl::oclMat::operator cv::_InputArray()
{
    _InputArray newInputArray;
    newInputArray.flags = cv::_InputArray::OCL_MAT;
    newInputArray.obj   = reinterpret_cast<void *>(this);
    return newInputArray;
}

cv::ocl::oclMat::operator cv::_OutputArray()
{
    _OutputArray newOutputArray;
    newOutputArray.flags = cv::_InputArray::OCL_MAT;
    newOutputArray.obj   = reinterpret_cast<void *>(this);
    return newOutputArray;
}

cv::ocl::oclMat& cv::ocl::getOclMatRef(InputArray src)
{
    CV_Assert(src.flags & cv::_InputArray::OCL_MAT);
    return *reinterpret_cast<oclMat*>(src.obj);
}

cv::ocl::oclMat& cv::ocl::getOclMatRef(OutputArray src)
{
    CV_Assert(src.flags & cv::_InputArray::OCL_MAT);
    return *reinterpret_cast<oclMat*>(src.obj);
}

void cv::ocl::oclMat::download(cv::Mat &m) const
{
    CV_DbgAssert(!this->empty());
    //   int t = type();
    //   if(download_channels == 3)
    //{
    //	t = CV_MAKETYPE(depth(), 3);
    //}
    m.create(wholerows, wholecols, type());

    if(m.channels() == 3)
    {
        int pitch = wholecols * 3 * m.elemSize1();
        int tail_padding = m.elemSize1() * 3072;
        int err;
        cl_mem temp = clCreateBuffer((cl_context)clCxt->oclContext(), CL_MEM_READ_WRITE,
                                     (pitch * wholerows + tail_padding - 1) / tail_padding * tail_padding, 0, &err);
        openCLVerifyCall(err);

        convert_C4C3(*this, temp);
        openCLMemcpy2D(clCxt, m.data, m.step, temp, pitch, wholecols * m.elemSize(), wholerows, clMemcpyDeviceToHost, 3);
        //int* cputemp=new int[wholecols*wholerows * 3];
        //int* cpudata=new int[this->step*this->wholerows/sizeof(int)];
        //openCLSafeCall(clEnqueueReadBuffer(clCxt->impl->clCmdQueue, temp, CL_TRUE,
        //						0, wholecols*wholerows * 3* sizeof(int), cputemp, 0, NULL, NULL));
        //openCLSafeCall(clEnqueueReadBuffer(clCxt->impl->clCmdQueue, (cl_mem)data, CL_TRUE,
        //						0, this->step*this->wholerows, cpudata, 0, NULL, NULL));
        //for(int i=0;i<wholerows;i++)
        //{
        //	int *a = cputemp+i*wholecols * 3,*b = cpudata + i*this->step/sizeof(int);
        //	for(int j=0;j<wholecols;j++)
        //	{
        //		if((a[3*j] != b[4*j])||(a[3*j+1] != b[4*j+1])||(a[3*j+2] != b[4*j+2]))
        //			printf("rows=%d,cols=%d,cputtemp=%d,%d,%d;cpudata=%d,%d,%d\n",
        //			i,j,a[3*j],a[3*j+1],a[3*j+2],b[4*j],b[4*j+1],b[4*j+2]);
        //	}
        //}
        //delete []cputemp;
        //delete []cpudata;
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

/////////////////////common//////////////////////////////////////
inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
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
    size_t globalThreads[3];

    globalThreads[0] = divUp(dst.cols, localThreads[0]) * localThreads[0];
    globalThreads[1] = divUp(dst.rows, localThreads[1]) * localThreads[1];
    globalThreads[2] = 1;

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

void cv::ocl::oclMat::copyTo( oclMat &m ) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    openCLCopyBuffer2D(clCxt, m.data, m.step, m.offset,
                       data, step, cols * elemSize(), rows, offset);
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
static void convert_run(const oclMat &src, oclMat &dst, double alpha, double beta)
{
    String kernelName = "convert_to_S";
    std::stringstream idxStr;
    idxStr << src.depth();
    kernelName = kernelName + idxStr.str().c_str();
    float alpha_f = alpha, beta_f = beta;
    CV_DbgAssert(src.rows == dst.rows && src.cols == dst.cols);
    std::vector<std::pair<size_t , const void *> > args;
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
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&srcstep_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&srcoffset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dststep_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dstoffset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_float) , (void *)&alpha_f ));
    args.push_back( std::make_pair( sizeof(cl_float) , (void *)&beta_f ));
    openCLExecuteKernel(dst.clCxt , &operator_convertTo, kernelName, globalThreads,
                        localThreads, args, dst.oclchannels(), dst.depth());
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
static void set_to_withoutmask_run(const oclMat &dst, const Scalar &scalar, String kernelName)
{
    std::vector<std::pair<size_t , const void *> > args;

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
    char compile_option[32];
    union sc
    {
        cl_uchar4 uval;
        cl_char4  cval;
        cl_ushort4 usval;
        cl_short4 shval;
        cl_int4 ival;
        cl_float4 fval;
        cl_double4 dval;
    } val;
    switch(dst.depth())
    {
    case CV_8U:
        val.uval.s[0] = saturate_cast<uchar>(scalar.val[0]);
        val.uval.s[1] = saturate_cast<uchar>(scalar.val[1]);
        val.uval.s[2] = saturate_cast<uchar>(scalar.val[2]);
        val.uval.s[3] = saturate_cast<uchar>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=uchar");
            args.push_back( std::make_pair( sizeof(cl_uchar) , (void *)&val.uval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=uchar4");
            args.push_back( std::make_pair( sizeof(cl_uchar4) , (void *)&val.uval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_8S:
        val.cval.s[0] = saturate_cast<char>(scalar.val[0]);
        val.cval.s[1] = saturate_cast<char>(scalar.val[1]);
        val.cval.s[2] = saturate_cast<char>(scalar.val[2]);
        val.cval.s[3] = saturate_cast<char>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=char");
            args.push_back( std::make_pair( sizeof(cl_char) , (void *)&val.cval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=char4");
            args.push_back( std::make_pair( sizeof(cl_char4) , (void *)&val.cval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_16U:
        val.usval.s[0] = saturate_cast<ushort>(scalar.val[0]);
        val.usval.s[1] = saturate_cast<ushort>(scalar.val[1]);
        val.usval.s[2] = saturate_cast<ushort>(scalar.val[2]);
        val.usval.s[3] = saturate_cast<ushort>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=ushort");
            args.push_back( std::make_pair( sizeof(cl_ushort) , (void *)&val.usval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=ushort4");
            args.push_back( std::make_pair( sizeof(cl_ushort4) , (void *)&val.usval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_16S:
        val.shval.s[0] = saturate_cast<short>(scalar.val[0]);
        val.shval.s[1] = saturate_cast<short>(scalar.val[1]);
        val.shval.s[2] = saturate_cast<short>(scalar.val[2]);
        val.shval.s[3] = saturate_cast<short>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=short");
            args.push_back( std::make_pair( sizeof(cl_short) , (void *)&val.shval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=short4");
            args.push_back( std::make_pair( sizeof(cl_short4) , (void *)&val.shval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_32S:
        val.ival.s[0] = saturate_cast<int>(scalar.val[0]);
        val.ival.s[1] = saturate_cast<int>(scalar.val[1]);
        val.ival.s[2] = saturate_cast<int>(scalar.val[2]);
        val.ival.s[3] = saturate_cast<int>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=int");
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&val.ival.s[0] ));
            break;
        case 2:
            sprintf(compile_option, "-D GENTYPE=int2");
            cl_int2 i2val;
            i2val.s[0] = val.ival.s[0];
            i2val.s[1] = val.ival.s[1];
            args.push_back( std::make_pair( sizeof(cl_int2) , (void *)&i2val ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=int4");
            args.push_back( std::make_pair( sizeof(cl_int4) , (void *)&val.ival ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_32F:
        val.fval.s[0] = scalar.val[0];
        val.fval.s[1] = scalar.val[1];
        val.fval.s[2] = scalar.val[2];
        val.fval.s[3] = scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=float");
            args.push_back( std::make_pair( sizeof(cl_float) , (void *)&val.fval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=float4");
            args.push_back( std::make_pair( sizeof(cl_float4) , (void *)&val.fval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_64F:
        val.dval.s[0] = scalar.val[0];
        val.dval.s[1] = scalar.val[1];
        val.dval.s[2] = scalar.val[2];
        val.dval.s[3] = scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=double");
            args.push_back( std::make_pair( sizeof(cl_double) , (void *)&val.dval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=double4");
            args.push_back( std::make_pair( sizeof(cl_double4) , (void *)&val.dval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unknown depth");
    }
#ifdef CL_VERSION_1_2
    //this enables backwards portability to
    //run on OpenCL 1.1 platform if library binaries are compiled with OpenCL 1.2 support
    if(Context::getContext()->supportsFeature(Context::CL_VER_1_2) &&
        dst.offset == 0 && dst.cols == dst.wholecols)
    {
        clEnqueueFillBuffer((cl_command_queue)dst.clCxt->oclCommandQueue(),
            (cl_mem)dst.data, args[0].second, args[0].first, 0, dst.step * dst.rows, 0, NULL, NULL);
    }
    else
#endif
    {
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset_in_pixel));
        openCLExecuteKernel(dst.clCxt , &operator_setTo, kernelName, globalThreads,
            localThreads, args, -1, -1, compile_option);
    }
}

static void set_to_withmask_run(const oclMat &dst, const Scalar &scalar, const oclMat &mask, String kernelName)
{
    CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols);
    std::vector<std::pair<size_t , const void *> > args;
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3];
    globalThreads[0] = (dst.cols + localThreads[0] - 1) / localThreads[0] * localThreads[0];
    globalThreads[1] = (dst.rows + localThreads[1] - 1) / localThreads[1] * localThreads[1];
    globalThreads[2] = 1;
    int step_in_pixel = dst.step / dst.elemSize(), offset_in_pixel = dst.offset / dst.elemSize();
    char compile_option[32];
    union sc
    {
        cl_uchar4 uval;
        cl_char4  cval;
        cl_ushort4 usval;
        cl_short4 shval;
        cl_int4 ival;
        cl_float4 fval;
        cl_double4 dval;
    } val;
    switch(dst.depth())
    {
    case CV_8U:
        val.uval.s[0] = saturate_cast<uchar>(scalar.val[0]);
        val.uval.s[1] = saturate_cast<uchar>(scalar.val[1]);
        val.uval.s[2] = saturate_cast<uchar>(scalar.val[2]);
        val.uval.s[3] = saturate_cast<uchar>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=uchar");
            args.push_back( std::make_pair( sizeof(cl_uchar) , (void *)&val.uval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=uchar4");
            args.push_back( std::make_pair( sizeof(cl_uchar4) , (void *)&val.uval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_8S:
        val.cval.s[0] = saturate_cast<char>(scalar.val[0]);
        val.cval.s[1] = saturate_cast<char>(scalar.val[1]);
        val.cval.s[2] = saturate_cast<char>(scalar.val[2]);
        val.cval.s[3] = saturate_cast<char>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=char");
            args.push_back( std::make_pair( sizeof(cl_char) , (void *)&val.cval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=char4");
            args.push_back( std::make_pair( sizeof(cl_char4) , (void *)&val.cval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_16U:
        val.usval.s[0] = saturate_cast<ushort>(scalar.val[0]);
        val.usval.s[1] = saturate_cast<ushort>(scalar.val[1]);
        val.usval.s[2] = saturate_cast<ushort>(scalar.val[2]);
        val.usval.s[3] = saturate_cast<ushort>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=ushort");
            args.push_back( std::make_pair( sizeof(cl_ushort) , (void *)&val.usval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=ushort4");
            args.push_back( std::make_pair( sizeof(cl_ushort4) , (void *)&val.usval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_16S:
        val.shval.s[0] = saturate_cast<short>(scalar.val[0]);
        val.shval.s[1] = saturate_cast<short>(scalar.val[1]);
        val.shval.s[2] = saturate_cast<short>(scalar.val[2]);
        val.shval.s[3] = saturate_cast<short>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=short");
            args.push_back( std::make_pair( sizeof(cl_short) , (void *)&val.shval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=short4");
            args.push_back( std::make_pair( sizeof(cl_short4) , (void *)&val.shval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_32S:
        val.ival.s[0] = saturate_cast<int>(scalar.val[0]);
        val.ival.s[1] = saturate_cast<int>(scalar.val[1]);
        val.ival.s[2] = saturate_cast<int>(scalar.val[2]);
        val.ival.s[3] = saturate_cast<int>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=int");
            args.push_back( std::make_pair( sizeof(cl_int) , (void *)&val.ival.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=int4");
            args.push_back( std::make_pair( sizeof(cl_int4) , (void *)&val.ival ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_32F:
        val.fval.s[0] = scalar.val[0];
        val.fval.s[1] = scalar.val[1];
        val.fval.s[2] = scalar.val[2];
        val.fval.s[3] = scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=float");
            args.push_back( std::make_pair( sizeof(cl_float) , (void *)&val.fval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=float4");
            args.push_back( std::make_pair( sizeof(cl_float4) , (void *)&val.fval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case CV_64F:
        val.dval.s[0] = scalar.val[0];
        val.dval.s[1] = scalar.val[1];
        val.dval.s[2] = scalar.val[2];
        val.dval.s[3] = scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=double");
            args.push_back( std::make_pair( sizeof(cl_double) , (void *)&val.dval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=double4");
            args.push_back( std::make_pair( sizeof(cl_double4) , (void *)&val.dval ));
            break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported channels");
        }
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unknown depth");
    }
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset_in_pixel ));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&mask.data ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.step ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&mask.offset ));
    openCLExecuteKernel(dst.clCxt , &operator_setToM, kernelName, globalThreads,
                        localThreads, args, -1, -1, compile_option);
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
        if(type() == CV_8UC1)
        {
            set_to_withoutmask_run(*this, scalar, "set_to_without_mask_C1_D0");
        }
        else
        {
            set_to_withoutmask_run(*this, scalar, "set_to_without_mask");
        }
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
                               DevMemRW rw_type, DevMemType mem_type, void* hptr)
{
    createEx(size.height, size.width, type, rw_type, mem_type, hptr);
}

void cv::ocl::oclMat::create(int _rows, int _cols, int _type)
{
    createEx(_rows, _cols, _type, gDeviceMemRW, gDeviceMemType);
}

void cv::ocl::oclMat::createEx(int _rows, int _cols, int _type,
                               DevMemRW rw_type, DevMemType mem_type, void* hptr)
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
        openCLMallocPitch(clCxt, &dev_ptr, &step, GPU_MATRIX_MALLOC_STEP(esz * cols),
                            rows, rw_type, mem_type, hptr);

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
