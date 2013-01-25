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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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
#include "mcwutil.hpp"
using namespace std;
using namespace cv;
using namespace cv::ocl;

#if !defined (HAVE_OPENCL)

void cv::ocl::PyrLKOpticalFlow::sparse(const oclMat &, const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat &) {  }
void cv::ocl::PyrLKOpticalFlow::dense(const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat *) {  }

#else /* !defined (HAVE_OPENCL) */

namespace cv
{
namespace ocl
{
///////////////////////////OpenCL kernel strings///////////////////////////
extern const char *pyrlk;
extern const char *pyrlk_no_image;
extern const char *operator_setTo;
extern const char *operator_convertTo;
extern const char *operator_copyToM;
extern const char *arithm_mul;
extern const char *pyr_down;
}
}

struct dim3
{
    unsigned int x, y, z;
};

struct float2
{
    float x, y;
};

struct int2
{
    int x, y;
};

namespace
{
void calcPatchSize(cv::Size winSize, int cn, dim3 &block, dim3 &patch, bool isDeviceArch11)
{
    winSize.width *= cn;

    if (winSize.width > 32 && winSize.width > 2 * winSize.height)
    {
        block.x = isDeviceArch11 ? 16 : 32;
        block.y = 8;
    }
    else
    {
        block.x = 16;
        block.y = isDeviceArch11 ? 8 : 16;
    }

    patch.x = (winSize.width  + block.x - 1) / block.x;
    patch.y = (winSize.height + block.y - 1) / block.y;

    block.z = patch.z = 1;
}
}

inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// ConvertTo ////////////////////////////////
///////////////////////////////////////////////////////////////////////////
static void convert_run_cus(const oclMat &src, oclMat &dst, double alpha, double beta)
{
    string kernelName = "convert_to_S";
    stringstream idxStr;
    idxStr << src.depth();
    kernelName += idxStr.str();
    float alpha_f = (float)alpha, beta_f = (float)beta;
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
    openCLExecuteKernel2(dst.clCxt , &operator_convertTo, kernelName, globalThreads,
                         localThreads, args, dst.oclchannels(), dst.depth(), CLFLUSH);
}
void convertTo( const oclMat &src, oclMat &m, int rtype, double alpha = 1, double beta = 0 );
void convertTo( const oclMat &src, oclMat &dst, int rtype, double alpha, double beta )
{
    //cout << "cv::ocl::oclMat::convertTo()" << endl;

    bool noScale = fabs(alpha - 1) < std::numeric_limits<double>::epsilon()
                   && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = src.type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), src.oclchannels());

    int sdepth = src.depth(), ddepth = CV_MAT_DEPTH(rtype);
    if( sdepth == ddepth && noScale )
    {
        src.copyTo(dst);
        return;
    }

    oclMat temp;
    const oclMat *psrc = &src;
    if( sdepth != ddepth && psrc == &dst )
        psrc = &(temp = src);

    dst.create( src.size(), rtype );
    convert_run_cus(*psrc, dst, alpha, beta);
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// setTo ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//oclMat &operator = (const Scalar &s)
//{
//    //cout << "cv::ocl::oclMat::=" << endl;
//    setTo(s);
//    return *this;
//}
static void set_to_withoutmask_run_cus(const oclMat &dst, const Scalar &scalar, string kernelName)
{
    vector<pair<size_t , const void *> > args;

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
    case 0:
        val.uval.s[0] = saturate_cast<uchar>(scalar.val[0]);
        val.uval.s[1] = saturate_cast<uchar>(scalar.val[1]);
        val.uval.s[2] = saturate_cast<uchar>(scalar.val[2]);
        val.uval.s[3] = saturate_cast<uchar>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=uchar");
            args.push_back( make_pair( sizeof(cl_uchar) , (void *)&val.uval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=uchar4");
            args.push_back( make_pair( sizeof(cl_uchar4) , (void *)&val.uval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 1:
        val.cval.s[0] = saturate_cast<char>(scalar.val[0]);
        val.cval.s[1] = saturate_cast<char>(scalar.val[1]);
        val.cval.s[2] = saturate_cast<char>(scalar.val[2]);
        val.cval.s[3] = saturate_cast<char>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=char");
            args.push_back( make_pair( sizeof(cl_char) , (void *)&val.cval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=char4");
            args.push_back( make_pair( sizeof(cl_char4) , (void *)&val.cval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 2:
        val.usval.s[0] = saturate_cast<ushort>(scalar.val[0]);
        val.usval.s[1] = saturate_cast<ushort>(scalar.val[1]);
        val.usval.s[2] = saturate_cast<ushort>(scalar.val[2]);
        val.usval.s[3] = saturate_cast<ushort>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=ushort");
            args.push_back( make_pair( sizeof(cl_ushort) , (void *)&val.usval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=ushort4");
            args.push_back( make_pair( sizeof(cl_ushort4) , (void *)&val.usval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 3:
        val.shval.s[0] = saturate_cast<short>(scalar.val[0]);
        val.shval.s[1] = saturate_cast<short>(scalar.val[1]);
        val.shval.s[2] = saturate_cast<short>(scalar.val[2]);
        val.shval.s[3] = saturate_cast<short>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=short");
            args.push_back( make_pair( sizeof(cl_short) , (void *)&val.shval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=short4");
            args.push_back( make_pair( sizeof(cl_short4) , (void *)&val.shval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 4:
        val.ival.s[0] = saturate_cast<int>(scalar.val[0]);
        val.ival.s[1] = saturate_cast<int>(scalar.val[1]);
        val.ival.s[2] = saturate_cast<int>(scalar.val[2]);
        val.ival.s[3] = saturate_cast<int>(scalar.val[3]);
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=int");
            args.push_back( make_pair( sizeof(cl_int) , (void *)&val.ival.s[0] ));
            break;
        case 2:
            sprintf(compile_option, "-D GENTYPE=int2");
            cl_int2 i2val;
            i2val.s[0] = val.ival.s[0];
            i2val.s[1] = val.ival.s[1];
            args.push_back( make_pair( sizeof(cl_int2) , (void *)&i2val ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=int4");
            args.push_back( make_pair( sizeof(cl_int4) , (void *)&val.ival ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 5:
        val.fval.s[0] = (float)scalar.val[0];
        val.fval.s[1] = (float)scalar.val[1];
        val.fval.s[2] = (float)scalar.val[2];
        val.fval.s[3] = (float)scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=float");
            args.push_back( make_pair( sizeof(cl_float) , (void *)&val.fval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=float4");
            args.push_back( make_pair( sizeof(cl_float4) , (void *)&val.fval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    case 6:
        val.dval.s[0] = scalar.val[0];
        val.dval.s[1] = scalar.val[1];
        val.dval.s[2] = scalar.val[2];
        val.dval.s[3] = scalar.val[3];
        switch(dst.oclchannels())
        {
        case 1:
            sprintf(compile_option, "-D GENTYPE=double");
            args.push_back( make_pair( sizeof(cl_double) , (void *)&val.dval.s[0] ));
            break;
        case 4:
            sprintf(compile_option, "-D GENTYPE=double4");
            args.push_back( make_pair( sizeof(cl_double4) , (void *)&val.dval ));
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "unsupported channels");
        }
        break;
    default:
        CV_Error(CV_StsUnsupportedFormat, "unknown depth");
    }
#ifdef CL_VERSION_1_2
    if(dst.offset == 0 && dst.cols == dst.wholecols)
    {
        clEnqueueFillBuffer(dst.clCxt->impl->clCmdQueue, (cl_mem)dst.data, args[0].second, args[0].first, 0, dst.step * dst.rows, 0, NULL, NULL);
    }
    else
    {
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.cols ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.rows ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&offset_in_pixel));
        openCLExecuteKernel2(dst.clCxt , &operator_setTo, kernelName, globalThreads,
                             localThreads, args, -1, -1, compile_option, CLFLUSH);
    }
#else
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.cols ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&dst.rows ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&step_in_pixel ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&offset_in_pixel));
    openCLExecuteKernel2(dst.clCxt , &operator_setTo, kernelName, globalThreads,
                         localThreads, args, -1, -1, compile_option, CLFLUSH);
#endif
}

static oclMat &setTo(oclMat &src, const Scalar &scalar)
{
    CV_Assert( src.depth() >= 0 && src.depth() <= 6 );
    CV_DbgAssert( !src.empty());

    if(src.type() == CV_8UC1)
    {
        set_to_withoutmask_run_cus(src, scalar, "set_to_without_mask_C1_D0");
    }
    else
    {
        set_to_withoutmask_run_cus(src, scalar, "set_to_without_mask");
    }

    return src;
}

///////////////////////////////////////////////////////////////////////////
////////////////////////////////// CopyTo /////////////////////////////////
///////////////////////////////////////////////////////////////////////////
// static void copy_to_with_mask_cus(const oclMat &src, oclMat &dst, const oclMat &mask, string kernelName)
// {
//     CV_DbgAssert( dst.rows == mask.rows && dst.cols == mask.cols &&
//                   src.rows == dst.rows && src.cols == dst.cols
//                   && mask.type() == CV_8UC1);

//     vector<pair<size_t , const void *> > args;

//     std::string string_types[4][7] = {{"uchar", "char", "ushort", "short", "int", "float", "double"},
//         {"uchar2", "char2", "ushort2", "short2", "int2", "float2", "double2"},
//         {"uchar3", "char3", "ushort3", "short3", "int3", "float3", "double3"},
//         {"uchar4", "char4", "ushort4", "short4", "int4", "float4", "double4"}
//     };
//     char compile_option[32];
//     sprintf(compile_option, "-D GENTYPE=%s", string_types[dst.oclchannels() - 1][dst.depth()].c_str());
//     size_t localThreads[3] = {16, 16, 1};
//     size_t globalThreads[3];

//     globalThreads[0] = divUp(dst.cols, localThreads[0]) * localThreads[0];
//     globalThreads[1] = divUp(dst.rows, localThreads[1]) * localThreads[1];
//     globalThreads[2] = 1;

//     int dststep_in_pixel = dst.step / dst.elemSize(), dstoffset_in_pixel = dst.offset / dst.elemSize();
//     int srcstep_in_pixel = src.step / src.elemSize(), srcoffset_in_pixel = src.offset / src.elemSize();

//     args.push_back( make_pair( sizeof(cl_mem) , (void *)&src.data ));
//     args.push_back( make_pair( sizeof(cl_mem) , (void *)&dst.data ));
//     args.push_back( make_pair( sizeof(cl_mem) , (void *)&mask.data ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&src.cols ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&src.rows ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&srcstep_in_pixel ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&srcoffset_in_pixel ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&dststep_in_pixel ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&dstoffset_in_pixel ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.step ));
//     args.push_back( make_pair( sizeof(cl_int) , (void *)&mask.offset ));

//     openCLExecuteKernel2(dst.clCxt , &operator_copyToM, kernelName, globalThreads,
//                          localThreads, args, -1, -1, compile_option, CLFLUSH);
// }

static void copyTo(const oclMat &src, oclMat &m )
{
    CV_DbgAssert(!src.empty());
    m.create(src.size(), src.type());
    openCLCopyBuffer2D(src.clCxt, m.data, m.step, m.offset,
                       src.data, src.step, src.cols * src.elemSize(), src.rows, src.offset);
}

// static void copyTo(const oclMat &src, oclMat &mat, const oclMat &mask)
// {
//     if (mask.empty())
//     {
//         copyTo(src, mat);
//     }
//     else
//     {
//         mat.create(src.size(), src.type());
//         copy_to_with_mask_cus(src, mat, mask, "copy_to_with_mask");
//     }
// }

static void arithmetic_run(const oclMat &src1, oclMat &dst, string kernelName, const char **kernelString, void *_scalar)
{
    if(src1.clCxt -> impl -> double_support == 0 && src1.type() == CV_64F)
    {
        CV_Error(CV_GpuNotSupported, "Selected device don't support double\r\n");
        return;
    }

    //dst.create(src1.size(), src1.type());
    //CV_Assert(src1.cols == src2.cols && src2.cols == dst.cols &&
    //          src1.rows == src2.rows && src2.rows == dst.rows);
    CV_Assert(src1.cols == dst.cols &&
              src1.rows == dst.rows);

    CV_Assert(src1.type() == dst.type());
    CV_Assert(src1.depth() != CV_8S);

    Context  *clCxt = src1.clCxt;
    //int channels = dst.channels();
    //int depth = dst.depth();

    //int vector_lengths[4][7] = {{4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1},
    //    {4, 0, 4, 4, 1, 1, 1}
    //};

    //size_t vector_length = vector_lengths[channels-1][depth];
    //int offset_cols = (dst.offset / dst.elemSize1()) & (vector_length - 1);
    //int cols = divUp(dst.cols * channels + offset_cols, vector_length);

    size_t localThreads[3]  = { 16, 16, 1 };
    //size_t globalThreads[3] = { divUp(cols, localThreads[0]) * localThreads[0],
    //                               divUp(dst.rows, localThreads[1]) * localThreads[1],
    //                               1
    //                             };
    size_t globalThreads[3] = { src1.cols,
                                src1.rows,
                                1
                              };

    int dst_step1 = dst.cols * dst.elemSize();
    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.offset ));
    //args.push_back( make_pair( sizeof(cl_mem), (void *)&src2.data ));
    //args.push_back( make_pair( sizeof(cl_int), (void *)&src2.step ));
    //args.push_back( make_pair( sizeof(cl_int), (void *)&src2.offset ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst_step1 ));

    //if(_scalar != NULL)
    //{
    float scalar1 = *((float *)_scalar);
    args.push_back( make_pair( sizeof(float), (float *)&scalar1 ));
    //}

    openCLExecuteKernel2(clCxt, kernelString, kernelName, globalThreads, localThreads, args, -1, src1.depth(), CLFLUSH);
}

static void multiply_cus(const oclMat &src1, oclMat &dst, float scalar)
{
    arithmetic_run(src1, dst, "arithm_muls", &arithm_mul, (void *)(&scalar));
}

static void pyrdown_run_cus(const oclMat &src, const oclMat &dst)
{

    CV_Assert(src.type() == dst.type());
    CV_Assert(src.depth() != CV_8S);

    Context  *clCxt = src.clCxt;

    string kernelName = "pyrDown";

    size_t localThreads[3]  = { 256, 1, 1 };
    size_t globalThreads[3] = { src.cols, dst.rows, 1};

    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.cols));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.cols));

    openCLExecuteKernel2(clCxt, &pyr_down, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth(), CLFLUSH);
}

static void pyrDown_cus(const oclMat &src, oclMat &dst)
{
    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);

    dst.create((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());

    pyrdown_run_cus(src, dst);
}


//struct MultiplyScalar
//{
//    MultiplyScalar(double val_, double scale_) : val(val_), scale(scale_) {}
//    double operator ()(double a) const
//    {
//        return (scale * a * val);
//    }
//    const double val;
//    const double scale;
//};
//
//void callF(const oclMat& src, oclMat& dst, MultiplyScalar op, int mask)
//{
//	Mat srcTemp;
//	Mat dstTemp;
//	src.download(srcTemp);
//	dst.download(dstTemp);
//
//	int i;
//	int j;
//	int k;
//	for(i = 0; i < srcTemp.rows; i++)
//	{
//		for(j = 0; j < srcTemp.cols; j++)
//		{
//			for(k = 0; k < srcTemp.channels(); k++)
//			{
//				((float*)dstTemp.data)[srcTemp.channels() * (i * srcTemp.rows + j) + k] = (float)op(((float*)srcTemp.data)[srcTemp.channels() * (i * srcTemp.rows + j) + k]);
//			}
//		}
//	}
//
//	dst = dstTemp;
//}
//
//static inline bool isAligned(const unsigned char* ptr, size_t size)
//{
//    return reinterpret_cast<size_t>(ptr) % size == 0;
//}
//
//static inline bool isAligned(size_t step, size_t size)
//{
//    return step % size == 0;
//}
//
//void callT(const oclMat& src, oclMat& dst, MultiplyScalar op, int mask)
//{
//    if (!isAligned(src.data, 4 * sizeof(double)) || !isAligned(src.step, 4 * sizeof(double)) ||
//        !isAligned(dst.data, 4 * sizeof(double)) || !isAligned(dst.step, 4 * sizeof(double)))
//    {
//        callF(src, dst, op, mask);
//        return;
//    }
//
//	Mat srcTemp;
//	Mat dstTemp;
//	src.download(srcTemp);
//	dst.download(dstTemp);
//
//	int x_shifted;
//
//	int i;
//	int j;
//	for(i = 0; i < srcTemp.rows; i++)
//	{
//		const double* srcRow = (const double*)srcTemp.data + i * srcTemp.rows;
//        double* dstRow = (double*)dstTemp.data + i * dstTemp.rows;;
//
//		for(j = 0; j < srcTemp.cols; j++)
//		{
//			x_shifted = j * 4;
//
//			if(x_shifted + 4 - 1 < srcTemp.cols)
//			{
//				dstRow[x_shifted    ] = op(srcRow[x_shifted    ]);
//				dstRow[x_shifted + 1] = op(srcRow[x_shifted + 1]);
//				dstRow[x_shifted + 2] = op(srcRow[x_shifted + 2]);
//				dstRow[x_shifted + 3] = op(srcRow[x_shifted + 3]);
//			}
//			else
//			{
//				for (int real_x = x_shifted; real_x < srcTemp.cols; ++real_x)
//				{
//					((float*)dstTemp.data)[i * srcTemp.rows + real_x] = op(((float*)srcTemp.data)[i * srcTemp.rows + real_x]);
//				}
//			}
//		}
//	}
//}
//
//void multiply(const oclMat& src1, double val, oclMat& dst, double scale = 1.0f);
//void multiply(const oclMat& src1, double val, oclMat& dst, double scale)
//{
//    MultiplyScalar op(val, scale);
//	//if(src1.channels() == 1 && dst.channels() == 1)
//	//{
//	//    callT(src1, dst, op, 0);
//	//}
//	//else
//	//{
//	    callF(src1, dst, op, 0);
//	//}
//}

static cl_mem bindTexture(const oclMat &mat, int depth, int channels)
{
    cl_mem texture;
    cl_image_format format;
    int err;
    if(depth == 0)
    {
        format.image_channel_data_type = CL_UNSIGNED_INT8;
    }
    else if(depth == 5)
    {
        format.image_channel_data_type = CL_FLOAT;
    }
    if(channels == 1)
    {
        format.image_channel_order     = CL_R;
    }
    else if(channels == 3)
    {
        format.image_channel_order     = CL_RGB;
    }
    else if(channels == 4)
    {
        format.image_channel_order     = CL_RGBA;
    }
#ifdef CL_VERSION_1_2
    cl_image_desc desc;
    desc.image_type       = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width      = mat.step / mat.elemSize();
    desc.image_height     = mat.rows;
    desc.image_depth      = NULL;
    desc.image_array_size = 1;
    desc.image_row_pitch  = 0;
    desc.image_slice_pitch = 0;
    desc.buffer           = NULL;
    desc.num_mip_levels   = 0;
    desc.num_samples      = 0;
    texture = clCreateImage(mat.clCxt->impl->clContext, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
#else
    texture = clCreateImage2D(
                  mat.clCxt->impl->clContext,
                  CL_MEM_READ_WRITE,
                  &format,
                  mat.step / mat.elemSize(),
                  mat.rows,
                  0,
                  NULL,
                  &err);
#endif
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { mat.step / mat.elemSize(), mat.rows, 1 };
    clEnqueueCopyBufferToImage(mat.clCxt->impl->clCmdQueue, (cl_mem)mat.data, texture, 0, origin, region, 0, NULL, 0);
    openCLSafeCall(err);

    return texture;
}

static void releaseTexture(cl_mem texture)
{
    openCLFree(texture);
}

static void lkSparse_run(oclMat &I, oclMat &J,
                  const oclMat &prevPts, oclMat &nextPts, oclMat &status, oclMat& err, bool /*GET_MIN_EIGENVALS*/, int ptcount,
                  int level, /*dim3 block, */dim3 patch, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;
    char platform[256] = {0};
    cl_platform_id pid;
    clGetDeviceInfo(*clCxt->impl->devices, CL_DEVICE_PLATFORM, sizeof(pid), &pid, NULL);
    clGetPlatformInfo(pid, CL_PLATFORM_NAME, 256, platform, NULL);
    std::string namestr = platform;
    bool isImageSupported = true;
    if(namestr.find("NVIDIA")!=string::npos || namestr.find("Intel")!=string::npos)
        isImageSupported = false;

    int elemCntPerRow = I.step / I.elemSize();

    string kernelName = "lkSparse";


    size_t localThreads[3]  = { 8, isImageSupported?8:32, 1 };
    size_t globalThreads[3] = { 8 * ptcount, isImageSupported?8:32, 1};

    int cn = I.oclchannels();

    char calcErr;
    if (level == 0)
    {
        calcErr = 1;
    }
    else
    {
        calcErr = 0;
    }

    vector<pair<size_t , const void *> > args;
    cl_mem ITex;
    cl_mem JTex;
    if (isImageSupported)
    {
        ITex = bindTexture(I, I.depth(), cn);
        JTex = bindTexture(J, J.depth(), cn);
    }
    else
    {
        ITex = (cl_mem)I.data;
        JTex = (cl_mem)J.data;
    }

    args.push_back( make_pair( sizeof(cl_mem), (void *)&ITex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&JTex ));
    //cl_mem clmD = clCreateBuffer(clCxt, CL_MEM_READ_WRITE, ptcount * sizeof(float), NULL, NULL);
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&nextPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&nextPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&status.data ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&err.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&level ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.cols ));
    if (!isImageSupported)
    {
        args.push_back( make_pair( sizeof(cl_int), (void *)&elemCntPerRow ) );
    }
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.x ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.y ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));
    //args.push_back( make_pair( sizeof(cl_char), (void *)&GET_MIN_EIGENVALS ));

    if (isImageSupported)
    {
        openCLExecuteKernel2(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), CLFLUSH);

        releaseTexture(ITex);
        releaseTexture(JTex);
    }
    else
    {
        //printf("Warning: The image2d_t is not supported by the device. Using alternative method!\n");
        openCLExecuteKernel2(clCxt, &pyrlk_no_image, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), CLFLUSH);
    }
}

void cv::ocl::PyrLKOpticalFlow::sparse(const oclMat &prevImg, const oclMat &nextImg, const oclMat &prevPts, oclMat &nextPts, oclMat &status, oclMat *err)
{
    if (prevPts.empty())
    {
        nextPts.release();
        status.release();
        //if (err) err->release();
        return;
    }

    derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);

    iters = std::min(std::max(iters, 0), 100);

    const int cn = prevImg.oclchannels();

    dim3 block, patch;
    calcPatchSize(winSize, cn, block, patch, isDeviceArch11_);

    CV_Assert(derivLambda >= 0);
    CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(patch.x > 0 && patch.x < 6 && patch.y > 0 && patch.y < 6);
    CV_Assert(prevPts.rows == 1 && prevPts.type() == CV_32FC2);

    if (useInitialFlow)
        CV_Assert(nextPts.size() == prevPts.size() && nextPts.type() == CV_32FC2);
    else
        ensureSizeIsEnough(1, prevPts.cols, prevPts.type(), nextPts);

    oclMat temp1 = (useInitialFlow ? nextPts : prevPts).reshape(1);
    oclMat temp2 = nextPts.reshape(1);
    //oclMat scalar(temp1.rows, temp1.cols, temp1.type(), Scalar(1.0f / (1 << maxLevel) / 2.0f));
    multiply_cus(temp1, temp2, 1.0f / (1 << maxLevel) / 2.0f);
    //::multiply(temp1, 1.0f / (1 << maxLevel) / 2.0f, temp2);

    ensureSizeIsEnough(1, prevPts.cols, CV_8UC1, status);
    //status.setTo(Scalar::all(1));
    setTo(status, Scalar::all(1));

    bool errMat = false;
    if (!err)
    {
        err = new oclMat(1, prevPts.cols, CV_32FC1);
        errMat = true;
    }
    else
        ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, *err);
    //ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, err);

    // build the image pyramids.

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    if (cn == 1 || cn == 4)
    {
        //prevImg.convertTo(prevPyr_[0], CV_32F);
        //nextImg.convertTo(nextPyr_[0], CV_32F);
        convertTo(prevImg, prevPyr_[0], CV_32F);
        convertTo(nextImg, nextPyr_[0], CV_32F);
    }
    else
    {
        //oclMat buf_;
        //      cvtColor(prevImg, buf_, COLOR_BGR2BGRA);
        //      buf_.convertTo(prevPyr_[0], CV_32F);

        //      cvtColor(nextImg, buf_, COLOR_BGR2BGRA);
        //      buf_.convertTo(nextPyr_[0], CV_32F);
    }

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown_cus(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown_cus(nextPyr_[level - 1], nextPyr_[level]);
    }

    // dI/dx ~ Ix, dI/dy ~ Iy

    for (int level = maxLevel; level >= 0; level--)
    {
        lkSparse_run(prevPyr_[level], nextPyr_[level],
                     prevPts, nextPts, status, *err, getMinEigenVals, prevPts.cols,
                     level, /*block, */patch, winSize, iters);
    }

    clFinish(prevImg.clCxt->impl->clCmdQueue);

    if(errMat)
        delete err;
}

static void lkDense_run(oclMat &I, oclMat &J, oclMat &u, oclMat &v,
                 oclMat &prevU, oclMat &prevV, oclMat *err, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;
    bool isImageSupported = clCxt->impl->devName.find("Intel(R) HD Graphics") == string::npos;
    int elemCntPerRow = I.step / I.elemSize();

    string kernelName = "lkDense";

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { I.cols, I.rows, 1};

    int cn = I.oclchannels();

    bool calcErr;
    if (err)
    {
        calcErr = true;
    }
    else
    {
        calcErr = false;
    }

    cl_mem ITex;
    cl_mem JTex;

    if (isImageSupported)
    {
        ITex = bindTexture(I, I.depth(), cn);
        JTex = bindTexture(J, J.depth(), cn);
    }
    else
    {
        ITex = (cl_mem)I.data;
        JTex = (cl_mem)J.data;
    }

    //int2 halfWin = {(winSize.width - 1) / 2, (winSize.height - 1) / 2};
    //const int patchWidth  = 16 + 2 * halfWin.x;
    //const int patchHeight = 16 + 2 * halfWin.y;
    //size_t smem_size = 3 * patchWidth * patchHeight * sizeof(int);

    vector<pair<size_t , const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&ITex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&JTex ));

    args.push_back( make_pair( sizeof(cl_mem), (void *)&u.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&u.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&v.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&v.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevU.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevU.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevV.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevV.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.cols ));
    //args.push_back( make_pair( sizeof(cl_mem), (void *)&(*err).data ));
    //args.push_back( make_pair( sizeof(cl_int), (void *)&(*err).step ));
    if (!isImageSupported)
    {
        args.push_back( make_pair( sizeof(cl_int), (void *)&elemCntPerRow ) );
    }
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    if (isImageSupported)
    {
        openCLExecuteKernel2(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), CLFLUSH);

        releaseTexture(ITex);
        releaseTexture(JTex);
    }
    else
    {
        //printf("Warning: The image2d_t is not supported by the device. Using alternative method!\n");
        openCLExecuteKernel2(clCxt, &pyrlk_no_image, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), CLFLUSH);
    }
}

void cv::ocl::PyrLKOpticalFlow::dense(const oclMat &prevImg, const oclMat &nextImg, oclMat &u, oclMat &v, oclMat *err)
{
    CV_Assert(prevImg.type() == CV_8UC1);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(maxLevel >= 0);
    CV_Assert(winSize.width > 2 && winSize.height > 2);

    if (err)
        err->create(prevImg.size(), CV_32FC1);

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    prevPyr_[0] = prevImg;
    //nextImg.convertTo(nextPyr_[0], CV_32F);
    convertTo(nextImg, nextPyr_[0], CV_32F);

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown_cus(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown_cus(nextPyr_[level - 1], nextPyr_[level]);
    }

    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[1]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[1]);
    //uPyr_[1].setTo(Scalar::all(0));
    //vPyr_[1].setTo(Scalar::all(0));
    setTo(uPyr_[1], Scalar::all(0));
    setTo(vPyr_[1], Scalar::all(0));

    Size winSize2i(winSize.width, winSize.height);

    int idx = 0;

    for (int level = maxLevel; level >= 0; level--)
    {
        int idx2 = (idx + 1) & 1;

        lkDense_run(prevPyr_[level], nextPyr_[level], uPyr_[idx], vPyr_[idx], uPyr_[idx2], vPyr_[idx2],
                    level == 0 ? err : 0, winSize2i, iters);

        if (level > 0)
            idx = idx2;
    }

    //uPyr_[idx].copyTo(u);
    //vPyr_[idx].copyTo(v);
    copyTo(uPyr_[idx], u);
    copyTo(vPyr_[idx], v);

    clFinish(prevImg.clCxt->impl->clCmdQueue);
}

#endif /* !defined (HAVE_CUDA) */
