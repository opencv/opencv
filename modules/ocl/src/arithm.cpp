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
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, jlyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
//    Peng Xiao, pengxiao@outlook.com
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

static std::vector<uchar> scalarToVector(const cv::Scalar & sc, int depth, int ocn, int cn)
{
    CV_Assert(ocn == cn || (ocn == 4 && cn == 3));

    static const int sizeMap[] = { sizeof(uchar), sizeof(char), sizeof(ushort),
                               sizeof(short), sizeof(int), sizeof(float), sizeof(double) };

    int elemSize1 = sizeMap[depth];
    int bufSize = elemSize1 * ocn;
    std::vector<uchar> _buf(bufSize);
    uchar * buf = &_buf[0];
    scalarToRawData(sc, buf, CV_MAKE_TYPE(depth, cn));
    memset(buf + elemSize1 * cn, 0, (ocn - cn) * elemSize1);

    return _buf;
}

//////////////////////////////////////////////////////////////////////////////
/////////////// add subtract multiply divide min max /////////////////////////
//////////////////////////////////////////////////////////////////////////////

enum { ADD = 0, SUB, MUL, DIV, ABS, ABS_DIFF, MIN, MAX };

static void arithmetic_run_generic(const oclMat &src1, const oclMat &src2, const Scalar & scalar, const oclMat & mask,
                            oclMat &dst, int op_type, bool use_scalar = false)
{
    Context *clCxt = src1.clCxt;
    bool hasDouble = clCxt->supportsFeature(FEATURE_CL_DOUBLE);
    if (!hasDouble && (src1.depth() == CV_64F || src2.depth() == CV_64F || dst.depth() == CV_64F))
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(src2.empty() || (!src2.empty() && src1.type() == src2.type() && src1.size() == src2.size()));
    CV_Assert(mask.empty() || (!mask.empty() && mask.type() == CV_8UC1 && mask.size() == src1.size()));
    CV_Assert(op_type >= ADD && op_type <= MAX);

    dst.create(src1.size(), src1.type());

    int oclChannels = src1.oclchannels(), depth = src1.depth();
    int src1step1 = src1.step / src1.elemSize(), src1offset1 = src1.offset / src1.elemSize();
    int src2step1 = src2.step / src2.elemSize(), src2offset1 = src2.offset / src2.elemSize();
    int maskstep1 = mask.step, maskoffset1 = mask.offset / mask.elemSize();
    int dststep1 = dst.step / dst.elemSize(), dstoffset1 = dst.offset / dst.elemSize();
    std::vector<uchar> m;

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    std::string kernelName = "arithm_binary_op";

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const WTypeMap[] = { "short", "short", "int", "int", "int", "float", "double" };
    const char * const funcMap[] = { "FUNC_ADD", "FUNC_SUB", "FUNC_MUL", "FUNC_DIV", "FUNC_ABS", "FUNC_ABS_DIFF", "FUNC_MIN", "FUNC_MAX" };
    const char * const channelMap[] = { "", "", "2", "4", "4" };
    bool haveScalar = use_scalar || src2.empty();

    int WDepth = depth;
    if (haveScalar)
        WDepth = hasDouble && WDepth == CV_64F ? CV_64F : CV_32F;
    if (op_type == DIV)
        WDepth = hasDouble ? CV_64F : CV_32F;
    else if (op_type == MUL)
        WDepth = hasDouble && (depth == CV_32S || depth == CV_64F) ? CV_64F : CV_32F;

    std::string buildOptions = format("-D T=%s%s -D WT=%s%s -D convertToT=convert_%s%s%s -D %s "
                                      "-D convertToWT=convert_%s%s",
                                      typeMap[depth], channelMap[oclChannels],
                                      WTypeMap[WDepth], channelMap[oclChannels],
                                      typeMap[depth], channelMap[oclChannels], (depth >= CV_32F ? "" : (depth == CV_32S ? "_rte" : "_sat_rte")),
                                      funcMap[op_type], WTypeMap[WDepth], channelMap[oclChannels]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1offset1 ));

    if (!src2.empty())
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2step1 ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2offset1 ));

        kernelName += "_mat";

        if (haveScalar)
            buildOptions += " -D HAVE_SCALAR";
    }

    if (haveScalar)
    {
        const int WDepthMap[] = { CV_16S, CV_16S, CV_32S, CV_32S, CV_32S, CV_32F, CV_64F };
        m = scalarToVector(scalar, WDepthMap[WDepth], oclChannels, src1.channels());

        args.push_back( std::make_pair( m.size(), (void *)&m[0]));

        kernelName += "_scalar";
    }

    if (!mask.empty())
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&maskstep1 ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&maskoffset1 ));

        kernelName += "_mask";
    }

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1 ));

    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));

    openCLExecuteKernel(clCxt, mask.empty() ?
                            (!src2.empty() ? &arithm_add : &arithm_add_scalar) :
                            (!src2.empty() ? &arithm_add_mask : &arithm_add_scalar_mask),
                        kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

void cv::ocl::add(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask)
{
    arithmetic_run_generic(src1, src2, Scalar(), mask, dst, ADD);
}

void cv::ocl::add(const oclMat &src1, const Scalar &src2, oclMat &dst, const oclMat &mask)
{
    arithmetic_run_generic(src1, oclMat(), src2, mask, dst, ADD);
}

void cv::ocl::subtract(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask)
{
    arithmetic_run_generic(src1, src2, Scalar(), mask, dst, SUB);
}

void cv::ocl::subtract(const oclMat &src1, const Scalar &src2, oclMat &dst, const oclMat &mask)
{
    arithmetic_run_generic(src1, oclMat(), src2, mask, dst, SUB);
}

void cv::ocl::multiply(const oclMat &src1, const oclMat &src2, oclMat &dst, double scalar)
{
    const bool use_scalar = !(std::abs(scalar - 1.0) < std::numeric_limits<double>::epsilon());
    arithmetic_run_generic(src1, src2, Scalar::all(scalar), oclMat(), dst, MUL, use_scalar);
}

void cv::ocl::multiply(double scalar, const oclMat &src, oclMat &dst)
{
    arithmetic_run_generic(src, oclMat(), Scalar::all(scalar), oclMat(), dst, MUL);
}

void cv::ocl::divide(const oclMat &src1, const oclMat &src2, oclMat &dst, double scalar)
{
    const bool use_scalar = !(std::abs(scalar - 1.0) < std::numeric_limits<double>::epsilon());
    arithmetic_run_generic(src1, src2, Scalar::all(scalar), oclMat(), dst, DIV, use_scalar);
}

void cv::ocl::divide(double scalar, const oclMat &src, oclMat &dst)
{
    arithmetic_run_generic(src, oclMat(), Scalar::all(scalar), oclMat(), dst, DIV);
}

void cv::ocl::min(const oclMat &src1, const oclMat &src2, oclMat &dst)
{
    arithmetic_run_generic(src1, src2, Scalar::all(0), oclMat(), dst, MIN);
}

void cv::ocl::max(const oclMat &src1, const oclMat &src2, oclMat &dst)
{
    arithmetic_run_generic(src1, src2, Scalar::all(0), oclMat(), dst, MAX);
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////Abs, Absdiff ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void cv::ocl::abs(const oclMat &src, oclMat &dst)
{
    // explicitly uses use_scalar (even if zero) so that the correct kernel is used
    arithmetic_run_generic(src, oclMat(), Scalar(), oclMat(), dst, ABS, true);
}

void cv::ocl::absdiff(const oclMat &src1, const oclMat &src2, oclMat &dst)
{
    arithmetic_run_generic(src1, src2, Scalar(), oclMat(), dst, ABS_DIFF);
}

void cv::ocl::absdiff(const oclMat &src1, const Scalar &src2, oclMat &dst)
{
    arithmetic_run_generic(src1, oclMat(), src2, oclMat(), dst, ABS_DIFF);
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////  compare ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void compare_run(const oclMat &src1, const oclMat &src2, oclMat &dst, int cmpOp,
                        String kernelName, const cv::ocl::ProgramEntry* source)
{
    dst.create(src1.size(), CV_8UC1);

    int depth = src1.depth();
    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    int src1step1 = src1.step1(), src1offset1 = src1.offset / src1.elemSize1();
    int src2step1 = src2.step1(), src2offset1 = src2.offset / src2.elemSize1();
    int dststep1 = dst.step1(), dstoffset1 = dst.offset / dst.elemSize1();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * operationMap[] = { "==", ">", ">=", "<", "<=", "!=" };
    std::string buildOptions = format("-D T=%s -D Operation=%s", typeMap[depth], operationMap[cmpOp]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1offset1 ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2offset1 ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));

    openCLExecuteKernel(src1.clCxt, source, kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

void cv::ocl::compare(const oclMat &src1, const oclMat &src2, oclMat &dst , int cmpOp)
{
    if (!src1.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src1.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(src1.type() == src2.type() && src1.channels() == 1);
    CV_Assert(cmpOp >= CMP_EQ && cmpOp <= CMP_NE);

    compare_run(src1, src2, dst, cmpOp, "arithm_compare", &arithm_compare);
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// sum  //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

enum { SUM = 0, ABS_SUM, SQR_SUM };

static void arithmetic_sum_buffer_run(const oclMat &src, cl_mem &dst, int groupnum, int type, int ddepth)
{
    int ochannels = src.oclchannels();
    int all_cols = src.step / src.elemSize();
    int pre_cols = (src.offset % src.step) / src.elemSize();
    int sec_cols = all_cols - (src.offset % src.step + src.cols * src.elemSize() - 1) / src.elemSize() - 1;
    int invalid_cols = pre_cols + sec_cols;
    int cols = all_cols - invalid_cols , elemnum = cols * src.rows;;
    int offset = src.offset / src.elemSize();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const funcMap[] = { "FUNC_SUM", "FUNC_ABS_SUM", "FUNC_SQR_SUM" };
    const char * const channelMap[] = { " ", " ", "2", "4", "4" };
    String buildOptions = format("-D srcT=%s%s -D dstT=%s%s -D convertToDstT=convert_%s%s -D %s",
                                 typeMap[src.depth()], channelMap[ochannels],
                                 typeMap[ddepth], channelMap[ochannels],
                                 typeMap[ddepth], channelMap[ochannels],
                                 funcMap[type]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&invalid_cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&elemnum));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&groupnum));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst ));
    size_t globalThreads[3] = { groupnum * 256, 1, 1 };
    size_t localThreads[3] = { 256, 1, 1 };

    openCLExecuteKernel(src.clCxt, &arithm_sum, "arithm_op_sum", globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

template <typename T>
Scalar arithmetic_sum(const oclMat &src, int type, int ddepth)
{
    CV_Assert(src.step % src.elemSize() == 0);

    size_t groupnum = src.clCxt->getDeviceInfo().maxComputeUnits;
    CV_Assert(groupnum != 0);

    int dbsize = groupnum * src.oclchannels();
    Context *clCxt = src.clCxt;

    AutoBuffer<T> _buf(dbsize);
    T *p = (T*)_buf;
    memset(p, 0, dbsize * sizeof(T));

    cl_mem dstBuffer = openCLCreateBuffer(clCxt, CL_MEM_WRITE_ONLY, dbsize * sizeof(T));
    arithmetic_sum_buffer_run(src, dstBuffer, groupnum, type, ddepth);
    openCLReadBuffer(clCxt, dstBuffer, (void *)p, dbsize * sizeof(T));
    openCLFree(dstBuffer);

    Scalar s = Scalar::all(0.0);
    for (int i = 0; i < dbsize;)
         for (int j = 0; j < src.oclchannels(); j++, i++)
            s.val[j] += p[i];

    return s;
}

typedef Scalar (*sumFunc)(const oclMat &src, int type, int ddepth);

Scalar cv::ocl::sum(const oclMat &src)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return Scalar::all(0);
    }
    static sumFunc functab[3] =
    {
        arithmetic_sum<int>,
        arithmetic_sum<float>,
        arithmetic_sum<double>
    };

    int ddepth = std::max(src.depth(), CV_32S);
    sumFunc func = functab[ddepth - CV_32S];
    return func(src, SUM, ddepth);
}

Scalar cv::ocl::absSum(const oclMat &src)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return cv::Scalar::all(0);
    }

    static sumFunc functab[3] =
    {
        arithmetic_sum<int>,
        arithmetic_sum<float>,
        arithmetic_sum<double>
    };

    int ddepth = std::max(src.depth(), CV_32S);
    sumFunc func = functab[ddepth - CV_32S];
    return func(src, ABS_SUM, ddepth);
}

Scalar cv::ocl::sqrSum(const oclMat &src)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return cv::Scalar::all(0);
    }
    static sumFunc functab[3] =
    {
        arithmetic_sum<int>,
        arithmetic_sum<float>,
        arithmetic_sum<double>
    };

    int ddepth = std::max(src.depth(), CV_32S);
    sumFunc func = functab[ddepth - CV_32S];
    return func(src, SQR_SUM, ddepth);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////// meanStdDev //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void cv::ocl::meanStdDev(const oclMat &src, Scalar &mean, Scalar &stddev)
{
    if (src.depth() == CV_64F && !src.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    double total = 1.0 / src.size().area();

    mean = sum(src);
    stddev = sqrSum(src);

    for (int i = 0; i < 4; ++i)
    {
        mean[i] *= total;
        stddev[i] = std::sqrt(std::max(stddev[i] * total - mean.val[i] * mean.val[i] , 0.));
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// minMax  /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <typename T, typename WT>
static void arithmetic_minMax_run(const oclMat &src, const oclMat & mask, cl_mem &dst, int groupnum, String kernelName)
{
    int all_cols = src.step / src.elemSize();
    int pre_cols = (src.offset % src.step) / src.elemSize();
    int sec_cols = all_cols - (src.offset % src.step + src.cols * src.elemSize() - 1) / src.elemSize() - 1;
    int invalid_cols = pre_cols + sec_cols;
    int cols = all_cols - invalid_cols , elemnum = cols * src.rows;
    int offset = src.offset / src.elemSize();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const channelMap[] = { " ", " ", "2", "4", "4" };

    std::ostringstream stream;
    stream << "-D T=" << typeMap[src.depth()] << channelMap[src.channels()];
    if (std::numeric_limits<T>::is_integer)
    {
        stream << " -D MAX_VAL=" << (WT)std::numeric_limits<T>::max();
        stream << " -D MIN_VAL=" << (WT)std::numeric_limits<T>::min();
    }
    else
        stream << " -D DEPTH_" << src.depth();
    std::string buildOptions = stream.str();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&invalid_cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&elemnum));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&groupnum));

    int minvalid_cols = 0, moffset = 0;
    if (!mask.empty())
    {
        int mall_cols = mask.step / mask.elemSize();
        int mpre_cols = (mask.offset % mask.step) / mask.elemSize();
        int msec_cols = mall_cols - (mask.offset % mask.step + mask.cols * mask.elemSize() - 1) / mask.elemSize() - 1;
        minvalid_cols = mpre_cols + msec_cols;
        moffset = mask.offset / mask.elemSize();

        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&mask.data ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&minvalid_cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&moffset ));

        kernelName = kernelName + "_mask";
    }

    size_t globalThreads[3] = {groupnum * 256, 1, 1};
    size_t localThreads[3] = {256, 1, 1};

    openCLExecuteKernel(src.clCxt, &arithm_minMax, kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

template <typename T, typename WT>
void arithmetic_minMax(const oclMat &src, double *minVal, double *maxVal, const oclMat &mask)
{
    size_t groupnum = src.clCxt->getDeviceInfo().maxComputeUnits;
    CV_Assert(groupnum != 0);

    int dbsize = groupnum * 2 * src.elemSize();
    oclMat buf;
    ensureSizeIsEnough(1, dbsize, CV_8UC1, buf);

    cl_mem buf_data = reinterpret_cast<cl_mem>(buf.data);
    arithmetic_minMax_run<T, WT>(src, mask, buf_data, groupnum, "arithm_op_minMax");

    Mat matbuf = Mat(buf);
    T *p = matbuf.ptr<T>();
    if (minVal != NULL)
    {
        *minVal = std::numeric_limits<double>::max();
        for (int i = 0, end = src.oclchannels() * (int)groupnum; i < end; i++)
            *minVal = *minVal < p[i] ? *minVal : p[i];
    }
    if (maxVal != NULL)
    {
        *maxVal = -std::numeric_limits<double>::max();
        for (int i = src.oclchannels() * (int)groupnum, end = i << 1; i < end; i++)
            *maxVal = *maxVal > p[i] ? *maxVal : p[i];
    }
}

typedef void (*minMaxFunc)(const oclMat &src, double *minVal, double *maxVal, const oclMat &mask);

void cv::ocl::minMax(const oclMat &src, double *minVal, double *maxVal, const oclMat &mask)
{
    CV_Assert(src.channels() == 1);
    CV_Assert(src.size() == mask.size() || mask.empty());
    CV_Assert(src.step % src.elemSize() == 0);

    if (minVal == NULL && maxVal == NULL)
        return;

    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    static minMaxFunc functab[] =
    {
        arithmetic_minMax<uchar, int>,
        arithmetic_minMax<char, int>,
        arithmetic_minMax<ushort, int>,
        arithmetic_minMax<short, int>,
        arithmetic_minMax<int, int>,
        arithmetic_minMax<float, float>,
        arithmetic_minMax<double, double>,
        0
    };

    minMaxFunc func = functab[src.depth()];
    CV_Assert(func != 0);

    func(src, minVal, maxVal, mask);
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// norm /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

double cv::ocl::norm(const oclMat &src1, int normType)
{
    CV_Assert((normType & NORM_RELATIVE) == 0);
    return norm(src1, oclMat(), normType);
}

static void arithm_absdiff_nonsaturate_run(const oclMat & src1, const oclMat & src2, oclMat & diff, int ntype)
{
    Context *clCxt = src1.clCxt;
    if (!clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src1.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }
    CV_Assert(src1.step % src1.elemSize() == 0 && (src2.empty() || src2.step % src2.elemSize() == 0));

    int ddepth = std::max(src1.depth(), CV_32S);
    if (ntype == NORM_L2)
        ddepth = std::max<int>(CV_32F, ddepth);

    diff.create(src1.size(), CV_MAKE_TYPE(ddepth, src1.channels()));
    CV_Assert(diff.step % diff.elemSize() == 0);

    int oclChannels = src1.oclchannels(), sdepth = src1.depth();
    int src1step1 = src1.step / src1.elemSize(), src1offset1 = src1.offset / src1.elemSize();
    int src2step1 = src2.step / src2.elemSize(), src2offset1 = src2.offset / src2.elemSize();
    int diffstep1 = diff.step / diff.elemSize(), diffoffset1 = diff.offset / diff.elemSize();

    String kernelName = "arithm_absdiff_nonsaturate";
    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { diff.cols, diff.rows, 1 };

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const channelMap[] = { "", "", "2", "4", "4" };

    std::string buildOptions = format("-D srcT=%s%s -D dstT=%s%s -D convertToDstT=convert_%s%s",
                                      typeMap[sdepth], channelMap[oclChannels],
                                      typeMap[ddepth], channelMap[oclChannels],
                                      typeMap[ddepth], channelMap[oclChannels]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1offset1 ));

    if (!src2.empty())
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2step1 ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2offset1 ));

        kernelName = kernelName + "_binary";
    }

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&diff.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&diffstep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&diffoffset1 ));

    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));

    openCLExecuteKernel(clCxt, &arithm_absdiff_nonsaturate,
                        kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

double cv::ocl::norm(const oclMat &src1, const oclMat &src2, int normType)
{
    if (!src1.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src1.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return -1;
    }
    CV_Assert(src2.empty() || (src1.type() == src2.type() && src1.size() == src2.size()));

    bool isRelative = (normType & NORM_RELATIVE) != 0;
    normType &= NORM_TYPE_MASK;
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);

    Scalar s;
    int cn = src1.channels();
    double r = 0;
    oclMat diff;

    arithm_absdiff_nonsaturate_run(src1, src2, diff, normType);

    switch (normType)
    {
    case NORM_INF:
        diff = diff.reshape(1);
        minMax(diff, NULL, &r);
        break;
    case NORM_L1:
        s = sum(diff);
        for (int i = 0; i < cn; ++i)
            r += s[i];
        break;
    case NORM_L2:
        s = sqrSum(diff);
        for (int i = 0; i < cn; ++i)
            r += s[i];
        r = std::sqrt(r);
        break;
    }
    if (isRelative)
        r = r / (norm(src2, normType) + DBL_EPSILON);

    return r;
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// flip //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

enum { FLIP_COLS = 1 << 0, FLIP_ROWS = 1 << 1, FLIP_BOTH = FLIP_ROWS | FLIP_COLS };

static void arithmetic_flip_run(const oclMat &src, oclMat &dst, String kernelName, int flipType)
{
    int cols = dst.cols, rows = dst.rows;
    if ((cols == 1 && flipType == FLIP_COLS) ||
            (rows == 1 && flipType == FLIP_ROWS) ||
            (rows == 1 && cols == 1 && flipType == FLIP_BOTH))
    {
        src.copyTo(dst);
        return;
    }

    cols = flipType == FLIP_COLS ? divUp(cols, 2) : cols;
    rows = flipType & FLIP_ROWS ? divUp(rows, 2) : rows;

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D T=%s%s", typeMap[dst.depth()], channelMap[dst.oclchannels()]);

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, rows, 1 };

    int elemSize = src.elemSize();
    int src_step = src.step / elemSize, src_offset = src.offset / elemSize;
    int dst_step = dst.step / elemSize, dst_offset = dst.offset / elemSize;

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));

    openCLExecuteKernel(src.clCxt, &arithm_flip, kernelName, globalThreads, localThreads, args,
                        -1, -1, buildOptions.c_str());
}

void cv::ocl::flip(const oclMat &src, oclMat &dst, int flipCode)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    dst.create(src.size(), src.type());

    if (flipCode == 0)
        arithmetic_flip_run(src, dst, "arithm_flip_rows", FLIP_ROWS);
    else if (flipCode > 0)
        arithmetic_flip_run(src, dst, "arithm_flip_cols", FLIP_COLS);
    else
        arithmetic_flip_run(src, dst, "arithm_flip_rows_cols", FLIP_BOTH);
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// LUT  //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_lut_run(const oclMat &src, const oclMat &lut, oclMat &dst, String kernelName)
{
    int sdepth = src.depth();
    int src_step1 = src.step1(), dst_step1 = dst.step1();
    int src_offset1 = src.offset / src.elemSize1(), dst_offset1 = dst.offset / dst.elemSize1();
    int lut_offset1 = lut.offset / lut.elemSize1() + (sdepth == CV_8U ? 0 : 128) * lut.channels();
    int cols1 = src.cols * src.oclchannels();

    size_t localSize[] = { 16, 16, 1 };
    size_t globalSize[] = { lut.channels() == 1 ? cols1 : src.cols, src.rows, 1 };

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D srcT=%s -D dstT=%s", typeMap[sdepth], typeMap[dst.depth()]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&lut.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols1));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&lut_offset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_offset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step1 ));

    openCLExecuteKernel(src.clCxt, &arithm_LUT, kernelName, globalSize, localSize,
                        args, lut.oclchannels(), -1, buildOptions.c_str());
}

void cv::ocl::LUT(const oclMat &src, const oclMat &lut, oclMat &dst)
{
    if (!lut.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && lut.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    int cn = src.channels(), depth = src.depth();

    CV_Assert(depth == CV_8U || depth == CV_8S);
    CV_Assert(lut.channels() == 1 || lut.channels() == src.channels());
    CV_Assert(lut.rows == 1 && lut.cols == 256);

    dst.create(src.size(), CV_MAKETYPE(lut.depth(), cn));
    arithmetic_lut_run(src, lut, dst, "LUT");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////// exp log /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_exp_log_run(const oclMat &src, oclMat &dst, String kernelName, const cv::ocl::ProgramEntry* source)
{
    Context  *clCxt = src.clCxt;
    if (!clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert( src.depth() == CV_32F || src.depth() == CV_64F);
    dst.create(src.size(), src.type());

    int ddepth = dst.depth();
    int cols1 = src.cols * src.oclchannels();
    int srcoffset1 = src.offset / src.elemSize1(), dstoffset1 = dst.offset / dst.elemSize1();
    int srcstep1 = src.step1(), dststep1 = dst.step1();

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    std::string buildOptions = format("-D srcT=%s",
                                      ddepth == CV_32F ? "float" : "double");

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcoffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcstep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));

    openCLExecuteKernel(clCxt, source, kernelName, globalThreads, localThreads,
                        args, src.oclchannels(), -1, buildOptions.c_str());
}

void cv::ocl::exp(const oclMat &src, oclMat &dst)
{
    arithmetic_exp_log_run(src, dst, "arithm_exp", &arithm_exp);
}

void cv::ocl::log(const oclMat &src, oclMat &dst)
{
    arithmetic_exp_log_run(src, dst, "arithm_log", &arithm_log);
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////// magnitude phase ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_magnitude_phase_run(const oclMat &src1, const oclMat &src2, oclMat &dst, String kernelName)
{
    int channels = dst.oclchannels();
    int depth = dst.depth();

    size_t vector_length = 1;
    int offset_cols = ((dst.offset % dst.step) / dst.elemSize1()) & (vector_length - 1);
    int cols = divUp(dst.cols * channels + offset_cols, vector_length);

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, dst.rows, 1 };

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));

    openCLExecuteKernel(src1.clCxt, &arithm_magnitude, kernelName, globalThreads, localThreads, args, -1, depth);
}

void cv::ocl::magnitude(const oclMat &src1, const oclMat &src2, oclMat &dst)
{
    if (!src1.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src1.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(src1.type() == src2.type() && src1.size() == src2.size() &&
              (src1.depth() == CV_32F || src1.depth() == CV_64F));

    dst.create(src1.size(), src1.type());
    arithmetic_magnitude_phase_run(src1, src2, dst, "arithm_magnitude");
}

static void arithmetic_phase_run(const oclMat &src1, const oclMat &src2, oclMat &dst, String kernelName, const cv::ocl::ProgramEntry* source)
{
    int depth = dst.depth(), cols1 = src1.cols * src1.oclchannels();
    int src1step1 = src1.step / src1.elemSize1(), src1offset1 = src1.offset / src1.elemSize1();
    int src2step1 = src2.step / src2.elemSize1(), src2offset1 = src2.offset / src2.elemSize1();
    int dststep1 = dst.step / dst.elemSize1(), dstoffset1 = dst.offset / dst.elemSize1();

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols1, dst.rows, 1 };

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1offset1 ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2offset1 ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows ));

    openCLExecuteKernel(src1.clCxt, source, kernelName, globalThreads, localThreads, args, -1, depth);
}

void cv::ocl::phase(const oclMat &x, const oclMat &y, oclMat &Angle, bool angleInDegrees)
{
    if (!x.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && x.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(x.type() == y.type() && x.size() == y.size() && (x.depth() == CV_32F || x.depth() == CV_64F));
    CV_Assert(x.step % x.elemSize() == 0 && y.step % y.elemSize() == 0);

    Angle.create(x.size(), x.type());
    arithmetic_phase_run(x, y, Angle, angleInDegrees ? "arithm_phase_indegrees" : "arithm_phase_inradians", &arithm_phase);
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// cartToPolar ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_cartToPolar_run(const oclMat &src1, const oclMat &src2, oclMat &dst_mag, oclMat &dst_cart,
                                String kernelName, bool angleInDegrees)
{
    int channels = src1.oclchannels();
    int depth = src1.depth();

    int cols = src1.cols * channels;

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, src1.rows, 1 };

    int tmp = angleInDegrees ? 1 : 0;
    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst_mag.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_mag.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_mag.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst_cart.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_cart.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_cart.offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tmp ));

    openCLExecuteKernel(src1.clCxt, &arithm_cartToPolar, kernelName, globalThreads, localThreads, args, -1, depth);
}

void cv::ocl::cartToPolar(const oclMat &x, const oclMat &y, oclMat &mag, oclMat &angle, bool angleInDegrees)
{
    if (!x.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && x.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(x.type() == y.type() && x.size() == y.size() && (x.depth() == CV_32F || x.depth() == CV_64F));

    mag.create(x.size(), x.type());
    angle.create(x.size(), x.type());

    arithmetic_cartToPolar_run(x, y, mag, angle, "arithm_cartToPolar", angleInDegrees);
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// polarToCart ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_ptc_run(const oclMat &src1, const oclMat &src2, oclMat &dst1, oclMat &dst2, bool angleInDegrees,
                        String kernelName)
{
    int channels = src2.oclchannels();
    int depth = src2.depth();

    int cols = src2.cols * channels;
    int rows = src2.rows;

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, rows, 1 };

    int tmp = angleInDegrees ? 1 : 0;
    std::vector<std::pair<size_t , const void *> > args;
    if (src1.data)
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));
    }
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst1.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst2.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst2.offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&tmp ));

    openCLExecuteKernel(src1.clCxt, &arithm_polarToCart, kernelName, globalThreads, localThreads, args, -1, depth);
}

void cv::ocl::polarToCart(const oclMat &magnitude, const oclMat &angle, oclMat &x, oclMat &y, bool angleInDegrees)
{
    if (!magnitude.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && magnitude.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(angle.depth() == CV_32F || angle.depth() == CV_64F);
    CV_Assert(magnitude.size() == angle.size() && magnitude.type() == angle.type());

    x.create(angle.size(), angle.type());
    y.create(angle.size(), angle.type());

    if ( magnitude.data )
        arithmetic_ptc_run(magnitude, angle, x, y, angleInDegrees, "arithm_polarToCart_mag");
    else
        arithmetic_ptc_run(magnitude, angle, x, y, angleInDegrees, "arithm_polarToCart");
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// minMaxLoc ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_minMaxLoc_run(const oclMat &src, cl_mem &dst, int vlen , int groupnum)
{
    std::vector<std::pair<size_t , const void *> > args;
    int all_cols = src.step / (vlen * src.elemSize1());
    int pre_cols = (src.offset % src.step) / (vlen * src.elemSize1());
    int sec_cols = all_cols - (src.offset % src.step + src.cols * src.elemSize1() - 1) / (vlen * src.elemSize1()) - 1;
    int invalid_cols = pre_cols + sec_cols;
    int cols = all_cols - invalid_cols , elemnum = cols * src.rows;;
    int offset = src.offset / (vlen * src.elemSize1());
    int repeat_s = src.offset / src.elemSize1() - offset * vlen;
    int repeat_e = (offset + cols) * vlen - src.offset / src.elemSize1() - src.cols;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&invalid_cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&elemnum));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&groupnum));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst ));
    char build_options[50];
    sprintf(build_options, "-D DEPTH_%d -D REPEAT_S%d -D REPEAT_E%d", src.depth(), repeat_s, repeat_e);
    size_t gt[3] = {groupnum * 256, 1, 1}, lt[3] = {256, 1, 1};
    openCLExecuteKernel(src.clCxt, &arithm_minMaxLoc, "arithm_op_minMaxLoc", gt, lt, args, -1, -1, build_options);
}

static void arithmetic_minMaxLoc_mask_run(const oclMat &src, const oclMat &mask, cl_mem &dst, int vlen, int groupnum)
{
    std::vector<std::pair<size_t , const void *> > args;
    size_t gt[3] = {groupnum * 256, 1, 1}, lt[3] = {256, 1, 1};
    char build_options[50];
    if (src.oclchannels() == 1)
    {
        int cols = (src.cols - 1) / vlen + 1;
        int invalid_cols = src.step / (vlen * src.elemSize1()) - cols;
        int offset = src.offset / src.elemSize1();
        int repeat_me = vlen - (mask.cols % vlen == 0 ? vlen : mask.cols % vlen);
        int minvalid_cols = mask.step / (vlen * mask.elemSize1()) - cols;
        int moffset = mask.offset / mask.elemSize1();
        int elemnum = cols * src.rows;
        sprintf(build_options, "-D DEPTH_%d -D REPEAT_E%d", src.depth(), repeat_me);
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&invalid_cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&elemnum));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&groupnum));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&minvalid_cols ));
        args.push_back( std::make_pair( sizeof(cl_int) , (void *)&moffset ));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&mask.data ));
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst ));

        openCLExecuteKernel(src.clCxt, &arithm_minMaxLoc_mask, "arithm_op_minMaxLoc_mask", gt, lt, args, -1, -1, build_options);
    }
}

template <typename T>
void arithmetic_minMaxLoc(const oclMat &src, double *minVal, double *maxVal,
                          Point *minLoc, Point *maxLoc, const oclMat &mask)
{
    CV_Assert(src.oclchannels() == 1);
    size_t groupnum = src.clCxt->getDeviceInfo().maxComputeUnits;
    CV_Assert(groupnum != 0);
    int minloc = -1 , maxloc = -1;
    int vlen = 4, dbsize = groupnum * vlen * 4 * sizeof(T) ;
    Context *clCxt = src.clCxt;
    cl_mem dstBuffer = openCLCreateBuffer(clCxt, CL_MEM_WRITE_ONLY, dbsize);
    *minVal = std::numeric_limits<double>::max() , *maxVal = -std::numeric_limits<double>::max();

    if (mask.empty())
        arithmetic_minMaxLoc_run(src, dstBuffer, vlen, groupnum);
    else
        arithmetic_minMaxLoc_mask_run(src, mask, dstBuffer, vlen, groupnum);

    AutoBuffer<T> _buf(groupnum * vlen * 4);
    T *p = (T*)_buf;
    memset(p, 0, dbsize);

    openCLReadBuffer(clCxt, dstBuffer, (void *)p, dbsize);
    for (int i = 0; i < vlen * (int)groupnum; i++)
    {
        *minVal = (*minVal < p[i] || p[i + 2 * vlen * groupnum] == -1) ? *minVal : p[i];
        minloc = (*minVal < p[i] || p[i + 2 * vlen * groupnum] == -1) ? minloc : cvRound(p[i + 2 * vlen * groupnum]);
    }
    for (int i = vlen * (int)groupnum; i < 2 * vlen * (int)groupnum; i++)
    {
        *maxVal = (*maxVal > p[i] || p[i + 2 * vlen * groupnum] == -1) ? *maxVal : p[i];
        maxloc = (*maxVal > p[i] || p[i + 2 * vlen * groupnum] == -1) ? maxloc : cvRound(p[i + 2 * vlen * groupnum]);
    }

    int pre_rows = src.offset / src.step;
    int pre_cols = (src.offset % src.step) / src.elemSize1();
    int wholecols = src.step / src.elemSize1();
    if ( minLoc )
    {
        if ( minloc >= 0 )
        {
            minLoc->y = minloc / wholecols - pre_rows;
            minLoc->x = minloc % wholecols - pre_cols;
        }
        else
            minLoc->x = minLoc->y = -1;
    }
    if ( maxLoc )
    {
        if ( maxloc >= 0 )
        {
            maxLoc->y = maxloc / wholecols - pre_rows;
            maxLoc->x = maxloc % wholecols - pre_cols;
        }
        else
            maxLoc->x = maxLoc->y = -1;
    }

    openCLSafeCall(clReleaseMemObject(dstBuffer));
}

typedef void (*minMaxLocFunc)(const oclMat &src, double *minVal, double *maxVal,
                              Point *minLoc, Point *maxLoc, const oclMat &mask);

void cv::ocl::minMaxLoc(const oclMat &src, double *minVal, double *maxVal,
                        Point *minLoc, Point *maxLoc, const oclMat &mask)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    static minMaxLocFunc functab[2] =
    {
        arithmetic_minMaxLoc<float>,
        arithmetic_minMaxLoc<double>
    };

    minMaxLocFunc func;
    func = functab[(int)src.clCxt->supportsFeature(FEATURE_CL_DOUBLE)];
    func(src, minVal, maxVal, minLoc, maxLoc, mask);
}

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// countNonZero ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_countNonZero_run(const oclMat &src, cl_mem &dst, int groupnum, String kernelName)
{
    int ochannels = src.oclchannels();
    int all_cols = src.step / src.elemSize();
    int pre_cols = (src.offset % src.step) / src.elemSize();
    int sec_cols = all_cols - (src.offset % src.step + src.cols * src.elemSize() - 1) / src.elemSize() - 1;
    int invalid_cols = pre_cols + sec_cols;
    int cols = all_cols - invalid_cols , elemnum = cols * src.rows;;
    int offset = src.offset / src.elemSize();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const channelMap[] = { " ", " ", "2", "4", "4" };
    String buildOptions = format("-D srcT=%s%s -D dstT=int%s", typeMap[src.depth()], channelMap[ochannels],
                                 channelMap[ochannels]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&invalid_cols ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&offset));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&elemnum));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&groupnum));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst ));

    size_t globalThreads[3] = { groupnum * 256, 1, 1 };
    size_t localThreads[3] = { 256, 1, 1 };

    openCLExecuteKernel(src.clCxt, &arithm_nonzero, kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

int cv::ocl::countNonZero(const oclMat &src)
{
    CV_Assert(src.step % src.elemSize() == 0);
    CV_Assert(src.channels() == 1);

    Context *clCxt = src.clCxt;
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "selected device doesn't support double");
        return -1;
    }

    size_t groupnum = src.clCxt->getDeviceInfo().maxComputeUnits;
    CV_Assert(groupnum != 0);
    int dbsize = groupnum;

    String kernelName = "arithm_op_nonzero";

    AutoBuffer<int> _buf(dbsize);
    int *p = (int*)_buf, nonzero = 0;
    memset(p, 0, dbsize * sizeof(int));

    cl_mem dstBuffer = openCLCreateBuffer(clCxt, CL_MEM_WRITE_ONLY, dbsize * sizeof(int));
    arithmetic_countNonZero_run(src, dstBuffer, groupnum, kernelName);
    openCLReadBuffer(clCxt, dstBuffer, (void *)p, dbsize * sizeof(int));

    for (int i = 0; i < dbsize; i++)
        nonzero += p[i];

    openCLSafeCall(clReleaseMemObject(dstBuffer));

    return nonzero;
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////bitwise_op////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void bitwise_unary_run(const oclMat &src1, oclMat &dst, String kernelName, const cv::ocl::ProgramEntry* source)
{
    dst.create(src1.size(), src1.type());

    int channels = dst.oclchannels();
    int depth = dst.depth();

    int vector_lengths[4][7] = {{4, 4, 4, 4, 1, 1, 1},
        {4, 4, 4, 4, 1, 1, 1},
        {4, 4, 4, 4, 1, 1, 1},
        {4, 4, 4, 4, 1, 1, 1}
    };

    size_t vector_length = vector_lengths[channels - 1][depth];
    int offset_cols = (dst.offset / dst.elemSize1()) & (vector_length - 1);
    int cols = divUp(dst.cols * channels + offset_cols, vector_length);

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, dst.rows, 1 };

    int dst_step1 = dst.cols * dst.elemSize();
    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step1 ));

    openCLExecuteKernel(src1.clCxt, source, kernelName, globalThreads, localThreads, args, -1, depth);
}

enum { AND = 0, OR, XOR };

static void bitwise_binary_run(const oclMat &src1, const oclMat &src2, const Scalar& src3, const oclMat &mask,
                               oclMat &dst, int operationType)
{
    CV_Assert(operationType >= AND && operationType <= XOR);
    CV_Assert(src2.empty() || (!src2.empty() && src1.type() == src2.type() && src1.size() == src2.size()));
    CV_Assert(mask.empty() || (!mask.empty() && mask.type() == CV_8UC1 && mask.size() == src1.size()));

    dst.create(src1.size(), src1.type());
    oclMat m;

    const char operationMap[] = { '&', '|', '^' };
    std::string kernelName("arithm_bitwise_binary");

    int vlen = std::min<int>(8, src1.elemSize1() * src1.oclchannels());
    std::string vlenstr = vlen > 1 ? format("%d", vlen) : "";
    std::string buildOptions = format("-D Operation=%c -D vloadn=vload%s -D vstoren=vstore%s -D elemSize=%d -D vlen=%d"
                                      " -D ucharv=uchar%s",
                                      operationMap[operationType], vlenstr.c_str(), vlenstr.c_str(),
                                      (int)src1.elemSize(), vlen, vlenstr.c_str());

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { dst.cols, dst.rows, 1 };

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));

    if (src2.empty())
    {
        m.create(1, 1, dst.type());
        m.setTo(src3);

        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&m.data ));

        kernelName += "_scalar";
    }
    else
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.step ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2.offset ));
    }

    if (!mask.empty())
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mask.step ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mask.offset ));

        kernelName += "_mask";
    }

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));

    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));

    openCLExecuteKernel(src1.clCxt, mask.empty() ? (!src2.empty() ? &arithm_bitwise_binary : &arithm_bitwise_binary_scalar) :
                                              (!src2.empty() ? &arithm_bitwise_binary_mask : &arithm_bitwise_binary_scalar_mask),
                        kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

void cv::ocl::bitwise_not(const oclMat &src, oclMat &dst)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    dst.create(src.size(), src.type());
    bitwise_unary_run(src, dst, "arithm_bitwise_not", &arithm_bitwise_not);
}

void cv::ocl::bitwise_or(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, src2, Scalar(), mask, dst, OR);
}

void cv::ocl::bitwise_or(const oclMat &src1, const Scalar &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, oclMat(), src2, mask, dst, OR);
}

void cv::ocl::bitwise_and(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, src2, Scalar(), mask, dst, AND);
}

void cv::ocl::bitwise_and(const oclMat &src1, const Scalar &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, oclMat(), src2, mask, dst, AND);
}

void cv::ocl::bitwise_xor(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, src2, Scalar(), mask, dst, XOR);
}

void cv::ocl::bitwise_xor(const oclMat &src1, const Scalar &src2, oclMat &dst, const oclMat &mask)
{
    bitwise_binary_run(src1, oclMat(), src2, mask, dst, XOR);
}

oclMat cv::ocl::operator ~ (const oclMat &src)
{
    return oclMatExpr(src, oclMat(), MAT_NOT);
}

oclMat cv::ocl::operator | (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, MAT_OR);
}

oclMat cv::ocl::operator & (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, MAT_AND);
}

oclMat cv::ocl::operator ^ (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, MAT_XOR);
}

cv::ocl::oclMatExpr cv::ocl::operator + (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, cv::ocl::MAT_ADD);
}

cv::ocl::oclMatExpr cv::ocl::operator - (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, cv::ocl::MAT_SUB);
}

cv::ocl::oclMatExpr cv::ocl::operator * (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, cv::ocl::MAT_MUL);
}

cv::ocl::oclMatExpr cv::ocl::operator / (const oclMat &src1, const oclMat &src2)
{
    return oclMatExpr(src1, src2, cv::ocl::MAT_DIV);
}

void oclMatExpr::assign(oclMat& m) const
{
    switch (op)
    {
        case MAT_ADD:
            add(a, b, m);
            break;
        case MAT_SUB:
            subtract(a, b, m);
            break;
        case MAT_MUL:
            multiply(a, b, m);
            break;
        case MAT_DIV:
            divide(a, b, m);
            break;
        case MAT_NOT:
            bitwise_not(a, m);
            break;
        case MAT_AND:
            bitwise_and(a, b, m);
            break;
        case MAT_OR:
            bitwise_or(a, b, m);
            break;
        case MAT_XOR:
            bitwise_xor(a, b, m);
            break;
    }
}

oclMatExpr::operator oclMat() const
{
    oclMat m;
    assign(m);
    return m;
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////// transpose ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#define TILE_DIM   (32)
#define BLOCK_ROWS (256 / TILE_DIM)

static void transpose_run(const oclMat &src, oclMat &dst, String kernelName, bool inplace = false)
{
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char channelsString[] = { ' ', ' ', '2', '4', '4' };
    std::string buildOptions = format("-D T=%s%c", typeMap[src.depth()],
                                      channelsString[src.channels()]);

    size_t localThreads[3]  = { TILE_DIM, BLOCK_ROWS, 1 };
    size_t globalThreads[3] = { src.cols, inplace ? src.rows : divUp(src.rows, TILE_DIM) * BLOCK_ROWS, 1 };

    int srcstep1 = src.step / src.elemSize(), dststep1 = dst.step / dst.elemSize();
    int srcoffset1 = src.offset / src.elemSize(), dstoffset1 = dst.offset / dst.elemSize();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcstep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&srcoffset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1 ));

    openCLExecuteKernel(src.clCxt, &arithm_transpose, kernelName, globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

void cv::ocl::transpose(const oclMat &src, oclMat &dst)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    if ( src.data == dst.data && src.cols == src.rows && dst.offset == src.offset
         && dst.size() == src.size())
        transpose_run( src, dst, "transpose_inplace", true);
    else
    {
        dst.create(src.cols, src.rows, src.type());
        transpose_run( src, dst, "transpose");
    }
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////// addWeighted ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void cv::ocl::addWeighted(const oclMat &src1, double alpha, const oclMat &src2, double beta, double gama, oclMat &dst)
{
    Context *clCxt = src1.clCxt;
    bool hasDouble = clCxt->supportsFeature(FEATURE_CL_DOUBLE);
    if (!hasDouble && src1.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(src1.size() ==  src2.size() && src1.type() == src2.type());
    dst.create(src1.size(), src1.type());

    int channels = dst.oclchannels();
    int depth = dst.depth();

    int cols1 = src1.cols * channels;
    int src1step1 = src1.step1(), src1offset1 = src1.offset / src1.elemSize1();
    int src2step1 = src2.step1(), src2offset1 = src2.offset / src1.elemSize1();
    int dststep1 = dst.step1(), dstoffset1 = dst.offset / dst.elemSize1();

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D T=%s -D WT=%s -D convertToT=convert_%s%s",
                                      typeMap[depth], hasDouble ? "double" : "float", typeMap[depth],
                                      depth >= CV_32F ? "" : "_sat_rte");

    size_t localThreads[3]  = { 256, 1, 1 };
    size_t globalThreads[3] = { cols1, dst.rows, 1};

    float alpha_f = static_cast<float>(alpha),
            beta_f = static_cast<float>(beta),
            gama_f = static_cast<float>(gama);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1offset1));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src2offset1));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dststep1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dstoffset1));

    if (!hasDouble)
    {
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&alpha_f ));
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&beta_f ));
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&gama_f ));
    }
    else
    {
        args.push_back( std::make_pair( sizeof(cl_double), (void *)&alpha ));
        args.push_back( std::make_pair( sizeof(cl_double), (void *)&beta ));
        args.push_back( std::make_pair( sizeof(cl_double), (void *)&gama ));
    }

    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.rows ));

    openCLExecuteKernel(clCxt, &arithm_addWeighted, "addWeighted", globalThreads, localThreads,
                        args, -1, -1, buildOptions.c_str());
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Pow //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void arithmetic_pow_run(const oclMat &src1, double p, oclMat &dst, String kernelName, const cv::ocl::ProgramEntry* source)
{
    int channels = dst.oclchannels();
    int depth = dst.depth();

    size_t vector_length = 1;
    int offset_cols = ((dst.offset % dst.step) / dst.elemSize1()) & (vector_length - 1);
    int cols = divUp(dst.cols * channels + offset_cols, vector_length);
    int rows = dst.rows;

    size_t localThreads[3]  = { 64, 4, 1 };
    size_t globalThreads[3] = { cols, rows, 1 };

    int dst_step1 = dst.cols * dst.elemSize();
    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src1.offset ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step1 ));

    float pf = static_cast<float>(p);
    if (!src1.clCxt->supportsFeature(FEATURE_CL_DOUBLE))
        args.push_back( std::make_pair( sizeof(cl_float), (void *)&pf ));
    else
        args.push_back( std::make_pair( sizeof(cl_double), (void *)&p ));

    openCLExecuteKernel(src1.clCxt, source, kernelName, globalThreads, localThreads, args, -1, depth);
}

void cv::ocl::pow(const oclMat &x, double p, oclMat &y)
{
    if (!x.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && x.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(x.depth() == CV_32F || x.depth() == CV_64F);
    y.create(x.size(), x.type());

    arithmetic_pow_run(x, p, y, "arithm_pow", &arithm_pow);
}

//////////////////////////////////////////////////////////////////////////////
/////////////////////////////// setIdentity //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void cv::ocl::setIdentity(oclMat& src, const Scalar & scalar)
{
    if (!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.depth() == CV_64F)
    {
        CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
        return;
    }

    CV_Assert(src.step % src.elemSize() == 0);

    int src_step1 = src.step / src.elemSize(), src_offset1 = src.offset / src.elemSize();
    size_t local_threads[] = { 16, 16, 1 };
    size_t global_threads[] = { src.cols, src.rows, 1 };

    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    const char * const channelMap[] = { "", "", "2", "4", "4" };
    String buildOptions = format("-D T=%s%s", typeMap[src.depth()], channelMap[src.oclchannels()]);

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_step1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src_offset1 ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows));

    oclMat sc(1, 1, src.type(), scalar);
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&sc.data ));

    openCLExecuteKernel(src.clCxt, &arithm_setidentity, "setIdentity", global_threads, local_threads,
                        args, -1, -1, buildOptions.c_str());
}

//////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Repeat ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void cv::ocl::repeat(const oclMat & src, int ny, int nx, oclMat & dst)
{
    CV_Assert(nx > 0 && ny > 0);
    dst.create(src.rows * ny, src.cols * nx, src.type());

    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
        {
            Rect roi(x * src.cols, y * src.rows, src.cols, src.rows);
            oclMat hdr = dst(roi);
            src.copyTo(hdr);
        }
}
