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

using namespace std;
using namespace cv;
using namespace cv::ocl;

#if !defined (HAVE_OPENCL)

void cv::ocl::PyrLKOpticalFlow::sparse(const oclMat&, const oclMat&, const oclMat&, oclMat&, oclMat&, oclMat*) {  }
void cv::ocl::PyrLKOpticalFlow::dense(const oclMat&, const oclMat&, oclMat&, oclMat&, oclMat*) {  }

#else /* !defined (HAVE_OPENCL) */

namespace cv
{
    namespace ocl
    {
        ///////////////////////////OpenCL kernel strings///////////////////////////
        extern const char *pyrlk;

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

void calcSharrDeriv_run(const oclMat& src, oclMat& dx_buf, oclMat& dy_buf, oclMat& dIdx, oclMat& dIdy, int cn)
{
    Context  *clCxt = src.clCxt;

    string kernelName = "calcSharrDeriv_vertical";

    size_t localThreads[3]  = { 32, 8, 1 };
    size_t globalThreads[3] = { src.cols, src.rows, 1};

    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx_buf.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy_buf.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.step ));

    openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, src.channels(), src.depth());

	kernelName = "calcSharrDeriv_horizontal";

    vector<pair<size_t , const void *> > args2;
    args2.push_back( make_pair( sizeof(cl_int), (void *)&src.rows ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&src.cols ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args2.push_back( make_pair( sizeof(cl_mem), (void *)&dx_buf.data ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.step ));
    args2.push_back( make_pair( sizeof(cl_mem), (void *)&dy_buf.data ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.step ));
    args2.push_back( make_pair( sizeof(cl_mem), (void *)&dIdx.data ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&dIdx.step ));
    args2.push_back( make_pair( sizeof(cl_mem), (void *)&dIdy.data ));
    args2.push_back( make_pair( sizeof(cl_int), (void *)&dIdy.step ));

    openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args2, src.channels(), src.depth());
}


void cv::ocl::PyrLKOpticalFlow::calcSharrDeriv(const oclMat& src, oclMat& dIdx, oclMat& dIdy)
{
    CV_Assert(src.rows > 1 && src.cols > 1);
    CV_Assert(src.depth() == CV_8U);

    const int cn = src.channels();

    ensureSizeIsEnough(src.size(), CV_MAKETYPE(CV_16S, cn), dx_calcBuf_);
    ensureSizeIsEnough(src.size(), CV_MAKETYPE(CV_16S, cn), dy_calcBuf_);

	calcSharrDeriv_run(src, dx_calcBuf_, dy_calcBuf_, dIdx, dIdy, cn);
}

void cv::ocl::PyrLKOpticalFlow::buildImagePyramid(const oclMat& img0, vector<oclMat>& pyr, bool withBorder)
{
    pyr.resize(maxLevel + 1);

    Size sz = img0.size();

	Mat img0Temp;
	img0.download(img0Temp);
	
	Mat pyrTemp;
	oclMat o;

    for (int level = 0; level <= maxLevel; ++level)
    {
        oclMat temp;

        if (withBorder)
        {
            temp.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, img0.type());
        }
        else
        {
            ensureSizeIsEnough(sz, img0.type(), pyr[level]);
        }

        if (level == 0)
			pyr[level] = img0Temp;
        else
            pyrDown(pyr[level - 1], pyr[level]);

        if (withBorder)
            copyMakeBorder(pyr[level], temp, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_REFLECT_101);

        sz = Size((sz.width + 1) / 2, (sz.height + 1) / 2);

        if (sz.width <= winSize.width || sz.height <= winSize.height)
        {
            maxLevel = level;
            break;
        }
    }
}

namespace
{
    void calcPatchSize(cv::Size winSize, int cn, dim3& block, dim3& patch, bool isDeviceArch11)
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

struct MultiplyScalar
{
    MultiplyScalar(double val_, double scale_) : val(val_), scale(scale_) {}
    double operator ()(double a) const
    {
        return (scale * a * val);
    }
    const double val;
    const double scale;
};

void callF(const oclMat& src, oclMat& dst, MultiplyScalar op, int mask)
{
	Mat srcTemp;
	Mat dstTemp;
	src.download(srcTemp);
	dst.download(dstTemp);

	int i;
	int j;
	int k;
	for(i = 0; i < srcTemp.rows; i++)
	{
		for(j = 0; j < srcTemp.cols; j++)
		{
			for(k = 0; k < srcTemp.channels(); k++)
			{
				((float*)dstTemp.data)[srcTemp.channels() * (i * srcTemp.rows + j) + k] = (float)op(((float*)srcTemp.data)[srcTemp.channels() * (i * srcTemp.rows + j) + k]);
			}
		}
	}

	dst = dstTemp;
}

static inline bool isAligned(const unsigned char* ptr, size_t size)
{
    return reinterpret_cast<size_t>(ptr) % size == 0;
}

static inline bool isAligned(size_t step, size_t size)
{
    return step % size == 0;
}

void callT(const oclMat& src, oclMat& dst, MultiplyScalar op, int mask)
{
    if (!isAligned(src.data, 4 * sizeof(double)) || !isAligned(src.step, 4 * sizeof(double)) || 
        !isAligned(dst.data, 4 * sizeof(double)) || !isAligned(dst.step, 4 * sizeof(double)))
    {
        callF(src, dst, op, mask);
        return;
    }

	Mat srcTemp;
	Mat dstTemp;
	src.download(srcTemp);
	dst.download(dstTemp);

	int x_shifted;

	int i;
	int j;
	for(i = 0; i < srcTemp.rows; i++)
	{
		const double* srcRow = (const double*)srcTemp.data + i * srcTemp.rows;
        double* dstRow = (double*)dstTemp.data + i * dstTemp.rows;;

		for(j = 0; j < srcTemp.cols; j++)
		{
			x_shifted = j * 4;

			if(x_shifted + 4 - 1 < srcTemp.cols)
			{
				dstRow[x_shifted    ] = op(srcRow[x_shifted    ]);
				dstRow[x_shifted + 1] = op(srcRow[x_shifted + 1]);
				dstRow[x_shifted + 2] = op(srcRow[x_shifted + 2]);
				dstRow[x_shifted + 3] = op(srcRow[x_shifted + 3]);
			}
			else
			{
				for (int real_x = x_shifted; real_x < srcTemp.cols; ++real_x)
				{
					((float*)dstTemp.data)[i * srcTemp.rows + real_x] = op(((float*)srcTemp.data)[i * srcTemp.rows + real_x]);
				}
			}
		}
	}
}

void multiply(const oclMat& src1, double val, oclMat& dst, double scale = 1.0f);
void multiply(const oclMat& src1, double val, oclMat& dst, double scale)
{
    MultiplyScalar op(val, scale);
	//if(src1.channels() == 1 && dst.channels() == 1)
	//{
	//    callT(src1, dst, op, 0);
	//}
	//else
	//{
	    callF(src1, dst, op, 0);
	//}
}

cl_mem bindTexture(const oclMat& mat, int depth, int channels)
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
#if CL_VERSION_1_2
    cl_image_desc desc;
    desc.image_type       = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width      = mat.cols;
    desc.image_height     = mat.rows;
    desc.image_depth      = NULL;
    desc.image_array_size = 1;
    desc.image_row_pitch  = 0;
    desc.image_slice_pitch= 0;
    desc.buffer           = NULL;
    desc.num_mip_levels   = 0;
    desc.num_samples      = 0;
	texture = clCreateImage(mat.clCxt->impl->clContext, CL_MEM_READ_WRITE, &format, &desc, NULL, &err); 
#else
    texture = clCreateImage2D(
        mat.clCxt->impl->clContext, 
        CL_MEM_READ_WRITE, 
        &format, 
        mat.cols, 
        mat.rows, 
        0, 
        NULL, 
        &err);
#endif
    size_t origin[] = { 0, 0, 0 }; 
    size_t region[] = { mat.cols, mat.rows, 1 }; 
	clEnqueueCopyBufferToImage(mat.clCxt->impl->clCmdQueue, (cl_mem)mat.data, texture, 0, origin, region, 0, NULL, 0);
    openCLSafeCall(err);

	return texture;
}

void lkSparse_run(oclMat& I, oclMat& J,
    const oclMat& prevPts, oclMat& nextPts, oclMat& status, oclMat* err, bool GET_MIN_EIGENVALS, int ptcount, 
    int level, dim3 block, dim3 patch, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;

    string kernelName = "lkSparse";

	size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { 16 * ptcount, 16, 1};

	int cn = I.channels();

	bool calcErr;
    if (err)
    {
		calcErr = true;
    }
    else
    {
		calcErr = false;
    }
	calcErr = true;

	cl_mem ITex = bindTexture(I, I.depth(), cn);
	cl_mem JTex = bindTexture(J, J.depth(), cn);

    vector<pair<size_t , const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&ITex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&JTex ));

    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&nextPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&nextPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&status.data ));
    //args.push_back( make_pair( sizeof(cl_mem), (void *)&(err->data) ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&level ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.x ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.y ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&GET_MIN_EIGENVALS ));

    openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.channels(), I.depth());
}

void cv::ocl::PyrLKOpticalFlow::sparse(const oclMat& prevImg, const oclMat& nextImg, const oclMat& prevPts, oclMat& nextPts, oclMat& status, oclMat* err)
{
    if (prevPts.empty())
    {
        nextPts.release();
        status.release();
        if (err) err->release();
        return;
    }

    derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);

    iters = std::min(std::max(iters, 0), 100);

    const int cn = prevImg.channels();

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
	//ocl::multiply(temp1, scalar, temp2);
	::multiply(temp1, 1.0f / (1 << maxLevel) / 2.0f, temp2);

    ensureSizeIsEnough(1, prevPts.cols, CV_8UC1, status);
    status.setTo(Scalar::all(1));

    if (err)
        ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, *err);

    // build the image pyramids.

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    if (cn == 1 || cn == 4)
    {
        prevImg.convertTo(prevPyr_[0], CV_32F);
        nextImg.convertTo(nextPyr_[0], CV_32F);
    }
    else
    {
		oclMat buf_;
        cvtColor(prevImg, buf_, COLOR_BGR2BGRA);
        buf_.convertTo(prevPyr_[0], CV_32F);

        cvtColor(nextImg, buf_, COLOR_BGR2BGRA);
        buf_.convertTo(nextPyr_[0], CV_32F);
    }

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    // dI/dx ~ Ix, dI/dy ~ Iy

    for (int level = maxLevel; level >= 0; level--)
    {
		lkSparse_run(prevPyr_[level], nextPyr_[level], 
			prevPts, nextPts, status, level == 0 && err ? err : 0, getMinEigenVals, prevPts.cols,
			level, block, patch, winSize, iters);
    }
}

void lkDense_run(oclMat& I, oclMat& J, oclMat& u, oclMat& v, 
    oclMat& prevU, oclMat& prevV, oclMat* err, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;

    string kernelName = "lkDense";

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { I.cols, I.rows, 1};

	int cn = I.channels();

	bool calcErr;
    if (err)
    {
		calcErr = true;
    }
    else
    {
		calcErr = false;
    }

	cl_mem ITex = bindTexture(I, I.depth(), cn);
	cl_mem JTex = bindTexture(J, J.depth(), cn);

	int2 halfWin = {(winSize.width - 1) / 2, (winSize.height - 1) / 2};
    const int patchWidth  = 16 + 2 * halfWin.x;
    const int patchHeight = 16 + 2 * halfWin.y;
    size_t smem_size = 3 * patchWidth * patchHeight * sizeof(int);

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
	args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
	args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.channels(), I.depth());
}

void cv::ocl::PyrLKOpticalFlow::dense(const oclMat& prevImg, const oclMat& nextImg, oclMat& u, oclMat& v, oclMat* err)
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
    nextImg.convertTo(nextPyr_[0], CV_32F);

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[1]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[1]);
    uPyr_[1].setTo(Scalar::all(0));
    vPyr_[1].setTo(Scalar::all(0));

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

    uPyr_[idx].copyTo(u);
    vPyr_[idx].copyTo(v);
}

#endif /* !defined (HAVE_CUDA) */
