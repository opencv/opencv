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

#include "ocl.hpp"

namespace cv{
	namespace ocl{

	extern cl_context ocl_context;
	extern cl_command_queue ocl_cmd_queue;

	OCL_EXPORTS void opticalFlowLK(const OclMat& imgA, const OclMat& imgB, CvSize winSize, OclMat& velX, OclMat& velY){
		
			if(imgA.rows != imgB.rows || imgA.cols != imgB.cols)
				return;

			int size = imgA.rows*imgA.cols;

			cl_program program;
			cl_kernel kernel;
			size_t global_work_size[1];
			size_t local_work_size[1];
			cl_uint size_ret = 0;
			cl_int err;

				
			//, __global const ushort* imWidth, __global const ushort* hradius
			const char* of_kernel_source[] =   {\
				"__kernel void derivatives (__global const uchar *imgA, __global const uchar* imgB, __global float* fx, __global float* fy, __global float* ft, __global float* u, __global float* v)"\
				"{"\
				"int tid = get_global_id(0);"\
				"float Ix = 0.0f, Iy = 0.0f, It = 0.0f;"\
				"ushort imageWidth = 1024;"\
				"ushort half_radius = 1;"\
				"Ix = ((imgA[tid+1] - imgA[tid] +  imgB[tid+1] - imgB[tid] )/2);"\
				"Iy = ((imgA[tid + imageWidth] - imgA[tid] + imgB[tid + imageWidth] - imgB[tid])/2);"	   
				"It = imgB[tid] - imgA[tid];"\
				"fx[tid] = Ix;"\
				"fy[tid] = Iy;"\
				"ft[tid] = It;"\
				"__local float s_data[3];"\
				"float A = 0.0f, B = 0.0f, C = 0.0f, D = 0.0f, E = 0.0f;"\
				"short i = 0;"\
				"for (i = -half_radius; i <= half_radius; i++){"\
				"for (short j = -half_radius; j <= half_radius; j++){"\
				"s_data[0] = fx[tid + i + j*imageWidth];"\
				"s_data[1] = fy[tid + i + j*imageWidth];"\
				"s_data[2] = ft[tid + i + j*imageWidth];"\
				"A = A + powf(s_data[0],2);"\
				"B = B + powf(s_data[1],2);"\
				"C = C + s_data[0] * s_data[1];"\
				"D = D + s_data[0] * s_data[2];"\
				"E = E + s_data[1] * s_data[2];"\
				"}}"\
				"u[tid] =  (D*B - E*C)/(A*B - C*C);"\
				"v[tid] =  (E*A - D*C)/(A*B - C*C);"\
				"}"
    };

			//program = clCreateProgramWithSource(imgA.ocl_context, 1, (const char**)&of_kernel_source, NULL, NULL);
			program = clCreateProgramWithSource(ocl_context, 1, (const char**)&of_kernel_source, NULL, NULL);

			err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Build failed, check the program source...\n",err);
				return;
			}
#endif

			kernel = clCreateKernel(program, "derivatives", NULL);

			//Creating additional temporary buffers fx, fy and ft
			//cl_mem fx = clCreateBuffer(imgA.ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);
			//cl_mem fy = clCreateBuffer(imgA.ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);
			//cl_mem ft = clCreateBuffer(imgA.ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);

			cl_mem fx = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);
			cl_mem fy = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);
			cl_mem ft = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, size*sizeof(cl_float), NULL, NULL);

			//Creating variables for imageWidth and half_window
			ushort x_radius = winSize.width/2;
			ushort cols = (ushort)imgA.cols;
		
			//cl_mem imageWidth = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_ushort), &cols, NULL);
			//cl_mem half_radius = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_ushort), &x_radius, NULL);


			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &imgA.data);
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &imgB.data);
			err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &fx);
			err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &fy);
			err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &ft);
			err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &velX.data);
			err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &velY.data);
			//err = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &imageWidth);
			//err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &half_radius);

#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Failed at setting kernel arguments...\n",err);
			return;
			}
#endif

			global_work_size[0] = size;
			local_work_size[0]= 1;

			//err = clEnqueueNDRangeKernel(imgA.ocl_cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
			err = clEnqueueNDRangeKernel(ocl_cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Kernel execution failed...\n",err);
				return;
			}
#endif
			clReleaseMemObject(fx);
			clReleaseMemObject(fy);
			clReleaseMemObject(ft);
			//clReleaseMemObject(imageWidth);
			//clReleaseMemObject(half_radius);
			clReleaseKernel(kernel);
			clReleaseProgram(program);

		}
	}
}
